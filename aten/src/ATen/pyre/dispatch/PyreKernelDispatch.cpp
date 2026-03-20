#include <ATen/pyre/dispatch/PyreKernelDispatch.h>
#include <ATen/pyre/PyreTensor.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreRuntime.h>
#include <c10/pyre/impl/PyreStorage.h>

#include <iree/hal/fence.h>
#include <iree/modules/hal/types.h>
#include <iree/vm/api.h>

namespace at::pyre {

namespace {

// Build the wait fence joining all input mutation timelines and
// (for native convention) the output mutation timeline.
iree_hal_fence_t* buildWaitFence(
    c10::ArrayRef<at::Tensor> inputs,
    at::Tensor* output_for_native,
    iree_allocator_t alloc) {
  iree_hal_fence_t* wait = nullptr;
  PYRE_CHECK_OK(iree_hal_fence_create(
      static_cast<iree_host_size_t>(inputs.size() + 2),
      alloc, &wait));
  for (const auto& t : inputs) {
    auto* ctx = static_cast<c10::pyre::PyreBufferContext*>(
        t.storage().data_ptr().get_context());
    if (ctx && ctx->mutation_sem && ctx->mutation_timepoint > 0)
      PYRE_CHECK_OK(iree_hal_fence_insert(
          wait, ctx->mutation_sem, ctx->mutation_timepoint));
  }
  if (output_for_native) {
    auto* ctx = static_cast<c10::pyre::PyreBufferContext*>(
        output_for_native->storage().data_ptr().get_context());
    if (ctx && ctx->mutation_sem && ctx->mutation_timepoint > 0)
      PYRE_CHECK_OK(iree_hal_fence_insert(
          wait, ctx->mutation_sem, ctx->mutation_timepoint));
  }
  return wait;
}

void recordTimeline(
    c10::ArrayRef<at::Tensor> inputs,
    at::Tensor& output,
    iree_hal_semaphore_t* sem,
    uint64_t timepoint) {
  for (const auto& t : inputs) {
    auto* ctx = static_cast<c10::pyre::PyreBufferContext*>(
        t.storage().data_ptr().get_context());
    if (ctx) ctx->recordUse(sem, timepoint);
  }
  auto* out_ctx = static_cast<c10::pyre::PyreBufferContext*>(
      output.storage().data_ptr().get_context());
  if (out_ctx) out_ctx->recordMutation(sem, timepoint);
}

} // namespace

void PyreKernelDispatch::invoke(
    CachedKernel* kernel,
    c10::ArrayRef<at::Tensor> inputs,
    at::Tensor& output,
    c10::pyre::PyreStreamContext& stream_ctx,
    const AbiConfig& abi) {
  PYRE_LOG(TRACE) << "invoke: " << inputs.size() << " inputs, "
                  << "output=" << output.sizes()
                  << " abi=" << (abi.isNative() ? "native" : "torch") << "\n";

  auto alloc = c10::pyre::PyreRuntime::get().hostAllocator();
  bool native = abi.isNative();

  // --- Build args list ---
  // Torch:  [output, inputs..., workspace, wait_fence, signal_fence]
  // Native: [inputs..., wait_fence, signal_fence]
  iree_host_size_t arg_count = native
      ? static_cast<iree_host_size_t>(inputs.size()) + 2
      : 1 + static_cast<iree_host_size_t>(inputs.size()) + 1 + 2;

  iree_vm_list_t* args = nullptr;
  PYRE_CHECK_OK(iree_vm_list_create(
      iree_vm_make_undefined_type_def(), arg_count, alloc, &args));

  auto push_view = [&](const at::Tensor& t) {
    auto view = abi.buildView(t);
    iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(view.release());
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  };

  // Torch convention: output first.
  if (!native) push_view(output);

  // Inputs.
  for (size_t i = 0; i < inputs.size(); ++i)
    push_view(inputs[i]);

  // Torch convention: workspace buffer for transients.
  iree_hal_buffer_t* workspace_buf = nullptr;
  if (!native) {
    if (kernel->has_transients_size) {
      // Query transient workspace size by calling the companion function.
      // It takes the same buffer_view args as the main function.
      iree_vm_list_t* size_args = nullptr;
      PYRE_CHECK_OK(iree_vm_list_create(
          iree_vm_make_undefined_type_def(),
          1 + static_cast<iree_host_size_t>(inputs.size()) + 1 + 2,
          alloc, &size_args));
      // Push same buffer views: output, inputs
      {
        auto ov = abi.buildView(output);
        iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(ov.release());
        PYRE_CHECK_OK(iree_vm_list_push_ref_move(size_args, &ref));
      }
      for (size_t i = 0; i < inputs.size(); ++i) {
        auto iv = abi.buildView(inputs[i]);
        iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(iv.release());
        PYRE_CHECK_OK(iree_vm_list_push_ref_move(size_args, &ref));
      }
      // Null workspace, null fences for the size query
      for (int i = 0; i < 3; ++i) {
        iree_vm_ref_t null_ref = iree_vm_ref_null();
        PYRE_CHECK_OK(iree_vm_list_push_ref_move(size_args, &null_ref));
      }

      iree_vm_list_t* size_rets = nullptr;
      PYRE_CHECK_OK(iree_vm_list_create(
          iree_vm_make_undefined_type_def(), 1, alloc, &size_rets));
      PYRE_CHECK_OK(iree_vm_invoke(
          kernel->context.get(), kernel->transients_size_fn,
          IREE_VM_INVOCATION_FLAG_NONE, nullptr,
          size_args, size_rets, alloc));

      iree_vm_value_t size_val;
      PYRE_CHECK_OK(iree_vm_list_get_value(size_rets, 0, &size_val));
      int64_t transient_size = size_val.i64;
      iree_vm_list_release(size_rets);
      iree_vm_list_release(size_args);

      if (transient_size > 0) {
        PYRE_LOG(WARN) << "kernel requires " << transient_size
                       << " bytes transient workspace\n";
        auto* device = c10::pyre::PyreDevice::get(0);
        iree_hal_buffer_params_t params = {};
        params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
        params.access = IREE_HAL_MEMORY_ACCESS_ALL;
        params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL
                    | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
        PYRE_CHECK_OK(iree_hal_allocator_allocate_buffer(
            device->allocator(), params,
            static_cast<iree_device_size_t>(transient_size),
            &workspace_buf));
        iree_vm_ref_t ref = iree_hal_buffer_move_ref(workspace_buf);
        PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
      } else {
        iree_vm_ref_t null_ref = iree_vm_ref_null();
        PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &null_ref));
      }
    } else {
      iree_vm_ref_t null_ref = iree_vm_ref_null();
      PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &null_ref));
    }
  }

  // Wait fence.
  {
    auto* wait = buildWaitFence(inputs,
        native ? &output : nullptr, alloc);
    iree_vm_ref_t ref = iree_hal_fence_move_ref(wait);
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  }

  // Signal fence.
  uint64_t signal_value = ++stream_ctx.timepoint;
  {
    iree_hal_fence_t* signal = nullptr;
    PYRE_CHECK_OK(iree_hal_fence_create_at(
        stream_ctx.timeline.get(), signal_value, alloc, &signal));
    iree_vm_ref_t ref = iree_hal_fence_move_ref(signal);
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  }

  // --- Invoke ---
  // Torch: void return. Native: returns output buffer_views.
  iree_vm_list_t* rets = nullptr;
  if (native) {
    PYRE_CHECK_OK(iree_vm_list_create(
        iree_vm_make_undefined_type_def(), 1, alloc, &rets));
  }

  PYRE_LOG(TRACE) << "  args=" << iree_vm_list_size(args)
                  << " signal=" << signal_value << "\n";

  PYRE_CHECK_OK(iree_vm_invoke(
      kernel->context.get(), kernel->function,
      IREE_VM_INVOCATION_FLAG_NONE, nullptr,
      args, rets, alloc));

  PYRE_LOG(TRACE) << "  invoke OK\n";

  // --- Timeline bookkeeping ---
  recordTimeline(inputs, output, stream_ctx.timeline.get(), signal_value);

  if (rets) iree_vm_list_release(rets);
  iree_vm_list_release(args);
}

} // namespace at::pyre
