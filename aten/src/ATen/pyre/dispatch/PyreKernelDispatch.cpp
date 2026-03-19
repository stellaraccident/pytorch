#include <ATen/pyre/dispatch/PyreKernelDispatch.h>
#include <ATen/pyre/PyreTensor.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreRuntime.h>
#include <c10/pyre/impl/PyreStorage.h>

#include <iree/hal/fence.h>
#include <iree/modules/hal/types.h>
#include <iree/vm/api.h>

namespace at::pyre {

void PyreKernelDispatch::invoke(
    CachedKernel* kernel,
    c10::ArrayRef<at::Tensor> inputs,
    at::Tensor& output,
    c10::pyre::PyreStreamContext& stream_ctx) {
  PYRE_LOG(TRACE) << "invoke: " << inputs.size() << " inputs, "
                  << "output=" << output.sizes() << "\n";

  auto alloc = c10::pyre::PyreRuntime::get().hostAllocator();

  // Async calling convention ($async entry point):
  //   args = [output, input0, ..., inputN, workspace, wait_fence, signal_fence]
  //   returns void — output written in-place.
  iree_host_size_t arg_count =
      1 + static_cast<iree_host_size_t>(inputs.size()) + 1 + 2;
  iree_vm_list_t* args = nullptr;
  PYRE_CHECK_OK(iree_vm_list_create(
      iree_vm_make_undefined_type_def(), arg_count, alloc, &args));

  // Output first.
  {
    auto view = buildBufferView(output);
    PYRE_LOG(TRACE) << "  out: "
        << iree_hal_buffer_byte_length(iree_hal_buffer_view_buffer(view.get()))
        << " bytes\n";
    iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(view.release());
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  }

  // Inputs.
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto view = buildBufferView(inputs[i]);
    iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(view.release());
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  }

  // Null workspace (no transients for elementwise/matmul ops).
  {
    iree_vm_ref_t null_ref = iree_vm_ref_null();
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &null_ref));
  }

  // Wait fence: join input mutation timelines.
  {
    iree_hal_fence_t* wait_fence = nullptr;
    PYRE_CHECK_OK(iree_hal_fence_create(
        static_cast<iree_host_size_t>(inputs.size() + 1),
        alloc, &wait_fence));
    for (const auto& input : inputs) {
      auto* ctx = static_cast<c10::pyre::PyreBufferContext*>(
          input.storage().data_ptr().get_context());
      if (ctx && ctx->mutation_sem && ctx->mutation_timepoint > 0) {
        PYRE_CHECK_OK(iree_hal_fence_insert(
            wait_fence, ctx->mutation_sem, ctx->mutation_timepoint));
      }
    }
    iree_vm_ref_t ref = iree_hal_fence_move_ref(wait_fence);
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  }

  // Signal fence: advance stream timeline.
  uint64_t signal_value = ++stream_ctx.timepoint;
  {
    iree_hal_fence_t* signal_fence = nullptr;
    PYRE_CHECK_OK(iree_hal_fence_create_at(
        stream_ctx.timeline.get(), signal_value,
        alloc, &signal_fence));
    iree_vm_ref_t ref = iree_hal_fence_move_ref(signal_fence);
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  }

  PYRE_LOG(TRACE) << "  args=" << iree_vm_list_size(args)
                  << " signal=" << signal_value << ", invoking\n";

  // Invoke $async — void return, work scheduled on device queue.
  PYRE_CHECK_OK(iree_vm_invoke(
      kernel->context.get(), kernel->function,
      IREE_VM_INVOCATION_FLAG_NONE, nullptr,
      args, nullptr, alloc));

  PYRE_LOG(TRACE) << "  invoke OK\n";

  // Record timeline bookkeeping.
  auto* sem = stream_ctx.timeline.get();
  for (const auto& input : inputs) {
    auto* ctx = static_cast<c10::pyre::PyreBufferContext*>(
        input.storage().data_ptr().get_context());
    if (ctx) ctx->recordUse(sem, signal_value);
  }
  auto* out_ctx = static_cast<c10::pyre::PyreBufferContext*>(
      output.storage().data_ptr().get_context());
  if (out_ctx) out_ctx->recordMutation(sem, signal_value);

  iree_vm_list_release(args);
}

} // namespace at::pyre
