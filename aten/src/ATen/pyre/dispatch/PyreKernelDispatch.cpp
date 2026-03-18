#include <ATen/pyre/dispatch/PyreKernelDispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/pyre/dispatch/PyreKernelCompiler.h>
#include <ATen/pyre/dispatch/PyreStringSplicer.h>
#include <ATen/pyre/PyreTensor.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreRuntime.h>
#include <c10/pyre/impl/PyreStorage.h>
#include <c10/util/Exception.h>

#include <iree/hal/fence.h>
#include <iree/modules/hal/types.h>
#include <iree/vm/api.h>

namespace at::pyre {

void PyreKernelDispatch::invokeAsync(
    CachedKernel* kernel,
    c10::ArrayRef<at::Tensor> inputs,
    at::Tensor& output,
    c10::pyre::PyreStreamContext& stream_ctx) {
  PYRE_LOG(TRACE) << "invokeAsync: " << inputs.size() << " inputs, "
                  << "output shape=" << output.sizes() << "\n";

  auto& runtime = c10::pyre::PyreRuntime::get();
  auto alloc = runtime.hostAllocator();

  // Torch-mlir calling convention (matching fusilli exactly):
  // Sync:  @func(out, in0, in1, workspace) -> ()
  // Async: @func$async(out, in0, in1, workspace, wait_fence, signal_fence) -> ()
  //
  // For now: sync CPU path. Output first, inputs next, null workspace.
  // Async path with fences to follow.

  iree_host_size_t arg_count =
      1 + static_cast<iree_host_size_t>(inputs.size()) + 1;  // out + ins + workspace
  iree_vm_list_t* args = nullptr;
  PYRE_CHECK_OK(iree_vm_list_create(
      iree_vm_make_undefined_type_def(),
      arg_count, alloc, &args));

  // Output buffer view FIRST.
  {
    iree_hal_buffer_view_t* output_view = buildBufferView(output);
    PYRE_LOG(TRACE) << "  out: " << iree_hal_buffer_byte_length(
                           iree_hal_buffer_view_buffer(output_view))
                    << " bytes\n";
    iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(output_view);
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  }

  // Input buffer views.
  for (size_t i = 0; i < inputs.size(); ++i) {
    iree_hal_buffer_view_t* view = buildBufferView(inputs[i]);
    PYRE_LOG(TRACE) << "  in[" << i << "]: "
                    << iree_hal_buffer_byte_length(
                           iree_hal_buffer_view_buffer(view))
                    << " bytes\n";
    iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(view);
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  }

  // Null workspace ref (no transients for elementwise ops).
  {
    iree_vm_ref_t null_ref = iree_vm_ref_null();
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &null_ref));
  }

  PYRE_LOG(TRACE) << "  args packed: " << iree_vm_list_size(args) << "\n";

  // Invoke (sync CPU path — function is void, no rets needed).
  PYRE_CHECK_OK(iree_vm_invoke(
      kernel->context.get(),
      kernel->function,
      IREE_VM_INVOCATION_FLAG_NONE,
      /*policy=*/nullptr,
      args,
      /*outputs=*/nullptr,
      alloc));

  PYRE_LOG(TRACE) << "  invoke OK\n";

  // Record timeline bookkeeping.
  // Sync invoke: the work is done, manually signal the stream semaphore
  // so that downstream synchronize() calls don't hang.
  ++stream_ctx.timepoint;
  auto* sem = stream_ctx.timeline.get();
  uint64_t tp = stream_ctx.timepoint;
  PYRE_CHECK_OK(iree_hal_semaphore_signal(sem, tp));
  for (const auto& input : inputs) {
    auto* ctx = static_cast<c10::pyre::PyreBufferContext*>(
        input.storage().data_ptr().get_context());
    if (ctx) ctx->recordUse(sem, tp);
  }
  auto* out_ctx = static_cast<c10::pyre::PyreBufferContext*>(
      output.storage().data_ptr().get_context());
  if (out_ctx) out_ctx->recordMutation(sem, tp);

  iree_vm_list_release(args);
}

at::Tensor PyreKernelDispatch::dispatch(
    const std::string& op_name,
    const std::string& template_key,
    c10::ArrayRef<at::Tensor> inputs,
    const at::TensorOptions& output_opts) {
  auto* device = c10::pyre::PyreDevice::get(0);
  auto& caps = device->capabilities();

  ArgShapeSpecializer specializer;
  auto decision = specializer.analyze(
      op_name, output_opts.dtype().toScalarType(), inputs, caps);

  auto adapted = applyAdapters(
      std::vector<at::Tensor>(inputs.begin(), inputs.end()),
      decision.arg_adapters);

  auto cache_key = decision.spec_key.toString()
                 + "::" + caps.cacheKey();

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, op_name);

  if (!kernel) {
    TORCH_CHECK(
        PyreKernelCompiler::isAvailable(),
        "pyre: kernel '", op_name, "' not in cache and IREE compiler "
        "not available. Set PYRE_IREE_COMPILE or PYRE_IREE_COMPILER_LIB.");

    auto vmfb = PyreKernelCompiler::compileSync(
        template_key, caps.compilerFlags());
    kernel = cache.store(cache_key, op_name, vmfb);
  }

  auto output_sizes = inputs[0].sizes().vec();
  if (inputs.size() == 2) {
    output_sizes = at::infer_size(inputs[0].sizes(), inputs[1].sizes());
  }

  auto output = at::empty(output_sizes, output_opts);

  c10::pyre::PyreStream stream(
      c10::pyre::getCurrentHostStream(0));
  auto& stream_ctx = stream.context();
  invokeAsync(kernel, adapted, output, stream_ctx);

  return output;
}

} // namespace at::pyre
