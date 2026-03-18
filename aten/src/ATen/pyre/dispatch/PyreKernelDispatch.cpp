#include <ATen/pyre/dispatch/PyreKernelDispatch.h>
#include <ATen/pyre/PyreTensor.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreRuntime.h>
#include <c10/pyre/impl/PyreStorage.h>

#include <iree/modules/hal/types.h>
#include <iree/vm/api.h>

namespace at::pyre {

void PyreKernelDispatch::invoke(
    CachedKernel* kernel,
    c10::ArrayRef<at::Tensor> inputs,
    at::Tensor& output,
    c10::pyre::PyreStreamContext& /*stream_ctx*/) {
  PYRE_LOG(TRACE) << "invoke: " << inputs.size() << " inputs, "
                  << "output=" << output.sizes() << "\n";

  auto alloc = c10::pyre::PyreRuntime::get().hostAllocator();

  // Torch-mlir calling convention (sync CPU path, matching fusilli):
  // args = [output, input0, input1, ..., null_workspace]
  // No fences for sync path. Void return.
  iree_host_size_t arg_count =
      1 + static_cast<iree_host_size_t>(inputs.size()) + 1;
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

  // Null workspace.
  {
    iree_vm_ref_t null_ref = iree_vm_ref_null();
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &null_ref));
  }

  PYRE_LOG(TRACE) << "  args=" << iree_vm_list_size(args) << ", invoking\n";

  // Sync invoke — void return, work completes before return.
  PYRE_CHECK_OK(iree_vm_invoke(
      kernel->context.get(), kernel->function,
      IREE_VM_INVOCATION_FLAG_NONE, nullptr,
      args, nullptr, alloc));

  PYRE_LOG(TRACE) << "  invoke OK\n";

  iree_vm_list_release(args);
}

} // namespace at::pyre
