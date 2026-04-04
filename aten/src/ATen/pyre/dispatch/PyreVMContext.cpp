#include <ATen/pyre/dispatch/PyreVMContext.h>
#include <c10/pyre/impl/PyreDevice.h>

namespace at::pyre {

CachedKernel loadKernel(
    std::shared_ptr<CompilerOutput> vmfb,
    const std::string& func_name,
    const AbiConfig& abi) {
  auto* device = c10::pyre::PyreDevice::get(0);

  PYRE_LOG(INFO) << "loading kernel, func=" << func_name
                 << " vmfb=" << vmfb->size() << " bytes\n";

  CachedKernel kernel;
  kernel.vmfb = std::move(vmfb);
  kernel.vmfb->loadInto(device->handle(), kernel.module.for_output());

  std::string full_name = abi.resolveFunction(func_name);
  PYRE_LOG(DEBUG) << "resolving: " << full_name << "\n";
  PYRE_CHECK_OK(pyre_module_lookup_function(
      kernel.module.get(), full_name.c_str(),
      kernel.function.for_output()));

  PYRE_LOG(INFO) << "kernel loaded: " << full_name << "\n";

  // Try to resolve the transients_size companion function.
  std::string size_fn_name = full_name + "_transients_size";
  pyre_status_t status = pyre_module_lookup_function(
      kernel.module.get(), size_fn_name.c_str(),
      kernel.transients_size_fn.for_output());
  if (pyre_status_is_ok(status)) {
    PYRE_LOG(DEBUG) << "resolved transients_size: " << size_fn_name << "\n";
  } else {
    pyre_status_ignore(status);
  }

  return kernel;
}

} // namespace at::pyre
