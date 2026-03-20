#include <ATen/pyre/dispatch/PyreVMContext.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreRuntime.h>

#include <iree/modules/hal/debugging.h>
#include <iree/modules/hal/module.h>
#include <iree/vm/bytecode/module.h>

namespace at::pyre {

CachedKernel loadKernel(
    std::shared_ptr<CompilerOutput> vmfb,
    const std::string& func_name,
    const AbiConfig& abi) {
  auto& runtime = c10::pyre::PyreRuntime::get();
  auto* device = c10::pyre::PyreDevice::get(0);
  auto alloc = runtime.hostAllocator();

  PYRE_LOG(INFO) << "loading kernel, func=" << func_name
                 << " vmfb=" << vmfb->size() << " bytes\n";

  CachedKernel kernel;
  kernel.vmfb = std::move(vmfb);  // keep data alive

  // The CachedKernel owns the CompilerOutput which owns the page-aligned
  // VMFB data. Pass null archive_allocator so IREE doesn't try to free it —
  // the data lifetime is managed by kernel.vmfb (shared_ptr).
  auto span = kernel.vmfb->span();
  iree_vm_module_t* bytecode_module = nullptr;
  PYRE_CHECK_OK(iree_vm_bytecode_module_create(
      runtime.instance(),
      IREE_VM_BYTECODE_MODULE_FLAG_NONE,
      span, iree_allocator_null(), alloc,
      &bytecode_module));
  kernel.module = c10::pyre::vm_module_ptr::steal(bytecode_module);

  iree_hal_device_group_t* device_group = nullptr;
  PYRE_CHECK_OK(iree_hal_device_group_create_from_device(
      device->halDevice(), alloc, &device_group));

  iree_vm_module_t* hal_module = nullptr;
  PYRE_CHECK_OK(iree_hal_module_create(
      runtime.instance(),
      iree_hal_module_device_policy_default(),
      device_group, IREE_HAL_MODULE_FLAG_NONE,
      iree_hal_module_debug_sink_null(),
      alloc, &hal_module));
  iree_hal_device_group_release(device_group);

  iree_vm_module_t* modules[] = {hal_module, bytecode_module};
  iree_vm_context_t* context = nullptr;
  PYRE_CHECK_OK(iree_vm_context_create_with_modules(
      runtime.instance(), IREE_VM_CONTEXT_FLAG_NONE,
      2, modules, alloc, &context));
  kernel.context = c10::pyre::vm_context_ptr::steal(context);
  iree_vm_module_release(hal_module);

  std::string full_name = abi.resolveFunction(func_name);
  PYRE_LOG(DEBUG) << "resolving: " << full_name << "\n";
  iree_string_view_t nv = {
      full_name.c_str(), static_cast<iree_host_size_t>(full_name.size())};
  PYRE_CHECK_OK(iree_vm_context_resolve_function(context, nv, &kernel.function));

  PYRE_LOG(INFO) << "kernel loaded: " << full_name
                 << " ordinal=" << kernel.function.ordinal << "\n";
  return kernel;
}

} // namespace at::pyre
