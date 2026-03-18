#include <ATen/pyre/dispatch/PyreVMContext.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreRuntime.h>
#include <c10/util/Exception.h>

#include <iree/modules/hal/debugging.h>
#include <iree/modules/hal/module.h>
#include <iree/vm/bytecode/module.h>

namespace at::pyre {

CachedKernel loadKernel(
    const std::vector<uint8_t>& vmfb,
    const std::string& func_name) {
  auto& runtime = c10::pyre::PyreRuntime::get();
  auto* device = c10::pyre::PyreDevice::get(0);
  auto alloc = runtime.hostAllocator();

  PYRE_LOG(INFO) << "loading kernel module, func=" << func_name
                 << " vmfb_size=" << vmfb.size() << "\n";

  CachedKernel kernel;

  // 1. Load the VMFB bytecode into a VM module.
  iree_const_byte_span_t vmfb_span = {
      vmfb.data(), static_cast<iree_host_size_t>(vmfb.size())};
  iree_vm_module_t* bytecode_module = nullptr;
  PYRE_CHECK_OK(iree_vm_bytecode_module_create(
      runtime.instance(),
      IREE_VM_BYTECODE_MODULE_FLAG_NONE,
      vmfb_span,
      /*archive_allocator=*/iree_allocator_null(),
      alloc,
      &bytecode_module));
  kernel.module = c10::pyre::vm_module_ptr::steal(bytecode_module);

  // 2. Create a HAL device group for the target device.
  iree_hal_device_group_t* device_group = nullptr;
  PYRE_CHECK_OK(iree_hal_device_group_create_from_device(
      device->halDevice(), alloc, &device_group));

  // 3. Create the HAL module.
  iree_vm_module_t* hal_module = nullptr;
  PYRE_CHECK_OK(iree_hal_module_create(
      runtime.instance(),
      iree_hal_module_device_policy_default(),
      device_group,
      IREE_HAL_MODULE_FLAG_NONE,
      iree_hal_module_debug_sink_null(),
      alloc,
      &hal_module));

  // device_group is retained by the hal_module, we can release.
  iree_hal_device_group_release(device_group);

  // 4. Create VM context linking HAL module + kernel module.
  iree_vm_module_t* modules[] = {hal_module, bytecode_module};
  iree_vm_context_t* context = nullptr;
  PYRE_CHECK_OK(iree_vm_context_create_with_modules(
      runtime.instance(),
      IREE_VM_CONTEXT_FLAG_NONE,
      /*module_count=*/2, modules,
      alloc, &context));
  kernel.context = c10::pyre::vm_context_ptr::steal(context);

  // Release hal_module (context retains it).
  iree_vm_module_release(hal_module);

  // 5. Resolve the entry point via context (not module lookup).
  // Following fusilli pattern: "module.<func_name>" for sync CPU path.
  std::string full_name = "module." + func_name;
  PYRE_LOG(DEBUG) << "resolving function: " << full_name << "\n";

  iree_string_view_t name_view = {
      full_name.c_str(),
      static_cast<iree_host_size_t>(full_name.size())};

  PYRE_CHECK_OK(iree_vm_context_resolve_function(
      context, name_view, &kernel.function));

  PYRE_LOG(INFO) << "kernel loaded: " << full_name << " (ordinal="
                 << kernel.function.ordinal << ")\n";

  return kernel;
}

} // namespace at::pyre
