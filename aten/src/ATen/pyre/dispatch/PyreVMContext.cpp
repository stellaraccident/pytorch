#include <ATen/pyre/dispatch/PyreVMContext.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreRuntime.h>
#include <c10/util/Exception.h>

#include <cstdlib>
#include <cstring>

#include <iree/io/file_contents.h>
#include <iree/modules/hal/debugging.h>
#include <iree/modules/hal/module.h>
#include <iree/vm/bytecode/module.h>

namespace at::pyre {

// Wrap in-memory VMFB bytes in an iree_io_file_contents_t with proper
// page alignment, matching what iree_io_file_contents_read produces.
// Ownership of the returned contents transfers to the bytecode module
// via iree_io_file_contents_deallocator.
static iree_io_file_contents_t* wrapVmfbAsFileContents(
    const std::vector<uint8_t>& vmfb,
    iree_allocator_t alloc) {
  // Allocate struct + page-aligned data in one block.
  // Layout: [iree_io_file_contents_t][padding][4096-aligned data][NUL]
  constexpr size_t kAlignment = 4096;
  size_t header_size = sizeof(iree_io_file_contents_t);
  size_t data_offset = (header_size + kAlignment - 1) & ~(kAlignment - 1);
  size_t total_size = data_offset + vmfb.size() + 1;  // +1 for NUL

  void* block = nullptr;
  PYRE_CHECK_OK(iree_allocator_malloc(alloc, total_size, &block));

  auto* contents = static_cast<iree_io_file_contents_t*>(block);
  contents->allocator = alloc;
  contents->buffer.data =
      static_cast<uint8_t*>(block) + data_offset;
  contents->buffer.data_length = vmfb.size();
  contents->mapping = nullptr;

  std::memcpy(contents->buffer.data, vmfb.data(), vmfb.size());
  contents->buffer.data[vmfb.size()] = 0;  // NUL terminator

  return contents;
}

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
  // Wrap in file_contents with page-aligned data — the bytecode module
  // holds a direct pointer into this memory for its lifetime.
  auto* contents = wrapVmfbAsFileContents(vmfb, alloc);
  iree_vm_module_t* bytecode_module = nullptr;
  iree_status_t status = iree_vm_bytecode_module_create(
      runtime.instance(),
      IREE_VM_BYTECODE_MODULE_FLAG_NONE,
      contents->const_buffer,
      iree_io_file_contents_deallocator(contents),
      alloc,
      &bytecode_module);
  if (!iree_status_is_ok(status)) {
    iree_io_file_contents_free(contents);
    PYRE_CHECK_OK(status);
  }
  // Ownership of contents transferred to module on success.
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
  iree_vm_module_release(hal_module);

  // 5. Resolve the entry point via context.
  // Following fusilli: "module.<func_name>" for sync CPU path.
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
