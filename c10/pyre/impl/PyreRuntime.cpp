#include <c10/pyre/impl/PyreRuntime.h>
#include <c10/pyre/impl/PyreDevice.h>

namespace c10::pyre {

PyreRuntime& PyreRuntime::get() {
  static PyreRuntime instance;
  return instance;
}

PyreRuntime::PyreRuntime() : host_allocator_(iree_allocator_system()) {
  initialize();
}

PyreRuntime::~PyreRuntime() {
  // Explicit deterministic destruction order:
  // 1. Devices (waits on all streams, releases HAL devices)
  devices_.clear();
  // 2. Drivers
  drivers_.clear();
  // 3. Proactor pool
  proactor_pool_.reset();
  // 4. VM instance
  instance_.reset();
}

void PyreRuntime::initialize() {
  PYRE_CHECK_OK(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));

  PYRE_CHECK_OK(iree_vm_instance_create(
      IREE_VM_TYPE_CAPACITY_DEFAULT, host_allocator_,
      instance_.for_output()));

  PYRE_CHECK_OK(iree_hal_module_register_all_types(instance_.get()));

  uint32_t node_id = 0;
  PYRE_CHECK_OK(iree_async_proactor_pool_create(
      /*node_count=*/1, &node_id,
      iree_async_proactor_pool_options_default(),
      host_allocator_, proactor_pool_.for_output()));

  // Create CPU device (local-task driver).
  hal_driver_ptr cpu_driver;
  PYRE_CHECK_OK(iree_hal_driver_registry_try_create(
      iree_hal_driver_registry_default(),
      iree_make_cstring_view("local-task"),
      host_allocator_, cpu_driver.for_output()));

  iree_hal_device_create_params_t device_params =
      iree_hal_device_create_params_default();
  device_params.proactor_pool = proactor_pool_.get();

  iree_hal_device_t* cpu_hal_device = nullptr;
  PYRE_CHECK_OK(iree_hal_driver_create_default_device(
      cpu_driver.get(), &device_params, host_allocator_, &cpu_hal_device));

  devices_.push_back(
      std::make_unique<PyreDevice>(cpu_hal_device, cpu_driver.get()));
  drivers_.push_back(std::move(cpu_driver));
}

PyreDevice* PyreRuntime::device(DeviceIndex index) {
  TORCH_CHECK(
      index >= 0 && static_cast<size_t>(index) < devices_.size(),
      "pyre: invalid device index ", index,
      ", expected 0..", devices_.size() - 1);
  return devices_[static_cast<size_t>(index)].get();
}

int32_t PyreRuntime::deviceCount() const {
  return static_cast<int32_t>(devices_.size());
}

} // namespace c10::pyre
