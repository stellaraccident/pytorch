#include <c10/pyre/impl/PyreRuntime.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/core/thread_pool.h>

#include <iree/hal/drivers/local_task/task_driver.h>
#include <iree/hal/local/loaders/registration/init.h>
#include <iree/task/api.h>


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

  // Create CPU device with explicit task topology.
  //
  // We bypass the flag-based factory (iree_task_executors_create_from_flags)
  // because IREE's cpuinfo is disabled in the pyre build, causing the factory
  // to fall back to 1 worker. Instead, we create the executor directly with
  // a topology sized to the physical core count.
  size_t worker_count = c10::TaskThreadPoolBase::defaultNumThreads();
  PYRE_LOG(INFO) << "task topology: " << worker_count << " workers";

  iree_task_topology_t topology;
  iree_task_topology_initialize_from_group_count(worker_count, &topology);

  iree_task_executor_options_t executor_options;
  iree_task_executor_options_initialize(&executor_options);

  iree_task_executor_t* executor = nullptr;
  PYRE_CHECK_OK(iree_task_executor_create(
      executor_options, &topology, host_allocator_, &executor));
  iree_task_topology_deinitialize(&topology);

  // Create executable loaders.
  iree_hal_executable_loader_t* loaders[8] = {nullptr};
  iree_host_size_t loader_count = 0;
  PYRE_CHECK_OK(iree_hal_create_all_available_executable_loaders(
      /*plugin_manager=*/nullptr,
      IREE_ARRAYSIZE(loaders), &loader_count, loaders,
      host_allocator_));

  // Device allocator (heap-backed for host buffers).
  iree_hal_allocator_t* device_allocator = nullptr;
  PYRE_CHECK_OK(iree_hal_allocator_create_heap(
      iree_make_cstring_view("local"), host_allocator_, host_allocator_,
      &device_allocator));

  // Assemble the local-task driver.
  iree_hal_task_device_params_t task_params;
  iree_hal_task_device_params_initialize(&task_params);

  hal_driver_ptr cpu_driver;
  PYRE_CHECK_OK(iree_hal_task_driver_create(
      iree_make_cstring_view("local-task"),
      &task_params,
      /*queue_count=*/1, &executor,
      loader_count, loaders,
      device_allocator,
      host_allocator_, cpu_driver.for_output()));

  // Driver took ownership references; release ours.
  iree_task_executor_release(executor);
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
  iree_hal_allocator_release(device_allocator);

  // Create device from driver.
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
