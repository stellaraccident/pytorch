#pragma once

// PyreRuntime: Singleton managing global IREE process state and device
// registry.
//
// Owns the VM instance, driver registry, and collection of PyreDevice
// instances. Per-device state (timelines, allocators) lives in PyreDevice.

#include <iree/hal/drivers/init.h>
#include <iree/modules/hal/module.h>

#include <c10/core/Device.h>
#include <c10/macros/Export.h>
#include <c10/pyre/impl/PyreHelpers.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace c10::pyre {

class PyreDevice;

class C10_PYRE_API PyreRuntime {
 public:
  static PyreRuntime& get();

  PyreRuntime(const PyreRuntime&) = delete;
  PyreRuntime& operator=(const PyreRuntime&) = delete;

  iree_vm_instance_t* instance() const { return instance_.get(); }

  // Central host allocator. All IREE allocation calls should use this
  // so we have one place to swap in a higher-performance allocator.
  iree_allocator_t hostAllocator() const { return host_allocator_; }

  // Device registry
  PyreDevice* device(DeviceIndex index);
  int32_t deviceCount() const;

 private:
  PyreRuntime();
  ~PyreRuntime();

  void initialize();

  iree_allocator_t host_allocator_;
  vm_instance_ptr instance_;
  proactor_pool_ptr proactor_pool_;

  // Owns all device instances.
  std::vector<std::unique_ptr<PyreDevice>> devices_;
  std::vector<hal_driver_ptr> drivers_;
};

} // namespace c10::pyre
