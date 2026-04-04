#pragma once

// PyreRuntime: Singleton managing global IREE process state and device
// registry.
//
// Owns the VM instance, driver registry, and collection of PyreDevice
// instances. Per-device state (timelines, allocators) lives in PyreDevice.

#include <c10/core/Device.h>
#include <c10/macros/Export.h>
#include <c10/pyre/impl/PyreHelpers.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <array>
#include <memory>
#include <vector>

namespace c10::pyre {

class PyreDevice;

class C10_PYRE_API PyreRuntime {
 public:
  static PyreRuntime& get();

  PyreRuntime(const PyreRuntime&) = delete;
  PyreRuntime& operator=(const PyreRuntime&) = delete;

  // Central host allocator. All IREE allocation calls should use this
  // so we have one place to swap in a higher-performance allocator.
  pyre_host_allocator_t hostAllocator() const { return host_allocator_; }

  // Device registry
  PyreDevice* device(DeviceType type, DeviceIndex index);
  int32_t deviceCount(DeviceType type) const;

  PyreDevice* device(DeviceIndex index);
  int32_t deviceCount() const;

 private:
  PyreRuntime();
  ~PyreRuntime();

  void initialize();
  std::vector<std::unique_ptr<PyreDevice>>& devicesForType(DeviceType type);
  const std::vector<std::unique_ptr<PyreDevice>>& devicesForType(
      DeviceType type) const;

  pyre_host_allocator_t host_allocator_;

  std::array<std::vector<std::unique_ptr<PyreDevice>>, 2> devices_by_type_;
  bool gpu_initialized_ = false;
};

} // namespace c10::pyre
