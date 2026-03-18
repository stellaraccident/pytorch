#pragma once

// PyreDevice: Per-device instance owning HAL device, allocator, and
// stream pool with timeline semaphores.
//
// Each accessible device gets its own instance with independent stream
// contexts. The PyreRuntime singleton manages the collection of devices.

#include <iree/hal/api.h>

#include <c10/core/Device.h>
#include <c10/macros/Export.h>
#include <c10/pyre/impl/PyreHelpers.h>
#include <c10/pyre/impl/PyreStream.h>

#include <array>
#include <atomic>
#include <cstdint>

namespace c10::pyre {

class C10_PYRE_API PyreDevice {
 public:
  PyreDevice(iree_hal_device_t* device, iree_hal_driver_t* driver);
  ~PyreDevice();

  PyreDevice(const PyreDevice&) = delete;
  PyreDevice& operator=(const PyreDevice&) = delete;

  // HAL access
  iree_hal_device_t* halDevice() const { return device_; }
  iree_hal_allocator_t* allocator() const { return allocator_; }
  iree_hal_driver_t* driver() const { return driver_; }

  // Stream pool — lazily creates stream contexts with timeline semaphores.
  PyreStreamContext& defaultStream() { return default_stream_; }
  PyreStreamContext& streamFromId(StreamId id);
  StreamId getStreamFromPool(bool high_priority = false);

  // Convenience: get device by index (delegates to PyreRuntime).
  static PyreDevice* get(DeviceIndex index = 0);
  static int32_t deviceCount();

 private:
  iree_hal_device_t* device_;
  iree_hal_driver_t* driver_;           // borrowed from runtime
  iree_hal_allocator_t* allocator_;     // borrowed from device

  // Stream pool
  PyreStreamContext default_stream_;
  std::array<PyreStreamContext, kStreamsPerPool> low_priority_streams_;
  std::array<PyreStreamContext, kStreamsPerPool> high_priority_streams_;
  std::atomic<uint32_t> low_priority_next_{0};
  std::atomic<uint32_t> high_priority_next_{0};

  // Lazily initialize a stream context with a new timeline semaphore.
  void initStreamContext(PyreStreamContext& ctx);

  // Wait on all stream timelines (used during teardown).
  void syncAllStreams();
};

} // namespace c10::pyre
