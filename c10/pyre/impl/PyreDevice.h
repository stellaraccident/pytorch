#pragma once

// PyreDevice: Per-device instance owning HAL device, allocator, stream pool,
// and compiler capabilities.
//
// Each accessible device gets its own instance with independent stream
// contexts. The PyreRuntime singleton manages the collection of devices.

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/macros/Export.h>
#include <c10/pyre/impl/PyreHelpers.h>
#include <c10/pyre/impl/PyreStream.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace c10::pyre {

// Per-device compiler flags and cache key. Lives on PyreDevice.
class C10_PYRE_API DeviceCapabilities {
 public:
  DeviceCapabilities(std::string backend, std::string target);

  const std::vector<std::string>& compilerFlags() const { return flags_; }
  const std::string& cacheKey() const { return cache_key_; }
  int64_t preferredVectorWidth(c10::ScalarType dtype) const;

 private:
  std::string backend_;
  std::string target_;
  std::vector<std::string> flags_;
  std::string cache_key_;
};

class C10_PYRE_API PyreDevice {
 public:
  explicit PyreDevice(pyre_device_t device);
  ~PyreDevice();

  PyreDevice(const PyreDevice&) = delete;
  PyreDevice& operator=(const PyreDevice&) = delete;

  // HAL access
  pyre_device_t handle() const { return device_.get(); }
  pyre_allocator_t allocator() const { return allocator_; }

  // Compiler capabilities for this device.
  const DeviceCapabilities& capabilities() const { return capabilities_; }

  // Stream pool — lazily creates stream contexts with timeline semaphores.
  PyreStreamContext& defaultStream() { return default_stream_; }
  PyreStreamContext& streamFromId(StreamId id);
  StreamId getStreamFromPool(bool high_priority = false);

  // Convenience: get device by index (delegates to PyreRuntime).
  static PyreDevice* get(DeviceType type, DeviceIndex index);
  static int32_t deviceCount(DeviceType type);
  static PyreDevice* get(DeviceIndex index = 0);
  static int32_t deviceCount();

 private:
  device_ptr device_;
  pyre_allocator_t allocator_;          // borrowed from device
  DeviceCapabilities capabilities_;

  // Stream pool
  PyreStreamContext default_stream_;
  std::array<PyreStreamContext, kStreamsPerPool> low_priority_streams_;
  std::array<PyreStreamContext, kStreamsPerPool> high_priority_streams_;
  std::atomic<uint32_t> low_priority_next_{0};
  std::atomic<uint32_t> high_priority_next_{0};

  void initStreamContext(PyreStreamContext& ctx);
  void syncAllStreams();
};

} // namespace c10::pyre
