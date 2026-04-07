#pragma once

// PyreStream: Stream pool and timeline context for pyre devices.
//
// c10::Stream is a value type holding (Device, StreamId). Platform state
// lives in PyreStreamContext structs owned by PyreDevice, indexed by
// StreamId. PyreStream is a wrapper that decodes StreamId to context.
//
// The local-task driver is async — queue_execute returns immediately
// and work executes on a task pool. Streams provide ordered execution
// via timeline semaphores.

#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/macros/Export.h>
#include <c10/pyre/impl/PyreHelpers.h>

#include <cstdint>

namespace c10::pyre {

// Per-stream state: timeline semaphore, current timepoint, and pending
// command buffer for batching native HAL operations.
// Owned by PyreDevice in its stream pool.
struct PyreStreamContext {
  stream_ptr stream;
  semaphore_ptr timeline;
  uint64_t timepoint = 0;
  pyre_queue_affinity_t affinity = 0;

  bool initialized() const { return static_cast<bool>(stream); }
};

// StreamId encoding — matches the CUDA/OpenReg pattern.
//
// Layout: [55 zero bits][5-bit pool index][3-bit type][1-bit ext/native]
//
// Type values:
//   0x0 = DEFAULT (StreamId 0)
//   0x1 = LOW_PRIORITY pool
//   0x2 = HIGH_PRIORITY pool
//   0x7 = EXTERNAL (StreamId is a raw pointer)

static constexpr int kStreamsPerPool = 32;
static constexpr DeviceIndex kMaxPyreDevices = 64;

enum class PyreStreamType : uint8_t {
  DEFAULT = 0x0,
  LOW_PRIORITY = 0x1,
  HIGH_PRIORITY = 0x2,
  EXTERNAL = 0x7,
};

// Encode a pool index and type into a StreamId.
inline StreamId encodeStreamId(PyreStreamType type, uint32_t index) {
  return static_cast<StreamId>(
      (static_cast<int64_t>(index & 0x1F) << 4) |
      (static_cast<int64_t>(type) << 1) |
      1);  // native bit
}

// Decode a StreamId into type and pool index.
inline void decodeStreamId(StreamId id, PyreStreamType& type, uint32_t& index) {
  type = static_cast<PyreStreamType>((id >> 1) & 0x7);
  index = static_cast<uint32_t>((id >> 4) & 0x1F);
}

// Thread-local current stream per device.
C10_PYRE_API DeviceIndex getCurrentPyreDeviceIndex(DeviceType device_type);
C10_PYRE_API void setCurrentPyreDeviceIndex(
    DeviceType device_type, DeviceIndex device_index);
C10_PYRE_API Stream getDefaultPyreStream(
    DeviceType device_type, DeviceIndex device_index = 0);
C10_PYRE_API Stream getCurrentPyreStream(
    DeviceType device_type, DeviceIndex device_index = 0);
C10_PYRE_API void setCurrentPyreStream(Stream stream);

C10_PYRE_API Stream getDefaultHostStream(DeviceIndex device_index = 0);
C10_PYRE_API Stream getCurrentHostStream(DeviceIndex device_index = 0);
C10_PYRE_API void setCurrentHostStream(Stream stream);

// Wrapper around c10::Stream that provides access to the PyreStreamContext.
// Composition, not inheritance — same pattern as CUDAStream.
class C10_PYRE_API PyreStream {
 public:
  explicit PyreStream(Stream stream);

  PyreStreamContext& context() const;

  pyre_stream_t handle() const { return context().stream.get(); }
  pyre_semaphore_t timeline() const { return context().timeline.get(); }
  uint64_t timepoint() const;
  uint64_t advance();
  void synchronize() const;

  // Flush the pending command buffer: end, queue_execute, reset.
  // No-op if no pending CB. Advances the timeline. Returns signal timepoint.
  uint64_t flush();

  // Flush if in EAGER mode (always, for now).
  uint64_t flushIfEager();

  Stream unwrap() const { return stream_; }
  DeviceIndex device_index() const { return stream_.device_index(); }

 private:
  void refreshTimeline() const;

  Stream stream_;
};

} // namespace c10::pyre
