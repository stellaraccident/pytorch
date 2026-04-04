#include <c10/pyre/impl/PyreStream.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreRuntime.h>

#include <array>

namespace c10::pyre {
namespace {

constexpr size_t kPyreDeviceTypeCount = 2;

size_t pyreDeviceTypeSlot(DeviceType device_type) {
  switch (device_type) {
    case DeviceType::PrivateUse1:
      return 0;
    case DeviceType::HIP:
      return 1;
    default:
      TORCH_CHECK(false, "pyre: unsupported stream device type ", device_type);
  }
}

thread_local std::array<DeviceIndex, kPyreDeviceTypeCount>
    current_device_indices = {0, 0};
thread_local std::array<std::array<StreamId, kMaxPyreDevices>,
                        kPyreDeviceTypeCount>
    current_stream_ids = {};

StreamId& currentPyreStreamId(DeviceType device_type, DeviceIndex device_index) {
  TORCH_CHECK(
      device_index >= 0 && device_index < kMaxPyreDevices,
      "pyre: invalid device index for stream TLS ", device_index);
  return current_stream_ids[pyreDeviceTypeSlot(device_type)]
                           [static_cast<size_t>(device_index)];
}

} // namespace

// Thread-local current stream per device.
// Default stream (StreamId 0) until explicitly changed.

DeviceIndex getCurrentPyreDeviceIndex(DeviceType device_type) {
  return current_device_indices[pyreDeviceTypeSlot(device_type)];
}

void setCurrentPyreDeviceIndex(
    DeviceType device_type, DeviceIndex device_index) {
  TORCH_CHECK(
      device_index >= 0 && device_index < PyreDevice::deviceCount(device_type),
      "pyre: invalid ", device_type, " device index ", device_index);
  current_device_indices[pyreDeviceTypeSlot(device_type)] = device_index;
}

Stream getDefaultPyreStream(DeviceType device_type, DeviceIndex device_index) {
  return Stream(Stream::DEFAULT, Device(device_type, device_index));
}

Stream getCurrentPyreStream(DeviceType device_type, DeviceIndex device_index) {
  return Stream(
      Stream::UNSAFE,
      Device(device_type, device_index),
      currentPyreStreamId(device_type, device_index));
}

void setCurrentPyreStream(Stream stream) {
  currentPyreStreamId(stream.device_type(), stream.device_index()) =
      stream.id();
}

Stream getDefaultHostStream(DeviceIndex device_index) {
  return getDefaultPyreStream(DeviceType::PrivateUse1, device_index);
}

Stream getCurrentHostStream(DeviceIndex device_index) {
  return getCurrentPyreStream(DeviceType::PrivateUse1, device_index);
}

void setCurrentHostStream(Stream stream) {
  setCurrentPyreStream(stream);
}

PyreStream::PyreStream(Stream stream) : stream_(stream) {
  TORCH_CHECK(
      stream_.device_type() == DeviceType::PrivateUse1 ||
          stream_.device_type() == DeviceType::HIP,
      "pyre: expected host or gpu device stream");
}

PyreStreamContext& PyreStream::context() const {
  auto* device = PyreDevice::get(
      stream_.device_type(), stream_.device_index());
  return device->streamFromId(stream_.id());
}

void PyreStream::refreshTimeline() const {
  auto& ctx = context();
  pyre_timeline_point_t position{};
  PYRE_CHECK_OK(
      pyre_stream_get_timeline_position(ctx.stream.get(), &position));
  ctx.timepoint = position.value;
}

uint64_t PyreStream::timepoint() const {
  refreshTimeline();
  return context().timepoint;
}

uint64_t PyreStream::advance() {
  auto& ctx = context();
  uint64_t value = 0;
  PYRE_CHECK_OK(pyre_stream_advance_timeline(ctx.stream.get(), &value));
  ctx.timepoint = value;
  return value;
}

void PyreStream::synchronize() const {
  auto& ctx = context();
  PYRE_CHECK_OK(pyre_stream_synchronize(ctx.stream.get()));
  refreshTimeline();
}

uint64_t PyreStream::flush() {
  auto& ctx = context();
  PYRE_CHECK_OK(pyre_stream_flush(ctx.stream.get()));
  refreshTimeline();
  return ctx.timepoint;
}

uint64_t PyreStream::flushIfEager() {
  return flush();
}

} // namespace c10::pyre
