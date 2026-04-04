#include <c10/pyre/impl/PyreStream.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreRuntime.h>

namespace c10::pyre {

// Thread-local current stream per device.
// Default stream (StreamId 0) until explicitly changed.
static thread_local StreamId current_stream_id = 0;

Stream getDefaultHostStream(DeviceIndex device_index) {
  return Stream(Stream::DEFAULT, Device(DeviceType::PrivateUse1, device_index));
}

Stream getCurrentHostStream(DeviceIndex device_index) {
  return Stream(
      Stream::UNSAFE,
      Device(DeviceType::PrivateUse1, device_index),
      current_stream_id);
}

void setCurrentHostStream(Stream stream) {
  current_stream_id = stream.id();
}

PyreStream::PyreStream(Stream stream) : stream_(stream) {
  TORCH_CHECK(stream_.device_type() == DeviceType::PrivateUse1,
      "pyre: expected host device stream");
}

PyreStreamContext& PyreStream::context() const {
  auto* device = PyreDevice::get(stream_.device_index());
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
