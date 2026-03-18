#include <c10/pyre/impl/PyreStream.h>
#include <c10/pyre/impl/PyreDevice.h>

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

void PyreStream::synchronize() const {
  auto& ctx = context();
  if (ctx.timepoint == 0) return;
  PYRE_CHECK_OK(iree_hal_semaphore_wait(
      ctx.timeline.get(), ctx.timepoint, iree_infinite_timeout(), 0));
}

} // namespace c10::pyre
