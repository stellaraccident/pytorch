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

void PyreStream::synchronize() const {
  // Flush pending CB before waiting — otherwise we'd wait on an older
  // timepoint while commands are still buffered.
  const_cast<PyreStream*>(this)->flush();

  auto& ctx = context();
  if (ctx.timepoint == 0) return;
  PYRE_CHECK_OK(iree_hal_semaphore_wait(
      ctx.timeline.get(), ctx.timepoint, iree_infinite_timeout(), 0));
}

iree_hal_command_buffer_t* PyreStream::getOrCreateCB(
    iree_hal_command_category_t category,
    iree_hal_fence_t* deps) {
  auto& ctx = context();
  auto* device = PyreDevice::get(stream_.device_index());
  auto alloc = PyreRuntime::get().hostAllocator();

  // Category mismatch — flush the existing CB first.
  if (ctx.hasPending() && ctx.active_category != category) {
    flush();
  }

  // Create new CB if needed.
  if (!ctx.hasPending()) {
    PYRE_CHECK_OK(iree_hal_command_buffer_create(
        device->halDevice(),
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        category,
        ctx.affinity,
        /*binding_capacity=*/0,
        ctx.active_cb.for_output()));
    PYRE_CHECK_OK(iree_hal_command_buffer_begin(ctx.active_cb.get()));

    // Initialize deps fence with the stream's current timepoint so the
    // CB orders after all previously submitted work on this stream.
    PYRE_CHECK_OK(iree_hal_fence_create(
        /*capacity=*/16, alloc, ctx.active_deps.for_output()));
    if (ctx.timepoint > 0) {
      PYRE_CHECK_OK(iree_hal_fence_insert(
          ctx.active_deps.get(), ctx.timeline.get(), ctx.timepoint));
    }
    ctx.active_category = category;
  }

  // Merge additional deps from the caller (e.g. buffer use barriers).
  if (deps) {
    PYRE_CHECK_OK(iree_hal_fence_extend(ctx.active_deps.get(), deps));
  }

  return ctx.active_cb.get();
}

uint64_t PyreStream::flush() {
  auto& ctx = context();
  if (!ctx.hasPending()) return ctx.timepoint;

  auto* device = PyreDevice::get(stream_.device_index());

  PYRE_CHECK_OK(iree_hal_command_buffer_end(ctx.active_cb.get()));

  uint64_t signal_value = ++ctx.timepoint;
  auto* sem = ctx.timeline.get();
  iree_hal_semaphore_list_t signal_list = {
      .count = 1, .semaphores = &sem, .payload_values = &signal_value};

  PYRE_CHECK_OK(iree_hal_device_queue_execute(
      device->halDevice(), ctx.affinity,
      iree_hal_fence_semaphore_list(ctx.active_deps.get()),
      signal_list,
      ctx.active_cb.get(),
      iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));

  ctx.active_cb.reset();
  ctx.active_deps.reset();
  return signal_value;
}

uint64_t PyreStream::flushIfEager() {
  return flush();
}

} // namespace c10::pyre
