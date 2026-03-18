#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreRuntime.h>

namespace c10::pyre {

PyreDevice::PyreDevice(iree_hal_device_t* device, iree_hal_driver_t* driver)
    : device_(device),
      driver_(driver),
      allocator_(iree_hal_device_allocator(device)) {
  // Initialize the default stream immediately.
  initStreamContext(default_stream_);
}

PyreDevice::~PyreDevice() {
  syncAllStreams();
  if (device_) iree_hal_device_release(device_);
}

void PyreDevice::initStreamContext(PyreStreamContext& ctx) {
  iree_hal_semaphore_t* sem = nullptr;
  PYRE_CHECK_OK(iree_hal_semaphore_create(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
      &sem));
  ctx.timeline = hal_semaphore_ptr::steal(sem);
  ctx.timepoint = 0;
  ctx.affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
}

PyreStreamContext& PyreDevice::streamFromId(StreamId id) {
  if (id == 0) return default_stream_;

  PyreStreamType type;
  uint32_t index;
  decodeStreamId(id, type, index);

  PyreStreamContext* ctx = nullptr;
  switch (type) {
    case PyreStreamType::DEFAULT:
      return default_stream_;
    case PyreStreamType::LOW_PRIORITY:
      ctx = &low_priority_streams_[index % kStreamsPerPool];
      break;
    case PyreStreamType::HIGH_PRIORITY:
      ctx = &high_priority_streams_[index % kStreamsPerPool];
      break;
    default:
      TORCH_CHECK(false, "pyre: unsupported stream type");
  }

  if (!ctx->initialized()) {
    initStreamContext(*ctx);
  }
  return *ctx;
}

StreamId PyreDevice::getStreamFromPool(bool high_priority) {
  if (high_priority) {
    uint32_t idx = high_priority_next_.fetch_add(1) % kStreamsPerPool;
    return encodeStreamId(PyreStreamType::HIGH_PRIORITY, idx);
  } else {
    uint32_t idx = low_priority_next_.fetch_add(1) % kStreamsPerPool;
    return encodeStreamId(PyreStreamType::LOW_PRIORITY, idx);
  }
}

void PyreDevice::syncAllStreams() {
  // TODO(pyre-workspace-k4t): Join all active semaphores into a single
  // multi-wait instead of waiting sequentially. This requires collecting
  // active semaphores into an iree_hal_semaphore_list_t.
  auto sync_ctx = [](PyreStreamContext& ctx) {
    if (ctx.initialized() && ctx.timepoint > 0) {
      pyre_log_status(
          iree_hal_semaphore_wait(
              ctx.timeline.get(), ctx.timepoint,
              iree_infinite_timeout(), 0),
          "failed to sync stream during device teardown");
    }
  };

  sync_ctx(default_stream_);
  for (auto& ctx : low_priority_streams_) sync_ctx(ctx);
  for (auto& ctx : high_priority_streams_) sync_ctx(ctx);
}

PyreDevice* PyreDevice::get(DeviceIndex index) {
  return PyreRuntime::get().device(index);
}

int32_t PyreDevice::deviceCount() {
  return PyreRuntime::get().deviceCount();
}

} // namespace c10::pyre
