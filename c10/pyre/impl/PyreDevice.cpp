#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreRuntime.h>

namespace c10::pyre {

// -------------------------------------------------------------------------- //
// DeviceCapabilities
// -------------------------------------------------------------------------- //

DeviceCapabilities::DeviceCapabilities(
    std::string backend, std::string target)
    : backend_(std::move(backend)), target_(std::move(target)) {
  flags_.push_back("--iree-hal-target-backends=" + backend_);
  flags_.push_back("--iree-input-type=torch");
  flags_.push_back("--iree-torch-externalize-transients");
  if (backend_ == "llvm-cpu") {
    flags_.push_back("--iree-llvmcpu-target-cpu=" + target_);
  } else if (backend_ == "rocm") {
    flags_.push_back("--iree-rocm-target-chip=" + target_);
  }
  cache_key_ = backend_ + "-" + target_;
}

int64_t DeviceCapabilities::preferredVectorWidth(c10::ScalarType dtype) const {
  switch (dtype) {
    case c10::ScalarType::Float: return 8;
    case c10::ScalarType::Double: return 4;
    case c10::ScalarType::Int: return 8;
    case c10::ScalarType::Long: return 4;
    case c10::ScalarType::Half:
    case c10::ScalarType::BFloat16: return 16;
    default: return 1;
  }
}

// -------------------------------------------------------------------------- //
// PyreDevice
// -------------------------------------------------------------------------- //

PyreDevice::PyreDevice(iree_hal_device_t* device, iree_hal_driver_t* driver)
    : device_(device),
      driver_(driver),
      allocator_(iree_hal_device_allocator(device)),
      capabilities_("llvm-cpu", "host") {
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
  auto flush_and_sync = [this](PyreStreamContext& ctx) {
    if (!ctx.initialized()) return;
    // Flush any pending command buffer before waiting.
    if (ctx.hasPending()) {
      // Build a temporary PyreStream to flush. We need the Stream value
      // type — use DEFAULT since we're operating directly on the context.
      // The flush path only needs device_index to find the PyreDevice,
      // and we already have it (this).
      pyre_log_status(
          iree_hal_command_buffer_end(ctx.active_cb.get()),
          "ending pending CB during device sync");
      auto* sem = ctx.timeline.get();
      uint64_t signal_value = ++ctx.timepoint;
      iree_hal_semaphore_list_t signal_list = {
          .count = 1, .semaphores = &sem, .payload_values = &signal_value};
      pyre_log_status(
          iree_hal_device_queue_execute(
              device_, ctx.affinity,
              iree_hal_fence_semaphore_list(ctx.active_deps.get()),
              signal_list,
              ctx.active_cb.get(),
              iree_hal_buffer_binding_table_empty(),
              IREE_HAL_EXECUTE_FLAG_NONE),
          "flushing pending CB during device sync");
      ctx.active_cb.reset();
      ctx.active_deps.reset();
    }
    if (ctx.timepoint > 0) {
      pyre_log_status(
          iree_hal_semaphore_wait(
              ctx.timeline.get(), ctx.timepoint,
              iree_infinite_timeout(), 0),
          "failed to sync stream during device teardown");
    }
  };

  flush_and_sync(default_stream_);
  for (auto& ctx : low_priority_streams_) flush_and_sync(ctx);
  for (auto& ctx : high_priority_streams_) flush_and_sync(ctx);
}

PyreDevice* PyreDevice::get(DeviceIndex index) {
  return PyreRuntime::get().device(index);
}

int32_t PyreDevice::deviceCount() {
  return PyreRuntime::get().deviceCount();
}

} // namespace c10::pyre
