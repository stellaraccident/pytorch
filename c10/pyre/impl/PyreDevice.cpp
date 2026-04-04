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

PyreDevice::PyreDevice(pyre_device_t device)
    : device_(device_ptr::borrow(device)),
      allocator_(pyre_device_allocator(device)),
      capabilities_([&] {
        pyre_accelerator_type_t type = PYRE_ACCELERATOR_CPU;
        PYRE_CHECK_OK(pyre_device_get_type(device, &type));
        char arch[64] = "host";
        pyre_status_t arch_status = pyre_device_get_property(
            device, PYRE_DEVICE_PROPERTY_ARCHITECTURE, arch, sizeof(arch));
        if (!pyre_status_is_ok(arch_status)) {
          pyre_status_ignore(arch_status);
        }
        return DeviceCapabilities(
            type == PYRE_ACCELERATOR_GPU ? "rocm" : "llvm-cpu",
            arch);
      }()) {
  initStreamContext(default_stream_);
}

PyreDevice::~PyreDevice() {
  syncAllStreams();
}

void PyreDevice::initStreamContext(PyreStreamContext& ctx) {
  pyre_stream_t stream = nullptr;
  PYRE_CHECK_OK(pyre_stream_create(device_.get(), /*flags=*/0, &stream));
  ctx.stream = stream_ptr::steal(stream);

  pyre_semaphore_t timeline = nullptr;
  PYRE_CHECK_OK(pyre_stream_get_semaphore(stream, &timeline));
  ctx.timeline = semaphore_ptr::borrow(timeline);

  pyre_timeline_point_t position{};
  PYRE_CHECK_OK(pyre_stream_get_timeline_position(stream, &position));
  ctx.timepoint = position.value;
  ctx.affinity = 0;
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
  auto flush_and_sync = [](PyreStreamContext& ctx) {
    if (!ctx.initialized()) return;
    pyre_log_status(
        pyre_stream_synchronize(ctx.stream.get()),
        "failed to sync stream during device teardown");
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
