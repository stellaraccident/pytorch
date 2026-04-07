#include <ATen/pyre/PyreTensor.h>
#include <ATen/pyre/dispatch/PyreTypeMapping.h>
#include <c10/pyre/impl/PyreRuntime.h>

#include <cstdint>
#include <cstring>

namespace at::pyre {
namespace {

bool isPyreDeviceType(c10::DeviceType device_type) {
  return device_type == c10::DeviceType::PrivateUse1 ||
      device_type == c10::DeviceType::HIP;
}

} // namespace

PyreTensor::PyreTensor(const at::Tensor& tensor)
    : device_(c10::pyre::PyreDevice::get(
          tensor.device().type(), tensor.device().index())),
      tensor_device_(tensor.device()) {
  TORCH_CHECK(
      isPyreDeviceType(tensor_device_.type()),
      "pyre: expected pyre device tensor");
  ctx_ = static_cast<c10::pyre::PyreBufferContext*>(
      tensor.storage().data_ptr().get_context());
  TORCH_CHECK(ctx_ && ctx_->buffer, "pyre: tensor has no IREE buffer");
}

c10::pyre::PyreStream PyreTensor::currentStream() const {
  return c10::pyre::PyreStream(
      c10::pyre::getCurrentPyreStream(
          tensor_device_.type(), tensor_device_.index()));
}

void PyreTensor::fill(
    const void* pattern, size_t pattern_length,
    size_t offset, size_t length) {
  c10::pyre::PyreStream stream = currentStream();
  PYRE_CHECK_OK(pyre_stream_fill_buffer(
      stream.handle(), buffer(), offset, length, pattern, pattern_length));
  stream.flushIfEager();
  ctx_->recordMutation(stream.timeline(), stream.timepoint());
}

void PyreTensor::copyFrom(
    const PyreTensor& src,
    size_t src_offset,
    size_t dst_offset,
    size_t length) {
  c10::pyre::PyreStream stream = currentStream();
  PYRE_CHECK_OK(pyre_stream_copy_buffer(
      stream.handle(), src.buffer(), src_offset, buffer(), dst_offset,
      length));
  stream.flushIfEager();
  src.ctx_->recordUse(stream.timeline(), stream.timepoint());
  ctx_->recordMutation(stream.timeline(), stream.timepoint());
}

void PyreTensor::updateFromHost(
    const void* data, size_t offset, size_t length) {
  c10::pyre::PyreStream stream = currentStream();

  // update_buffer stores data inline in the command buffer arena. The max
  // depends on the block builder's block size — 2KB on the new local-task
  // (block_builder.c). Use a conservative limit to stay within one block.
  static const size_t kInlineUpdateMax = 1024;
  if (length <= kInlineUpdateMax) {
    PYRE_CHECK_OK(pyre_stream_update_buffer(
        stream.handle(), data, length, buffer(), offset));
  } else {
    pyre_buffer_t staging = nullptr;
    PYRE_CHECK_OK(pyre_buffer_allocate(
        stream.handle(), length,
        PYRE_MEMORY_TYPE_HOST_LOCAL | PYRE_MEMORY_TYPE_DEVICE_VISIBLE,
        PYRE_BUFFER_USAGE_DEFAULT | PYRE_BUFFER_USAGE_MAPPING_SCOPED,
        &staging));
    uint64_t staging_ready = stream.timepoint();
    PYRE_CHECK_OK(
        pyre_semaphore_wait(stream.timeline(), staging_ready, UINT64_MAX));
    void* mapped = nullptr;
    PYRE_CHECK_OK(
        pyre_buffer_map(staging, PYRE_MAP_WRITE, 0, length, &mapped));
    std::memcpy(mapped, data, length);
    PYRE_CHECK_OK(pyre_buffer_unmap(staging));
    PYRE_CHECK_OK(pyre_stream_copy_buffer(
        stream.handle(), staging, 0, buffer(), offset, length));
    pyre_buffer_release(staging);
  }
  stream.flushIfEager();
  ctx_->recordMutation(stream.timeline(), stream.timepoint());
}

void PyreTensor::readToHost(
    void* data, size_t offset, size_t length) {
  c10::pyre::PyreStream stream = currentStream();
  stream.synchronize();
  if (tensor_device_.type() == c10::DeviceType::HIP) {
    // HACK: GPU buffers are DEVICE_LOCAL and not directly mappable here.
    // Replace with an async staging transfer once the runtime exposes one.
    PYRE_CHECK_OK(pyre_synchronous_d2h(
        device_->handle(), buffer(), offset, data, length));
    return;
  }
  void* mapped = nullptr;
  PYRE_CHECK_OK(
      pyre_buffer_map(buffer(), PYRE_MAP_READ, offset, length, &mapped));
  std::memcpy(data, mapped, length);
  PYRE_CHECK_OK(pyre_buffer_unmap(buffer()));
}

// -------------------------------------------------------------------------- //
// Batched copy plan execution (Tier 0 / Tier 1)
// -------------------------------------------------------------------------- //

void executeCopyPlan(
    const CopyPlan& plan,
    pyre_buffer_t src_buffer,
    pyre_buffer_t dst_buffer,
    c10::Device device,
    c10::pyre::PyreBufferContext* src_ctx,
    c10::pyre::PyreBufferContext* dst_ctx) {
  TORCH_CHECK(plan.tier == CopyPlan::kSingleCopy ||
              plan.tier == CopyPlan::kDecomposed,
      "pyre: executeCopyPlan only handles Tier 0/1 plans");
  TORCH_CHECK(!plan.chunks.empty(), "pyre: empty copy plan");

  PYRE_LOG(INFO) << "executeCopyPlan: " << plan.chunks.size()
                 << " chunks, tier=" << static_cast<int>(plan.tier) << "\n";

  // Record all chunks into the current stream and submit once.
  c10::pyre::PyreStream stream(
      c10::pyre::getCurrentPyreStream(device.type(), device.index()));
  for (const auto& chunk : plan.chunks) {
    PYRE_CHECK_OK(pyre_stream_copy_buffer(
        stream.handle(),
        src_buffer,
        static_cast<size_t>(chunk.src_offset),
        dst_buffer,
        static_cast<size_t>(chunk.dst_offset),
        static_cast<size_t>(chunk.length)));
  }
  uint64_t signal_value = stream.flush();

  // Record timeline barriers.
  src_ctx->recordUse(stream.timeline(), signal_value);
  dst_ctx->recordMutation(stream.timeline(), signal_value);
}

// -------------------------------------------------------------------------- //
// Buffer view bridge
// -------------------------------------------------------------------------- //

bool hasPyreBuffer(const at::Tensor& tensor) {
  if (!isPyreDeviceType(tensor.device().type())) return false;
  void* ctx = tensor.storage().data_ptr().get_context();
  if (!ctx) return false;
  auto* pctx = static_cast<c10::pyre::PyreBufferContext*>(ctx);
  return pctx->magic == c10::pyre::PyreBufferContext::kMagic && pctx->buffer;
}

} // namespace at::pyre
