#include <ATen/pyre/PyreTensor.h>
#include <ATen/pyre/dispatch/PyreTypeMapping.h>
#include <c10/pyre/impl/PyreRuntime.h>

#include <cstring>

namespace at::pyre {

PyreTensor::PyreTensor(const at::Tensor& tensor)
    : device_(c10::pyre::PyreDevice::get(tensor.device().index())) {
  TORCH_CHECK(tensor.is_privateuseone(), "pyre: expected pyre device tensor");
  ctx_ = static_cast<c10::pyre::PyreBufferContext*>(
      tensor.storage().data_ptr().get_context());
  TORCH_CHECK(ctx_ && ctx_->buffer, "pyre: tensor has no IREE buffer");
}

void PyreTensor::submitTransfer(
    const std::function<void(iree_hal_command_buffer_t*)>& record_fn) {
  auto* hal_device = device_->halDevice();

  iree_hal_command_buffer_t* cmd = nullptr;
  PYRE_CHECK_OK(iree_hal_command_buffer_create(
      hal_device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      0, &cmd));

  PYRE_CHECK_OK(iree_hal_command_buffer_begin(cmd));
  record_fn(cmd);
  PYRE_CHECK_OK(iree_hal_command_buffer_end(cmd));

  c10::pyre::PyreStream stream(c10::pyre::getCurrentHostStream(0));
  auto& ctx = stream.context();
  auto* sem = ctx.timeline.get();
  uint64_t wait_value = ctx.timepoint;
  uint64_t signal_value = stream.advance();

  iree_hal_semaphore_list_t wait_list = {
      .count = 1, .semaphores = &sem, .payload_values = &wait_value};
  iree_hal_semaphore_list_t signal_list = {
      .count = 1, .semaphores = &sem, .payload_values = &signal_value};

  PYRE_CHECK_OK(iree_hal_device_queue_execute(
      hal_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      wait_list, signal_list, cmd,
      iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));

  iree_hal_command_buffer_release(cmd);
}

void PyreTensor::fill(
    const void* pattern, iree_host_size_t pattern_length,
    iree_device_size_t offset, iree_device_size_t length) {
  auto target_ref = iree_hal_make_buffer_ref(buffer(), offset, length);
  submitTransfer([&](iree_hal_command_buffer_t* cmd) {
    PYRE_CHECK_OK(iree_hal_command_buffer_fill_buffer(
        cmd, target_ref, pattern, pattern_length, IREE_HAL_FILL_FLAG_NONE));
  });
  auto& sctx = c10::pyre::PyreStream(
      c10::pyre::getCurrentHostStream(0)).context();
  ctx_->recordMutation(sctx.timeline.get(), sctx.timepoint);
}

void PyreTensor::copyFrom(
    const PyreTensor& src,
    iree_device_size_t src_offset,
    iree_device_size_t dst_offset,
    iree_device_size_t length) {
  auto src_ref = iree_hal_make_buffer_ref(src.buffer(), src_offset, length);
  auto dst_ref = iree_hal_make_buffer_ref(buffer(), dst_offset, length);
  submitTransfer([&](iree_hal_command_buffer_t* cmd) {
    PYRE_CHECK_OK(iree_hal_command_buffer_copy_buffer(
        cmd, src_ref, dst_ref, IREE_HAL_COPY_FLAG_NONE));
  });
  auto& sctx = c10::pyre::PyreStream(
      c10::pyre::getCurrentHostStream(0)).context();
  src.ctx_->recordUse(sctx.timeline.get(), sctx.timepoint);
  ctx_->recordMutation(sctx.timeline.get(), sctx.timepoint);
}

void PyreTensor::updateFromHost(
    const void* data, iree_device_size_t offset, iree_device_size_t length) {
  if (length <= IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE) {
    auto target_ref = iree_hal_make_buffer_ref(buffer(), offset, length);
    submitTransfer([&](iree_hal_command_buffer_t* cmd) {
      PYRE_CHECK_OK(iree_hal_command_buffer_update_buffer(
          cmd, data, 0, target_ref, IREE_HAL_UPDATE_FLAG_NONE));
    });
  } else {
    auto* hal_alloc = device_->allocator();
    iree_hal_buffer_params_t params = {};
    params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT |
                   IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                  IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    iree_hal_buffer_t* staging = nullptr;
    PYRE_CHECK_OK(iree_hal_allocator_allocate_buffer(
        hal_alloc, params, length, &staging));
    iree_hal_buffer_mapping_t mapping;
    PYRE_CHECK_OK(iree_hal_buffer_map_range(
        staging, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_WRITE, 0, length, &mapping));
    std::memcpy(mapping.contents.data, data, length);
    iree_hal_buffer_unmap_range(&mapping);
    auto staging_ref = iree_hal_make_buffer_ref(staging, 0, length);
    auto target_ref = iree_hal_make_buffer_ref(buffer(), offset, length);
    submitTransfer([&](iree_hal_command_buffer_t* cmd) {
      PYRE_CHECK_OK(iree_hal_command_buffer_copy_buffer(
          cmd, staging_ref, target_ref, IREE_HAL_COPY_FLAG_NONE));
    });
    iree_hal_buffer_release(staging);
  }
  auto& sctx = c10::pyre::PyreStream(
      c10::pyre::getCurrentHostStream(0)).context();
  ctx_->recordMutation(sctx.timeline.get(), sctx.timepoint);
}

void PyreTensor::readToHost(
    void* data, iree_device_size_t offset, iree_device_size_t length) {
  c10::pyre::PyreStream stream(c10::pyre::getCurrentHostStream(0));
  stream.synchronize();
  iree_hal_buffer_mapping_t mapping;
  PYRE_CHECK_OK(iree_hal_buffer_map_range(
      buffer(), IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, offset, length, &mapping));
  std::memcpy(data, mapping.contents.data, length);
  iree_hal_buffer_unmap_range(&mapping);
}

// -------------------------------------------------------------------------- //
// Buffer view bridge
// -------------------------------------------------------------------------- //

bool hasPyreBuffer(const at::Tensor& tensor) {
  if (!tensor.is_privateuseone()) return false;
  void* ctx = tensor.storage().data_ptr().get_context();
  if (!ctx) return false;
  auto* pctx = static_cast<c10::pyre::PyreBufferContext*>(ctx);
  return pctx->magic == c10::pyre::PyreBufferContext::kMagic && pctx->buffer;
}

c10::pyre::hal_buffer_view_ptr buildBufferView(const at::Tensor& tensor) {
  TORCH_CHECK(hasPyreBuffer(tensor),
      "pyre: tensor has no IREE buffer (CPU fallback tensor?)");

  auto* ctx = static_cast<c10::pyre::PyreBufferContext*>(
      tensor.storage().data_ptr().get_context());
  auto iree_dtype = scalarTypeToHalElement(tensor.scalar_type());
  auto sizes = tensor.sizes();
  std::vector<iree_hal_dim_t> shape(sizes.begin(), sizes.end());

  iree_hal_buffer_view_t* view = nullptr;
  PYRE_CHECK_OK(iree_hal_buffer_view_create(
      ctx->buffer.get(), shape.size(), shape.data(),
      iree_dtype, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      c10::pyre::PyreRuntime::get().hostAllocator(), &view));

  return c10::pyre::hal_buffer_view_ptr::steal(view);
}

} // namespace at::pyre
