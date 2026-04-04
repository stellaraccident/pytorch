#pragma once

// PyreTensor: Non-owning view over a pyre tensor's HAL buffer.
//
// Provides buffer operations (fill, copy, host transfer) that compose
// through the current stream's timeline semaphore.

#include <ATen/core/Tensor.h>
#include <ATen/pyre/dispatch/StridedCopyPlan.h>
#include <c10/pyre/impl/PyreStorage.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreStream.h>

namespace at::pyre {

class PyreTensor {
 public:
  explicit PyreTensor(const at::Tensor& tensor);

  pyre_buffer_t buffer() const { return ctx_->buffer.get(); }
  c10::pyre::PyreDevice* device() const { return device_; }

  void fill(const void* pattern, size_t pattern_length,
            size_t offset, size_t length);
  void copyFrom(const PyreTensor& src,
                size_t src_offset,
                size_t dst_offset,
                size_t length);
  void updateFromHost(const void* data,
                      size_t offset,
                      size_t length);
  void readToHost(void* data,
                  size_t offset,
                  size_t length);

 private:
  c10::pyre::PyreBufferContext* ctx_;
  c10::pyre::PyreDevice* device_;
};

// Check if a tensor has a real pyre IREE buffer (vs CPU fallback backing).
bool hasPyreBuffer(const at::Tensor& tensor);

// Execute a Tier 0 or Tier 1 copy plan as batched HAL transfer commands.
void executeCopyPlan(
    const CopyPlan& plan,
    pyre_buffer_t src_buffer,
    pyre_buffer_t dst_buffer,
    c10::pyre::PyreDevice* device,
    c10::pyre::PyreBufferContext* src_ctx,
    c10::pyre::PyreBufferContext* dst_ctx);

} // namespace at::pyre
