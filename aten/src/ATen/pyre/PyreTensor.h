#pragma once

// PyreTensor: Non-owning view over a pyre tensor's HAL buffer.
//
// Provides buffer operations (fill, copy, host transfer) that compose
// through the current stream's timeline semaphore. All operations use
// IREE command buffers with semaphore-based synchronization — no
// assumption that device memory is host-local.
//
// Ownership: the backing buffer is owned by PyTorch's Storage layer.
// The chain is Tensor → Storage → StorageImpl → DataPtr → PyreBufferContext.
// PyreTensor extracts the buffer from DataPtr context for HAL operations
// but does not own it. See docs/design/iree_pytorch_api_mapping.md §3-4.

#include <iree/hal/api.h>

#include <ATen/core/Tensor.h>
#include <c10/pyre/impl/PyreStorage.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreStream.h>

#include <functional>

namespace at::pyre {

class PyreTensor {
 public:
  explicit PyreTensor(const at::Tensor& tensor);

  iree_hal_buffer_t* buffer() const { return ctx_->buffer.get(); }
  c10::pyre::PyreDevice* device() const { return device_; }

  // Fill buffer region with a repeating byte pattern.
  // pattern_length must be 1, 2, or 4.
  void fill(const void* pattern, iree_host_size_t pattern_length,
            iree_device_size_t offset, iree_device_size_t length);

  // Copy from another PyreTensor (device-to-device on same device).
  void copyFrom(const PyreTensor& src,
                iree_device_size_t src_offset,
                iree_device_size_t dst_offset,
                iree_device_size_t length);

  // Write host data into this buffer (host-to-device).
  void updateFromHost(const void* data,
                      iree_device_size_t offset,
                      iree_device_size_t length);

  // Read buffer data to host memory (device-to-host).
  // Synchronizes the current stream before reading.
  void readToHost(void* data,
                  iree_device_size_t offset,
                  iree_device_size_t length);

 private:
  c10::pyre::PyreBufferContext* ctx_;
  c10::pyre::PyreDevice* device_;

  // Submit a one-shot transfer command buffer on the current stream's timeline.
  void submitTransfer(
      const std::function<void(iree_hal_command_buffer_t*)>& record_fn);
};

// Map PyTorch ScalarType to IREE HAL element type.
iree_hal_element_type_t toIreeElementType(c10::ScalarType dtype);

// Build an IREE buffer view wrapping a PyTorch tensor's HAL buffer.
// Lightweight metadata wrapper — no data copy. The underlying buffer
// is shared between the PyTorch tensor and the IREE buffer view.
// Caller owns the returned buffer view (must release when done).
iree_hal_buffer_view_t* buildBufferView(const at::Tensor& tensor);

} // namespace at::pyre
