#pragma once

// PyreStorage: Buffer context and allocator for pyre device tensors.
//
// Each device type needs a differently configured allocator because they
// use different memory types (HOST_LOCAL vs DEVICE_LOCAL) and buffer
// usage flags. The allocator is parameterized with iree_hal_buffer_params_t.
//
// PyreBufferContext owns the IREE buffer and tracks timeline barriers
// for safe async deallocation. See docs/design/iree_pytorch_api_mapping.md §4.

#include <iree/hal/api.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/macros/Export.h>
#include <c10/pyre/impl/PyreHelpers.h>

#include <vector>

namespace c10::pyre {

class PyreDevice;

// Context attached to each DataPtr. Owns the IREE buffer and tracks
// timeline barriers for safe async deallocation.
//
// Ownership: Storage → StorageImpl → DataPtr → PyreBufferContext.
// The DataPtr deleter destroys this context when the tensor is freed.
struct C10_PYRE_API PyreBufferContext {
  hal_buffer_ptr buffer;
  iree_hal_buffer_mapping_t mapping = {};
  PyreDevice* device = nullptr;  // non-owning back-reference

  // Timeline accounting — tracks async operations on this buffer.
  // Mutation barrier: the last write (readers wait on this).
  iree_hal_semaphore_t* mutation_sem = nullptr;  // non-owning
  uint64_t mutation_timepoint = 0;

  // Use barrier: all reads and writes (writers + destructor wait on this).
  struct UseEntry {
    iree_hal_semaphore_t* sem;  // non-owning
    uint64_t timepoint;
  };
  std::vector<UseEntry> use_entries;

  // Record a write operation on this buffer.
  void recordMutation(iree_hal_semaphore_t* sem, uint64_t timepoint);

  // Record any operation (read or write) on this buffer.
  void recordUse(iree_hal_semaphore_t* sem, uint64_t timepoint);

  ~PyreBufferContext();
};

// Allocator parameterized for a specific device type's memory requirements.
class C10_PYRE_API PyreStorageAllocator final : public Allocator {
 public:
  PyreStorageAllocator(
      DeviceType device_type,
      iree_hal_buffer_params_t buffer_params,
      bool map_on_allocate);

  // Host device allocator (HOST_LOCAL | DEVICE_VISIBLE, mapped).
  static PyreStorageAllocator& hostAllocator();

  DataPtr allocate(size_t n) override;
  DeleterFnPtr raw_deleter() const override;
  void copy_data(void* dest, const void* src, std::size_t count) const override;

 private:
  DeviceType device_type_;
  iree_hal_buffer_params_t buffer_params_;
  bool map_on_allocate_;  // whether to map buffers for host access

  static void deleter(void* ctx);
};

} // namespace c10::pyre
