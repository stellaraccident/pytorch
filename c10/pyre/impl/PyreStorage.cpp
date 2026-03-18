#include <c10/pyre/impl/PyreStorage.h>
#include <c10/pyre/impl/PyreDevice.h>

namespace c10::pyre {

// --- PyreBufferContext ---

void PyreBufferContext::recordMutation(
    iree_hal_semaphore_t* sem, uint64_t timepoint) {
  mutation_sem = sem;
  mutation_timepoint = timepoint;
  recordUse(sem, timepoint);
}

void PyreBufferContext::recordUse(
    iree_hal_semaphore_t* sem, uint64_t timepoint) {
  // Update existing entry for the same semaphore, or add new one.
  for (auto& entry : use_entries) {
    if (entry.sem == sem) {
      if (timepoint > entry.timepoint) {
        entry.timepoint = timepoint;
      }
      return;
    }
  }
  use_entries.push_back({sem, timepoint});
}

PyreBufferContext::~PyreBufferContext() {
  // Wait on all pending uses before releasing the buffer.
  for (auto& entry : use_entries) {
    pyre_log_status(
        iree_hal_semaphore_wait(
            entry.sem, entry.timepoint, iree_infinite_timeout(), 0),
        "waiting on buffer use barrier in destructor");
  }
  if (mapping.contents.data) {
    iree_hal_buffer_unmap_range(&mapping);
  }
  // buffer released by hal_buffer_ptr destructor
}

// --- PyreStorageAllocator ---

PyreStorageAllocator::PyreStorageAllocator(
    DeviceType device_type,
    iree_hal_buffer_params_t buffer_params,
    bool map_on_allocate)
    : device_type_(device_type),
      buffer_params_(buffer_params),
      map_on_allocate_(map_on_allocate) {}

PyreStorageAllocator& PyreStorageAllocator::hostAllocator() {
  static PyreStorageAllocator instance(
      DeviceType::PrivateUse1,
      iree_hal_buffer_params_t{
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT |
                   IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                  IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      },
      /*map_on_allocate=*/true);
  return instance;
}

DataPtr PyreStorageAllocator::allocate(size_t n) {
  if (n == 0) {
    return {nullptr, nullptr, &deleter, Device(device_type_, 0)};
  }

  auto* pyre_device = PyreDevice::get(0);
  auto* hal_alloc = pyre_device->allocator();

  iree_hal_buffer_t* buffer = nullptr;
  PYRE_CHECK_OK(iree_hal_allocator_allocate_buffer(
      hal_alloc, buffer_params_, static_cast<iree_device_size_t>(n), &buffer));

  auto* ctx = new PyreBufferContext();
  ctx->buffer = hal_buffer_ptr::steal(buffer);
  ctx->device = pyre_device;

  void* ptr = nullptr;
  if (map_on_allocate_) {
    PYRE_CHECK_OK(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_ALL, 0,
        static_cast<iree_device_size_t>(n), &ctx->mapping));
    ptr = ctx->mapping.contents.data;
  }

  return {ptr, ctx, &deleter, Device(device_type_, 0)};
}

DeleterFnPtr PyreStorageAllocator::raw_deleter() const {
  // data != context, so raw_allocate/raw_deallocate are not supported.
  return nullptr;
}

void PyreStorageAllocator::copy_data(
    void* dest, const void* src, std::size_t count) const {
  default_copy_data(dest, src, count);
}

void PyreStorageAllocator::deleter(void* ctx) {
  delete static_cast<PyreBufferContext*>(ctx);
}

} // namespace c10::pyre
