#include <c10/pyre/impl/PyreStorage.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreRuntime.h>
#include <c10/pyre/impl/PyreStream.h>

#include <cstdint>

namespace c10::pyre {

// --- PyreBufferContext ---

void PyreBufferContext::recordMutation(
    pyre_semaphore_t sem, uint64_t timepoint) {
  mutation_sem = semaphore_ptr::borrow(sem);
  mutation_timepoint = timepoint;
  recordUse(sem, timepoint);
}

void PyreBufferContext::recordUse(
    pyre_semaphore_t sem, uint64_t timepoint) {
  // Update existing entry for the same semaphore, or add new one.
  for (auto& entry : use_entries) {
    if (entry.sem.get() == sem) {
      if (timepoint > entry.timepoint) {
        entry.timepoint = timepoint;
      }
      return;
    }
  }
  use_entries.push_back({semaphore_ptr::borrow(sem), timepoint});
}

PyreBufferContext::~PyreBufferContext() {
  if (!buffer) return;

  if (mapped_ptr) {
    pyre_log_status(
        pyre_buffer_unmap(buffer.get()),
        "unmapping buffer in destructor");
  }

  // Synchronous wait on all pending uses before releasing.
  // queue_dealloca is deferred to GPU backends with true async dealloc —
  // on local-task it's synchronous anyway, and advancing the timeline
  // from destructors (which fire at GC time, potentially mid-dispatch)
  // corrupts the timeline ordering.
  for (auto& entry : use_entries) {
    pyre_log_status(
        pyre_semaphore_wait(entry.sem.get(), entry.timepoint, UINT64_MAX),
        "waiting on buffer use barrier in destructor");
  }
  // buffer released by buffer_ptr destructor
}

// --- PyreStorageAllocator ---

PyreStorageAllocator::PyreStorageAllocator(
    DeviceType device_type,
    pyre_buffer_params_t buffer_params,
    bool map_on_allocate)
    : device_type_(device_type),
      buffer_params_(buffer_params),
      map_on_allocate_(map_on_allocate) {}

PyreStorageAllocator& PyreStorageAllocator::hostAllocator() {
  static PyreStorageAllocator instance(
      DeviceType::PrivateUse1,
      pyre_buffer_params_t{
          .type = PYRE_MEMORY_TYPE_HOST_LOCAL |
                  PYRE_MEMORY_TYPE_DEVICE_VISIBLE,
          .access = PYRE_MEMORY_ACCESS_ALL,
          .usage = PYRE_BUFFER_USAGE_DEFAULT |
                   PYRE_BUFFER_USAGE_MAPPING_SCOPED,
          .queue_affinity = 0,
      },
      /*map_on_allocate=*/true);
  return instance;
}

PyreStorageAllocator& PyreStorageAllocator::gpuAllocator() {
  static PyreStorageAllocator instance(
      DeviceType::HIP,
      pyre_buffer_params_t{
          .type = PYRE_MEMORY_TYPE_DEVICE_LOCAL,
          .access = PYRE_MEMORY_ACCESS_ALL,
          .usage = PYRE_BUFFER_USAGE_DEFAULT,
          .queue_affinity = 0,
      },
      /*map_on_allocate=*/false);
  return instance;
}

DataPtr PyreStorageAllocator::allocate(size_t n) {
  DeviceIndex device_index = getCurrentPyreDeviceIndex(device_type_);
  Device device(device_type_, device_index);
  if (n == 0) {
    return {nullptr, nullptr, &deleter, device};
  }

  auto* pyre_device = PyreDevice::get(device_type_, device_index);
  // Stream-ordered allocation via queue_alloca.
  PyreStream stream(getCurrentPyreStream(device_type_, device_index));

  pyre_buffer_t buffer = nullptr;
  PYRE_CHECK_OK(pyre_buffer_allocate(
      stream.handle(),
      n,
      buffer_params_.type,
      buffer_params_.usage,
      &buffer));

  auto* ctx = new PyreBufferContext();
  ctx->buffer = buffer_ptr::steal(buffer);
  ctx->device = pyre_device;
  uint64_t signal_value = stream.timepoint();
  ctx->recordMutation(stream.timeline(), signal_value);

  void* ptr = nullptr;
  if (map_on_allocate_) {
    // Wait for the alloca to commit the buffer before mapping.
    // With async local-task, queue_alloca returns immediately but the
    // buffer isn't committed until the signal semaphore fires.
    PYRE_CHECK_OK(
        pyre_semaphore_wait(stream.timeline(), signal_value, UINT64_MAX));
    PYRE_CHECK_OK(
        pyre_buffer_map(buffer, PYRE_MAP_READ | PYRE_MAP_WRITE, 0, n, &ptr));
    ctx->mapped_ptr = ptr;
  }

  return {ptr, ctx, &deleter, device};
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
