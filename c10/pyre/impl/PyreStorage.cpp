#include <c10/pyre/impl/PyreStorage.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreRuntime.h>
#include <c10/pyre/impl/PyreStream.h>

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
  if (!buffer) return;

  if (mapping.contents.data) {
    iree_hal_buffer_unmap_range(&mapping);
  }

  // Check if this buffer was allocated via queue_alloca.
  iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(buffer.get());

  if (iree_all_bits_set(placement.flags,
      IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS) && device) {
    // Stream-ordered deallocation: non-blocking.
    auto* hal_device = device->halDevice();
    auto alloc = PyreRuntime::get().hostAllocator();

    // Build wait fence from use entries.
    iree_hal_fence_t* wait_fence = nullptr;
    iree_status_t status = iree_hal_fence_create(
        static_cast<iree_host_size_t>(use_entries.size() + 1),
        alloc, &wait_fence);
    if (iree_status_is_ok(status)) {
      for (auto& entry : use_entries) {
        iree_hal_fence_insert(wait_fence, entry.sem, entry.timepoint);
      }

      // Signal on default stream — deallocation is not latency-sensitive.
      auto& default_ctx = device->defaultStream();
      uint64_t signal_value = ++default_ctx.timepoint;
      auto* sem = default_ctx.timeline.get();
      iree_hal_semaphore_list_t signal_list = {
          .count = 1, .semaphores = &sem, .payload_values = &signal_value};

      status = iree_hal_device_queue_dealloca(
          hal_device,
          IREE_HAL_QUEUE_AFFINITY_ANY,
          iree_hal_fence_semaphore_list(wait_fence),
          signal_list,
          buffer.get(),
          IREE_HAL_DEALLOCA_FLAG_NONE);
      iree_hal_fence_release(wait_fence);
    }

    if (!iree_status_is_ok(status)) {
      // Fallback: synchronous wait + release.
      pyre_log_status(status, "queue_dealloca failed, falling back");
      for (auto& entry : use_entries) {
        iree_hal_semaphore_wait(
            entry.sem, entry.timepoint, iree_infinite_timeout(), 0);
      }
    }
  } else {
    // Synchronous fallback for non-async buffers.
    for (auto& entry : use_entries) {
      pyre_log_status(
          iree_hal_semaphore_wait(
              entry.sem, entry.timepoint, iree_infinite_timeout(), 0),
          "waiting on buffer use barrier in destructor");
    }
  }
  // buffer released by hal_buffer_ptr destructor after dealloca queued.
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
  auto* hal_device = pyre_device->halDevice();

  // Stream-ordered allocation via queue_alloca.
  PyreStream stream(getCurrentHostStream(0));
  auto& stream_ctx = stream.context();

  // Flush pending CB — alloca is a queue op, not a CB command.
  stream.flush();

  uint64_t wait_value = stream_ctx.timepoint;
  auto* sem = stream_ctx.timeline.get();
  iree_hal_semaphore_list_t wait_list = {
      .count = (wait_value > 0) ? 1u : 0u,
      .semaphores = &sem,
      .payload_values = &wait_value};

  uint64_t signal_value = ++stream_ctx.timepoint;
  iree_hal_semaphore_list_t signal_list = {
      .count = 1, .semaphores = &sem, .payload_values = &signal_value};

  iree_hal_buffer_t* buffer = nullptr;
  PYRE_CHECK_OK(iree_hal_device_queue_alloca(
      hal_device,
      stream_ctx.affinity,
      wait_list, signal_list,
      IREE_HAL_ALLOCATOR_POOL_DEFAULT,
      buffer_params_,
      static_cast<iree_device_size_t>(n),
      IREE_HAL_ALLOCA_FLAG_NONE,
      &buffer));

  auto* ctx = new PyreBufferContext();
  ctx->buffer = hal_buffer_ptr::steal(buffer);
  ctx->device = pyre_device;
  ctx->recordMutation(sem, signal_value);

  void* ptr = nullptr;
  if (map_on_allocate_) {
    // For HOST_LOCAL buffers, map immediately. On local-task, alloca
    // completed synchronously so the buffer is ready. On GPU, HOST_LOCAL
    // buffers are in host memory and immediately mappable.
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
