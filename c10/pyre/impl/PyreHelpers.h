#pragma once

// Common IREE C++ interop utilities for pyre.
//
// Adapted from shortfin/support/iree_helpers.h. Provides:
// - RAII wrappers for IREE retain/release types
// - Status checking (throwable and non-throwable contexts)
// - Status string formatting (dynamic allocation, no fixed buffer)

#include <iree/async/util/proactor_pool.h>
#include <iree/hal/api.h>
#include <iree/hal/fence.h>
#include <iree/vm/api.h>

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

#include <cstdlib>
#include <iostream>
#include <string>

namespace c10::pyre {

// -------------------------------------------------------------------------- //
// Logging
//
// Controlled by PYRE_LOG_LEVEL env var:
//   0 or unset: silent (default)
//   1: errors and warnings
//   2: info (dispatch decisions, cache hits/misses, compilation)
//   3: debug (template expansion, MLIR text, arg details)
//   4: trace (every function entry, VM invocation args)
// -------------------------------------------------------------------------- //

enum class PyreLogLevel : int {
  SILENT = 0,
  WARN = 1,
  INFO = 2,
  DEBUG = 3,
  TRACE = 4,
};

inline PyreLogLevel pyreLogLevel() {
  static PyreLogLevel level = [] {
    const char* env = std::getenv("PYRE_LOG_LEVEL");
    if (!env) return PyreLogLevel::SILENT;
    int v = std::atoi(env);
    if (v < 0) v = 0;
    if (v > 4) v = 4;
    return static_cast<PyreLogLevel>(v);
  }();
  return level;
}

inline bool pyreLogEnabled(PyreLogLevel level) {
  return static_cast<int>(pyreLogLevel()) >= static_cast<int>(level);
}

// Usage: PYRE_LOG(INFO) << "cache hit for " << key;
// Compiles to nothing when log level check fails (short-circuit).
#define PYRE_LOG(level)                                             \
  if (::c10::pyre::pyreLogEnabled(::c10::pyre::PyreLogLevel::level)) \
  std::cerr << "[pyre:" #level "] "

// -------------------------------------------------------------------------- //
// Status handling
// -------------------------------------------------------------------------- //

// Format an iree_status_t into a human-readable string.
// Consumes (ignores) the status. Uses dynamic allocation (no fixed buffer).
inline std::string formatIreeStatus(iree_status_t status) {
  if (iree_status_is_ok(status)) return "OK";
  iree_allocator_t alloc = iree_allocator_system();
  char* buf = nullptr;
  iree_host_size_t len = 0;
  if (iree_status_to_string(status, &alloc, &buf, &len)) {
    std::string result(buf, len);
    iree_allocator_free(alloc, buf);
    return result;
  }
  iree_status_ignore(status);
  return "<<could not format iree_status_t>>";
}

// Check an iree_status_t in a throwable context.
// Throws via TORCH_CHECK on failure with a formatted message.
inline void pyre_check_ok(iree_status_t status, const char* expr) {
  if (IREE_LIKELY(iree_status_is_ok(status))) return;
  auto msg = formatIreeStatus(status);
  TORCH_CHECK(false, "pyre: ", expr, " — ", msg);
}

#define PYRE_CHECK_OK(expr) ::c10::pyre::pyre_check_ok((expr), #expr)

// Log an iree_status_t failure in a non-throwable context (destructors).
// Does not throw. Logs the error and consumes the status.
inline void pyre_log_status(iree_status_t status, const char* context) {
  if (IREE_LIKELY(iree_status_is_ok(status))) return;
  auto msg = formatIreeStatus(status);
  LOG(ERROR) << "pyre: " << context << " — " << msg;
}

// -------------------------------------------------------------------------- //
// RAII wrappers for IREE retain/release types
// -------------------------------------------------------------------------- //

template <
    typename T,
    void (*RetainFn)(T*),
    void (*ReleaseFn)(T*)>
class iree_ptr {
 public:
  iree_ptr() : ptr_(nullptr) {}
  explicit iree_ptr(T* owned) : ptr_(owned) {}

  iree_ptr(const iree_ptr& other) : ptr_(other.ptr_) {
    if (ptr_) RetainFn(ptr_);
  }
  iree_ptr(iree_ptr&& other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }

  iree_ptr& operator=(const iree_ptr&) = delete;
  iree_ptr& operator=(iree_ptr&& other) noexcept {
    reset();
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
    return *this;
  }

  ~iree_ptr() { reset(); }

  static iree_ptr steal(T* ptr) { return iree_ptr(ptr); }

  static iree_ptr borrow(T* ptr) {
    if (ptr) RetainFn(ptr);
    return iree_ptr(ptr);
  }

  T** for_output() {
    reset();
    return &ptr_;
  }

  void reset() {
    if (ptr_) {
      ReleaseFn(ptr_);
      ptr_ = nullptr;
    }
  }

  T* get() const { return ptr_; }
  T* release() { T* p = ptr_; ptr_ = nullptr; return p; }
  operator T*() const { return ptr_; }
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  T* ptr_;
};

// IREE HAL types
using hal_device_ptr = iree_ptr<
    iree_hal_device_t,
    iree_hal_device_retain,
    iree_hal_device_release>;

using hal_semaphore_ptr = iree_ptr<
    iree_hal_semaphore_t,
    iree_hal_semaphore_retain,
    iree_hal_semaphore_release>;

using hal_command_buffer_ptr = iree_ptr<
    iree_hal_command_buffer_t,
    iree_hal_command_buffer_retain,
    iree_hal_command_buffer_release>;

using hal_buffer_ptr = iree_ptr<
    iree_hal_buffer_t,
    iree_hal_buffer_retain,
    iree_hal_buffer_release>;

using hal_driver_ptr = iree_ptr<
    iree_hal_driver_t,
    iree_hal_driver_retain,
    iree_hal_driver_release>;

// IREE HAL buffer view
using hal_buffer_view_ptr = iree_ptr<
    iree_hal_buffer_view_t,
    iree_hal_buffer_view_retain,
    iree_hal_buffer_view_release>;

// IREE HAL fence
using hal_fence_ptr = iree_ptr<
    iree_hal_fence_t,
    iree_hal_fence_retain,
    iree_hal_fence_release>;

// IREE VM types
using vm_instance_ptr = iree_ptr<
    iree_vm_instance_t,
    iree_vm_instance_retain,
    iree_vm_instance_release>;

using vm_context_ptr = iree_ptr<
    iree_vm_context_t,
    iree_vm_context_retain,
    iree_vm_context_release>;

using vm_module_ptr = iree_ptr<
    iree_vm_module_t,
    iree_vm_module_retain,
    iree_vm_module_release>;

using vm_list_ptr = iree_ptr<
    iree_vm_list_t,
    iree_vm_list_retain,
    iree_vm_list_release>;

// IREE async types
using proactor_pool_ptr = iree_ptr<
    iree_async_proactor_pool_t,
    iree_async_proactor_pool_retain,
    iree_async_proactor_pool_release>;

} // namespace c10::pyre
