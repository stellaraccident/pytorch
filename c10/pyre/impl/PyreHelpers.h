#pragma once

// Common IREE C++ interop utilities for pyre.
//
// Adapted from shortfin/support/iree_helpers.h. Provides:
// - RAII wrappers for IREE retain/release types
// - Status checking (throwable and non-throwable contexts)
// - Status string formatting (dynamic allocation, no fixed buffer)

#include <pyre_runtime.h>
#include <pyre_runtime_cxx.h>

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
// Uses ternary+voidify to avoid dangling-else binding.
struct PyreLogVoidify { void operator&(std::ostream&) const {} };
#define PYRE_LOG(level)                                               \
  !::c10::pyre::pyreLogEnabled(::c10::pyre::PyreLogLevel::level)     \
      ? (void)0                                                       \
      : ::c10::pyre::PyreLogVoidify() &                               \
            std::cerr << "[pyre:" #level "] "

// -------------------------------------------------------------------------- //
// Status handling
// -------------------------------------------------------------------------- //

// Format a pyre_status_t into a human-readable string.
// Does not consume status.
inline std::string formatPyreStatus(pyre_status_t status) {
  return ::pyre::runtime::format_status(status);
}

// Check a pyre_status_t in a throwable context.
// Consumes the status on failure and throws a formatted TORCH_CHECK.
inline void pyre_check_ok(pyre_status_t status, const char* expr) {
  if (pyre_status_is_ok(status)) return;
  auto msg = formatPyreStatus(status);
  pyre_status_ignore(status);
  TORCH_CHECK(false, "pyre: ", expr, " — ", msg);
}

#define PYRE_CHECK_OK(expr) ::c10::pyre::pyre_check_ok((expr), #expr)

inline void pyre_log_status(pyre_status_t status, const char* context) {
  if (pyre_status_is_ok(status)) return;
  auto msg = formatPyreStatus(status);
  pyre_status_ignore(status);
  LOG(ERROR) << "pyre: " << context << " — " << msg;
}

using ::pyre::runtime::allocator_ptr;
using ::pyre::runtime::buffer_ptr;
using ::pyre::runtime::buffer_view_ptr;
using ::pyre::runtime::device_ptr;
using ::pyre::runtime::fence_ptr;
using ::pyre::runtime::function_ptr;
using ::pyre::runtime::module_ptr;
using ::pyre::runtime::pyre_ptr;
using ::pyre::runtime::semaphore_ptr;
using ::pyre::runtime::stream_ptr;
using ::pyre::runtime::value_list_ptr;

} // namespace c10::pyre
