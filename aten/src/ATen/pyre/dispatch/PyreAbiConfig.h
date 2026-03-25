#pragma once

// AbiConfig: compiler flag and function resolution selection.
//
// Two conventions:
// - kNativeOpaque: native IREE flags, direct entry points (copy/fill).
// - kEnvelope: torch-mlir flags, direct entry points (all arithmetic ops).

#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>

#include <string>

namespace at::pyre {

class AbiConfig {
 public:
  enum class Convention { kNative, kEnvelope };

  // Compiler flags for this ABI. Cached after first call.
  c10::ArrayRef<std::string> compilerFlags() const;

  // Resolve exported function name.
  std::string resolveFunction(const std::string& func_name) const;

  bool isNative() const { return convention_ == Convention::kNative; }

  // Native IREE convention (copy/fill kernels).
  static const AbiConfig kNativeOpaque;

  // Envelope convention (all arithmetic/index ops).
  static const AbiConfig kEnvelope;

 private:
  AbiConfig(Convention c) : convention_(c) {}
  Convention convention_;
  mutable c10::SmallVector<std::string, 8> cached_flags_;
  mutable bool flags_cached_ = false;
};

} // namespace at::pyre
