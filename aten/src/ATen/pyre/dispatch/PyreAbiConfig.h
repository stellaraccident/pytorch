#pragma once

// AbiConfig: compiler flag and function resolution selection.
//
// Two conventions:
// - kNativeOpaque: native IREE flags, direct entry points (copy/fill).
// - kEnvelope: torch-mlir flags, direct entry points (all arithmetic ops).

#include <c10/pyre/impl/PyreDevice.h>

#include <string>
#include <vector>

namespace at::pyre {

class AbiConfig {
 public:
  enum class Convention { kNative, kEnvelope };

  // Compiler flags for this ABI on a specific runtime device profile.
  std::vector<std::string> compilerFlags(
      const c10::pyre::PyreDevice& device) const;

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
};

} // namespace at::pyre
