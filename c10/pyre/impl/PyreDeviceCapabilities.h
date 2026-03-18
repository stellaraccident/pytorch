#pragma once

// Per-device compiler flags and cache key for kernel compilation.
//
// Each device returns its complete compilation flag vector — there is no
// separate "backend flags" concept. The cacheKey() is a hash of the sorted
// flags, ensuring consistency between compilation and cache lookup.

#include <c10/core/ScalarType.h>
#include <c10/macros/Export.h>

#include <cstdint>
#include <string>
#include <vector>

namespace c10::pyre {

class C10_PYRE_API PyreDeviceCapabilities {
 public:
  // Device driver/arch identifiers (used to construct flags and cache key).
  PyreDeviceCapabilities(
      std::string backend,
      std::string target_cpu);

  // Complete compilation flags for this device. The device is the authority.
  const std::vector<std::string>& compilerFlags() const { return flags_; }

  // Cache key fragment identifying this device category (e.g., "llvm-cpu-znver4").
  // Consistent with flags — same flags always produce the same cache key.
  const std::string& cacheKey() const { return cache_key_; }

  // Shape specialization hints.
  int64_t preferredVectorWidth(c10::ScalarType dtype) const;
  std::vector<int64_t> divisibilityRequirements(
      const std::string& op_name) const;

 private:
  std::string backend_;
  std::string target_cpu_;
  std::vector<std::string> flags_;
  std::string cache_key_;

  void buildFlags();
  void buildCacheKey();
};

} // namespace c10::pyre
