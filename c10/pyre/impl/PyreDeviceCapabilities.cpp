#include <c10/pyre/impl/PyreDeviceCapabilities.h>
#include <c10/util/Exception.h>

#include <algorithm>

namespace c10::pyre {

PyreDeviceCapabilities::PyreDeviceCapabilities(
    std::string backend,
    std::string target_cpu)
    : backend_(std::move(backend)),
      target_cpu_(std::move(target_cpu)) {
  buildFlags();
  buildCacheKey();
}

void PyreDeviceCapabilities::buildFlags() {
  flags_.clear();
  flags_.push_back("--iree-hal-target-backends=" + backend_);
  flags_.push_back("--iree-input-type=torch");
  flags_.push_back("--iree-torch-externalize-transients");

  if (backend_ == "llvm-cpu") {
    flags_.push_back("--iree-llvmcpu-target-cpu=" + target_cpu_);
  } else if (backend_ == "rocm") {
    // GPU flags will include target chip and runtime flags.
    // For now, target_cpu_ is repurposed as target chip (e.g., "gfx942").
    flags_.push_back("--iree-rocm-target-chip=" + target_cpu_);
  }
}

void PyreDeviceCapabilities::buildCacheKey() {
  // Hash of sorted flags — ensures same flags always produce same key.
  auto sorted = flags_;
  std::sort(sorted.begin(), sorted.end());

  // Build a deterministic key string: "backend-target"
  // This is human-readable and filesystem-safe.
  cache_key_ = backend_ + "-" + target_cpu_;
}

int64_t PyreDeviceCapabilities::preferredVectorWidth(
    c10::ScalarType dtype) const {
  // Conservative defaults for local-task CPU.
  // Future: query from device HAL properties or CPU feature detection.
  switch (dtype) {
    case c10::ScalarType::Float:
      return 8;  // AVX2: 256-bit / 32-bit = 8
    case c10::ScalarType::Double:
      return 4;  // AVX2: 256-bit / 64-bit = 4
    case c10::ScalarType::Int:
      return 8;
    case c10::ScalarType::Long:
      return 4;
    case c10::ScalarType::Half:
    case c10::ScalarType::BFloat16:
      return 16; // AVX2: 256-bit / 16-bit = 16
    default:
      return 1;
  }
}

std::vector<int64_t> PyreDeviceCapabilities::divisibilityRequirements(
    const std::string& /*op_name*/) const {
  // Elementwise ops in Epic 1 don't need divisibility specialization.
  // This becomes important for matmul/conv (tiling dimensions).
  return {};
}

} // namespace c10::pyre
