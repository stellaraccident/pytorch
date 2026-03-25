#include <ATen/pyre/dispatch/PyreAbiConfig.h>
#include <c10/pyre/impl/PyreDevice.h>

namespace at::pyre {

const AbiConfig AbiConfig::kNativeOpaque{Convention::kNative};
const AbiConfig AbiConfig::kEnvelope{Convention::kEnvelope};

c10::ArrayRef<std::string> AbiConfig::compilerFlags() const {
  if (flags_cached_) return cached_flags_;

  auto& base = c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags();
  if (convention_ == Convention::kEnvelope) {
    cached_flags_.assign(base.begin(), base.end());
  } else {
    for (const auto& f : base) {
      if (f.find("--iree-input-type=") != std::string::npos) continue;
      if (f.find("--iree-torch-") != std::string::npos) continue;
      cached_flags_.push_back(f);
    }
    cached_flags_.push_back("--iree-input-type=none");
  }
  flags_cached_ = true;
  return cached_flags_;
}

std::string AbiConfig::resolveFunction(const std::string& func_name) const {
  // All conventions use direct name (no $async suffix).
  return "module." + func_name;
}

} // namespace at::pyre
