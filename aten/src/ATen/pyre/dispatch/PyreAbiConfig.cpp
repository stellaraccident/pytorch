#include <ATen/pyre/dispatch/PyreAbiConfig.h>
namespace at::pyre {

const AbiConfig AbiConfig::kNativeOpaque{Convention::kNative};
const AbiConfig AbiConfig::kEnvelope{Convention::kEnvelope};

std::vector<std::string> AbiConfig::compilerFlags(
    const c10::pyre::PyreDevice& device) const {
  const auto& base = device.capabilities().compilerFlags();
  if (convention_ == Convention::kEnvelope) {
    return std::vector<std::string>(base.begin(), base.end());
  }

  std::vector<std::string> flags;
  flags.reserve(base.size() + 1);
  for (const auto& f : base) {
    if (f.find("--iree-input-type=") != std::string::npos) continue;
    if (f.find("--iree-torch-") != std::string::npos) continue;
    flags.push_back(f);
  }
  flags.push_back("--iree-input-type=none");
  return flags;
}

std::string AbiConfig::resolveFunction(const std::string& func_name) const {
  // All conventions use direct name (no $async suffix).
  return "module." + func_name;
}

} // namespace at::pyre
