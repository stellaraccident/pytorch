#include <ATen/pyre/dispatch/PyreAbiConfig.h>
#include <ATen/pyre/PyreTensor.h>
#include <c10/pyre/impl/PyreDevice.h>

namespace at::pyre {

const AbiConfig AbiConfig::kTorchTyped{
    Convention::kTorch, DtypeMapping::kTyped};
const AbiConfig AbiConfig::kNativeOpaque{
    Convention::kNative, DtypeMapping::kOpaque};

c10::ArrayRef<std::string> AbiConfig::compilerFlags() const {
  if (flags_cached_) return cached_flags_;

  auto& base = c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags();
  if (convention_ == Convention::kTorch) {
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
  if (convention_ == Convention::kTorch)
    return "module." + func_name + "$async";
  return "module." + func_name;
}

c10::pyre::hal_buffer_view_ptr AbiConfig::buildView(
    const at::Tensor& t) const {
  if (dtype_mapping_ == DtypeMapping::kOpaque)
    return buildOpaqueBufferView(t);
  return buildBufferView(t);
}

} // namespace at::pyre
