#include <ATen/pyre/dispatch/PyreArgShapeSpecializer.h>

namespace at::pyre {

SpecDecision ArgShapeSpecializer::analyze(
    const std::string& /*op_name*/,
    c10::ScalarType /*dtype*/,
    c10::ArrayRef<at::Tensor> inputs,
    const c10::pyre::DeviceCapabilities& /*caps*/) {
  TORCH_CHECK(!inputs.empty(), "pyre: need at least one input");

  std::vector<ArgAdapter> arg_adapters;
  for (const auto& t : inputs)
    arg_adapters.push_back(ArgAdapter::analyze(t));

  return {std::move(arg_adapters)};
}

} // namespace at::pyre
