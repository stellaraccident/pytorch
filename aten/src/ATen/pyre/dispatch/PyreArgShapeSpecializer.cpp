#include <ATen/pyre/dispatch/PyreArgShapeSpecializer.h>

namespace at::pyre {

SpecDecision ArgShapeSpecializer::analyze(
    const std::string& op_name,
    c10::ScalarType dtype,
    c10::ArrayRef<at::Tensor> inputs,
    const c10::pyre::PyreDeviceCapabilities& device_caps) {
  TORCH_CHECK(!inputs.empty(), "pyre: need at least one input");

  int64_t rank = inputs[0].dim();

  // Build per-arg shape info and adapters.
  std::vector<ArgShapeInfo> arg_shapes;
  std::vector<ArgAdapter> arg_adapters;
  arg_shapes.reserve(inputs.size());
  arg_adapters.reserve(inputs.size());

  for (const auto& t : inputs) {
    ArgShapeInfo info;
    info.rank = t.dim();
    info.dims = buildDimSpecs(t, device_caps, op_name);
    info.sizes = t.sizes().vec();
    arg_shapes.push_back(std::move(info));
    arg_adapters.push_back(ArgAdapter::analyze(t));
  }

  // Detect broadcasting (binary ops).
  std::vector<BroadcastEntry> broadcast_mask;
  if (inputs.size() == 2) {
    broadcast_mask = detectBroadcast(inputs[0], inputs[1]);
  }

  // Use the output rank for the dim_specs in the spec key.
  // For binary ops, output rank = max input rank (after broadcast).
  std::vector<DimSpec> dim_specs;
  if (!arg_shapes.empty()) {
    dim_specs = arg_shapes[0].dims;
  }

  PyreSpecKey spec_key(
      op_name, dtype, rank,
      std::move(dim_specs), std::move(broadcast_mask));

  return {std::move(spec_key), std::move(arg_shapes), std::move(arg_adapters)};
}

std::vector<BroadcastEntry> ArgShapeSpecializer::detectBroadcast(
    const at::Tensor& lhs, const at::Tensor& rhs) {
  std::vector<BroadcastEntry> mask;
  auto lsizes = lhs.sizes();
  auto rsizes = rhs.sizes();
  int64_t ndim = std::max(lsizes.size(), rsizes.size());

  for (int64_t d = 0; d < ndim; ++d) {
    // Align from the right (broadcasting semantics).
    int64_t li = static_cast<int64_t>(lsizes.size()) - ndim + d;
    int64_t ri = static_cast<int64_t>(rsizes.size()) - ndim + d;
    int64_t ls = (li >= 0) ? lsizes[li] : 1;
    int64_t rs = (ri >= 0) ? rsizes[ri] : 1;

    if (ls == 1 && rs != 1) {
      mask.push_back({0, static_cast<int>(d)});
    } else if (rs == 1 && ls != 1) {
      mask.push_back({1, static_cast<int>(d)});
    }
  }
  return mask;
}

std::vector<DimSpec> ArgShapeSpecializer::buildDimSpecs(
    const at::Tensor& tensor,
    const c10::pyre::PyreDeviceCapabilities& /*device_caps*/,
    const std::string& /*op_name*/) {
  std::vector<DimSpec> specs;
  specs.reserve(tensor.dim());
  for (int64_t d = 0; d < tensor.dim(); ++d) {
    int64_t size = tensor.size(d);
    if (size == 1) {
      specs.push_back({DimSpec::kBroadcast, 1});
    } else {
      specs.push_back({DimSpec::kDynamic, 0});
    }
  }
  return specs;
}

} // namespace at::pyre
