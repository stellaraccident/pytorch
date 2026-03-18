#include <ATen/pyre/dispatch/PyreArgShapeSpecializer.h>

namespace at::pyre {

SpecDecision ArgShapeSpecializer::analyze(
    const std::string& op_name,
    c10::ScalarType dtype,
    c10::ArrayRef<at::Tensor> inputs,
    const c10::pyre::DeviceCapabilities& /*caps*/) {
  TORCH_CHECK(!inputs.empty(), "pyre: need at least one input");

  int64_t rank = inputs[0].dim();

  std::vector<ArgShapeInfo> arg_shapes;
  std::vector<ArgAdapter> arg_adapters;
  for (const auto& t : inputs) {
    ArgShapeInfo info;
    info.rank = t.dim();
    info.dims = buildDimSpecs(t);
    info.sizes = t.sizes().vec();
    arg_shapes.push_back(std::move(info));
    arg_adapters.push_back(ArgAdapter::analyze(t));
  }

  std::vector<BroadcastEntry> broadcast_mask;
  if (inputs.size() == 2)
    broadcast_mask = detectBroadcast(inputs[0], inputs[1]);

  auto dim_specs = arg_shapes.empty()
      ? std::vector<DimSpec>{} : arg_shapes[0].dims;

  PyreSpecKey spec_key(
      op_name, dtype, rank,
      std::move(dim_specs), std::move(broadcast_mask));

  return {std::move(spec_key), std::move(arg_shapes), std::move(arg_adapters)};
}

std::vector<BroadcastEntry> ArgShapeSpecializer::detectBroadcast(
    const at::Tensor& lhs, const at::Tensor& rhs) {
  std::vector<BroadcastEntry> mask;
  auto ls = lhs.sizes(), rs = rhs.sizes();
  int64_t ndim = std::max(ls.size(), rs.size());
  for (int64_t d = 0; d < ndim; ++d) {
    int64_t li = static_cast<int64_t>(ls.size()) - ndim + d;
    int64_t ri = static_cast<int64_t>(rs.size()) - ndim + d;
    int64_t lv = (li >= 0) ? ls[li] : 1;
    int64_t rv = (ri >= 0) ? rs[ri] : 1;
    if (lv == 1 && rv != 1) mask.push_back({0, static_cast<int>(d)});
    else if (rv == 1 && lv != 1) mask.push_back({1, static_cast<int>(d)});
  }
  return mask;
}

std::vector<DimSpec> ArgShapeSpecializer::buildDimSpecs(
    const at::Tensor& tensor) {
  std::vector<DimSpec> specs;
  for (int64_t d = 0; d < tensor.dim(); ++d) {
    if (tensor.size(d) == 1)
      specs.push_back({DimSpec::kBroadcast, 1});
    else
      specs.push_back({DimSpec::kDynamic, 0});
  }
  return specs;
}

} // namespace at::pyre
