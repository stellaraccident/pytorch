#include <ATen/pyre/dispatch/PyreArgAdapter.h>

namespace at::pyre {

std::string ArgAdapter::generateBody(const std::string& type_name) const {
  switch (kind) {
    case kIdentity:
      return "    util.return %arg : " + type_name;

    case kTranspose:
      // Future: generate linalg.transpose with permutation.
      // For now, fall through to contiguous.
      return "    util.return %arg : " + type_name;

    case kContiguous:
      // The adapter is identity in MLIR — the actual contiguous() call
      // happens on the PyTorch side before dispatch (see applyAdapters).
      return "    util.return %arg : " + type_name;
  }
  return "    util.return %arg : " + type_name;
}

ArgAdapter ArgAdapter::analyze(const at::Tensor& tensor) {
  if (tensor.is_contiguous()) {
    return {kIdentity, {}};
  }

  // Future: detect transpose/permute patterns from strides.
  // For Epic 1, all non-contiguous tensors get forced contiguous.
  return {kContiguous, {}};
}

std::vector<at::Tensor> applyAdapters(
    const std::vector<at::Tensor>& inputs,
    const std::vector<ArgAdapter>& adapters) {
  std::vector<at::Tensor> result;
  result.reserve(inputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& adapter = (i < adapters.size())
        ? adapters[i]
        : ArgAdapter{ArgAdapter::kIdentity, {}};

    switch (adapter.kind) {
      case ArgAdapter::kContiguous:
        result.push_back(inputs[i].contiguous());
        break;
      default:
        result.push_back(inputs[i]);
        break;
    }
  }
  return result;
}

} // namespace at::pyre
