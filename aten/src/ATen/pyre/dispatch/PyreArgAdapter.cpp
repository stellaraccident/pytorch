#include <ATen/pyre/dispatch/PyreArgAdapter.h>

#include <algorithm>
#include <numeric>

namespace at::pyre {

ArgAdapter ArgAdapter::analyze(const at::Tensor& tensor) {
  if (tensor.is_contiguous())
    return {kIdentity, {}};

  int64_t ndim = tensor.dim();
  if (ndim < 2)
    return {kContiguous, {}};

  // Check if strides represent a pure axis permutation of a contiguous
  // tensor. Sort dims by stride (descending) and verify the sorted
  // strides match a contiguous layout with the permuted sizes.
  std::vector<int64_t> perm(ndim);
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
    return tensor.stride(a) > tensor.stride(b);
  });

  // Check identity permutation — would be contiguous (already handled above).
  bool is_identity = true;
  for (int64_t i = 0; i < ndim; ++i) {
    if (perm[i] != i) { is_identity = false; break; }
  }
  if (is_identity)
    return {kContiguous, {}};

  // Verify: sorted strides must form a contiguous layout for the
  // permuted sizes.
  int64_t expected_stride = 1;
  bool is_permutation = true;
  for (int64_t i = ndim - 1; i >= 0; --i) {
    if (tensor.stride(perm[i]) != expected_stride) {
      is_permutation = false;
      break;
    }
    expected_stride *= tensor.size(perm[i]);
  }

  if (is_permutation)
    return {kPermute, std::move(perm)};

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
      case ArgAdapter::kPermute: {
        // The adapter's perm maps physical position → logical dim
        // (sorted by stride). Applying it to the logical-shape tensor
        // recovers the physical (contiguous) layout.
        result.push_back(inputs[i].permute(adapter.permutation));
        break;
      }
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
