#include <ATen/pyre/dispatch/PyreArgAdapter.h>

#include <algorithm>
#include <numeric>

namespace at::pyre {

ArgAdapter ArgAdapter::analyze(const at::Tensor& tensor) {
  // HACK(pyre-workspace-blp): Non-zero storage offset (from split/narrow)
  // means the tensor's data doesn't start at byte 0 of the IREE buffer.
  // Force clone into a fresh buffer at offset 0. The proper fix is
  // iree_hal_buffer_subspan, but compiled kernels crash with non-zero
  // binding offsets (see ticket for details).
  if (tensor.storage_offset() != 0)
    return {kContiguous, {}};

  // PyTorch is_contiguous() can return true for tensors with non-standard
  // strides when size-1 dims are involved (e.g. [1,2,1,32] strides
  // (64,32,64,1) from transpose). We need strict row-major verification
  // because IREE buffer views are always DENSE_ROW_MAJOR.
  if (tensor.is_contiguous()) {
    bool strict_row_major = true;
    int64_t expected = 1;
    for (int64_t i = tensor.dim() - 1; i >= 0; --i) {
      if (tensor.size(i) > 1 && tensor.stride(i) != expected) {
        strict_row_major = false;
        break;
      }
      expected *= tensor.size(i);
    }
    if (strict_row_major)
      return {kIdentity, {}};
    // Fall through to permutation detection for "contiguous" tensors
    // with non-standard strides.
  }

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
      case ArgAdapter::kContiguous: {
        // HACK(pyre-workspace-blp): .contiguous() is a no-op when strides
        // are contiguous, even if storage_offset != 0. Use .clone() to
        // ensure the data starts at byte 0 in a fresh buffer.
        if (inputs[i].storage_offset() != 0) {
          result.push_back(inputs[i].clone());
        } else {
          result.push_back(inputs[i].contiguous());
        }
        break;
      }
      default:
        result.push_back(inputs[i]);
        break;
    }
  }
  return result;
}

} // namespace at::pyre
