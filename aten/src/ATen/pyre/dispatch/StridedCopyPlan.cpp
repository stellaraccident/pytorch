#include <ATen/pyre/dispatch/StridedCopyPlan.h>

#include <algorithm>

namespace at::pyre {

c10::SmallVector<CoalescedDim, 6> coalesceDims(
    c10::IntArrayRef shape,
    c10::IntArrayRef src_strides,
    c10::IntArrayRef dst_strides) {
  int64_t ndim = static_cast<int64_t>(shape.size());

  // Build triples, skipping size-1 dims.
  c10::SmallVector<CoalescedDim, 6> dims;
  for (int64_t i = 0; i < ndim; ++i) {
    if (shape[i] != 1) {
      dims.push_back({shape[i], src_strides[i], dst_strides[i]});
    }
  }

  if (dims.empty()) {
    // Scalar tensor: single element.
    return {{1, 1, 1}};
  }

  // Sort by src_stride ascending. Broadcast dims (stride=0) sort last
  // so the contiguous inner dim is first for chunk decomposition.
  std::sort(dims.begin(), dims.end(),
            [](const CoalescedDim& a, const CoalescedDim& b) {
              bool a_bcast = (a.src_stride == 0);
              bool b_bcast = (b.src_stride == 0);
              if (a_bcast != b_bcast) return b_bcast;  // non-broadcast first
              return a.src_stride < b.src_stride;
            });

  // Merge adjacent dims.
  c10::SmallVector<CoalescedDim, 6> merged;
  merged.push_back(dims[0]);
  for (size_t i = 1; i < dims.size(); ++i) {
    auto& prev = merged.back();
    if (prev.src_stride * prev.size == dims[i].src_stride &&
        prev.dst_stride * prev.size == dims[i].dst_stride) {
      prev.size *= dims[i].size;
    } else {
      merged.push_back(dims[i]);
    }
  }

  return merged;
}

CopyPlan planCopy(
    c10::IntArrayRef shape,
    c10::IntArrayRef src_strides,
    c10::IntArrayRef dst_strides,
    int64_t src_offset,
    int64_t dst_offset,
    int64_t element_size) {
  CopyPlan plan;

  // Compute numel.
  plan.numel = 1;
  for (auto s : shape) plan.numel *= s;

  // 0-dim scalar.
  if (shape.empty()) {
    plan.numel = 1;
    plan.tier = CopyPlan::kSingleCopy;
    plan.chunks.push_back({
        src_offset * element_size,
        dst_offset * element_size,
        element_size});
    return plan;
  }

  auto dims = coalesceDims(shape, src_strides, dst_strides);

  // Tier 0: single coalesced dim with both strides == 1.
  if (dims.size() == 1 && dims[0].src_stride == 1 && dims[0].dst_stride == 1) {
    plan.tier = CopyPlan::kSingleCopy;
    plan.chunks.push_back({
        src_offset * element_size,
        dst_offset * element_size,
        dims[0].size * element_size});
    return plan;
  }

  // Check innermost dim (first after sort, smallest src_stride).
  bool inner_contiguous =
      dims[0].src_stride == 1 && dims[0].dst_stride == 1;

  if (!inner_contiguous) {
    // Tier 2: innermost is strided → must compile.
    plan.tier = CopyPlan::kCompiledKernel;
    plan.dims = std::move(dims);
    return plan;
  }

  // Innermost is contiguous. Enumerate chunks from outer dims.
  int64_t chunk_bytes = dims[0].size * element_size;

  // Count total chunks = product of outer dim sizes.
  int64_t n_chunks = 1;
  for (size_t i = 1; i < dims.size(); ++i) {
    n_chunks *= dims[i].size;
  }

  if (n_chunks > kMaxCopyChunks) {
    plan.tier = CopyPlan::kCompiledKernel;
    plan.dims = std::move(dims);
    return plan;
  }

  // Tier 1: decomposed. Enumerate all outer coordinate combinations.
  plan.tier = CopyPlan::kDecomposed;
  plan.chunks.reserve(static_cast<size_t>(n_chunks));

  // Use a coordinate counter over dims[1..N-1].
  size_t n_outer = dims.size() - 1;
  c10::SmallVector<int64_t, 6> coords(n_outer, 0);

  for (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
    int64_t src_off = src_offset * element_size;
    int64_t dst_off = dst_offset * element_size;
    for (size_t d = 0; d < n_outer; ++d) {
      src_off += coords[d] * dims[d + 1].src_stride * element_size;
      dst_off += coords[d] * dims[d + 1].dst_stride * element_size;
    }
    plan.chunks.push_back({src_off, dst_off, chunk_bytes});

    // Advance coordinate counter (innermost outer dim first).
    for (size_t d = 0; d < n_outer; ++d) {
      if (++coords[d] < dims[d + 1].size) break;
      coords[d] = 0;
    }
  }

  return plan;
}

} // namespace at::pyre
