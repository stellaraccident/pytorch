#pragma once

// Strided copy planning: dimension coalescing and tier classification.
//
// Pure C++ utility. No IREE or device dependency.
// See docs/design/epic2_strided_copy.md §5.1.

#include <c10/util/SmallVector.h>
#include <c10/util/ArrayRef.h>

#include <cstdint>

namespace at::pyre {

struct CoalescedDim {
  int64_t size;
  int64_t src_stride;  // in elements
  int64_t dst_stride;  // in elements
};

struct CopyChunk {
  int64_t src_offset;  // in bytes
  int64_t dst_offset;  // in bytes
  int64_t length;      // in bytes
};

struct CopyPlan {
  enum Tier { kSingleCopy, kDecomposed, kCompiledKernel };
  Tier tier;

  // Tier 0/1: contiguous chunks to copy.
  c10::SmallVector<CopyChunk, 64> chunks;

  // Tier 2: coalesced dimensions for kernel parametrization.
  c10::SmallVector<CoalescedDim, 6> dims;

  // Tier 2: storage offsets in elements (baked into compiled kernel).
  int64_t src_base_offset = 0;
  int64_t dst_base_offset = 0;

  int64_t numel;
};

static constexpr int kMaxCopyChunks = 128;

// Sort by src_stride ascending, merge adjacent dims where
// src_stride[i]*size[i]==src_stride[i+1] AND same for dst.
// Skips size-1 dims.
c10::SmallVector<CoalescedDim, 6> coalesceDims(
    c10::IntArrayRef shape,
    c10::IntArrayRef src_strides,
    c10::IntArrayRef dst_strides);

// Classify copy into tier and produce chunks or coalesced dims.
CopyPlan planCopy(
    c10::IntArrayRef shape,
    c10::IntArrayRef src_strides,
    c10::IntArrayRef dst_strides,
    int64_t src_offset,
    int64_t dst_offset,
    int64_t element_size);

} // namespace at::pyre
