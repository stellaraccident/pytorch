#pragma once

// Stride analysis for kernel arg adaptation.
//
// Detects pure axis permutations from stride patterns and provides
// the permutation vector. AbiPacker uses this to encode physical layout
// in the cache key; AbiGenerator emits linalg.transpose in the envelope.

#include <ATen/Tensor.h>

#include <cstdint>
#include <vector>

namespace at::pyre {

struct ArgAdapter {
  enum Kind {
    kIdentity,    // Contiguous (row-major) — free
    kPermute,     // Pure axis permutation — compiler fuses, zero copy
    kContiguous,  // Arbitrary strides — force contiguous (data movement)
  };

  Kind kind = kIdentity;
  std::vector<int64_t> permutation;  // for kPermute: logical→physical mapping

  // Analyze a tensor's strides and choose the appropriate adapter.
  static ArgAdapter analyze(const at::Tensor& tensor);
};

} // namespace at::pyre
