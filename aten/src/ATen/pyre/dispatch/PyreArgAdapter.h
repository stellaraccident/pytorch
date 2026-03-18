#pragma once

// Stride analysis and MLIR adapter function generation.
//
// Templates have a two-level type system:
//   - Input type: physical layout as received from PyTorch (possibly strided)
//   - Compute type: linearized canonical layout for the core kernel
//
// Adapters transform from input type to compute type. When the adapter is
// identity, the types are the same and the compiler sees through it.
//
// Epic 1: identity adapter (active) + contiguous fallback.
// Transpose/permute adapter structure in place for future.
// See epic1_kernel_dispatch.md §4.3.

#include <ATen/Tensor.h>

#include <cstdint>
#include <string>
#include <vector>

namespace at::pyre {

struct ArgAdapter {
  enum Kind {
    kIdentity,    // Contiguous (row-major) — free
    kTranspose,   // Reversed/permuted strides — compiler fuses
    kContiguous,  // Arbitrary strides — force contiguous (data movement)
  };

  Kind kind = kIdentity;
  std::vector<int64_t> permutation;  // for kTranspose

  // Generate the MLIR body for this adapter function.
  // type_name is the MLIR type alias (e.g., "!lhs_compute_type").
  std::string generateBody(const std::string& type_name) const;

  // Analyze a tensor's strides and choose the appropriate adapter.
  static ArgAdapter analyze(const at::Tensor& tensor);
};

// Apply adapters to inputs: identity is no-op, contiguous calls .contiguous().
std::vector<at::Tensor> applyAdapters(
    const std::vector<at::Tensor>& inputs,
    const std::vector<ArgAdapter>& adapters);

} // namespace at::pyre
