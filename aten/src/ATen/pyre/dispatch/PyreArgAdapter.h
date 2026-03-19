#pragma once

// Stride analysis for kernel arg adaptation.
//
// Detects pure axis permutations from stride patterns and provides
// the permutation vector. applyAdapters() un-permutes to recover the
// physical (contiguous) layout. Template expansion emits
// torch.aten.permute to reconstruct the logical shape inside the kernel.
//
// See epic1_kernel_dispatch.md §10.3 P1-C.

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

// Apply adapters to inputs:
//   kIdentity  — no-op
//   kPermute   — inverse-permute to physical (contiguous) layout
//   kContiguous — force .contiguous() (data copy)
std::vector<at::Tensor> applyAdapters(
    const std::vector<at::Tensor>& inputs,
    const std::vector<ArgAdapter>& adapters);

} // namespace at::pyre
