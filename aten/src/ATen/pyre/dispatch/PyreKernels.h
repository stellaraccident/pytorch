#pragma once

// Bespoke kernel generators: compiled kernels for data movement and
// other operations that don't use the template-based torch-mlir path.
//
// Each generator has its own .cpp file. This header collects all
// declarations so callers include one file.

#include <ATen/core/Tensor.h>
#include <ATen/pyre/dispatch/StridedCopyPlan.h>
#include <c10/core/Scalar.h>

namespace at::pyre {

// Tier 2 compiled strided copy (transpose, general permutations).
void executeCompiledCopy(
    const CopyPlan& plan,
    const at::Tensor& src,
    const at::Tensor& dst);

// Compiled fill for non-contiguous tensors and 8-byte element types.
void executeCompiledFill(
    const at::Tensor& dst,
    const at::Scalar& value);

} // namespace at::pyre
