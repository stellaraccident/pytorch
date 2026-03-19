#pragma once

// Compiled strided-copy kernel for Tier 2 copy plans.
//
// Generates rank-specialized MLIR that iterates linearly over output
// elements, recovers N-D indices via divmod, and computes source offsets
// via stride arithmetic. Uses opaque integer types by bitwidth (i8/i16/
// i32/i64) — dtype-agnostic, pure data movement.
//
// See docs/design/epic2_strided_copy.md §5.3.

#include <ATen/core/Tensor.h>
#include <ATen/pyre/dispatch/StridedCopyPlan.h>

#include <string>

namespace at::pyre {

// Generate MLIR for a strided copy kernel from coalesced dimensions.
// element_size selects the opaque integer type (1→i8, 2→i16, 4→i32, 8→i64).
// src_numel/dst_numel are total flat element counts for the 1D tensor types.
std::string generateStridedCopyMlir(
    const std::string& func_name,
    c10::ArrayRef<CoalescedDim> dims,
    int64_t src_numel,
    int64_t dst_numel,
    int64_t element_size);

// Execute a Tier 2 (compiled kernel) copy plan.
void executeCompiledCopy(
    const CopyPlan& plan,
    const at::Tensor& src,
    const at::Tensor& dst);

} // namespace at::pyre
