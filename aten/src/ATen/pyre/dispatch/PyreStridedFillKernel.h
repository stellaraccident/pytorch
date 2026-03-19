#pragma once

// Compiled strided-fill kernel for non-contiguous and 8-byte types.
//
// Same infrastructure as the strided copy kernel: rank-specialized MLIR,
// opaque integer types by bitwidth, content-hash caching.
//
// See docs/design/epic2_strided_copy.md §5.4.

#include <ATen/core/Tensor.h>
#include <ATen/pyre/dispatch/StridedCopyPlan.h>
#include <c10/core/Scalar.h>

#include <string>

namespace at::pyre {

// Generate MLIR for a strided fill kernel.
std::string generateStridedFillMlir(
    const std::string& func_name,
    c10::ArrayRef<CoalescedDim> dims,
    int64_t dst_numel,
    int64_t element_size,
    int64_t fill_pattern);

// Execute a compiled strided fill.
void executeCompiledFill(
    const at::Tensor& dst,
    const at::Scalar& value);

} // namespace at::pyre
