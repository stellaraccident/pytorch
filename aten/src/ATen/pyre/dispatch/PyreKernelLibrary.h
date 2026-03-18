#pragma once

// PyreKernelLibrary: registry mapping op names to MLIR template expansion.
//
// Each op has: embedded MLIR template, scalar op name(s), supported dtypes.
// The library expands templates with concrete type/shape information from
// the specialization decision.
//
// See epic1_kernel_dispatch.md §4.2 and §6.

#include <ATen/pyre/dispatch/PyreArgShapeSpecializer.h>
#include <ATen/pyre/dispatch/PyreSpecKey.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>

#include <string>
#include <vector>

namespace at::pyre {

// Expand a binary elementwise template for the given op and shapes.
std::string expandBinaryTemplate(
    const std::string& func_name,
    const std::string& linalg_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    const std::vector<ArgAdapter>& adapters);

// Expand an alpha-fused binary template (for add/sub with alpha != 1).
std::string expandBinaryAlphaTemplate(
    const std::string& func_name,
    const std::string& alpha_add_op,
    const std::string& alpha_mul_op,
    double alpha_value,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    const std::vector<ArgAdapter>& adapters);

// Expand a unary elementwise template.
std::string expandUnaryTemplate(
    const std::string& func_name,
    const std::string& scalar_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    c10::ArrayRef<int64_t> out_shape,
    const ArgAdapter& adapter);

// Helper: build dynamic shape string (e.g., "?x?x?" for rank 3).
std::string dynamicShapeStr(int64_t rank);

// Helper: build dim variable string (e.g., "d0, d1, d2" for rank 3).
std::string dimVarsStr(int64_t rank);

// Helper: build parallel iterator types string for given rank.
std::string parallelTypesStr(int64_t rank);

} // namespace at::pyre
