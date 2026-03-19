#pragma once

// Kernel ASM builder: expand MLIR templates with concrete types and shapes.

#include <ATen/pyre/dispatch/PyreArgAdapter.h>
#include <ATen/pyre/dispatch/PyreTypeMapping.h>
#include <c10/util/ArrayRef.h>

#include <string>
#include <utility>
#include <vector>

namespace at::pyre {

// Expanded kernel: MLIR text + metadata for content-hash cache key.
struct ExpandedKernel {
  std::string mlir;
  const char* template_sha1;
  std::vector<std::pair<std::string, std::string>> substitutions;
};

// Content-hash cache key: SHA1 of template digest + substitutions + flags.
std::string contentHashCacheKey(
    const char* aten_name,
    const char* template_sha1,
    const std::vector<std::pair<std::string, std::string>>& substitutions,
    c10::ArrayRef<std::string> compiler_flags);

ExpandedKernel expandBinaryTemplate(
    const std::string& func_name,
    const std::string& linalg_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    const std::vector<ArgAdapter>& adapters);

ExpandedKernel expandBinaryAlphaTemplate(
    const std::string& func_name,
    const std::string& alpha_add_op,
    const std::string& alpha_mul_op,
    double alpha_value,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape,
    c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    const std::vector<ArgAdapter>& adapters);

ExpandedKernel expandUnaryTemplate(
    const std::string& func_name,
    const std::string& scalar_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    c10::ArrayRef<int64_t> out_shape,
    const ArgAdapter& adapter);

ExpandedKernel expandAddmmTemplate(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape,
    c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape);

ExpandedKernel expandAddmmTransposedTemplate(
    const std::string& func_name,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape,
    c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_orig_shape,
    c10::ArrayRef<int64_t> out_shape);

} // namespace at::pyre
