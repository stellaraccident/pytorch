#pragma once

// Kernel ASM builder: expand MLIR templates with concrete types and shapes.
//
// Split into two phases for hot-path performance:
//   buildKernelSpec — cheap, builds substitution pairs for cache key hashing
//   generateMlir   — expensive, called only on cache miss

#include <ATen/pyre/dispatch/PyreArgAdapter.h>
#include <ATen/pyre/dispatch/PyreTypeMapping.h>
#include <c10/util/ArrayRef.h>

#include <string>
#include <utility>
#include <vector>

namespace at::pyre {

// Substitution pairs + template identity — enough to compute cache key.
struct KernelSpec {
  const char* template_sha1;
  std::vector<std::pair<std::string, std::string>> substitutions;
};

// Content-hash cache key: SHA1 of template digest + substitutions + flags.
std::string contentHashCacheKey(
    const char* template_sha1,
    const std::vector<std::pair<std::string, std::string>>& substitutions,
    c10::ArrayRef<std::string> compiler_flags);

// --- Binary ops ---

KernelSpec buildBinaryKernelSpec(
    const std::string& func_name, const std::string& linalg_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    const std::vector<ArgAdapter>& adapters);

std::string generateBinaryMlir(
    const std::string& func_name, const std::string& linalg_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    const std::vector<ArgAdapter>& adapters);

// --- Binary with alpha ---

KernelSpec buildBinaryAlphaKernelSpec(
    const std::string& func_name, const std::string& alpha_add_op,
    double alpha_value, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    const std::vector<ArgAdapter>& adapters);

std::string generateBinaryAlphaMlir(
    const std::string& func_name, const std::string& alpha_add_op,
    double alpha_value, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    const std::vector<ArgAdapter>& adapters);

// --- Unary ops ---

KernelSpec buildUnaryKernelSpec(
    const std::string& func_name, const std::string& scalar_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> out_shape,
    const ArgAdapter& adapter);

std::string generateUnaryMlir(
    const std::string& func_name, const std::string& scalar_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> out_shape,
    const ArgAdapter& adapter);

// --- mm ---

KernelSpec buildMmKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> mat1_shape, c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape);

std::string generateMmMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> mat1_shape, c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape);

// --- addmm ---

KernelSpec buildAddmmKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape, c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_shape, c10::ArrayRef<int64_t> out_shape,
    double beta, double alpha);

std::string generateAddmmMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape, c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_shape, c10::ArrayRef<int64_t> out_shape,
    double beta, double alpha);

// --- addmm transposed ---

KernelSpec buildAddmmTransposedKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape, c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_orig_shape, c10::ArrayRef<int64_t> out_shape,
    double beta, double alpha);

std::string generateAddmmTransposedMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape, c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_orig_shape, c10::ArrayRef<int64_t> out_shape,
    double beta, double alpha);

} // namespace at::pyre
