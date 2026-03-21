#pragma once

// Kernel ASM builder: expand MLIR templates with concrete types and shapes.
//
// Split into two phases for hot-path performance:
//   buildKernelSpec — cheap, builds substitution pairs for cache key hashing
//   generateMlir   — expensive, called only on cache miss

#include <ATen/pyre/dispatch/PyreArgAdapter.h>
#include <ATen/pyre/dispatch/PyreStringSplicer.h>
#include <ATen/pyre/dispatch/PyreTypeMapping.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>
#include <c10/util/hash.h>

#include <functional>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

namespace at::pyre {

// Substitution pairs + template identity — enough to compute cache key.
using SubstPairs = c10::SmallVector<std::pair<std::string, std::string>, 16>;

struct KernelSpec {
  const char* template_sha1;
  SubstPairs substitutions;
};

// Content-hash cache key: SHA1 of template digest + substitutions + flags.
std::string contentHashCacheKey(
    const char* template_sha1,
    const SubstPairs& substitutions,
    c10::ArrayRef<std::string> compiler_flags);

// --- embedding ---

KernelSpec buildEmbeddingKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> weight_shape, c10::ArrayRef<int64_t> indices_shape,
    c10::ArrayRef<int64_t> out_shape);

std::string generateEmbeddingMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> weight_shape, c10::ArrayRef<int64_t> indices_shape,
    c10::ArrayRef<int64_t> out_shape);

// --- index_select ---

KernelSpec buildIndexSelectKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> out_shape, int64_t dim);

std::string generateIndexSelectMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> out_shape, int64_t dim);

// --- gather ---

KernelSpec buildGatherKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> out_shape, int64_t dim);

std::string generateGatherMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> out_shape, int64_t dim);

// --- arange ---

KernelSpec buildArangeKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    int64_t out_size, double start, double end, double step);

std::string generateArangeMlir(
    const std::string& func_name, c10::ScalarType dtype,
    int64_t out_size, double start, double end, double step);

// --- Type cast ---

KernelSpec buildTypeCastKernelSpec(
    const std::string& func_name,
    c10::ScalarType in_dtype, c10::ScalarType out_dtype,
    c10::ArrayRef<int64_t> shape);

std::string generateTypeCastMlir(
    const std::string& func_name,
    c10::ScalarType in_dtype, c10::ScalarType out_dtype,
    c10::ArrayRef<int64_t> shape);

// --- bmm ---

KernelSpec buildBmmKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> mat1_shape, c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape);

std::string generateBmmMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> mat1_shape, c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape);

// --- where ---

KernelSpec buildWhereKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> cond_shape,
    c10::ArrayRef<int64_t> self_shape,
    c10::ArrayRef<int64_t> other_shape,
    c10::ArrayRef<int64_t> out_shape);

std::string generateWhereMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> cond_shape,
    c10::ArrayRef<int64_t> self_shape,
    c10::ArrayRef<int64_t> other_shape,
    c10::ArrayRef<int64_t> out_shape);

// --- Reduction ops ---

KernelSpec buildReductionKernelSpec(
    const std::string& func_name, const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<int64_t> dims, bool keepdim,
    const std::string& extra_arg_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_arg_types = "");

std::string generateReductionMlir(
    const std::string& func_name, const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<int64_t> dims, bool keepdim,
    const std::string& extra_arg_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_arg_types = "");

// --- Single-dim reduction ops ---

KernelSpec buildSingleDimReductionKernelSpec(
    const std::string& func_name, const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> out_shape,
    int64_t dim, bool keepdim,
    const std::string& extra_arg_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_arg_types = "");

std::string generateSingleDimReductionMlir(
    const std::string& func_name, const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> out_shape,
    int64_t dim, bool keepdim,
    const std::string& extra_arg_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_arg_types = "");

// --- Comparison ops (tensor-tensor) ---

KernelSpec buildComparisonKernelSpec(
    const std::string& func_name, const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape);

std::string generateComparisonMlir(
    const std::string& func_name, const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape);

// --- Comparison ops (tensor-scalar) ---

KernelSpec buildComparisonScalarKernelSpec(
    const std::string& func_name, const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    double scalar_value);

std::string generateComparisonScalarMlir(
    const std::string& func_name, const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    double scalar_value);

// --- Scalar binary ops ---

KernelSpec buildScalarBinaryKernelSpec(
    const std::string& func_name, const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    double scalar_value,
    const std::string& extra_arg_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_arg_types = "");

std::string generateScalarBinaryMlir(
    const std::string& func_name, const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape,
    double scalar_value,
    const std::string& extra_arg_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_arg_types = "");

// --- Binary ops ---

KernelSpec buildBinaryKernelSpec(
    const std::string& func_name, const std::string& linalg_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters);

std::string generateBinaryMlir(
    const std::string& func_name, const std::string& linalg_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters);

// --- Binary with alpha ---

KernelSpec buildBinaryAlphaKernelSpec(
    const std::string& func_name, const std::string& alpha_add_op,
    double alpha_value, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters);

std::string generateBinaryAlphaMlir(
    const std::string& func_name, const std::string& alpha_add_op,
    double alpha_value, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters);

// --- Unary ops ---

KernelSpec buildUnaryKernelSpec(
    const std::string& func_name, const std::string& scalar_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> out_shape,
    const ArgAdapter& adapter,
    const std::string& extra_arg_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_arg_types = "");

std::string generateUnaryMlir(
    const std::string& func_name, const std::string& scalar_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> out_shape,
    const ArgAdapter& adapter,
    const std::string& extra_arg_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_arg_types = "");

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

// ---------------------------------------------------------------------------
// PyreKernelAsmFragments + PyreKernelAsmBuilder
//
// Two-mode builder for procedurally generated MLIR. Separates cache key
// hashing (hot path, no string generation) from MLIR generation (miss path).
// ---------------------------------------------------------------------------

class PyreKernelAsmBuilder;

class PyreKernelAsmFragments {
 public:
  PyreKernelAsmFragments(std::initializer_list<std::string_view> fragments);

  std::string_view fragment(size_t index) const { return fragments_[index]; }

  // SHA1 of all fragment texts concatenated. Computed once at construction.
  const std::string& combinedDigest() const { return combined_digest_; }

  // Hot path: replay recipe in digest mode. Returns cache key string.
  std::string digest(
      c10::ArrayRef<std::string> compiler_flags,
      const std::function<void(PyreKernelAsmBuilder&)>& recipe) const;

  // Miss path: replay recipe in generate mode. Returns full MLIR string.
  std::string generateMlir(
      const std::function<void(PyreKernelAsmBuilder&)>& recipe) const;

 private:
  c10::SmallVector<std::string_view, 8> fragments_;
  std::string combined_digest_;
};

class PyreKernelAsmBuilder {
 public:
  enum class Mode { kDigest, kGenerate };

  // Append a fragment with substitutions. Inline so the compiler can
  // optimize out the mode branch and dead code per call site.
  inline void appendFragment(
      size_t index,
      std::initializer_list<std::pair<std::string_view, std::string_view>> substs) {
    if (mode_ == Mode::kDigest) {
      // Accumulate only substitution values for hashing.
      // Fragment identity is covered by the combined digest.
      for (const auto& [key, value] : substs) {
        hash_input_.append(value.data(), value.size());
        hash_input_ += '\0';
      }
    } else {
      // Splice fragment text with substitutions, append to output.
      mlir_ += pyreSplice(fragments_->fragment(index), substs);
    }
  }

 private:
  friend class PyreKernelAsmFragments;
  static constexpr size_t kHashInputReserve = 256;

  PyreKernelAsmBuilder(const PyreKernelAsmFragments& frags, Mode mode)
      : fragments_(&frags), mode_(mode) {
    if (mode == Mode::kDigest) {
      hash_input_.reserve(kHashInputReserve);
      hash_input_ = frags.combinedDigest();
    }
  }

  std::string finish() {
    if (mode_ == Mode::kDigest) return c10::sha1(hash_input_).str();
    return std::move(mlir_);
  }

  const PyreKernelAsmFragments* fragments_;
  Mode mode_;
  std::string hash_input_;  // kDigest: combined digest + substitution values
  std::string mlir_;        // kGenerate: accumulated MLIR text
};

} // namespace at::pyre
