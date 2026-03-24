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

// Compute inverse permutation.
c10::SmallVector<int64_t, 6> inversePerm(c10::ArrayRef<int64_t> perm);

// Emit torch.aten.permute MLIR lines for an arg adapter.
std::string emitPermuteLines(
    const std::string& dst_name,
    const std::string& src_name,
    c10::ArrayRef<int64_t> perm,
    const std::string& src_type,
    const std::string& dst_type);

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

// ---------------------------------------------------------------------------
// ComputeBody — extracted torch op body without module/func wrapper.
//
// Contains only the logical torch dialect ops between input SSA values and
// result SSA value. No module wrapper, no func.func, no type aliases, no
// torch.overwrite.tensor.contents, no permute adapters.
//
// AbiGenerator (T2) wraps this with envelope function + compute func.func
// shell. The backward-compat wrapComputeBody() re-wraps for the old path.
// ---------------------------------------------------------------------------

struct ComputeBody {
  // Torch dialect ops only (indented, newline-terminated).
  // Uses SSA names from input_names for inputs, produces %result.
  std::string mlir_ops;

  // Per-input: logical type strings (post-permutation shapes).
  c10::SmallVector<std::string, 4> input_vtensor_types;
  c10::SmallVector<std::string, 4> input_tensor_types;

  // Per-input: SSA names referenced in mlir_ops.
  c10::SmallVector<std::string, 4> input_names;

  // Output type strings.
  std::string output_vtensor_type;
  std::string output_tensor_type;
};

// Wrap a ComputeBody back into a complete MLIR module for backward compat.
// Adds type aliases, module, func.func, adapters, and overwrite epilogue.
// This is the migration bridge — remove once T4 wires AbiGenerator.
std::string wrapComputeBody(
    const std::string& func_name,
    const ComputeBody& body,
    c10::ArrayRef<ArgAdapter> adapters,
    c10::ArrayRef<std::string> physical_input_vtensor_types,
    c10::ArrayRef<std::string> physical_input_names);

// --- Compute body generators (new API for AbiGenerator) ---

ComputeBody generateBinaryComputeBody(
    const std::string& linalg_op, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape);

ComputeBody generateBinaryAlphaComputeBody(
    const std::string& alpha_add_op, double alpha_value,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape);

ComputeBody generateUnaryComputeBody(
    const std::string& scalar_op, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> out_shape,
    const std::string& extra_arg_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_arg_types = "");

ComputeBody generateMmComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> mat1_shape, c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape);

ComputeBody generateSoftmaxComputeBody(
    const std::string& softmax_op, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> shape, int64_t dim);

ComputeBody generateComparisonComputeBody(
    const std::string& torch_op, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape);

ComputeBody generateComparisonScalarComputeBody(
    const std::string& torch_op, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, double scalar_value);

ComputeBody generateBmmComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> mat1_shape, c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape);

ComputeBody generateWhereComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> cond_shape,
    c10::ArrayRef<int64_t> self_shape,
    c10::ArrayRef<int64_t> other_shape,
    c10::ArrayRef<int64_t> out_shape);

ComputeBody generateAddmmComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> bias_shape, c10::ArrayRef<int64_t> mat1_shape,
    c10::ArrayRef<int64_t> mat2_shape, c10::ArrayRef<int64_t> out_shape,
    double beta, double alpha);

ComputeBody generateReductionComputeBody(
    const std::string& torch_op, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<int64_t> dims, bool keepdim,
    const std::string& extra_arg_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_arg_types = "");

ComputeBody generateSingleDimReductionComputeBody(
    const std::string& torch_op, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> out_shape,
    int64_t dim, bool keepdim,
    const std::string& extra_arg_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_arg_types = "");

ComputeBody generateEmbeddingComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> weight_shape, c10::ArrayRef<int64_t> indices_shape,
    c10::ArrayRef<int64_t> out_shape);

ComputeBody generateIndexSelectComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> out_shape, int64_t dim);

ComputeBody generateGatherComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> out_shape, int64_t dim);

ComputeBody generateScatterSrcComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim);

ComputeBody generateScatterAddComputeBody(
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim);

ComputeBody generateTypeCastComputeBody(
    c10::ScalarType in_dtype, c10::ScalarType out_dtype,
    c10::ArrayRef<int64_t> shape);

ComputeBody generateArangeComputeBody(
    c10::ScalarType dtype,
    int64_t out_size, double start, double end, double step);

// --- softmax ---

KernelSpec buildSoftmaxKernelSpec(
    const std::string& func_name, const std::string& softmax_op,
    c10::ScalarType dtype, c10::ArrayRef<int64_t> shape, int64_t dim);

std::string generateSoftmaxMlir(
    const std::string& func_name, const std::string& softmax_op,
    c10::ScalarType dtype, c10::ArrayRef<int64_t> shape, int64_t dim);

// --- scatter_src ---

KernelSpec buildScatterSrcKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim);

std::string generateScatterSrcMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim);

// --- scatter_add ---

KernelSpec buildScatterAddKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim);

std::string generateScatterAddMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> input_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim);

// --- scatter_src_inplace / scatter_add_inplace ---

KernelSpec buildScatterSrcInplaceKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> out_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim);

std::string generateScatterSrcInplaceMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> out_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim);

KernelSpec buildScatterAddInplaceKernelSpec(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> out_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim);

std::string generateScatterAddInplaceMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> out_shape, c10::ArrayRef<int64_t> index_shape,
    c10::ArrayRef<int64_t> src_shape, int64_t dim);

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
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters = {});

std::string generateComparisonMlir(
    const std::string& func_name, const std::string& torch_op,
    c10::ScalarType dtype,
    c10::ArrayRef<int64_t> lhs_shape, c10::ArrayRef<int64_t> rhs_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters = {});

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
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters = {});

std::string generateMmMlir(
    const std::string& func_name, c10::ScalarType dtype,
    c10::ArrayRef<int64_t> mat1_shape, c10::ArrayRef<int64_t> mat2_shape,
    c10::ArrayRef<int64_t> out_shape,
    c10::ArrayRef<ArgAdapter> adapters = {});

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
