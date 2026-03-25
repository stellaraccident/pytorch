#pragma once

// AbiGenerator: miss-path MLIR generation for pyre native ABI.
//
// Generates the complete MLIR module: envelope function (util.func public)
// wrapping a compute function (func.func private). The envelope handles
// buffer aliasing, byte offsets, permutations, transients, and fences.
// The compute function contains only logical torch dialect ops.
//
// See docs/design/epic5_pyre_abi_envelope.md §5.

#include <ATen/Tensor.h>
#include <ATen/pyre/dispatch/PyreAbiPacker.h>
#include <ATen/pyre/dispatch/PyreKernelAsmBuilder.h>
#include <c10/util/SmallVector.h>

#include <cstdint>
#include <string>

namespace at::pyre {

class AbiGenerator {
 public:
  // Visit input/output tensors. Must be called in the same order as AbiPacker.
  void visitInput(const at::Tensor& t);
  void visitOutput(const at::Tensor& t);

  // Generate the complete MLIR module.
  // envelope_name: function name for the envelope (e.g. "add_f32").
  // body: the compute body from generateXxxComputeBody().
  // Returns complete MLIR module string ready for iree-compile.
  std::string generateModule(
      const std::string& envelope_name,
      const ComputeBody& body) const;

 private:
  void visitTensor(const at::Tensor& t, bool is_output);

  // Per-tensor record for MLIR generation.
  struct TensorInfo {
    int buf_idx;            // index into unique_bufs_
    int64_t element_offset; // storage_offset()
    int byte_alignment;     // gcd alignment class
    bool is_output;
    ArgAdapter adapter;     // permutation detection
    // Tensor shape/type info for MLIR generation.
    c10::SmallVector<int64_t, 6> sizes;       // logical (what PyTorch reports)
    c10::SmallVector<int64_t, 6> phys_sizes;  // physical (memory layout order)
    c10::ScalarType dtype;
    int64_t elem_size;      // bytes per element
  };

  struct UniqueBuffer {
    c10::StorageImpl* storage;
    int64_t total_elements;  // total elements in the allocation
    c10::ScalarType dtype;   // element type (for flat 1D import)
  };

  c10::SmallVector<UniqueBuffer, 4> unique_bufs_;
  c10::SmallVector<TensorInfo, 8> tensors_;

  // MLIR generation helpers.
  std::string emitComputeFunction(
      const ComputeBody& body) const;
  std::string emitEnvelopeFunction(
      const std::string& envelope_name,
      const ComputeBody& body) const;

  // Generate the flat tensor type string for a buffer (e.g. "tensor<?xf32>").
  static std::string flatTensorType(c10::ScalarType dtype);
  // Generate shaped tensor type string (e.g. "tensor<?x4xf32>").
  static std::string shapedTensorType(
      c10::ArrayRef<int64_t> sizes, c10::ScalarType dtype);
  // Convert torch-mlir element to builtin (f32, i32, bf16, etc.)
  static std::string builtinElementType(c10::ScalarType dtype);
};

} // namespace at::pyre
