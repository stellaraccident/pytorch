#pragma once

// AbiPacker: hot-path class for compiled kernel dispatch.
//
// Visits input/output tensors, builds a cache key string encoding buffer
// topology (storage aliasing, element offsets, alignment) and dim pattern
// (static/dynamic dims, divisibility). Packs VM args for IREE dispatch.
//
// No MLIR generation — that's AbiGenerator's job on cache miss.
// See docs/design/epic5_pyre_abi_envelope.md §4.

#include <ATen/Tensor.h>
#include <ATen/pyre/dispatch/PyreArgAdapter.h>
#include <ATen/pyre/dispatch/PyreKernelAsmBuilder.h>
#include <c10/pyre/impl/PyreHelpers.h>
#include <c10/pyre/impl/PyreStorage.h>
#include <c10/util/SmallVector.h>
#include <c10/util/hash.h>

#include <iree/hal/api.h>
#include <iree/vm/api.h>

#include <cstdint>
#include <string>

namespace at::pyre {

// Allocator alignment guarantee for non-sub-allocated tensors.
static constexpr int kAllocatorAlignment = 64;

// Per-tensor record of storage identity, offset, and adaptation.
struct TensorSlot {
  int buf_idx;            // index into unique_bufs_
  int64_t element_offset; // storage_offset() value
  int byte_alignment;     // gcd(kAllocatorAlignment, byte_offset)
  bool is_output;
  ArgAdapter adapter;     // permutation/identity from shape oracle
};

class AbiPacker {
 public:
  // Visit an input tensor. Detects storage identity, deduplicates buffers,
  // computes element offset and byte alignment class. Appends to cache key
  // material. Also runs the shape oracle (permutation detection).
  void visitInput(const at::Tensor& t);

  // Visit the output tensor. Detects if output aliases an input.
  void visitOutput(const at::Tensor& t);

  // After all visits: compute the cache key.
  // Incorporates compute identity + buffer topology + dim pattern + flags.
  std::string cacheKey(
      const char* compute_sha1,
      const SubstPairs& compute_subs,
      c10::ArrayRef<std::string> compiler_flags) const;

  // Pack args into VM list for dispatch (envelope calling convention).
  // Order: [bufs..., offsets..., dims..., out_bufs..., transients, wait, signal]
  void packArgs(
      iree_vm_list_t* args,
      iree_hal_buffer_t* transients,
      iree_hal_fence_t* wait,
      iree_hal_fence_t* signal) const;

  // Access the recorded slots (for AbiGenerator to use matching analysis).
  c10::ArrayRef<TensorSlot> slots() const { return slots_; }

  // Access per-tensor dynamic dim values (flattened, in visit order).
  const c10::SmallVector<int64_t, 16>& dynamicDims() const {
    return dynamic_dims_;
  }

  // Number of unique buffers.
  int numUniqueBuffers() const {
    return static_cast<int>(unique_bufs_.size());
  }

  // Cache key material accessors (for testing).
  const std::string& bufTopology() const { return buf_topology_; }
  const std::string& dimPattern() const { return dim_pattern_; }

  // Compute byte alignment: gcd(kAllocatorAlignment, element_offset * elem_size).
  static int computeByteAlignment(int64_t element_offset, int64_t elem_size);

 private:
  // Shared visit logic for input and output.
  void visitTensor(const at::Tensor& t, bool is_output);

  // Storage dedup: StorageImpl* → (buffer index, iree_hal_buffer_t*).
  struct UniqueBuffer {
    c10::StorageImpl* storage;
    iree_hal_buffer_t* buffer;  // non-owning
    int64_t total_elements;     // total elements in the allocation
  };
  c10::SmallVector<UniqueBuffer, 4> unique_bufs_;

  // Per-tensor slots in visit order.
  c10::SmallVector<TensorSlot, 8> slots_;

  // Dynamic dim values (flattened across all tensors, in visit order).
  c10::SmallVector<int64_t, 16> dynamic_dims_;

  // Cache key material — built incrementally during visits.
  std::string buf_topology_;
  std::string dim_pattern_;
};

} // namespace at::pyre
