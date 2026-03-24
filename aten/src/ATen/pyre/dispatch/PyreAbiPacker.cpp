#include <ATen/pyre/dispatch/PyreAbiPacker.h>
#include <ATen/pyre/PyreTensor.h>
#include <ATen/pyre/dispatch/PyreTypeMapping.h>
#include <c10/pyre/impl/PyreRuntime.h>

#include <iree/hal/buffer_view.h>
#include <iree/modules/hal/types.h>

#include <algorithm>

namespace at::pyre {

// ---------------------------------------------------------------------------
// Byte alignment computation
// ---------------------------------------------------------------------------

int AbiPacker::computeByteAlignment(
    int64_t element_offset, int64_t elem_size) {
  if (element_offset == 0) return kAllocatorAlignment;
  int64_t byte_offset = element_offset * elem_size;
  // GCD of allocator alignment and byte offset.
  int64_t a = kAllocatorAlignment;
  int64_t b = byte_offset < 0 ? -byte_offset : byte_offset;
  while (b != 0) {
    int64_t t = b;
    b = a % b;
    a = t;
  }
  // Clamp to power-of-two alignment values the compiler understands.
  // Common values: 1, 2, 4, 8, 16, 32, 64.
  int align = static_cast<int>(a);
  // Round down to nearest power of 2.
  int result = 1;
  while (result * 2 <= align) result *= 2;
  return std::min(result, kAllocatorAlignment);
}

// ---------------------------------------------------------------------------
// Tensor visit
// ---------------------------------------------------------------------------

void AbiPacker::visitTensor(const at::Tensor& t, bool is_output) {
  auto* storage_impl = t.storage().unsafeGetStorageImpl();

  // Deduplicate by StorageImpl*.
  int buf_idx = -1;
  for (int i = 0; i < static_cast<int>(unique_bufs_.size()); ++i) {
    if (unique_bufs_[i].storage == storage_impl) {
      buf_idx = i;
      break;
    }
  }
  if (buf_idx < 0) {
    buf_idx = static_cast<int>(unique_bufs_.size());
    auto* ctx = static_cast<c10::pyre::PyreBufferContext*>(
        t.storage().data_ptr().get_context());
    TORCH_CHECK(ctx && ctx->buffer,
        "pyre: AbiPacker requires tensors with IREE buffers");
    int64_t total_bytes = static_cast<int64_t>(t.storage().nbytes());
    int64_t total_elems = total_bytes / t.element_size();
    unique_bufs_.push_back({storage_impl, ctx->buffer.get(), total_elems});
  }

  int64_t element_offset = t.storage_offset();
  int64_t elem_size = t.element_size();
  int byte_alignment = computeByteAlignment(element_offset, elem_size);

  // Shape oracle: detect permutations.
  ArgAdapter adapter = ArgAdapter::analyze(t);

  // Compute physical sizes BEFORE moving adapter into slot.
  // For permuted tensors, physical layout differs from logical.
  auto logical_sizes = t.sizes();
  c10::SmallVector<int64_t, 6> phys_sizes;
  bool is_permuted = (adapter.kind == ArgAdapter::kPermute);
  if (is_permuted) {
    phys_sizes.resize(logical_sizes.size());
    for (size_t d = 0; d < logical_sizes.size(); ++d)
      phys_sizes[d] = logical_sizes[adapter.permutation[d]];
  }
  auto sizes = is_permuted
      ? c10::ArrayRef<int64_t>(phys_sizes)
      : c10::ArrayRef<int64_t>(logical_sizes);

  slots_.push_back({buf_idx, element_offset, byte_alignment, is_output,
                    std::move(adapter)});

  // --- Build buf_topology_ ---
  if (!buf_topology_.empty()) buf_topology_ += ',';
  buf_topology_ += 'b';
  buf_topology_ += std::to_string(buf_idx);
  buf_topology_ += ':';
  if (element_offset == 0) {
    buf_topology_ += '0';
  } else {
    buf_topology_ += 'c';
    buf_topology_ += std::to_string(byte_alignment);
  }
  // Encode permutation in topology (affects envelope MLIR).
  if (is_permuted) {
    buf_topology_ += 'p';
    const auto& perm = slots_.back().adapter.permutation;
    for (size_t i = 0; i < perm.size(); ++i) {
      buf_topology_ += std::to_string(perm[i]);
    }
  }

  for (int64_t i = 0; i < t.dim(); ++i) {
    if (!dim_pattern_.empty()) dim_pattern_ += ',';
    int64_t s = sizes[i];
    if (s == 1) {
      dim_pattern_ += '1';
    } else {
      dim_pattern_ += '?';
      if (!is_output) {
        dynamic_dims_.push_back(s);
      }
    }
  }
}

void AbiPacker::visitInput(const at::Tensor& t) {
  visitTensor(t, /*is_output=*/false);
}

void AbiPacker::visitOutput(const at::Tensor& t) {
  visitTensor(t, /*is_output=*/true);
}

// ---------------------------------------------------------------------------
// Cache key
// ---------------------------------------------------------------------------

std::string AbiPacker::cacheKey(
    const char* compute_sha1,
    const SubstPairs& compute_subs,
    c10::ArrayRef<std::string> compiler_flags) const {
  std::string to_hash;
  to_hash.reserve(256);

  // Compute identity.
  to_hash += compute_sha1;
  to_hash += '\0';
  for (const auto& [key, value] : compute_subs) {
    to_hash += key;
    to_hash += '=';
    to_hash += value;
    to_hash += '\0';
  }

  // Buffer topology.
  to_hash += buf_topology_;
  to_hash += '\0';

  // Dim pattern.
  to_hash += dim_pattern_;
  to_hash += '\0';

  // Compiler flags.
  for (const auto& flag : compiler_flags) {
    to_hash += flag;
    to_hash += '\0';
  }

  return c10::sha1(to_hash).str();
}

// ---------------------------------------------------------------------------
// Arg packing
// ---------------------------------------------------------------------------

void AbiPacker::packArgs(
    iree_vm_list_t* args,
    iree_hal_buffer_t* transients,
    iree_hal_fence_t* wait,
    iree_hal_fence_t* signal) const {
  // Arg order (envelope convention):
  // [buf0, buf0_elems, buf1, buf1_elems, ...,
  //  element_offsets..., dynamic_dims...,
  //  output_buffers..., transients, wait_fence, signal_fence]

  // Determine which unique buffers are used by inputs (not output-only).
  c10::SmallVector<bool, 4> buf_used_by_input(unique_bufs_.size(), false);
  for (const auto& slot : slots_) {
    if (!slot.is_output)
      buf_used_by_input[slot.buf_idx] = true;
  }

  // 1. Unique input buffers as opaque !hal.buffer + parent element count.
  // Output-only buffers go in the output_buffers section.
  for (int i = 0; i < static_cast<int>(unique_bufs_.size()); ++i) {
    if (!buf_used_by_input[i]) continue;
    // Push !hal.buffer ref.
    iree_vm_ref_t ref = iree_hal_buffer_retain_ref(unique_bufs_[i].buffer);
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
    // Push parent element count (index).
    iree_vm_value_t val = iree_vm_value_make_i64(
        unique_bufs_[i].total_elements);
    PYRE_CHECK_OK(iree_vm_list_push_value(args, &val));
  }

  // 2. Element offsets (only for non-zero offsets).
  for (const auto& slot : slots_) {
    if (slot.element_offset != 0) {
      iree_vm_value_t val = iree_vm_value_make_i64(slot.element_offset);
      PYRE_CHECK_OK(iree_vm_list_push_value(args, &val));
    }
  }

  // 3. Dynamic dims.
  for (int64_t d : dynamic_dims_) {
    iree_vm_value_t val = iree_vm_value_make_i64(d);
    PYRE_CHECK_OK(iree_vm_list_push_value(args, &val));
  }

  // 4. Output buffers (as !hal.buffer, not buffer_view).
  for (const auto& slot : slots_) {
    if (slot.is_output) {
      iree_hal_buffer_t* buf = unique_bufs_[slot.buf_idx].buffer;
      iree_vm_ref_t ref = iree_hal_buffer_retain_ref(buf);
      PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
    }
  }

  // 5. Transients.
  if (transients) {
    iree_vm_ref_t ref = iree_hal_buffer_retain_ref(transients);
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  } else {
    iree_vm_ref_t null_ref = iree_vm_ref_null();
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &null_ref));
  }

  // 6. Wait fence.
  if (wait) {
    iree_vm_ref_t ref = iree_hal_fence_retain_ref(wait);
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  } else {
    iree_vm_ref_t ref = iree_vm_ref_null();
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  }

  // 7. Signal fence.
  if (signal) {
    iree_vm_ref_t ref = iree_hal_fence_retain_ref(signal);
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  } else {
    iree_vm_ref_t ref = iree_vm_ref_null();
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  }
}

} // namespace at::pyre
