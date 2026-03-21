// Pyre kernel registrations for PrivateUse1 (host device).
//
// Non-compiled ops (empty, fill, copy, view) live here.
// Compiled ops are registered via PyreOps.h / PyreOps.cpp.

#include <ATen/core/Tensor.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorShape.h>
#include <ATen/pyre/PyreTensor.h>
#include <ATen/pyre/PyreOps.h>
#include <ATen/pyre/dispatch/PyreKernels.h>
#include <ATen/pyre/dispatch/StridedCopyPlan.h>

#ifdef AT_PER_OPERATOR_HEADERS
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/chunk_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/expand_native.h>
#include <ATen/ops/narrow_native.h>
#include <ATen/ops/permute_native.h>
#include <ATen/ops/reshape_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/select_native.h>
#include <ATen/ops/slice_native.h>
#include <ATen/ops/split_native.h>
#include <ATen/ops/split_with_sizes_native.h>
#include <ATen/ops/t_native.h>
#include <ATen/ops/transpose_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/unsqueeze_native.h>
#include <ATen/ops/view_native.h>
#else
#include <ATen/NativeFunctions.h>
#endif

#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>

#include <torch/library.h>

#include <algorithm>
#include <cstring>

namespace at::pyre {
namespace {

// --- empty.memory_format ---

at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "pyre: only strided layout");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt),
      "pyre: no pinned memory");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_generic(
      size, allocator, pu1_dks, dtype, memory_format_opt);
}

// --- empty_strided ---

at::Tensor empty_strided(
    c10::IntArrayRef size, c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_strided_generic(
      size, stride, allocator, pu1_dks, dtype);
}

// --- fill_.Scalar ---

at::Tensor& fill_scalar(at::Tensor& self, const at::Scalar& value) {
  if (self.numel() == 0) return self;

  // Fast path: contiguous with element_size ≤ 4 → HAL pattern fill.
  // TODO: use is_non_overlapping_and_dense() instead of is_contiguous()
  // to also cover Fortran-contiguous and channels-last packed tensors.
  if (self.is_contiguous() && self.element_size() <= 4) {
    PyreTensor pt(self);
    uint8_t pattern[4] = {};
    AT_DISPATCH_ALL_TYPES_AND2(
        at::kHalf, at::kBFloat16, self.scalar_type(), "pyre_fill", [&] {
          if constexpr (sizeof(scalar_t) <= 4) {
            scalar_t val = value.to<scalar_t>();
            std::memcpy(pattern, &val, sizeof(scalar_t));
          }
        });
    pt.fill(pattern, self.element_size(),
            self.storage_offset() * self.element_size(), self.nbytes());
    return self;
  }

  // Compiled fill: handles non-contiguous and 8-byte types.
  executeCompiledFill(self, value);
  return self;
}

// Read a (possibly non-contiguous) device tensor to CPU as a contiguous tensor.
// Reads the entire underlying storage buffer, reconstructs the strided view
// on CPU, then calls .contiguous() on the CPU side. This avoids the recursion
// that .contiguous() on a device tensor would cause (it dispatches to _copy_from).
at::Tensor pyreReadToCpuContiguous(const at::Tensor& src) {
  PyreTensor pt(src);
  auto storage_bytes = src.storage().nbytes();
  auto cpu_storage = at::empty(
      {static_cast<int64_t>(storage_bytes / src.element_size())},
      src.options().device(at::kCPU));
  pt.readToHost(cpu_storage.data_ptr(), 0, storage_bytes);
  // Reconstruct the same strided view on CPU, then make contiguous.
  auto cpu_view = cpu_storage.as_strided(
      src.sizes(), src.strides(), src.storage_offset());
  return cpu_view.contiguous();
}

// --- _copy_from ---

at::Tensor pyre_copy_from(
    const at::Tensor& self, const at::Tensor& dst, bool /*non_blocking*/) {
  if (self.is_privateuseone() && dst.is_privateuseone()) {
    if (self.dtype() != dst.dtype()) {
      // Cross-dtype: route through CPU.
      auto cpu_src = self.to(at::kCPU).to(dst.dtype()).contiguous();
      PyreTensor dst_pt(dst);
      dst_pt.updateFromHost(cpu_src.data_ptr(),
          dst.storage_offset() * dst.element_size(),
          cpu_src.nbytes());
    } else {
      // Same-dtype device-to-device: three-tier copy strategy.
      auto plan = planCopy(
          self.sizes(), self.strides(), dst.strides(),
          self.storage_offset(), dst.storage_offset(),
          self.element_size());
      switch (plan.tier) {
        case CopyPlan::kSingleCopy:
        case CopyPlan::kDecomposed: {
          PyreTensor src_pt(self), dst_pt(dst);
          auto* src_ctx = static_cast<c10::pyre::PyreBufferContext*>(
              self.storage().data_ptr().get_context());
          auto* dst_ctx = static_cast<c10::pyre::PyreBufferContext*>(
              dst.storage().data_ptr().get_context());
          executeCopyPlan(plan, src_pt.buffer(), dst_pt.buffer(),
                          dst_pt.device(), src_ctx, dst_ctx);
          break;
        }
        case CopyPlan::kCompiledKernel:
          executeCompiledCopy(plan, self, dst);
          break;
      }
    }
  } else if (self.is_cpu()) {
    // CPU→device: make src contiguous, upload to dst.
    // If dst is non-contiguous, upload to temp then d2d copy.
    auto src = self.contiguous().to(dst.dtype());
    if (dst.is_contiguous()) {
      PyreTensor dst_pt(dst);
      dst_pt.updateFromHost(src.data_ptr(),
          dst.storage_offset() * dst.element_size(), src.nbytes());
    } else {
      auto tmp = at::empty(src.sizes(), dst.options());
      PyreTensor tmp_pt(tmp);
      tmp_pt.updateFromHost(src.data_ptr(), 0, src.nbytes());
      // d2d copy from contiguous tmp → non-contiguous dst.
      pyre_copy_from(tmp, dst, false);
    }
  } else {
    // pyre → CPU
    if (self.is_contiguous()) {
      PyreTensor src_pt(self);
      at::Tensor tmp = at::empty(self.sizes(), dst.options().dtype(self.dtype()));
      src_pt.readToHost(tmp.data_ptr(),
          self.storage_offset() * self.element_size(), self.nbytes());
      dst.copy_(tmp);
    } else {
      auto cpu_copy = pyreReadToCpuContiguous(self);
      dst.copy_(cpu_copy);
    }
  }
  return dst;
}

at::Tensor pyre_copy_from_and_resize(
    const at::Tensor& self, const at::Tensor& dst) {
  at::native::resize_(dst, self.sizes(), std::nullopt);
  return pyre_copy_from(self, dst, false);
}

// --- _local_scalar_dense ---

at::Scalar pyre_local_scalar_dense(const at::Tensor& self) {
  PyreTensor pt(self);
  alignas(8) uint8_t data[8] = {};
  pt.readToHost(data,
      self.storage_offset() * self.element_size(), self.element_size());
  return AT_DISPATCH_ALL_TYPES_AND3(
      at::kHalf, at::kBFloat16, at::kBool, self.scalar_type(),
      "pyre_local_scalar_dense", [&] {
        scalar_t val;
        std::memcpy(&val, data, sizeof(scalar_t));
        return at::Scalar(val);
      });
}

// --- View ops and clone (sorted by op name) ---

at::Tensor pyre_as_strided(
    const at::Tensor& self, c10::IntArrayRef size,
    c10::IntArrayRef stride, std::optional<int64_t> storage_offset) {
  return at::native::as_strided_tensorimpl(self, size, stride, storage_offset);
}
std::vector<at::Tensor> pyre_chunk(
    const at::Tensor& self, int64_t chunks, int64_t dim) {
  return at::native::chunk(self, chunks, dim);
}
at::Tensor pyre_clone(
    const at::Tensor& self,
    std::optional<at::MemoryFormat> memory_format) {
  TORCH_CHECK(!memory_format.has_value() ||
              *memory_format == at::MemoryFormat::Contiguous ||
              *memory_format == at::MemoryFormat::Preserve,
      "pyre: clone only supports Contiguous/Preserve memory format");
  auto output = at::empty(self.sizes(), self.options());
  output.copy_(self);
  return output;
}
at::Tensor pyre_expand(
    const at::Tensor& self, c10::IntArrayRef size, bool implicit) {
  return at::native::expand(self, size, implicit);
}
at::Tensor pyre_narrow_symint(
    const at::Tensor& self, int64_t dim,
    c10::SymInt start, c10::SymInt length) {
  return at::native::narrow_symint(self, dim, start, length);
}
at::Tensor pyre_permute(const at::Tensor& self, c10::IntArrayRef dims) {
  return at::native::permute(self, dims);
}
at::Tensor pyre_reshape(const at::Tensor& self, c10::SymIntArrayRef shape) {
  return at::native::reshape_symint(self, shape);
}
const at::Tensor& pyre_resize_(
    const at::Tensor& self, c10::SymIntArrayRef size,
    std::optional<at::MemoryFormat> memory_format) {
  return at::native::resize__symint(self, size, memory_format);
}
at::Tensor pyre_select(const at::Tensor& self, int64_t dim, c10::SymInt index) {
  return at::native::select_symint(self, dim, index);
}
at::Tensor pyre_slice(
    const at::Tensor& self, int64_t dim,
    std::optional<int64_t> start, std::optional<int64_t> end, int64_t step) {
  return at::native::slice(self, dim, start, end, step);
}
std::vector<at::Tensor> pyre_split(
    const at::Tensor& self, int64_t split_size, int64_t dim) {
  return at::native::split(self, split_size, dim);
}
std::vector<at::Tensor> pyre_split_with_sizes(
    const at::Tensor& self, at::IntArrayRef split_sizes, int64_t dim) {
  return at::native::split_with_sizes(self, split_sizes, dim);
}
at::Tensor pyre_t(const at::Tensor& self) {
  return at::native::t(self);
}
at::Tensor pyre_transpose(const at::Tensor& self, int64_t dim0, int64_t dim1) {
  return at::native::transpose(self, dim0, dim1);
}
at::Tensor pyre_unfold(const at::Tensor& self, int64_t dim, int64_t size, int64_t step) {
  return at::native::unfold(self, dim, size, step);
}
at::Tensor pyre_unsqueeze(const at::Tensor& self, int64_t dim) {
  return at::native::unsqueeze(self, dim);
}
at::Tensor pyre_view(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

// --- arange overload forwarding ---

at::Tensor pyre_arange(
    const at::Scalar& end,
    std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
    std::optional<at::Device> device, std::optional<bool> pin_memory) {
  return at::arange(at::Scalar(0), end, at::Scalar(1),
                    dtype, layout, device, pin_memory);
}

at::Tensor pyre_arange_start(
    const at::Scalar& start, const at::Scalar& end,
    std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
    std::optional<at::Device> device, std::optional<bool> pin_memory) {
  return at::arange(start, end, at::Scalar(1),
                    dtype, layout, device, pin_memory);
}

// --- CPU fallback ---

void cpu_fallback(
    const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // Factory ops
  m.impl("arange", pyre_arange);
  m.impl("arange.start", pyre_arange_start);

  // Storage and data movement
  m.impl("_copy_from", pyre_copy_from);
  m.impl("_copy_from_and_resize", pyre_copy_from_and_resize);
  m.impl("_local_scalar_dense", pyre_local_scalar_dense);
  m.impl("empty.memory_format", empty_memory_format);
  m.impl("empty_strided", empty_strided);
  m.impl("fill_.Scalar", fill_scalar);
  m.impl("resize_", pyre_resize_);

  // View ops and clone (sorted by op name)
  m.impl("as_strided", pyre_as_strided);
  m.impl("chunk", pyre_chunk);
  m.impl("clone", pyre_clone);
  m.impl("expand", pyre_expand);
  m.impl("narrow", pyre_narrow_symint);
  m.impl("permute", pyre_permute);
  m.impl("reshape", pyre_reshape);
  m.impl("select.int", pyre_select);
  m.impl("slice.Tensor", pyre_slice);
  m.impl("split.Tensor", pyre_split);
  m.impl("split_with_sizes", pyre_split_with_sizes);
  m.impl("t", pyre_t);
  m.impl("transpose.int", pyre_transpose);
  m.impl("unfold", pyre_unfold);
  m.impl("unsqueeze", pyre_unsqueeze);
  m.impl("view", pyre_view);

  // Compiled ops (CRTP registry)
  registerCompiledOps(m);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

} // namespace at::pyre
