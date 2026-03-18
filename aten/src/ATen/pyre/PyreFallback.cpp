// Pyre kernel registrations for PrivateUse1 (host device).
//
// Phase 0: empty, fill, copy, and a CPU fallback for everything else.
// Fill and copy use HAL command buffer primitives with semaphore-based
// synchronization — no assumption that device memory is host-local.

#include <ATen/core/Tensor.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorShape.h>
#include <ATen/pyre/PyreTensor.h>

#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>

#include <torch/library.h>

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
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "pyre: only strided layout is supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "pyre: pinned memory not supported on host device");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_generic(
      size, allocator, pu1_dks, dtype, memory_format_opt);
}

// --- empty_strided ---

at::Tensor empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
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
  TORCH_CHECK(self.is_contiguous(),
      "pyre: fill on non-contiguous tensors not yet supported");
  TORCH_CHECK(self.element_size() <= 4,
      "pyre: fill not yet supported for ", self.dtype(),
      " (element size ", self.element_size(), " > 4 bytes). "
      "Use CPU tensor and copy.");

  PyreTensor pt(self);
  alignas(4) uint8_t pattern[4] = {};
  AT_DISPATCH_ALL_TYPES_AND2(
      at::kHalf, at::kBFloat16, self.scalar_type(), "pyre_fill", [&] {
        if constexpr (sizeof(scalar_t) <= 4) {
          scalar_t val = value.to<scalar_t>();
          std::memcpy(pattern, &val, sizeof(scalar_t));
        }
      });

  pt.fill(pattern, self.element_size(),
          self.storage_offset() * self.element_size(),
          self.nbytes());
  return self;
}

// --- _copy_from (PrivateUse1 ↔ CPU) ---

at::Tensor pyre_copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  if (self.is_privateuseone() && dst.is_privateuseone()) {
    // pyre → pyre
    TORCH_CHECK(self.is_contiguous() && dst.is_contiguous(),
        "pyre: non-contiguous device-to-device copy not yet supported");
    if (self.dtype() != dst.dtype()) {
      // Cross-dtype on device: route through CPU for conversion.
      auto tmp = self.to(at::kCPU).to(dst.dtype()).to(dst.device());
      PyreTensor tmp_pt(tmp), dst_pt(dst);
      dst_pt.copyFrom(tmp_pt,
          tmp.storage_offset() * tmp.element_size(),
          dst.storage_offset() * dst.element_size(),
          tmp.nbytes());
    } else {
      PyreTensor src_pt(self), dst_pt(dst);
      dst_pt.copyFrom(src_pt,
          self.storage_offset() * self.element_size(),
          dst.storage_offset() * dst.element_size(),
          self.nbytes());
    }
  } else if (self.is_cpu()) {
    // CPU → pyre
    TORCH_CHECK(dst.is_contiguous(),
        "pyre: copy to non-contiguous device tensor not yet supported");
    auto src = self.contiguous().to(dst.dtype());  // CPU-only ops
    PyreTensor dst_pt(dst);
    dst_pt.updateFromHost(src.data_ptr(),
        dst.storage_offset() * dst.element_size(),
        src.nbytes());
  } else {
    // pyre → CPU
    TORCH_CHECK(self.is_contiguous(),
        "pyre: copy from non-contiguous device tensor not yet supported");
    PyreTensor src_pt(self);
    at::Tensor tmp = at::empty(
        self.sizes(), dst.options().dtype(self.dtype()));
    src_pt.readToHost(tmp.data_ptr(),
        self.storage_offset() * self.element_size(),
        self.nbytes());
    // Handle dtype conversion and strides on CPU side.
    dst.copy_(tmp);
  }
  return dst;
}

// --- _copy_from_and_resize ---

at::Tensor pyre_copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  at::native::resize_(dst, self.sizes(), std::nullopt);
  return pyre_copy_from(self, dst, false);
}

// --- _local_scalar_dense ---

at::Scalar pyre_local_scalar_dense(const at::Tensor& self) {
  PyreTensor pt(self);
  alignas(8) uint8_t data[8] = {};
  pt.readToHost(data,
      self.storage_offset() * self.element_size(),
      self.element_size());

  return AT_DISPATCH_ALL_TYPES_AND3(
      at::kHalf, at::kBFloat16, at::kBool, self.scalar_type(),
      "pyre_local_scalar_dense", [&] {
        scalar_t val;
        std::memcpy(&val, data, sizeof(scalar_t));
        return at::Scalar(val);
      });
}

// --- View ops ---
// View ops are metadata-only: they share the same Storage (and IREE buffer)
// with different sizes/strides/offset.
//
// These must be registered natively because the CPU fallback cannot handle
// view ops correctly. The fallback copies data to CPU, runs the op, and
// tags the result as device='host:0' — but the result's Storage points at
// CPU memory, not an IREE buffer. Any subsequent pyre operation on that
// tensor fails because PyreBufferContext is absent from the DataPtr.
// See CPUFallback.cpp "Note [CPU Fallback Does Not Handle View Operators]".

at::Tensor pyre_as_strided(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<int64_t> storage_offset) {
  return at::native::as_strided_tensorimpl(self, size, stride, storage_offset);
}

at::Tensor pyre_view(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

at::Tensor pyre_reshape(const at::Tensor& self, c10::SymIntArrayRef shape) {
  return at::native::reshape_symint(self, shape);
}

at::Tensor pyre_expand(
    const at::Tensor& self,
    c10::IntArrayRef size,
    bool implicit) {
  return at::native::expand(self, size, implicit);
}

at::Tensor pyre_permute(const at::Tensor& self, c10::IntArrayRef dims) {
  return at::native::permute(self, dims);
}

at::Tensor pyre_t(const at::Tensor& self) {
  return at::native::t(self);
}

at::Tensor pyre_unsqueeze(const at::Tensor& self, int64_t dim) {
  return at::native::unsqueeze(self, dim);
}

at::Tensor pyre_slice(
    const at::Tensor& self,
    int64_t dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    int64_t step) {
  return at::native::slice(self, dim, start, end, step);
}

const at::Tensor& pyre_resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    std::optional<at::MemoryFormat> memory_format) {
  return at::native::resize__symint(self, size, memory_format);
}

// --- Functional unary op helpers ---
// Some functional ops (e.g. abs) create a size-0 output then call the
// .out variant. The CPU fallback can't handle resize on a pyre output.
// Fix: allocate the output with the correct size before delegating.

at::Tensor pyre_abs(const at::Tensor& self) {
  auto result = at::empty_like(self);
  return at::abs_out(result, self);
}

// --- CPU fallback ---

void cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

} // namespace

// Register specific implementations.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", empty_memory_format);
  m.impl("empty_strided", empty_strided);
  m.impl("fill_.Scalar", fill_scalar);
  m.impl("_copy_from", pyre_copy_from);
  m.impl("_copy_from_and_resize", pyre_copy_from_and_resize);
  m.impl("_local_scalar_dense", pyre_local_scalar_dense);

  // View ops (metadata-only, share storage)
  m.impl("as_strided", pyre_as_strided);
  m.impl("view", pyre_view);
  m.impl("reshape", pyre_reshape);
  m.impl("expand", pyre_expand);
  m.impl("permute", pyre_permute);
  m.impl("t", pyre_t);
  m.impl("unsqueeze", pyre_unsqueeze);
  m.impl("slice.Tensor", pyre_slice);
  m.impl("resize_", pyre_resize_);

  // Functional ops that need proper output allocation
  m.impl("abs", pyre_abs);
}

// Fallback: anything not registered above goes through CPU fallback.
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

} // namespace at::pyre
