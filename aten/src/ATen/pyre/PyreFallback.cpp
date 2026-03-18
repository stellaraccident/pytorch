// Pyre kernel registrations for PrivateUse1 (host device).

#include <ATen/core/Tensor.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorShape.h>
#include <ATen/pyre/PyreTensor.h>
#include <ATen/pyre/PyreOpJit.h>

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
  TORCH_CHECK(self.is_contiguous(),
      "pyre: fill on non-contiguous not supported");
  TORCH_CHECK(self.element_size() <= 4,
      "pyre: fill not supported for ", self.dtype());

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
          self.storage_offset() * self.element_size(), self.nbytes());
  return self;
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
    } else if (!self.is_contiguous() || !dst.is_contiguous()) {
      TORCH_CHECK(false, "pyre: non-contiguous d2d copy not yet supported. "
          "Use .contiguous() before copying between device tensors.");
    } else {
      PyreTensor src_pt(self), dst_pt(dst);
      dst_pt.copyFrom(src_pt,
          self.storage_offset() * self.element_size(),
          dst.storage_offset() * dst.element_size(), self.nbytes());
    }
  } else if (self.is_cpu()) {
    TORCH_CHECK(dst.is_contiguous(), "pyre: copy to non-contiguous not supported");
    auto src = self.contiguous().to(dst.dtype());
    PyreTensor dst_pt(dst);
    dst_pt.updateFromHost(src.data_ptr(),
        dst.storage_offset() * dst.element_size(), src.nbytes());
  } else {
    // pyre → CPU: make contiguous if needed.
    auto self_c = self.is_contiguous() ? self : self.contiguous();
    PyreTensor src_pt(self_c);
    at::Tensor tmp = at::empty(self_c.sizes(), dst.options().dtype(self_c.dtype()));
    src_pt.readToHost(tmp.data_ptr(),
        self_c.storage_offset() * self_c.element_size(), self_c.nbytes());
    dst.copy_(tmp);
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

// --- View ops ---

at::Tensor pyre_as_strided(
    const at::Tensor& self, c10::IntArrayRef size,
    c10::IntArrayRef stride, std::optional<int64_t> storage_offset) {
  return at::native::as_strided_tensorimpl(self, size, stride, storage_offset);
}
at::Tensor pyre_view(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}
at::Tensor pyre_reshape(const at::Tensor& self, c10::SymIntArrayRef shape) {
  return at::native::reshape_symint(self, shape);
}
at::Tensor pyre_expand(
    const at::Tensor& self, c10::IntArrayRef size, bool implicit) {
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
    const at::Tensor& self, int64_t dim,
    std::optional<int64_t> start, std::optional<int64_t> end, int64_t step) {
  return at::native::slice(self, dim, start, end, step);
}
const at::Tensor& pyre_resize_(
    const at::Tensor& self, c10::SymIntArrayRef size,
    std::optional<at::MemoryFormat> memory_format) {
  return at::native::resize__symint(self, size, memory_format);
}
at::Tensor pyre_abs(const at::Tensor& self) {
  auto result = at::empty_like(self);
  return at::abs_out(result, self);
}

// --- Compiled elementwise ops ---

at::Tensor pyre_add(const at::Tensor& self, const at::Tensor& other,
                     const at::Scalar& alpha) {
  if (!jitAvailable()) {
    return at::add(self.cpu(), other.cpu(), alpha).to(self.device());
  }
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(other),
      "pyre: add requires tensors with IREE buffers, not CPU fallback tensors");
  return jitAddOp(self, other, alpha);
}

at::Tensor pyre_sub(const at::Tensor& self, const at::Tensor& other,
                     const at::Scalar& alpha) {
  if (!jitAvailable())
    return at::sub(self.cpu(), other.cpu(), alpha).to(self.device());
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(other),
      "pyre: sub requires tensors with IREE buffers");
  return jitSubOp(self, other, alpha);
}

at::Tensor pyre_mul(const at::Tensor& self, const at::Tensor& other) {
  if (!jitAvailable())
    return at::mul(self.cpu(), other.cpu()).to(self.device());
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(other),
      "pyre: mul requires tensors with IREE buffers");
  return jitBinaryOp(self, other, "pyre_mul", "mul");
}

at::Tensor pyre_div(const at::Tensor& self, const at::Tensor& other) {
  if (!jitAvailable())
    return at::div(self.cpu(), other.cpu()).to(self.device());
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(other),
      "pyre: div requires tensors with IREE buffers");
  return jitBinaryOp(self, other, "pyre_div", "div");
}

at::Tensor pyre_neg(const at::Tensor& self) {
  if (!jitAvailable())
    return (-self.cpu()).to(self.device());
  TORCH_CHECK(hasPyreBuffer(self),
      "pyre: neg requires tensor with IREE buffer");
  return jitUnaryOp(self, "pyre_neg", "torch.aten.neg");
}

at::Tensor pyre_addmm(
    const at::Tensor& bias, const at::Tensor& mat1, const at::Tensor& mat2,
    const at::Scalar& beta, const at::Scalar& alpha) {
  if (!jitAvailable())
    return at::addmm(bias.cpu(), mat1.cpu(), mat2.cpu(), beta, alpha).to(mat1.device());
  TORCH_CHECK(hasPyreBuffer(bias) && hasPyreBuffer(mat1) && hasPyreBuffer(mat2),
      "pyre: addmm requires tensors with IREE buffers");
  return jitAddmmOp(bias, mat1, mat2, beta, alpha);
}

at::Tensor pyre_mm(const at::Tensor& self, const at::Tensor& other) {
  if (!jitAvailable())
    return at::mm(self.cpu(), other.cpu()).to(self.device());
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(other),
      "pyre: mm requires tensors with IREE buffers");
  return jitMmOp(self, other);
}

at::Tensor pyre_relu(const at::Tensor& self) {
  if (!jitAvailable())
    return self.cpu().relu().to(self.device());
  TORCH_CHECK(hasPyreBuffer(self),
      "pyre: relu requires tensor with IREE buffer");
  return jitUnaryOp(self, "pyre_relu", "torch.aten.relu");
}

// --- CPU fallback ---

void cpu_fallback(
    const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", empty_memory_format);
  m.impl("empty_strided", empty_strided);
  m.impl("fill_.Scalar", fill_scalar);
  m.impl("_copy_from", pyre_copy_from);
  m.impl("_copy_from_and_resize", pyre_copy_from_and_resize);
  m.impl("_local_scalar_dense", pyre_local_scalar_dense);

  // View ops
  m.impl("as_strided", pyre_as_strided);
  m.impl("view", pyre_view);
  m.impl("reshape", pyre_reshape);
  m.impl("expand", pyre_expand);
  m.impl("permute", pyre_permute);
  m.impl("t", pyre_t);
  m.impl("unsqueeze", pyre_unsqueeze);
  m.impl("slice.Tensor", pyre_slice);
  m.impl("resize_", pyre_resize_);
  m.impl("abs", pyre_abs);

  // Compiled elementwise ops
  m.impl("add.Tensor", pyre_add);
  m.impl("sub.Tensor", pyre_sub);
  m.impl("mul.Tensor", pyre_mul);
  m.impl("div.Tensor", pyre_div);
  m.impl("addmm", pyre_addmm);
  m.impl("mm", pyre_mm);
  m.impl("neg", pyre_neg);
  m.impl("relu", pyre_relu);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

} // namespace at::pyre
