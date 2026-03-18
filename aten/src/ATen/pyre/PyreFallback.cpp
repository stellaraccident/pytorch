// Pyre kernel registrations for PrivateUse1 (host device).
//
// Phase 0: empty, fill, copy, and a CPU fallback for everything else.
// Fill and copy use HAL command buffer primitives with semaphore-based
// synchronization — no assumption that device memory is host-local.

#include <ATen/core/Tensor.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorShape.h>
#include <ATen/pyre/PyreTensor.h>
#include <ATen/pyre/dispatch/PyreArgAdapter.h>
#include <ATen/pyre/dispatch/PyreKernelCache.h>
#include <ATen/pyre/dispatch/PyreKernelCompiler.h>
#include <ATen/pyre/dispatch/PyreKernelDispatch.h>
#include <ATen/pyre/dispatch/PyreKernelLibrary.h>
#include <ATen/pyre/dispatch/PyreSpecKey.h>
#include <c10/pyre/impl/PyreDevice.h>

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

// --- Compiled elementwise ops ---
// These replace CPU fallback with IREE-compiled kernels.

bool compiled_dispatch_available() {
  static bool checked = false;
  static bool available = false;
  if (!checked) {
    available = PyreKernelCompiler::isAvailable();
    checked = true;
  }
  return available;
}

at::Tensor compiled_binary_op(
    const at::Tensor& self,
    const at::Tensor& other,
    const std::string& op_name,
    const std::string& linalg_op,
    const at::Scalar& alpha = at::Scalar(1)) {
  if (!compiled_dispatch_available()) {
    // Fall through to CPU fallback if compiler unavailable.
    auto result_cpu = at::add(self.cpu(), other.cpu(), alpha);
    return result_cpu.to(self.device());
  }

  auto dtype = self.scalar_type();
  auto out_shape = at::infer_size(self.sizes(), other.sizes());
  std::vector<ArgAdapter> adapters = {
      ArgAdapter::analyze(self),
      ArgAdapter::analyze(other),
  };
  auto self_c = adapters[0].kind == ArgAdapter::kContiguous
      ? self.contiguous() : self;
  auto other_c = adapters[1].kind == ArgAdapter::kContiguous
      ? other.contiguous() : other;

  PYRE_LOG(INFO) << "compiled_binary_op: " << op_name << " dtype="
                 << c10::toString(dtype) << " shape=" << self.sizes()
                 << " x " << other.sizes() << "\n";

  double alpha_val = alpha.toDouble();
  std::string mlir;
  std::string func_name = op_name;

  if (std::abs(alpha_val - 1.0) < 1e-12) {
    mlir = expandBinaryTemplate(
        func_name, linalg_op, dtype,
        self_c.sizes(), other_c.sizes(), out_shape, adapters);
  } else {
    func_name = op_name + "_alpha";
    std::string add_op = pyreIsFloatingType(dtype) ? "arith.addf" : "arith.addi";
    std::string mul_op = pyreIsFloatingType(dtype) ? "arith.mulf" : "arith.muli";
    if (linalg_op == "sub") {
      add_op = pyreIsFloatingType(dtype) ? "arith.subf" : "arith.subi";
    }
    mlir = expandBinaryAlphaTemplate(
        func_name, add_op, mul_op, alpha_val, dtype,
        self_c.sizes(), other_c.sizes(), out_shape, adapters);
  }

  auto& device = *c10::pyre::PyreDevice::get(0);
  auto cache_key = func_name + "::" + scalarTypeToMlir(dtype)
                 + "::" + device.capabilities().cacheKey();
  PYRE_LOG(DEBUG) << "cache_key=" << cache_key << " func=" << func_name << "\n";

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);

  if (kernel) {
    PYRE_LOG(INFO) << "cache HIT: " << cache_key << "\n";
  } else {
    PYRE_LOG(INFO) << "cache MISS: " << cache_key << ", compiling...\n";
    PYRE_LOG(DEBUG) << "expanded MLIR:\n" << mlir << "\n";
    auto vmfb = PyreKernelCompiler::compileSync(
        mlir, device.capabilities().compilerFlags());
    PYRE_LOG(INFO) << "compilation done, VMFB " << vmfb.size() << " bytes\n";
    kernel = cache.store(cache_key, func_name, vmfb);
  }

  auto output = at::empty(
      out_shape, self.options());
  c10::pyre::PyreStream stream(
      c10::pyre::getCurrentHostStream(0));
  auto& stream_ctx = stream.context();

  std::vector<at::Tensor> inputs = {self_c, other_c};
  PyreKernelDispatch::invokeAsync(kernel, inputs, output, stream_ctx);

  return output;
}

at::Tensor compiled_unary_op(
    const at::Tensor& self,
    const std::string& op_name,
    const std::string& scalar_op) {
  if (!compiled_dispatch_available()) {
    if (op_name == "neg") return (-self.cpu()).to(self.device());
    if (op_name == "abs") return self.cpu().abs().to(self.device());
    if (op_name == "relu") return self.cpu().relu().to(self.device());
    return self.cpu().to(self.device());
  }

  PYRE_LOG(INFO) << "compiled_unary_op: " << op_name << " dtype="
                 << c10::toString(self.scalar_type())
                 << " shape=" << self.sizes() << "\n";

  auto dtype = self.scalar_type();
  ArgAdapter adapter = ArgAdapter::analyze(self);
  auto self_c = adapter.kind == ArgAdapter::kContiguous
      ? self.contiguous() : self;

  auto mlir = expandUnaryTemplate(
      op_name, scalar_op, dtype,
      self_c.sizes(), self_c.sizes(), adapter);

  auto& device = *c10::pyre::PyreDevice::get(0);
  auto cache_key = op_name + "::" + scalarTypeToMlir(dtype)
                 + "::" + device.capabilities().cacheKey();
  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, op_name);

  if (!kernel) {
    auto vmfb = PyreKernelCompiler::compileSync(
        mlir, device.capabilities().compilerFlags());
    kernel = cache.store(cache_key, op_name, vmfb);
  }

  auto output = at::empty_like(self_c);
  c10::pyre::PyreStream stream(
      c10::pyre::getCurrentHostStream(0));
  auto& stream_ctx = stream.context();

  std::vector<at::Tensor> inputs = {self_c};
  PyreKernelDispatch::invokeAsync(kernel, inputs, output, stream_ctx);

  return output;
}

// --- Compiled op wrappers ---

at::Tensor pyre_add(const at::Tensor& self, const at::Tensor& other,
                     const at::Scalar& alpha) {
  return compiled_binary_op(self, other, "pyre_add", "add", alpha);
}

at::Tensor pyre_sub(const at::Tensor& self, const at::Tensor& other,
                     const at::Scalar& alpha) {
  return compiled_binary_op(self, other, "pyre_sub", "sub", alpha);
}

at::Tensor pyre_mul(const at::Tensor& self, const at::Tensor& other) {
  return compiled_binary_op(self, other, "pyre_mul", "mul");
}

at::Tensor pyre_div(const at::Tensor& self, const at::Tensor& other) {
  return compiled_binary_op(self, other, "pyre_div", "div");
}

at::Tensor pyre_neg(const at::Tensor& self) {
  return compiled_unary_op(self, "pyre_neg", "torch.aten.neg");
}

at::Tensor pyre_relu(const at::Tensor& self) {
  return compiled_unary_op(self, "pyre_relu", "torch.aten.relu");
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

  // Compiled elementwise ops (replace CPU fallback)
  m.impl("add.Tensor", pyre_add);
  m.impl("sub.Tensor", pyre_sub);
  m.impl("mul.Tensor", pyre_mul);
  m.impl("div.Tensor", pyre_div);
  m.impl("neg", pyre_neg);
  m.impl("relu", pyre_relu);
}

// Fallback: anything not registered above goes through CPU fallback.
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

} // namespace at::pyre
