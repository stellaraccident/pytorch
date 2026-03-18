#include <ATen/pyre/PyreOpJit.h>
#include <ATen/pyre/PyreTensor.h>
#include <ATen/pyre/dispatch/PyreArgAdapter.h>
#include <ATen/pyre/dispatch/PyreKernelAsmBuilder.h>
#include <ATen/pyre/dispatch/PyreKernelCache.h>
#include <ATen/pyre/dispatch/PyreKernelCompiler.h>
#include <ATen/pyre/dispatch/PyreKernelDispatch.h>
#include <ATen/pyre/dispatch/PyreTypeMapping.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <c10/pyre/impl/PyreDevice.h>

namespace at::pyre {

bool jitAvailable() {
  return PyreKernelCompiler::isAvailable();
}

// Compile-and-cache a kernel, returning the cached entry.
static CachedKernel* getOrCompile(
    const std::string& cache_key,
    const std::string& func_name,
    const std::string& mlir) {
  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (kernel) {
    PYRE_LOG(INFO) << "cache HIT: " << cache_key << "\n";
    return kernel;
  }

  PYRE_LOG(INFO) << "cache MISS: " << cache_key << ", compiling\n";
  PYRE_LOG(DEBUG) << "MLIR:\n" << mlir << "\n";

  auto vmfb = PyreKernelCompiler::compileSync(
      std::string(mlir),
      std::vector<std::string>(
          c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags().begin(),
          c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags().end()));

  PYRE_LOG(INFO) << "compiled " << vmfb->size() << " bytes\n";
  return cache.store(cache_key, func_name, std::move(vmfb));
}

static at::Tensor invokeKernel(
    CachedKernel* kernel,
    const std::vector<at::Tensor>& inputs,
    c10::IntArrayRef out_shape,
    const at::TensorOptions& opts) {
  auto output = at::empty(out_shape, opts);
  c10::pyre::PyreStream stream(c10::pyre::getCurrentHostStream(0));
  auto& ctx = stream.context();
  PyreKernelDispatch::invoke(kernel, inputs, output, ctx);
  return output;
}

// -------------------------------------------------------------------------- //
// Binary ops
// -------------------------------------------------------------------------- //

at::Tensor jitBinaryOp(
    const at::Tensor& self,
    const at::Tensor& other,
    const std::string& func_name,
    const std::string& linalg_op) {
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(other),
      "pyre: compiled ops require tensors with IREE buffers");

  auto dtype = self.scalar_type();
  auto out_shape = at::infer_size(self.sizes(), other.sizes());
  std::vector<ArgAdapter> adapters = {
      ArgAdapter::analyze(self), ArgAdapter::analyze(other)};
  auto self_c = adapters[0].kind == ArgAdapter::kContiguous
      ? self.contiguous() : self;
  auto other_c = adapters[1].kind == ArgAdapter::kContiguous
      ? other.contiguous() : other;

  PYRE_LOG(INFO) << func_name << " dtype=" << c10::toString(dtype)
                 << " " << self.sizes() << " x " << other.sizes() << "\n";

  auto mlir = expandBinaryTemplate(
      func_name, linalg_op, dtype,
      self_c.sizes(), other_c.sizes(), out_shape, adapters);

  auto cache_key = func_name + "::" + scalarTypeToTorchMlir(dtype)
                 + "::" + c10::pyre::PyreDevice::get(0)->capabilities().cacheKey();

  auto* kernel = getOrCompile(cache_key, func_name, mlir);
  return invokeKernel(kernel, {self_c, other_c}, out_shape, self.options());
}

at::Tensor jitAddOp(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(other),
      "pyre: compiled ops require tensors with IREE buffers");

  double alpha_val = alpha.toDouble();
  auto dtype = self.scalar_type();
  auto out_shape = at::infer_size(self.sizes(), other.sizes());

  std::string func_name = (std::abs(alpha_val - 1.0) < 1e-12)
      ? "pyre_add" : "pyre_add_alpha";

  std::vector<ArgAdapter> adapters = {
      ArgAdapter::analyze(self), ArgAdapter::analyze(other)};
  auto self_c = adapters[0].kind == ArgAdapter::kContiguous
      ? self.contiguous() : self;
  auto other_c = adapters[1].kind == ArgAdapter::kContiguous
      ? other.contiguous() : other;

  std::string mlir;
  if (func_name == "pyre_add") {
    mlir = expandBinaryTemplate(
        func_name, "add", dtype,
        self_c.sizes(), other_c.sizes(), out_shape, adapters);
  } else {
    mlir = expandBinaryAlphaTemplate(
        func_name,
        isFloatDtype(dtype) ? "arith.addf" : "arith.addi",
        isFloatDtype(dtype) ? "arith.mulf" : "arith.muli",
        alpha_val, dtype,
        self_c.sizes(), other_c.sizes(), out_shape, adapters);
  }

  auto cache_key = func_name + "::" + scalarTypeToTorchMlir(dtype)
                 + "::" + c10::pyre::PyreDevice::get(0)->capabilities().cacheKey();

  auto* kernel = getOrCompile(cache_key, func_name, mlir);
  return invokeKernel(kernel, {self_c, other_c}, out_shape, self.options());
}

at::Tensor jitSubOp(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(other),
      "pyre: compiled ops require tensors with IREE buffers");

  double alpha_val = alpha.toDouble();
  auto dtype = self.scalar_type();
  auto out_shape = at::infer_size(self.sizes(), other.sizes());

  std::string func_name = (std::abs(alpha_val - 1.0) < 1e-12)
      ? "pyre_sub" : "pyre_sub_alpha";

  std::vector<ArgAdapter> adapters = {
      ArgAdapter::analyze(self), ArgAdapter::analyze(other)};
  auto self_c = adapters[0].kind == ArgAdapter::kContiguous
      ? self.contiguous() : self;
  auto other_c = adapters[1].kind == ArgAdapter::kContiguous
      ? other.contiguous() : other;

  std::string mlir;
  if (func_name == "pyre_sub") {
    mlir = expandBinaryTemplate(
        func_name, "sub", dtype,
        self_c.sizes(), other_c.sizes(), out_shape, adapters);
  } else {
    mlir = expandBinaryAlphaTemplate(
        func_name,
        isFloatDtype(dtype) ? "arith.subf" : "arith.subi",
        isFloatDtype(dtype) ? "arith.mulf" : "arith.muli",
        alpha_val, dtype,
        self_c.sizes(), other_c.sizes(), out_shape, adapters);
  }

  auto cache_key = func_name + "::" + scalarTypeToTorchMlir(dtype)
                 + "::" + c10::pyre::PyreDevice::get(0)->capabilities().cacheKey();

  auto* kernel = getOrCompile(cache_key, func_name, mlir);
  return invokeKernel(kernel, {self_c, other_c}, out_shape, self.options());
}

// -------------------------------------------------------------------------- //
// Unary ops
// -------------------------------------------------------------------------- //

at::Tensor jitUnaryOp(
    const at::Tensor& self,
    const std::string& func_name,
    const std::string& torch_op) {
  TORCH_CHECK(hasPyreBuffer(self),
      "pyre: compiled ops require tensors with IREE buffers");

  auto dtype = self.scalar_type();
  ArgAdapter adapter = ArgAdapter::analyze(self);
  auto self_c = adapter.kind == ArgAdapter::kContiguous
      ? self.contiguous() : self;

  auto mlir = expandUnaryTemplate(
      func_name, torch_op, dtype,
      self_c.sizes(), self_c.sizes(), adapter);

  auto cache_key = func_name + "::" + scalarTypeToTorchMlir(dtype)
                 + "::" + c10::pyre::PyreDevice::get(0)->capabilities().cacheKey();

  auto* kernel = getOrCompile(cache_key, func_name, mlir);
  return invokeKernel(kernel, {self_c}, self_c.sizes(), self.options());
}

} // namespace at::pyre
