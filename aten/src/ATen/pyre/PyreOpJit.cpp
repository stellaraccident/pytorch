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

// Encode broadcast pattern for cache key — "1" for size-1 dims, "d" otherwise.
// Must be consistent with broadcastAwareShapeStr in PyreKernelAsmBuilder.
static std::string broadcastKey(c10::IntArrayRef sizes) {
  std::string r;
  for (auto d : sizes) r += (d == 1) ? "1" : "d";
  return r;
}

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
                 + "::" + std::to_string(out_shape.size())
                 + "::" + broadcastKey(self_c.sizes()) + "+" + broadcastKey(other_c.sizes())
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

  // Alpha value is baked into the MLIR constant — different values need
  // different compiled kernels.
  std::string alpha_suffix;
  if (func_name != "pyre_add")
    alpha_suffix = "::a" + std::to_string(static_cast<int64_t>(alpha_val));

  auto cache_key = func_name + "::" + scalarTypeToTorchMlir(dtype)
                 + "::" + std::to_string(out_shape.size())
                 + "::" + broadcastKey(self_c.sizes()) + "+" + broadcastKey(other_c.sizes())
                 + alpha_suffix
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

  std::string alpha_suffix;
  if (func_name != "pyre_sub")
    alpha_suffix = "::a" + std::to_string(static_cast<int64_t>(alpha_val));

  auto cache_key = func_name + "::" + scalarTypeToTorchMlir(dtype)
                 + "::" + std::to_string(out_shape.size())
                 + "::" + broadcastKey(self_c.sizes()) + "+" + broadcastKey(other_c.sizes())
                 + alpha_suffix
                 + "::" + c10::pyre::PyreDevice::get(0)->capabilities().cacheKey();

  auto* kernel = getOrCompile(cache_key, func_name, mlir);
  return invokeKernel(kernel, {self_c, other_c}, out_shape, self.options());
}

// -------------------------------------------------------------------------- //
// Matrix multiply
// -------------------------------------------------------------------------- //

at::Tensor jitMmOp(
    const at::Tensor& self,
    const at::Tensor& other) {
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(other),
      "pyre: mm requires tensors with IREE buffers");
  TORCH_CHECK(self.dim() == 2 && other.dim() == 2,
      "pyre: mm requires 2D tensors");
  TORCH_CHECK(self.size(1) == other.size(0),
      "pyre: mm dimension mismatch");

  auto dtype = self.scalar_type();
  std::vector<int64_t> out_shape = {self.size(0), other.size(1)};

  auto mlir = expandBinaryTemplate(
      "pyre_mm", "mm", dtype,
      self.sizes(), other.sizes(), out_shape, {});

  auto cache_key = std::string("pyre_mm::") + scalarTypeToTorchMlir(dtype)
                 + "::2"
                 + "::" + broadcastKey(self.sizes()) + "+" + broadcastKey(other.sizes())
                 + "::" + c10::pyre::PyreDevice::get(0)->capabilities().cacheKey();

  auto* kernel = getOrCompile(cache_key, "pyre_mm", mlir);
  return invokeKernel(kernel, {self, other}, out_shape, self.options());
}

at::Tensor jitAddmmOp(
    const at::Tensor& bias,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  TORCH_CHECK(beta.toDouble() == 1.0 && alpha.toDouble() == 1.0,
      "pyre: addmm only supports beta=1, alpha=1 currently");
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2,
      "pyre: addmm requires 2D matrices");
  TORCH_CHECK(hasPyreBuffer(bias) && hasPyreBuffer(mat1) && hasPyreBuffer(mat2),
      "pyre: addmm requires tensors with IREE buffers");

  auto dtype = mat1.scalar_type();

  // Detect rank-2 transpose: strides are swapped relative to contiguous.
  auto isTransposed2D = [](const at::Tensor& t) {
    return t.dim() == 2 && t.stride(0) == 1 && t.stride(1) == t.size(0);
  };

  TORCH_CHECK(mat1.is_contiguous(),
      "pyre: addmm mat1 must be contiguous");

  std::string mlir;
  std::string func_name;
  std::string cache_variant;
  at::Tensor mat2_for_invoke;

  if (isTransposed2D(mat2)) {
    // mat2 is a transpose view — pass the underlying contiguous storage
    // with the original (un-transposed) shape. The kernel does the
    // transpose in MLIR via torch.aten.t.
    func_name = "pyre_addmm_t";
    cache_variant = "t";
    // The underlying storage has shape [N,K] (mat2.T is [K,N]).
    // mat2.sizes() is [K,N], original is [N,K].
    std::vector<int64_t> orig_shape = {mat2.size(1), mat2.size(0)};
    std::vector<int64_t> out_shape = {mat1.size(0), mat2.size(1)};
    mlir = expandAddmmTransposedTemplate(
        func_name, dtype,
        bias.sizes(), mat1.sizes(), orig_shape, out_shape);
    // Pass the underlying contiguous tensor (the one before .T).
    mat2_for_invoke = mat2.t();  // undo the transpose to get original layout
  } else {
    TORCH_CHECK(mat2.is_contiguous(),
        "pyre: addmm mat2 must be contiguous or transposed");
    func_name = "pyre_addmm";
    cache_variant = "c";
    std::vector<int64_t> out_shape_v = {mat1.size(0), mat2.size(1)};
    mlir = expandAddmmTemplate(
        func_name, dtype,
        bias.sizes(), mat1.sizes(), mat2.sizes(), out_shape_v);
    mat2_for_invoke = mat2;
  }

  std::vector<int64_t> out_shape = {mat1.size(0),
      isTransposed2D(mat2) ? mat2.size(1) : mat2.size(1)};

  auto cache_key = func_name + "::" + scalarTypeToTorchMlir(dtype)
                 + "::" + cache_variant
                 + "::" + broadcastKey(bias.sizes())
                 + "+" + broadcastKey(mat1.sizes())
                 + "+" + broadcastKey(mat2_for_invoke.sizes())
                 + "::" + c10::pyre::PyreDevice::get(0)->capabilities().cacheKey();

  auto* kernel = getOrCompile(cache_key, func_name, mlir);

  auto output = at::empty(out_shape, mat1.options());
  c10::pyre::PyreStream stream(c10::pyre::getCurrentHostStream(0));
  auto& ctx = stream.context();
  PyreKernelDispatch::invoke(kernel, {bias, mat1, mat2_for_invoke}, output, ctx);
  return output;
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
                 + "::" + std::to_string(self_c.dim())
                 + "::" + c10::pyre::PyreDevice::get(0)->capabilities().cacheKey();

  auto* kernel = getOrCompile(cache_key, func_name, mlir);
  return invokeKernel(kernel, {self_c}, self_c.sizes(), self.options());
}

} // namespace at::pyre
