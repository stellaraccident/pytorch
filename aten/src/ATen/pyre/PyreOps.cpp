#include <ATen/pyre/PyreOps.h>
#include <ATen/pyre/dispatch/PyreAlgorithmicKernels.h>

#ifdef AT_PER_OPERATOR_HEADERS
#include <ATen/ops/_to_copy_native.h>
#else
#include <ATen/NativeFunctions.h>
#endif

#include <cmath>
#include <set>

namespace at::pyre {

// ---------------------------------------------------------------------------
// Stock helper implementations
// ---------------------------------------------------------------------------

c10::DimVector inferShapeBroadcast(const OpContext& ctx) {
  TORCH_CHECK(ctx.raw_inputs.size() >= 2);
  return c10::DimVector(
      at::infer_size(ctx.raw_inputs[0].sizes(), ctx.raw_inputs[1].sizes()));
}

c10::DimVector inferShapeIdentity(const OpContext& ctx) {
  return c10::DimVector(ctx.raw_inputs[0].sizes());
}

bool promoteScalarTensors(
    c10::ArrayRef<at::Tensor> raw_inputs,
    c10::SmallVector<at::Tensor, 4>& out,
    c10::ArrayRef<at::Tensor>& effective) {
  c10::Device target_device = c10::kCPU;
  for (const auto& t : raw_inputs) {
    if (t.device().type() != c10::DeviceType::CPU) {
      target_device = t.device();
      break;
    }
  }
  if (target_device.type() == c10::DeviceType::CPU) {
    effective = raw_inputs;
    return false;
  }
  bool need_promote = false;
  for (const auto& t : raw_inputs) {
    if (t.dim() == 0 && t.device().type() == c10::DeviceType::CPU) {
      need_promote = true;
      break;
    }
  }
  if (!need_promote) {
    effective = raw_inputs;
    return false;
  }
  auto target_dtype = raw_inputs[0].scalar_type();
  for (const auto& t : raw_inputs) {
    if (t.device() == target_device) {
      target_dtype = t.scalar_type();
      break;
    }
  }
  for (const auto& t : raw_inputs) {
    if (t.dim() == 0 && t.device().type() == c10::DeviceType::CPU)
      out.push_back(t.to(target_device, target_dtype));
    else
      out.push_back(t);
  }
  effective = out;
  return true;
}

std::string funcNameDefault(const char* aten_name) {
  std::string s = "pyre_";
  for (const char* p = aten_name; *p; ++p) {
    if (*p == '.') s += '_';
    else s += *p;
  }
  return s;
}

// Binary: common logic for building args used by both spec and mlir paths.
static void binaryArgs(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx,
    std::string& linalg_op, std::string& alpha_add_op,
    double& alpha_val, bool& has_alpha,
    c10::DimVector& out_shape) {
  has_alpha = !ctx.scalars.empty() &&
      std::abs(ctx.scalars[0].toDouble() - 1.0) >= 1e-12;
  alpha_val = has_alpha ? ctx.scalars[0].toDouble() : 1.0;
  linalg_op = torch_op;
  if (has_alpha) {
    if (std::string(torch_op) == "sub")
      alpha_add_op = isFloatDtype(ctx.dtype) ? "arith.subf" : "arith.subi";
    else
      alpha_add_op = isFloatDtype(ctx.dtype) ? "arith.addf" : "arith.addi";
  }
  out_shape = c10::DimVector(
      at::infer_size(ctx.raw_inputs[0].sizes(), ctx.raw_inputs[1].sizes()));
}

KernelSpec buildBinarySpec(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx) {
  std::string linalg_op, alpha_add_op;
  double alpha_val;
  bool has_alpha;
  c10::DimVector out_shape;
  binaryArgs(torch_op, func_name, ctx, linalg_op, alpha_add_op,
             alpha_val, has_alpha, out_shape);

  if (has_alpha) {
    return buildBinaryAlphaKernelSpec(
        func_name, alpha_add_op, alpha_val, ctx.dtype,
        ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
        out_shape, ctx.decision.arg_adapters);
  }
  return buildBinaryKernelSpec(
      func_name, linalg_op, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
      out_shape, ctx.decision.arg_adapters);
}

std::string buildBinaryMlir(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx) {
  std::string linalg_op, alpha_add_op;
  double alpha_val;
  bool has_alpha;
  c10::DimVector out_shape;
  binaryArgs(torch_op, func_name, ctx, linalg_op, alpha_add_op,
             alpha_val, has_alpha, out_shape);

  if (has_alpha) {
    return generateBinaryAlphaMlir(
        func_name, alpha_add_op, alpha_val, ctx.dtype,
        ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
        out_shape, ctx.decision.arg_adapters);
  }
  return generateBinaryMlir(
      func_name, linalg_op, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
      out_shape, ctx.decision.arg_adapters);
}

KernelSpec buildUnarySpec(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  ArgAdapter adapter = ctx.decision.arg_adapters.empty()
      ? ArgAdapter{ArgAdapter::kIdentity, {}}
      : ctx.decision.arg_adapters[0];
  return buildUnaryKernelSpec(
      func_name, torch_op, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[0].sizes(), adapter,
      extra_arg_decls, extra_args, extra_arg_types);
}

std::string buildUnaryMlir(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  ArgAdapter adapter = ctx.decision.arg_adapters.empty()
      ? ArgAdapter{ArgAdapter::kIdentity, {}}
      : ctx.decision.arg_adapters[0];
  return generateUnaryMlir(
      func_name, torch_op, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[0].sizes(), adapter,
      extra_arg_decls, extra_args, extra_arg_types);
}

// ---------------------------------------------------------------------------
// Reduction helpers
// ---------------------------------------------------------------------------

c10::DimVector inferReducedShape(
    c10::IntArrayRef input_shape, c10::IntArrayRef dims, bool keepdim) {
  std::set<int64_t> reduce_set;
  for (int64_t d : dims) {
    if (d < 0) d += static_cast<int64_t>(input_shape.size());
    reduce_set.insert(d);
  }
  c10::DimVector out;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (reduce_set.count(static_cast<int64_t>(i))) {
      if (keepdim) out.push_back(1);
    } else {
      out.push_back(input_shape[i]);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Scalar binary helpers
// ---------------------------------------------------------------------------

KernelSpec buildScalarBinarySpec(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx, double scalar_value,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  return buildScalarBinaryKernelSpec(
      func_name, torch_op, ctx.dtype,
      ctx.inputs[0].sizes(), scalar_value,
      extra_arg_decls, extra_args, extra_arg_types);
}

std::string buildScalarBinaryMlirHelper(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx, double scalar_value,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  return generateScalarBinaryMlir(
      func_name, torch_op, ctx.dtype,
      ctx.inputs[0].sizes(), scalar_value,
      extra_arg_decls, extra_args, extra_arg_types);
}

// ---------------------------------------------------------------------------
// getOrCompile / invokeKernel
// ---------------------------------------------------------------------------

CachedKernel* getOrCompile(
    const std::string& cache_key,
    const std::string& func_name,
    const std::string& mlir,
    const AbiConfig& abi) {
  PYRE_LOG(INFO) << "cache MISS: " << cache_key << ", compiling\n";
  PYRE_LOG(DEBUG) << "MLIR:\n" << mlir << "\n";

  auto vmfb = PyreKernelCompiler::compileSync(
      std::string(mlir), abi.compilerFlags());

  PYRE_LOG(INFO) << "compiled " << vmfb->size() << " bytes\n";
  auto& cache = PyreKernelCache::get();
  return cache.store(cache_key, func_name, std::move(vmfb), abi);
}

at::Tensor invokeKernel(
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

// HACK(pyre-workspace-blp): Check if a tensor is strictly dense row-major
// at offset 0. The offset check is a workaround — with proper subspan
// support, non-zero offsets would be handled by the buffer view.
static bool isDenseRowMajorAtZero(const at::Tensor& t) {
  if (t.storage_offset() != 0) return false;
  int64_t expected = 1;
  for (int64_t i = t.dim() - 1; i >= 0; --i) {
    if (t.size(i) > 1 && t.stride(i) != expected) return false;
    expected *= t.size(i);
  }
  return true;
}

void invokeKernelInplace(
    CachedKernel* kernel,
    const std::vector<at::Tensor>& inputs,
    at::Tensor& self) {
  c10::pyre::PyreStream stream(c10::pyre::getCurrentHostStream(0));
  auto& ctx = stream.context();
  if (isDenseRowMajorAtZero(self)) {
    PyreKernelDispatch::invoke(kernel, inputs, self, ctx);
  } else {
    auto tmp = at::empty(self.sizes(), self.options());
    PyreKernelDispatch::invoke(kernel, inputs, tmp, ctx);
    self.copy_(tmp);
  }
}

// ---------------------------------------------------------------------------
// MmOp
// ---------------------------------------------------------------------------

KernelSpec MmOp::buildKernelSpec(
    const std::string& func_name, const OpContext& ctx) {
  auto out_shape = inferShape(ctx);
  return buildMmKernelSpec(
      func_name, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(), out_shape,
      ctx.decision.arg_adapters);
}

std::string MmOp::generateMlir(
    const std::string& func_name, const OpContext& ctx) {
  auto out_shape = inferShape(ctx);
  return generateMmMlir(
      func_name, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(), out_shape,
      ctx.decision.arg_adapters);
}

// ---------------------------------------------------------------------------
// AddmmOp
// ---------------------------------------------------------------------------

static void addmmArgs(const OpContext& ctx, double& beta, double& alpha,
                      bool& mat2_permuted) {
  beta = ctx.scalars.size() >= 1 ? ctx.scalars[0].toDouble() : 1.0;
  alpha = ctx.scalars.size() >= 2 ? ctx.scalars[1].toDouble() : 1.0;
  mat2_permuted = ctx.decision.arg_adapters.size() > 2
      && ctx.decision.arg_adapters[2].kind == ArgAdapter::kPermute;
}

KernelSpec AddmmOp::buildKernelSpec(
    const std::string& func_name, const OpContext& ctx) {
  double beta, alpha;
  bool mat2_permuted;
  addmmArgs(ctx, beta, alpha, mat2_permuted);
  if (mat2_permuted) {
    return buildAddmmTransposedKernelSpec(func_name, ctx.dtype,
        ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
        ctx.inputs[2].sizes(), inferShape(ctx), beta, alpha);
  }
  return buildAddmmKernelSpec(func_name, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
      ctx.inputs[2].sizes(), inferShape(ctx), beta, alpha);
}

std::string AddmmOp::generateMlir(
    const std::string& func_name, const OpContext& ctx) {
  double beta, alpha;
  bool mat2_permuted;
  addmmArgs(ctx, beta, alpha, mat2_permuted);
  if (mat2_permuted) {
    return generateAddmmTransposedMlir(func_name, ctx.dtype,
        ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
        ctx.inputs[2].sizes(), inferShape(ctx), beta, alpha);
  }
  return generateAddmmMlir(func_name, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
      ctx.inputs[2].sizes(), inferShape(ctx), beta, alpha);
}

// ---------------------------------------------------------------------------
// Multi-dim reduction dispatch helper
// ---------------------------------------------------------------------------

at::Tensor dispatchMultiDimReduction(
    const char* aten_name, const char* torch_op,
    const at::Tensor& self, c10::ArrayRef<int64_t> dims, bool keepdim,
    bool has_dtype_arg) {
  TORCH_CHECK(hasPyreBuffer(self), "pyre: ", aten_name,
      " requires IREE buffers");

  auto dtype = self.scalar_type();
  auto decision = ArgShapeSpecializer().analyze(
      aten_name, dtype, {self},
      c10::pyre::PyreDevice::get(0)->capabilities());
  auto adapted = applyAdapters({self}, decision.arg_adapters);

  c10::SmallVector<int64_t, 6> norm_dims;
  for (auto d : dims) {
    if (d < 0) d += self.dim();
    norm_dims.push_back(d);
  }

  auto out_shape = inferReducedShape(self.sizes(), norm_dims, keepdim);
  auto func_name = funcNameDefault(aten_name);

  std::string extra_decls, extra_args, extra_types;
  if (has_dtype_arg) {
    extra_decls = "    %none = torch.constant.none";
    extra_args = ", %none";
    extra_types = ", !torch.none";
  }

  auto spec = buildReductionKernelSpec(
      func_name, torch_op, dtype,
      adapted[0].sizes(), out_shape, norm_dims, keepdim,
      extra_decls, extra_args, extra_types);

  auto cache_key = contentHashCacheKey(
      spec.template_sha1, spec.substitutions,
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags());

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = generateReductionMlir(
        func_name, torch_op, dtype,
        adapted[0].sizes(), out_shape, norm_dims, keepdim,
        extra_decls, extra_args, extra_types);
    kernel = getOrCompile(cache_key, func_name, mlir);
  }

  return invokeKernel(kernel, adapted, out_shape, self.options());
}

// ---------------------------------------------------------------------------
// Single-dim reduction dispatch helper
// ---------------------------------------------------------------------------

at::Tensor dispatchSingleDimReduction(
    const char* aten_name, const char* torch_op,
    const at::Tensor& self, int64_t dim, bool keepdim,
    const std::string& extra_decls,
    const std::string& extra_args,
    const std::string& extra_types) {
  TORCH_CHECK(hasPyreBuffer(self), "pyre: ", aten_name,
      " requires IREE buffers");

  auto dtype = self.scalar_type();
  if (dim < 0) dim += self.dim();

  auto out_shape = inferReducedShape(self.sizes(), {dim}, keepdim);
  auto func_name = funcNameDefault(aten_name);

  auto spec = buildSingleDimReductionKernelSpec(
      func_name, torch_op, dtype,
      self.sizes(), out_shape, dim, keepdim,
      extra_decls, extra_args, extra_types);

  auto cache_key = contentHashCacheKey(
      spec.template_sha1, spec.substitutions,
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags());

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = generateSingleDimReductionMlir(
        func_name, torch_op, dtype,
        self.sizes(), out_shape, dim, keepdim,
        extra_decls, extra_args, extra_types);
    kernel = getOrCompile(cache_key, func_name, mlir);
  }

  return invokeKernel(kernel, {self}, out_shape, self.options());
}

// ---------------------------------------------------------------------------
// EmbeddingOp
// ---------------------------------------------------------------------------

at::Tensor EmbeddingOp::impl(
    const at::Tensor& weight, const at::Tensor& indices,
    int64_t padding_idx, bool /*scale_grad_by_freq*/, bool /*sparse*/) {
  TORCH_CHECK(hasPyreBuffer(weight), "pyre: embedding requires IREE weight");
  TORCH_CHECK(hasPyreBuffer(indices), "pyre: embedding requires IREE indices");
  TORCH_CHECK(weight.dim() == 2, "pyre: embedding weight must be 2D");
  TORCH_CHECK(padding_idx == -1,
      "pyre: embedding with custom padding_idx not yet supported");

  auto dtype = weight.scalar_type();
  c10::DimVector out_shape(indices.sizes().begin(), indices.sizes().end());
  out_shape.push_back(weight.size(1));

  auto func_name = funcNameDefault(aten_name);
  auto spec = buildEmbeddingKernelSpec(
      func_name, dtype, weight.sizes(), indices.sizes(), out_shape);
  auto cache_key = contentHashCacheKey(
      spec.template_sha1, spec.substitutions,
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags());

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = generateEmbeddingMlir(
        func_name, dtype, weight.sizes(), indices.sizes(), out_shape);
    kernel = getOrCompile(cache_key, func_name, mlir);
  }

  return invokeKernel(kernel, {weight, indices}, out_shape, weight.options());
}

// ---------------------------------------------------------------------------
// IndexSelectOp
// ---------------------------------------------------------------------------

at::Tensor IndexSelectOp::impl(
    const at::Tensor& self, int64_t dim, const at::Tensor& index) {
  TORCH_CHECK(hasPyreBuffer(self), "pyre: index_select requires IREE self");
  TORCH_CHECK(hasPyreBuffer(index), "pyre: index_select requires IREE index");

  if (dim < 0) dim += self.dim();
  auto dtype = self.scalar_type();

  c10::DimVector out_shape(self.sizes().begin(), self.sizes().end());
  out_shape[dim] = index.size(0);

  auto func_name = funcNameDefault(aten_name);
  auto spec = buildIndexSelectKernelSpec(
      func_name, dtype, self.sizes(), index.sizes(), out_shape, dim);
  auto cache_key = contentHashCacheKey(
      spec.template_sha1, spec.substitutions,
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags());

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = generateIndexSelectMlir(
        func_name, dtype, self.sizes(), index.sizes(), out_shape, dim);
    kernel = getOrCompile(cache_key, func_name, mlir);
  }

  return invokeKernel(kernel, {self, index}, out_shape, self.options());
}

// ---------------------------------------------------------------------------
// GatherOp
// ---------------------------------------------------------------------------

at::Tensor GatherOp::impl(
    const at::Tensor& self, int64_t dim, const at::Tensor& index,
    bool /*sparse_grad*/) {
  TORCH_CHECK(hasPyreBuffer(self), "pyre: gather requires IREE self");
  TORCH_CHECK(hasPyreBuffer(index), "pyre: gather requires IREE index");

  if (dim < 0) dim += self.dim();
  auto dtype = self.scalar_type();
  c10::DimVector out_shape(index.sizes().begin(), index.sizes().end());

  auto func_name = funcNameDefault(aten_name);
  auto spec = buildGatherKernelSpec(
      func_name, dtype, self.sizes(), index.sizes(), out_shape, dim);
  auto cache_key = contentHashCacheKey(
      spec.template_sha1, spec.substitutions,
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags());

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = generateGatherMlir(
        func_name, dtype, self.sizes(), index.sizes(), out_shape, dim);
    kernel = getOrCompile(cache_key, func_name, mlir);
  }

  return invokeKernel(kernel, {self, index}, out_shape, self.options());
}

// ---------------------------------------------------------------------------
// SoftmaxOp / LogSoftmaxOp — compiled kernel with first-class _softmax
// ---------------------------------------------------------------------------

static at::Tensor dispatchSoftmax(
    const char* aten_name, const std::string& softmax_op,
    const at::Tensor& self, int64_t dim) {
  TORCH_CHECK(hasPyreBuffer(self), "pyre: ", aten_name,
      " requires IREE buffers");
  if (dim < 0) dim += self.dim();
  auto dtype = self.scalar_type();
  auto func_name = funcNameDefault(aten_name);

  auto spec = buildSoftmaxKernelSpec(func_name, softmax_op, dtype, self.sizes(), dim);
  auto cache_key = contentHashCacheKey(
      spec.template_sha1, spec.substitutions,
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags());

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = generateSoftmaxMlir(func_name, softmax_op, dtype, self.sizes(), dim);
    kernel = getOrCompile(cache_key, func_name, mlir);
  }
  return invokeKernel(kernel, {self}, c10::DimVector(self.sizes()), self.options());
}

at::Tensor SoftmaxOp::impl(
    const at::Tensor& self, int64_t dim, bool /*half_to_float*/) {
  return dispatchSoftmax(aten_name, "_softmax", self, dim);
}

at::Tensor LogSoftmaxOp::impl(
    const at::Tensor& self, int64_t dim, bool /*half_to_float*/) {
  return dispatchSoftmax(aten_name, "_log_softmax", self, dim);
}

// ---------------------------------------------------------------------------
// ScatterSrcOp
// ---------------------------------------------------------------------------

static at::Tensor dispatchScatterSrc(
    const at::Tensor& self, int64_t dim,
    const at::Tensor& index, const at::Tensor& src) {
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(index) && hasPyreBuffer(src),
      "pyre: scatter.src requires IREE buffers");
  if (dim < 0) dim += self.dim();
  auto dtype = self.scalar_type();
  auto func_name = funcNameDefault("scatter_src");

  auto spec = buildScatterSrcKernelSpec(
      func_name, dtype, self.sizes(), index.sizes(), src.sizes(), dim);
  auto cache_key = contentHashCacheKey(
      spec.template_sha1, spec.substitutions,
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags());

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = generateScatterSrcMlir(
        func_name, dtype, self.sizes(), index.sizes(), src.sizes(), dim);
    kernel = getOrCompile(cache_key, func_name, mlir);
  }
  return invokeKernel(kernel, {self, index, src},
                      c10::DimVector(self.sizes()), self.options());
}

at::Tensor ScatterSrcOp::impl(
    const at::Tensor& self, int64_t dim,
    const at::Tensor& index, const at::Tensor& src) {
  return dispatchScatterSrc(self, dim, index, src);
}

at::Tensor& ScatterSrcOp::impl_inplace(
    at::Tensor& self, int64_t dim,
    const at::Tensor& index, const at::Tensor& src) {
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(index) && hasPyreBuffer(src),
      "pyre: scatter_.src requires IREE buffers");
  if (dim < 0) dim += self.dim();
  auto dtype = self.scalar_type();
  auto func_name = funcNameDefault("scatter_src_inplace");

  auto spec = buildScatterSrcInplaceKernelSpec(
      func_name, dtype, self.sizes(), index.sizes(), src.sizes(), dim);
  auto cache_key = contentHashCacheKey(
      spec.template_sha1, spec.substitutions,
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags());

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = generateScatterSrcInplaceMlir(
        func_name, dtype, self.sizes(), index.sizes(), src.sizes(), dim);
    kernel = getOrCompile(cache_key, func_name, mlir);
  }

  // Reads and writes to self. The in-place template accesses self through
  // %out_ via torch.copy.to_vtensor, so only index and src are inputs.
  invokeKernelInplace(kernel, {index, src}, self);
  return self;
}

// ---------------------------------------------------------------------------
// ScatterAddOp
// ---------------------------------------------------------------------------

static at::Tensor dispatchScatterAdd(
    const at::Tensor& self, int64_t dim,
    const at::Tensor& index, const at::Tensor& src) {
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(index) && hasPyreBuffer(src),
      "pyre: scatter_add requires IREE buffers");
  if (dim < 0) dim += self.dim();
  auto dtype = self.scalar_type();
  auto func_name = funcNameDefault("scatter_add");

  auto spec = buildScatterAddKernelSpec(
      func_name, dtype, self.sizes(), index.sizes(), src.sizes(), dim);
  auto cache_key = contentHashCacheKey(
      spec.template_sha1, spec.substitutions,
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags());

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = generateScatterAddMlir(
        func_name, dtype, self.sizes(), index.sizes(), src.sizes(), dim);
    kernel = getOrCompile(cache_key, func_name, mlir);
  }
  return invokeKernel(kernel, {self, index, src},
                      c10::DimVector(self.sizes()), self.options());
}

at::Tensor ScatterAddOp::impl(
    const at::Tensor& self, int64_t dim,
    const at::Tensor& index, const at::Tensor& src) {
  return dispatchScatterAdd(self, dim, index, src);
}

at::Tensor& ScatterAddOp::impl_inplace(
    at::Tensor& self, int64_t dim,
    const at::Tensor& index, const at::Tensor& src) {
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(index) && hasPyreBuffer(src),
      "pyre: scatter_add_ requires IREE buffers");
  if (dim < 0) dim += self.dim();
  auto dtype = self.scalar_type();
  auto func_name = funcNameDefault("scatter_add_inplace");

  auto spec = buildScatterAddInplaceKernelSpec(
      func_name, dtype, self.sizes(), index.sizes(), src.sizes(), dim);
  auto cache_key = contentHashCacheKey(
      spec.template_sha1, spec.substitutions,
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags());

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = generateScatterAddInplaceMlir(
        func_name, dtype, self.sizes(), index.sizes(), src.sizes(), dim);
    kernel = getOrCompile(cache_key, func_name, mlir);
  }

  invokeKernelInplace(kernel, {index, src}, self);
  return self;
}

// ---------------------------------------------------------------------------
// IndexPutOp — algorithmic MLIR for variable-length index list
// ---------------------------------------------------------------------------

// Fragment accessors in dispatch/PyreIndexKernels.cpp.

static at::Tensor dispatchIndexPut(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values, bool accumulate) {
  TORCH_CHECK(hasPyreBuffer(self), "pyre: index_put requires IREE self");
  TORCH_CHECK(hasPyreBuffer(values), "pyre: index_put requires IREE values");

  // Collect non-None indices, promoting CPU to device.
  c10::SmallVector<std::pair<int64_t, at::Tensor>, 4> non_none;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
    if (indices[i].has_value() && indices[i]->defined()) {
      auto idx = *indices[i];
      if (!hasPyreBuffer(idx))
        idx = idx.to(self.device());
      non_none.push_back({i, idx});
    }
  }
  TORCH_CHECK(!non_none.empty(),
      "pyre: index_put requires at least one index tensor");

  auto dtype = self.scalar_type();
  std::string elt = scalarTypeToTorchMlir(dtype);
  auto func_name = funcNameDefault("index_put");

  auto dynShape = [](c10::IntArrayRef sizes) {
    std::string s;
    for (size_t i = 0; i < sizes.size(); ++i) {
      if (i > 0) s += ",";
      s += (sizes[i] == 1) ? "1" : "?";
    }
    return s;
  };

  // Adapt self and values for non-standard layouts (permuted strides,
  // storage offsets).
  auto self_adapter = ArgAdapter::analyze(self);
  auto vals_adapter = ArgAdapter::analyze(values);
  at::Tensor self_phys = self;
  at::Tensor vals_phys = values;
  if (self_adapter.kind == ArgAdapter::kPermute) {
    self_phys = self.permute(self_adapter.permutation);
  } else if (self_adapter.kind == ArgAdapter::kContiguous) {
    // HACK(pyre-workspace-blp): clone to get offset-0 buffer.
    self_phys = self.storage_offset() != 0 ? self.clone() : self.contiguous();
  }
  if (vals_adapter.kind == ArgAdapter::kPermute) {
    vals_phys = values.permute(vals_adapter.permutation);
  } else if (vals_adapter.kind == ArgAdapter::kContiguous) {
    // HACK(pyre-workspace-blp): clone to get offset-0 buffer.
    vals_phys = values.storage_offset() != 0 ? values.clone() : values.contiguous();
  }

  std::string self_s = dynShape(self_phys.sizes());
  std::string vals_s = dynShape(vals_phys.sizes());

  c10::SmallVector<std::string, 8> idx_shapes;
  for (const auto& [dim, idx] : non_none)
    idx_shapes.push_back(dynShape(idx.sizes()));

  // Build opt decls, names, types for the index list
  std::string opt_decls, opt_names, opt_types;
  int tensor_counter = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
    if (i > 0) { opt_names += ", "; opt_types += ", "; }
    opt_names += "%opt" + std::to_string(i);
    opt_types += "!torch.optional<vtensor>";

    if (indices[i].has_value() && indices[i]->defined()) {
      std::string tidx = std::to_string(tensor_counter);
      std::string ish = idx_shapes[tensor_counter];
      opt_decls += "\n    %opt" + std::to_string(i) +
          " = torch.derefine %idx" + tidx +
          " : !torch.vtensor<[" + ish + "], si64> to !torch.optional<vtensor>";
      tensor_counter++;
    } else {
      opt_decls += "\n    %none" + std::to_string(i) + " = torch.constant.none" +
          "\n    %opt" + std::to_string(i) +
          " = torch.derefine %none" + std::to_string(i) +
          " : !torch.none to !torch.optional<vtensor>";
    }
  }

  std::string accum_str = accumulate ? "true" : "false";

  // Logical shapes for the index_put op (what the op operates on).
  std::string self_log_s = dynShape(self.sizes());
  std::string vals_log_s = dynShape(values.sizes());
  std::string self_log_t = "!torch.vtensor<[" + self_log_s + "], " + elt + ">";
  std::string vals_log_t = "!torch.vtensor<[" + vals_log_s + "], " + elt + ">";

  // Physical shapes for function signature (what the buffer view has).
  std::string self_phys_t = "!torch.vtensor<[" + self_s + "], " + elt + ">";
  std::string vals_phys_t = "!torch.vtensor<[" + vals_s + "], " + elt + ">";

  // Use PyreKernelAsmFragments for digest/generate split.
  // The recipe lambda replays identically in both modes.
  auto recipe = [&](PyreKernelAsmBuilder& b) {
    // Fragment 0: header (output + self)
    b.appendFragment(0, {{"FUNC", func_name}, {"SELF_LOG", self_log_s},
                         {"ELT", elt}, {"SELF_PHYS_T", self_phys_t},
                         {"VALS_PHYS_T", vals_phys_t}});
    // Fragment 1: per-index input param (repeated)
    for (size_t i = 0; i < non_none.size(); ++i) {
      b.appendFragment(1, {{"IDX", std::to_string(i)},
                           {"IDX_SHAPE", idx_shapes[i]}});
    }
    // Fragment 2: body
    std::string self_name = self_adapter.kind == ArgAdapter::kPermute ? "self" : "self_raw";
    std::string vals_name = vals_adapter.kind == ArgAdapter::kPermute ? "values" : "values_raw";
    std::string self_adapt_body, vals_adapt_body;
    if (self_adapter.kind == ArgAdapter::kPermute) {
      self_adapt_body = emitPermuteLines("self", "self_raw",
          inversePerm(self_adapter.permutation), self_phys_t, self_log_t) + "\n";
    }
    if (vals_adapter.kind == ArgAdapter::kPermute) {
      vals_adapt_body = emitPermuteLines("values", "values_raw",
          inversePerm(vals_adapter.permutation), vals_phys_t, vals_log_t) + "\n";
    }
    b.appendFragment(2, {
        {"VALS_PHYS_T", vals_phys_t},
        {"SELF_ADAPT", self_adapt_body}, {"VALS_ADAPT", vals_adapt_body},
        {"OPT_DECLS", opt_decls},
        {"ACCUM", accum_str},
        {"OPT_NAMES", opt_names}, {"OPT_TYPES", opt_types},
        {"SELF_NAME", self_name}, {"VALS_NAME", vals_name},
        {"SELF_LOG_T", self_log_t}, {"VALS_LOG_T", vals_log_t},
        {"SELF_LOG", self_log_s}, {"ELT", elt}});
  };

  auto& frags = indexPutFragments();
  auto cache_key = frags.digest(
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags(), recipe);

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = frags.generateMlir(recipe);
    kernel = getOrCompile(cache_key, func_name, mlir);
  }

  std::vector<at::Tensor> inputs;
  inputs.push_back(self_phys);
  for (const auto& [dim, idx] : non_none)
    inputs.push_back(idx);
  inputs.push_back(vals_phys);

  return invokeKernel(kernel, inputs, c10::DimVector(self.sizes()), self.options());
}

at::Tensor IndexPutOp::impl(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values, bool accumulate) {
  return dispatchIndexPut(self, indices, values, accumulate);
}

at::Tensor& IndexPutOp::impl_inplace(
    at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values, bool accumulate) {
  TORCH_CHECK(hasPyreBuffer(self), "pyre: index_put_ requires IREE self");
  TORCH_CHECK(hasPyreBuffer(values), "pyre: index_put_ requires IREE values");

  c10::SmallVector<std::pair<int64_t, at::Tensor>, 4> non_none;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
    if (indices[i].has_value() && indices[i]->defined()) {
      auto idx = *indices[i];
      if (!hasPyreBuffer(idx)) {
        idx = idx.to(self.device());
      }
      non_none.push_back({i, idx});
    }
  }
  TORCH_CHECK(!non_none.empty(),
      "pyre: index_put_ requires at least one index tensor");

  auto dtype = self.scalar_type();
  std::string elt = scalarTypeToTorchMlir(dtype);
  auto func_name = funcNameDefault("index_put_inplace");

  auto dynShape = [](c10::IntArrayRef sizes) {
    std::string s;
    for (size_t i = 0; i < sizes.size(); ++i) {
      if (i > 0) s += ",";
      s += (sizes[i] == 1) ? "1" : "?";
    }
    return s;
  };

  // Adapt values for non-standard layouts.
  auto vals_adapter = ArgAdapter::analyze(values);
  at::Tensor vals_phys = values;
  if (vals_adapter.kind == ArgAdapter::kPermute) {
    vals_phys = values.permute(vals_adapter.permutation);
  } else if (vals_adapter.kind == ArgAdapter::kContiguous) {
    // HACK(pyre-workspace-blp): clone to get offset-0 buffer.
    vals_phys = values.storage_offset() != 0 ? values.clone() : values.contiguous();
  }

  std::string self_log_s = dynShape(self.sizes());
  std::string vals_s = dynShape(vals_phys.sizes());
  std::string vals_log_s = dynShape(values.sizes());
  std::string self_log_t = "!torch.vtensor<[" + self_log_s + "], " + elt + ">";
  std::string vals_phys_t = "!torch.vtensor<[" + vals_s + "], " + elt + ">";
  std::string vals_log_t = "!torch.vtensor<[" + vals_log_s + "], " + elt + ">";

  c10::SmallVector<std::string, 8> idx_shapes;
  for (const auto& [dim, idx] : non_none)
    idx_shapes.push_back(dynShape(idx.sizes()));

  std::string opt_decls, opt_names, opt_types;
  int tensor_counter = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
    if (i > 0) { opt_names += ", "; opt_types += ", "; }
    opt_names += "%opt" + std::to_string(i);
    opt_types += "!torch.optional<vtensor>";
    if (indices[i].has_value() && indices[i]->defined()) {
      std::string tidx = std::to_string(tensor_counter);
      std::string ish = idx_shapes[tensor_counter];
      opt_decls += "\n    %opt" + std::to_string(i) +
          " = torch.derefine %idx" + tidx +
          " : !torch.vtensor<[" + ish + "], si64> to !torch.optional<vtensor>";
      tensor_counter++;
    } else {
      opt_decls += "\n    %none" + std::to_string(i) + " = torch.constant.none" +
          "\n    %opt" + std::to_string(i) +
          " = torch.derefine %none" + std::to_string(i) +
          " : !torch.none to !torch.optional<vtensor>";
    }
  }

  std::string accum_str = accumulate ? "true" : "false";

  std::string vals_adapt_body;
  std::string vals_name = "values_raw";
  if (vals_adapter.kind == ArgAdapter::kPermute) {
    vals_adapt_body = emitPermuteLines("values", "values_raw",
        inversePerm(vals_adapter.permutation), vals_phys_t, vals_log_t) + "\n";
    vals_name = "values";
  }

  auto recipe = [&](PyreKernelAsmBuilder& b) {
    b.appendFragment(0, {{"FUNC", func_name}, {"SELF_LOG", self_log_s},
                         {"ELT", elt}});
    for (size_t i = 0; i < non_none.size(); ++i) {
      b.appendFragment(1, {{"IDX", std::to_string(i)},
                           {"IDX_SHAPE", idx_shapes[i]}});
    }
    b.appendFragment(2, {
        {"VALS_PHYS_T", vals_phys_t},
        {"SELF_LOG_T", self_log_t},
        {"VALS_ADAPT", vals_adapt_body},
        {"OPT_DECLS", opt_decls},
        {"ACCUM", accum_str},
        {"OPT_NAMES", opt_names}, {"OPT_TYPES", opt_types},
        {"VALS_NAME", vals_name},
        {"VALS_LOG_T", vals_log_t},
        {"SELF_LOG", self_log_s}, {"ELT", elt}});
  };

  auto& frags = indexPutInplaceFragments();
  auto cache_key = frags.digest(
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags(), recipe);

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = frags.generateMlir(recipe);
    kernel = getOrCompile(cache_key, func_name, mlir);
  }

  // In-place: self is the output, inputs are {index tensors..., values}.
  std::vector<at::Tensor> inputs;
  for (const auto& [dim, idx] : non_none)
    inputs.push_back(idx);
  inputs.push_back(vals_phys);

  invokeKernelInplace(kernel, inputs, self);
  return self;
}

// ---------------------------------------------------------------------------
// IndexTensorOp — algorithmic MLIR generation for variable index lists
// ---------------------------------------------------------------------------

// Fragment accessors in dispatch/PyreIndexKernels.cpp.

at::Tensor IndexTensorOp::impl(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices) {
  // Promote CPU self to device if indices are on device (common for
  // model attributes that aren't registered buffers, e.g. causal_mask).
  at::Tensor self_effective = self;
  if (!hasPyreBuffer(self) && self.is_cpu()) {
    c10::Device target = c10::kCPU;
    for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
      if (indices[i].has_value() && indices[i]->defined() &&
          indices[i]->is_privateuseone()) {
        target = indices[i]->device();
        break;
      }
    }
    if (target.type() != c10::DeviceType::CPU)
      self_effective = self.to(target);
  }
  TORCH_CHECK(hasPyreBuffer(self_effective),
      "pyre: index.Tensor requires IREE self");

  // Collect non-None indices, promoting CPU indices to device.
  // Note: __getitem__ passes undefined tensors for None positions.
  c10::SmallVector<std::pair<int64_t, at::Tensor>, 4> non_none;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
    if (indices[i].has_value() && indices[i]->defined()) {
      auto idx = *indices[i];
      if (!hasPyreBuffer(idx))
        idx = idx.to(self_effective.device());
      non_none.push_back({i, idx});
    }
  }
  TORCH_CHECK(!non_none.empty(),
      "pyre: index.Tensor requires at least one index tensor");

  // Fast path: single 1D index → decompose to index_select.
  if (non_none.size() == 1 && non_none[0].second.dim() == 1) {
    return at::index_select(self_effective, non_none[0].first, non_none[0].second);
  }

  // General case: algorithmic MLIR generation.
  auto dtype = self_effective.scalar_type();
  std::string elt = scalarTypeToTorchMlir(dtype);

  // Compute output shape: broadcast index shapes + trailing self dims.
  // Broadcast all index tensor shapes together.
  c10::DimVector bcast_shape;
  for (const auto& [dim, idx] : non_none) {
    if (bcast_shape.empty()) {
      bcast_shape.assign(idx.sizes().begin(), idx.sizes().end());
    } else {
      bcast_shape = c10::DimVector(
          at::infer_size(bcast_shape, idx.sizes()));
    }
  }
  // Result shape: broadcast_shape + trailing dims after last indexed dim.
  int64_t last_indexed = non_none.back().first;
  c10::DimVector out_shape(bcast_shape.begin(), bcast_shape.end());
  for (int64_t d = last_indexed + 1; d < self_effective.dim(); ++d)
    out_shape.push_back(self_effective.size(d));

  auto func_name = funcNameDefault(aten_name);

  auto dynShape = [](c10::IntArrayRef sizes) {
    std::string s;
    for (size_t i = 0; i < sizes.size(); ++i) {
      if (i > 0) s += ",";
      s += (sizes[i] == 1) ? "1" : "?";
    }
    return s;
  };

  std::string self_s = dynShape(self_effective.sizes());
  std::string out_s = dynShape(out_shape);

  // Build per-index shape strings and the opt decls/names/types.
  c10::SmallVector<std::string, 8> idx_shapes;
  for (const auto& [dim, idx] : non_none)
    idx_shapes.push_back(dynShape(idx.sizes()));

  std::string opt_decls, opt_names, opt_types;
  int tensor_counter = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
    if (i > 0) { opt_names += ", "; opt_types += ", "; }
    opt_names += "%opt" + std::to_string(i);
    opt_types += "!torch.optional<vtensor>";

    if (indices[i].has_value()) {
      // Find index into non_none
      std::string tidx = std::to_string(tensor_counter);
      std::string ish = idx_shapes[tensor_counter];
      opt_decls += "\n    %opt" + std::to_string(i) +
          " = torch.derefine %idx" + tidx +
          " : !torch.vtensor<[" + ish + "], si64> to !torch.optional<vtensor>";
      tensor_counter++;
    } else {
      opt_decls += "\n    %none" + std::to_string(i) + " = torch.constant.none" +
          "\n    %opt" + std::to_string(i) +
          " = torch.derefine %none" + std::to_string(i) +
          " : !torch.none to !torch.optional<vtensor>";
    }
  }

  auto recipe = [&](PyreKernelAsmBuilder& b) {
    b.appendFragment(0, {{"FUNC", func_name}, {"OUT_SHAPE", out_s},
                         {"ELT", elt}, {"SELF_SHAPE", self_s}});
    for (size_t i = 0; i < non_none.size(); ++i) {
      b.appendFragment(1, {{"IDX", std::to_string(i)},
                           {"IDX_SHAPE", idx_shapes[i]}});
    }
    b.appendFragment(4, {{"OPT_DECLS", opt_decls},
                         {"OPT_NAMES", opt_names}, {"OPT_TYPES", opt_types},
                         {"SELF_SHAPE", self_s}, {"ELT", elt},
                         {"OUT_SHAPE", out_s}});
  };

  auto& frags = indexFragments();
  auto cache_key = frags.digest(
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags(), recipe);

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = frags.generateMlir(recipe);
    kernel = getOrCompile(cache_key, func_name, mlir);
  }

  std::vector<at::Tensor> inputs;
  inputs.push_back(self_effective);
  for (const auto& [dim, idx] : non_none)
    inputs.push_back(idx);

  return invokeKernel(kernel, inputs, out_shape, self_effective.options());
}

// ---------------------------------------------------------------------------
// ArangeOp
// ---------------------------------------------------------------------------

at::Tensor ArangeOp::impl(
    const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout>,
    std::optional<at::Device> device,
    std::optional<bool>) {
  double start_d = start.toDouble();
  double end_d = end.toDouble();
  double step_d = step.toDouble();
  TORCH_CHECK(step_d != 0, "pyre: arange step must be non-zero");
  TORCH_CHECK((end_d - start_d) / step_d >= 0,
      "pyre: arange upper bound and step mismatch");

  int64_t out_size = static_cast<int64_t>(
      std::ceil((end_d - start_d) / step_d));
  if (out_size < 0) out_size = 0;

  auto out_dtype = dtype.value_or(
      start.isFloatingPoint() || end.isFloatingPoint() || step.isFloatingPoint()
          ? at::kFloat : at::kLong);
  auto out_device = c10::device_or_default(device);

  auto func_name = funcNameDefault(aten_name);
  auto spec = buildArangeKernelSpec(
      func_name, out_dtype, out_size, start_d, end_d, step_d);
  auto cache_key = contentHashCacheKey(
      spec.template_sha1, spec.substitutions,
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags());

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = generateArangeMlir(
        func_name, out_dtype, out_size, start_d, end_d, step_d);
    kernel = getOrCompile(cache_key, func_name, mlir);
  }

  // Zero tensor inputs — invoke with empty input list.
  return invokeKernel(kernel, {}, {out_size},
                      at::TensorOptions().dtype(out_dtype).device(out_device));
}

// ---------------------------------------------------------------------------
// TypeCastOp
// ---------------------------------------------------------------------------

at::Tensor TypeCastOp::impl(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<at::MemoryFormat> memory_format) {
  if (!hasPyreBuffer(self) || !dtype.has_value() || *dtype == self.scalar_type())
    return at::native::_to_copy(self, dtype, layout, device, pin_memory,
                                 non_blocking, memory_format);

  auto in_dtype = self.scalar_type();
  auto out_dtype = *dtype;
  auto func_name = funcNameDefault(aten_name);

  auto spec = buildTypeCastKernelSpec(func_name, in_dtype, out_dtype, self.sizes());
  auto cache_key = contentHashCacheKey(
      spec.template_sha1, spec.substitutions,
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags());

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = generateTypeCastMlir(func_name, in_dtype, out_dtype, self.sizes());
    kernel = getOrCompile(cache_key, func_name, mlir);
  }

  return invokeKernel(kernel, {self}, c10::DimVector(self.sizes()),
                      self.options().dtype(out_dtype));
}

// ---------------------------------------------------------------------------
// CatOp — algorithmic MLIR generation using PyreKernelAsmFragments
// ---------------------------------------------------------------------------

// Fragment accessors in dispatch/PyreCatKernel.cpp.

at::Tensor CatOp::impl(const at::ITensorListRef& tensors, int64_t dim) {
  auto materialized = tensors.materialize();
  TORCH_CHECK(!materialized.empty(), "pyre: cat requires at least one tensor");

  for (const auto& t : materialized)
    TORCH_CHECK(hasPyreBuffer(t.get()), "pyre: cat requires IREE buffers");

  auto dtype = materialized[0].get().scalar_type();
  if (dim < 0) dim += materialized[0].get().dim();

  c10::DimVector out_shape(materialized[0].get().sizes().begin(),
                           materialized[0].get().sizes().end());
  out_shape[dim] = 0;
  for (const auto& t : materialized)
    out_shape[dim] += t.get().size(dim);

  auto func_name = funcNameDefault(aten_name);
  std::string elt = scalarTypeToTorchMlir(dtype);
  std::string dim_str = std::to_string(dim);

  auto dynShape = [](const at::Tensor& t) {
    std::string s;
    for (int64_t i = 0; i < t.dim(); ++i) {
      if (i > 0) s += ",";
      s += (t.size(i) == 1) ? "1" : "?";
    }
    return s;
  };

  std::string out_s;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    if (i > 0) out_s += ",";
    out_s += (out_shape[i] == 1) ? "1" : "?";
  }

  // Pre-compute per-input shape strings
  c10::SmallVector<std::string, 8> input_shapes;
  for (const auto& t : materialized)
    input_shapes.push_back(dynShape(t.get()));

  // Build input_names and input_types strings
  std::string input_names, input_types;
  for (size_t i = 0; i < materialized.size(); ++i) {
    if (i > 0) { input_names += ", "; input_types += ", "; }
    input_names += "%input" + std::to_string(i);
    input_types += "!torch.vtensor<[" + input_shapes[i] + "], " + elt + ">";
  }

  // Recipe lambda: replayed in both digest and generate modes
  auto recipe = [&](PyreKernelAsmBuilder& b) {
    b.appendFragment(0, {{"FUNC", func_name}, {"OUT_SHAPE", out_s}, {"ELT", elt}});
    for (size_t i = 0; i < materialized.size(); ++i) {
      b.appendFragment(1, {
          {"IDX", std::to_string(i)},
          {"INPUT_SHAPE", input_shapes[i]},
          {"ELT", elt}});
    }
    b.appendFragment(2, {
        {"INPUT_NAMES", input_names}, {"INPUT_TYPES", input_types},
        {"DIM", dim_str}, {"OUT_SHAPE", out_s}, {"ELT", elt}});
  };

  auto& frags = catFragments();
  auto cache_key = frags.digest(
      c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags(), recipe);

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name);
  if (!kernel) {
    auto mlir = frags.generateMlir(recipe);
    kernel = getOrCompile(cache_key, func_name, mlir);
  }

  std::vector<at::Tensor> inputs;
  inputs.reserve(materialized.size());
  for (const auto& t : materialized)
    inputs.push_back(t.get());

  return invokeKernel(kernel, inputs, out_shape,
                      materialized[0].get().options());
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void registerCompiledOps(torch::Library& m) {
  // --- Binary ops (functional, sorted) ---
  AddOp::register_impl(m);
  AddmmOp::register_impl(m);
  Atan2Op::register_impl(m);
  BitwiseAndOp::register_impl(m);
  BitwiseOrOp::register_impl(m);
  BitwiseXorOp::register_impl(m);
  DivOp::register_impl(m);
  FmodOp::register_impl(m);
  MaximumOp::register_impl(m);
  MinimumOp::register_impl(m);
  MmOp::register_impl(m);
  MulOp::register_impl(m);
  PowTensorOp::register_impl(m);
  RemainderOp::register_impl(m);
  SubOp::register_impl(m);

  // --- Binary ops (in-place, sorted) ---
  AddOp::register_inplace(m);
  Atan2Op::register_inplace(m);
  BitwiseAndOp::register_inplace(m);
  BitwiseOrOp::register_inplace(m);
  BitwiseXorOp::register_inplace(m);
  DivOp::register_inplace(m);
  FmodOp::register_inplace(m);
  MulOp::register_inplace(m);
  PowTensorOp::register_inplace(m);
  RemainderOp::register_inplace(m);
  SubOp::register_inplace(m);

  // --- Unary ops (functional, sorted) ---
  AbsOp::register_impl(m);
  BitwiseNotOp::register_impl(m);
  CeilOp::register_impl(m);
  CosOp::register_impl(m);
  EluOp::register_impl(m);
  ErfOp::register_impl(m);
  ExpOp::register_impl(m);
  FloorOp::register_impl(m);
  GeluOp::register_impl(m);
  HardtanhOp::register_impl(m);
  LeakyReluOp::register_impl(m);
  LogOp::register_impl(m);
  LogicalNotOp::register_impl(m);
  NegOp::register_impl(m);
  ReciprocalOp::register_impl(m);
  ReluOp::register_impl(m);
  RoundOp::register_impl(m);
  RsqrtOp::register_impl(m);
  SigmoidOp::register_impl(m);
  SignOp::register_impl(m);
  SiluOp::register_impl(m);
  SinOp::register_impl(m);
  SqrtOp::register_impl(m);
  TanhOp::register_impl(m);

  // --- Unary ops (in-place, sorted) ---
  AbsOp::register_inplace(m);
  BitwiseNotOp::register_inplace(m);
  CeilOp::register_inplace(m);
  CosOp::register_inplace(m);
  EluOp::register_inplace(m);
  ErfOp::register_inplace(m);
  ExpOp::register_inplace(m);
  FloorOp::register_inplace(m);
  GeluOp::register_inplace(m);
  HardtanhOp::register_inplace(m);
  LeakyReluOp::register_inplace(m);
  LogOp::register_inplace(m);
  LogicalNotOp::register_inplace(m);
  NegOp::register_inplace(m);
  ReciprocalOp::register_inplace(m);
  ReluOp::register_inplace(m);
  RoundOp::register_inplace(m);
  RsqrtOp::register_inplace(m);
  SigmoidOp::register_inplace(m);
  SignOp::register_inplace(m);
  SiluOp::register_inplace(m);
  SinOp::register_inplace(m);
  SqrtOp::register_inplace(m);
  TanhOp::register_inplace(m);

  // --- Scalar binary ops (functional, sorted) ---
  AddScalarOp::register_impl(m);
  DivScalarOp::register_impl(m);
  MulScalarOp::register_impl(m);
  PowScalarOp::register_impl(m);
  SubScalarOp::register_impl(m);

  // --- Scalar binary ops (in-place, sorted) ---
  AddScalarOp::register_inplace(m);
  DivScalarOp::register_inplace(m);
  MulScalarOp::register_inplace(m);
  PowScalarOp::register_inplace(m);
  SubScalarOp::register_inplace(m);

  // --- Comparison ops (no in-place — output dtype differs) ---
  EqScalarOp::register_impl(m);
  EqTensorOp::register_impl(m);
  GeScalarOp::register_impl(m);
  GeTensorOp::register_impl(m);
  GtScalarOp::register_impl(m);
  GtTensorOp::register_impl(m);
  LeScalarOp::register_impl(m);
  LeTensorOp::register_impl(m);
  LtScalarOp::register_impl(m);
  LtTensorOp::register_impl(m);
  NeScalarOp::register_impl(m);
  NeTensorOp::register_impl(m);

  // --- Reduction ops (no in-place — output shape differs) ---
  AmaxOp::register_impl(m);
  AminOp::register_impl(m);
  MeanOp::register_impl(m);
  ProdOp::register_impl(m);
  SumOp::register_impl(m);

  // --- Custom ops ---
  ArangeOp::register_impl(m);
  BmmOp::register_impl(m);
  CatOp::register_impl(m);
  EmbeddingOp::register_impl(m);
  GatherOp::register_impl(m);
  IndexSelectOp::register_impl(m);
  IndexTensorOp::register_impl(m);
  LogSoftmaxOp::register_impl(m);
  SoftmaxOp::register_impl(m);
  m.impl("index_put", &IndexPutOp::impl);
  m.impl("index_put_", &IndexPutOp::impl_inplace);
  m.impl("scatter.src", &ScatterSrcOp::impl);
  m.impl("scatter_.src", &ScatterSrcOp::impl_inplace);
  m.impl("scatter_add", &ScatterAddOp::impl);
  m.impl("scatter_add_", &ScatterAddOp::impl_inplace);
  TypeCastOp::register_impl(m);
  WhereOp::register_impl(m);
}

} // namespace at::pyre
