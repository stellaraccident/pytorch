#include <ATen/pyre/PyreOp.h>

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

// ---------------------------------------------------------------------------
// MmOp
// ---------------------------------------------------------------------------

KernelSpec MmOp::buildKernelSpec(
    const std::string& func_name, const OpContext& ctx) {
  auto out_shape = inferShape(ctx);
  return buildMmKernelSpec(
      func_name, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(), out_shape);
}

std::string MmOp::generateMlir(
    const std::string& func_name, const OpContext& ctx) {
  auto out_shape = inferShape(ctx);
  return generateMmMlir(
      func_name, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(), out_shape);
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
// Registration
// ---------------------------------------------------------------------------

void registerCompiledOps(torch::Library& m) {
  // Binary ops
  AddOp::register_impl(m);
  SubOp::register_impl(m);
  MulOp::register_impl(m);
  DivOp::register_impl(m);
  MmOp::register_impl(m);
  AddmmOp::register_impl(m);
  PowTensorOp::register_impl(m);
  MaximumOp::register_impl(m);
  MinimumOp::register_impl(m);
  RemainderOp::register_impl(m);
  FmodOp::register_impl(m);
  BitwiseAndOp::register_impl(m);
  BitwiseOrOp::register_impl(m);
  BitwiseXorOp::register_impl(m);
  Atan2Op::register_impl(m);

  // Pure unary ops
  NegOp::register_impl(m);
  ReluOp::register_impl(m);
  AbsOp::register_impl(m);
  SiluOp::register_impl(m);
  SigmoidOp::register_impl(m);
  TanhOp::register_impl(m);
  RsqrtOp::register_impl(m);
  ExpOp::register_impl(m);
  LogOp::register_impl(m);
  SqrtOp::register_impl(m);
  SinOp::register_impl(m);
  CosOp::register_impl(m);
  CeilOp::register_impl(m);
  FloorOp::register_impl(m);
  RoundOp::register_impl(m);
  ReciprocalOp::register_impl(m);
  ErfOp::register_impl(m);
  BitwiseNotOp::register_impl(m);
  LogicalNotOp::register_impl(m);
  SignOp::register_impl(m);

  // Parameterized unary ops
  GeluOp::register_impl(m);
  HardtanhOp::register_impl(m);
  LeakyReluOp::register_impl(m);
  EluOp::register_impl(m);

  // Scalar binary ops
  AddScalarOp::register_impl(m);
  SubScalarOp::register_impl(m);
  MulScalarOp::register_impl(m);
  DivScalarOp::register_impl(m);
  PowScalarOp::register_impl(m);

  // Comparison ops (tensor-tensor)
  EqTensorOp::register_impl(m);
  NeTensorOp::register_impl(m);
  LtTensorOp::register_impl(m);
  LeTensorOp::register_impl(m);
  GtTensorOp::register_impl(m);
  GeTensorOp::register_impl(m);

  // Comparison ops (tensor-scalar)
  EqScalarOp::register_impl(m);
  NeScalarOp::register_impl(m);
  LtScalarOp::register_impl(m);
  LeScalarOp::register_impl(m);
  GtScalarOp::register_impl(m);
  GeScalarOp::register_impl(m);

  // Reduction ops
  SumOp::register_impl(m);
  MeanOp::register_impl(m);
  AmaxOp::register_impl(m);
  AminOp::register_impl(m);
  // ProdOp skipped: prod.dim_int takes single int, not list — needs separate template

  // Type cast, bmm, where
  TypeCastOp::register_impl(m);
  BmmOp::register_impl(m);
  WhereOp::register_impl(m);
}

} // namespace at::pyre
