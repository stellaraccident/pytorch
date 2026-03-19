#include <ATen/pyre/PyreOp.h>

namespace at::pyre {

// ---------------------------------------------------------------------------
// Stock helper implementations
// ---------------------------------------------------------------------------

c10::DimVector inferShapeBroadcast(const OpContext& ctx) {
  TORCH_CHECK(ctx.raw_inputs.size() >= 2);
  // Use raw (logical) shapes for broadcast inference.
  return c10::DimVector(
      at::infer_size(ctx.raw_inputs[0].sizes(), ctx.raw_inputs[1].sizes()));
}

c10::DimVector inferShapeIdentity(const OpContext& ctx) {
  // Use raw (logical) shape — physical shape may differ for permuted inputs.
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

ExpandedKernel expandBinaryStandard(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx) {
  bool has_alpha = !ctx.scalars.empty() &&
      std::abs(ctx.scalars[0].toDouble() - 1.0) >= 1e-12;

  if (has_alpha) {
    std::string arith_op;
    if (std::string(torch_op) == "sub")
      arith_op = isFloatDtype(ctx.dtype) ? "arith.subf" : "arith.subi";
    else
      arith_op = isFloatDtype(ctx.dtype) ? "arith.addf" : "arith.addi";

    auto out_shape = at::infer_size(
        ctx.raw_inputs[0].sizes(), ctx.raw_inputs[1].sizes());
    return expandBinaryAlphaTemplate(
        func_name, arith_op,
        isFloatDtype(ctx.dtype) ? "arith.mulf" : "arith.muli",
        ctx.scalars[0].toDouble(), ctx.dtype,
        ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
        out_shape, ctx.decision.arg_adapters);
  }

  auto out_shape = at::infer_size(
      ctx.raw_inputs[0].sizes(), ctx.raw_inputs[1].sizes());
  return expandBinaryTemplate(
      func_name, torch_op, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
      out_shape, ctx.decision.arg_adapters);
}

ExpandedKernel expandUnaryStandard(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx) {
  ArgAdapter adapter = ctx.decision.arg_adapters.empty()
      ? ArgAdapter{ArgAdapter::kIdentity, {}}
      : ctx.decision.arg_adapters[0];
  return expandUnaryTemplate(
      func_name, torch_op, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[0].sizes(), adapter);
}

// ---------------------------------------------------------------------------
// getOrCompile / invokeKernel
// ---------------------------------------------------------------------------

CachedKernel* getOrCompile(
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
// MmOp::expandTemplate
// ---------------------------------------------------------------------------

ExpandedKernel MmOp::expandTemplate(
    const std::string& func_name, const OpContext& ctx) {
  auto out_shape = inferShape(ctx);
  return expandBinaryTemplate(
      func_name, torch_op, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
      out_shape, ctx.decision.arg_adapters);
}

// ---------------------------------------------------------------------------
// AddmmOp::expandTemplate
// ---------------------------------------------------------------------------

ExpandedKernel AddmmOp::expandTemplate(
    const std::string& func_name, const OpContext& ctx) {
  bool mat2_permuted = ctx.decision.arg_adapters.size() > 2
      && ctx.decision.arg_adapters[2].kind == ArgAdapter::kPermute;
  if (mat2_permuted) {
    // ctx.inputs[2] is the physical (contiguous) layout after un-permuting.
    // expandAddmmTransposedTemplate expects the physical shape and emits
    // torch.aten.t inside the kernel to reconstruct logical shape.
    return expandAddmmTransposedTemplate(func_name, ctx.dtype,
        ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
        ctx.inputs[2].sizes(), inferShape(ctx));
  }
  return expandAddmmTemplate(func_name, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
      ctx.inputs[2].sizes(), inferShape(ctx));
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void registerCompiledOps(torch::Library& m) {
  AddOp::register_impl(m);
  SubOp::register_impl(m);
  MulOp::register_impl(m);
  DivOp::register_impl(m);
  MmOp::register_impl(m);
  AddmmOp::register_impl(m);
  NegOp::register_impl(m);
  ReluOp::register_impl(m);
}

} // namespace at::pyre
