#pragma once

// CRTP-based op registry for compiled kernel dispatch.
//
// PyreOp<Derived> provides the dispatch skeleton. Intermediate CRTP
// templates (RegularBinaryOp, AlphaBinaryOp, RegularUnaryOp) provide
// behavior statics for common patterns. Concrete ops carry only data.
//
// See epic1_kernel_dispatch.md §10.3 P1-A.

#include <ATen/Tensor.h>
#include <ATen/pyre/PyreTensor.h>
#include <ATen/pyre/dispatch/PyreArgAdapter.h>
#include <ATen/pyre/dispatch/PyreArgShapeSpecializer.h>
#include <ATen/pyre/dispatch/PyreKernelAsmBuilder.h>
#include <ATen/pyre/dispatch/PyreKernelCache.h>
#include <ATen/pyre/dispatch/PyreKernelCompiler.h>
#include <ATen/pyre/dispatch/PyreKernelDispatch.h>
#include <ATen/pyre/dispatch/PyreTypeMapping.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <c10/core/Scalar.h>
#include <c10/pyre/impl/PyreDevice.h>

#include <torch/library.h>

#include <sstream>
#include <string>
#include <vector>

namespace at::pyre {

struct OpContext {
  c10::ArrayRef<at::Tensor> inputs;       // post-adapter (physical layout)
  c10::ArrayRef<at::Tensor> raw_inputs;   // pre-adapter (logical layout)
  c10::ArrayRef<at::Scalar> scalars;
  c10::ScalarType dtype;
  const SpecDecision& decision;
};

// Stock helper functions (non-templated, do the real work).
c10::DimVector inferShapeBroadcast(const OpContext& ctx);
c10::DimVector inferShapeIdentity(const OpContext& ctx);
std::string funcNameDefault(const char* aten_name);

// Promote 0-dim CPU scalar tensors to the target device with matching
// dtype (matches CUDA behavior). Returns true if promotion occurred,
// in which case `out` holds the promoted tensors and `effective` points
// into it. Otherwise `effective` points to `raw_inputs` unchanged.
bool promoteScalarTensors(
    c10::ArrayRef<at::Tensor> raw_inputs,
    c10::SmallVector<at::Tensor, 4>& out,
    c10::ArrayRef<at::Tensor>& effective);

// Stock helpers — split into spec (cheap) + mlir (expensive).
KernelSpec buildBinarySpec(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx);
std::string buildBinaryMlir(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx);

KernelSpec buildUnarySpec(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx,
    const std::string& extra_arg_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_arg_types = "");
std::string buildUnaryMlir(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx,
    const std::string& extra_arg_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_arg_types = "");

// Scalar binary helpers — split into spec + mlir.
KernelSpec buildScalarBinarySpec(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx, double scalar_value,
    const std::string& extra_arg_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_arg_types = "");
std::string buildScalarBinaryMlirHelper(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx, double scalar_value,
    const std::string& extra_arg_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_arg_types = "");

// Compile-and-cache, returning the cached entry.
// AbiConfig controls compiler flags and function resolution.
CachedKernel* getOrCompile(
    const std::string& cache_key,
    const std::string& func_name,
    const std::string& mlir,
    const AbiConfig& abi = AbiConfig::kTorchTyped);

// Invoke a kernel with the given inputs, producing an output tensor.
at::Tensor invokeKernel(
    CachedKernel* kernel,
    const std::vector<at::Tensor>& inputs,
    c10::IntArrayRef out_shape,
    const at::TensorOptions& opts);

// Invoke a kernel in-place: writes result into self's buffer (contiguous fast
// path) or into a contiguous temp then copies back (non-contiguous slow path).
void invokeKernelInplace(
    CachedKernel* kernel,
    const std::vector<at::Tensor>& inputs,
    at::Tensor& self);

// ---------------------------------------------------------------------------
// CRTP base
// ---------------------------------------------------------------------------

template <typename Derived>
struct PyreOp {
  static void register_impl(torch::Library& m) {
    m.impl(Derived::aten_name, &Derived::impl);
  }

  static void register_inplace(torch::Library& m) {
    m.impl(Derived::aten_name_inplace, &Derived::impl_inplace);
  }

  static at::Tensor& dispatch_inplace(
      at::Tensor& self,
      c10::ArrayRef<at::Tensor> other_inputs,
      c10::ArrayRef<at::Scalar> scalars) {
    c10::SmallVector<at::Tensor, 4> all_inputs;
    all_inputs.push_back(self);
    all_inputs.insert(all_inputs.end(),
                      other_inputs.begin(), other_inputs.end());

    c10::SmallVector<at::Tensor, 4> promoted_storage;
    c10::ArrayRef<at::Tensor> effective_inputs;
    promoteScalarTensors(all_inputs, promoted_storage, effective_inputs);

    for (const auto& t : effective_inputs)
      TORCH_CHECK(hasPyreBuffer(t), "pyre: ", Derived::aten_name,
          " (in-place) requires tensors with IREE buffers");

    auto dtype = effective_inputs[0].scalar_type();

    auto decision = ArgShapeSpecializer().analyze(
        Derived::aten_name, dtype, effective_inputs,
        c10::pyre::PyreDevice::get(0)->capabilities());

    auto adapted = applyAdapters(
        {effective_inputs.begin(), effective_inputs.end()},
        decision.arg_adapters);

    OpContext ctx{adapted, effective_inputs, scalars, dtype, decision};

    auto func_name = Derived::buildFuncName(ctx);
    auto spec = Derived::buildKernelSpec(func_name, ctx);
    auto cache_key = contentHashCacheKey(
        spec.template_sha1, spec.substitutions,
        c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags());

    auto& cache = PyreKernelCache::get();
    auto* kernel = cache.lookup(cache_key, func_name);
    if (!kernel) {
      auto mlir = Derived::generateMlir(func_name, ctx);
      kernel = getOrCompile(cache_key, func_name, mlir);
    }

    invokeKernelInplace(kernel, adapted, self);
    return self;
  }

  static at::Tensor dispatch(
      c10::ArrayRef<at::Tensor> raw_inputs,
      c10::ArrayRef<at::Scalar> scalars) {
    c10::SmallVector<at::Tensor, 4> promoted_storage;
    c10::ArrayRef<at::Tensor> effective_inputs;
    promoteScalarTensors(raw_inputs, promoted_storage, effective_inputs);

    for (const auto& t : effective_inputs)
      TORCH_CHECK(hasPyreBuffer(t), "pyre: ", Derived::aten_name,
          " requires tensors with IREE buffers");

    auto dtype = effective_inputs[0].scalar_type();

    auto decision = ArgShapeSpecializer().analyze(
        Derived::aten_name, dtype, effective_inputs,
        c10::pyre::PyreDevice::get(0)->capabilities());

    auto adapted = applyAdapters(
        {effective_inputs.begin(), effective_inputs.end()},
        decision.arg_adapters);

    OpContext ctx{adapted, effective_inputs, scalars, dtype, decision};

    auto out_shape  = Derived::inferShape(ctx);
    auto func_name  = Derived::buildFuncName(ctx);

    // Phase 1: build spec (cheap) and hash for cache lookup.
    auto spec = Derived::buildKernelSpec(func_name, ctx);
    auto cache_key = contentHashCacheKey(
        spec.template_sha1, spec.substitutions,
        c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags());

    // Phase 2: lookup. Generate MLIR only on cache miss.
    auto& cache = PyreKernelCache::get();
    auto* kernel = cache.lookup(cache_key, func_name);
    if (!kernel) {
      auto mlir = Derived::generateMlir(func_name, ctx);
      kernel = getOrCompile(cache_key, func_name, mlir);
    }

    return invokeKernel(kernel, adapted, out_shape,
                        Derived::outputOptions(ctx));
  }

  // Default: output options from first input. Override for comparison ops.
  static at::TensorOptions outputOptions(const OpContext& ctx) {
    return ctx.raw_inputs[0].options();
  }
};

// ---------------------------------------------------------------------------
// Intermediate: RegularBinaryOp — two tensor inputs, no scalars, broadcast
// ---------------------------------------------------------------------------

template <typename Derived>
struct RegularBinaryOp : PyreOp<Derived> {
  static at::Tensor impl(const at::Tensor& self, const at::Tensor& other) {
    return PyreOp<Derived>::dispatch({self, other}, {});
  }
  static at::Tensor& impl_inplace(at::Tensor& self, const at::Tensor& other) {
    return PyreOp<Derived>::dispatch_inplace(self, {other}, {});
  }

  static c10::DimVector inferShape(const OpContext& ctx) {
    return inferShapeBroadcast(ctx);
  }
  static KernelSpec buildKernelSpec(
      const std::string& func_name, const OpContext& ctx) {
    return buildBinarySpec(Derived::torch_op, func_name, ctx);
  }
  static std::string generateMlir(
      const std::string& func_name, const OpContext& ctx) {
    return buildBinaryMlir(Derived::torch_op, func_name, ctx);
  }
  static std::string buildFuncName(const OpContext&) {
    return funcNameDefault(Derived::aten_name);
  }
};

// ---------------------------------------------------------------------------
// Intermediate: AlphaBinaryOp — two tensor inputs + alpha scalar, broadcast
// ---------------------------------------------------------------------------

template <typename Derived>
struct AlphaBinaryOp : PyreOp<Derived> {
  static at::Tensor impl(
      const at::Tensor& self, const at::Tensor& other,
      const at::Scalar& alpha) {
    return PyreOp<Derived>::dispatch({self, other}, {alpha});
  }
  static at::Tensor& impl_inplace(
      at::Tensor& self, const at::Tensor& other,
      const at::Scalar& alpha) {
    return PyreOp<Derived>::dispatch_inplace(self, {other}, {alpha});
  }

  static c10::DimVector inferShape(const OpContext& ctx) {
    return inferShapeBroadcast(ctx);
  }
  static KernelSpec buildKernelSpec(
      const std::string& func_name, const OpContext& ctx) {
    return buildBinarySpec(Derived::torch_op, func_name, ctx);
  }
  static std::string generateMlir(
      const std::string& func_name, const OpContext& ctx) {
    return buildBinaryMlir(Derived::torch_op, func_name, ctx);
  }
  static std::string buildFuncName(const OpContext& ctx) {
    auto base = funcNameDefault(Derived::aten_name);
    if (!ctx.scalars.empty() &&
        std::abs(ctx.scalars[0].toDouble() - 1.0) >= 1e-12)
      base += "_alpha";
    return base;
  }
};

// ---------------------------------------------------------------------------
// Intermediate: RegularUnaryOp — single tensor input, identity output shape
// ---------------------------------------------------------------------------

template <typename Derived>
struct RegularUnaryOp : PyreOp<Derived> {
  static at::Tensor impl(const at::Tensor& self) {
    return PyreOp<Derived>::dispatch({self}, {});
  }
  static at::Tensor& impl_inplace(at::Tensor& self) {
    return PyreOp<Derived>::dispatch_inplace(self, {}, {});
  }

  static c10::DimVector inferShape(const OpContext& ctx) {
    return inferShapeIdentity(ctx);
  }
  static KernelSpec buildKernelSpec(
      const std::string& func_name, const OpContext& ctx) {
    return buildUnarySpec(Derived::torch_op, func_name, ctx);
  }
  static std::string generateMlir(
      const std::string& func_name, const OpContext& ctx) {
    return buildUnaryMlir(Derived::torch_op, func_name, ctx);
  }
  static std::string buildFuncName(const OpContext&) {
    return funcNameDefault(Derived::aten_name);
  }
};

// ---------------------------------------------------------------------------
// Intermediate: ParameterizedUnaryOp — single input + constant extra args
// ---------------------------------------------------------------------------

template <typename Derived>
struct ParameterizedUnaryOp : PyreOp<Derived> {
  static at::Tensor impl(const at::Tensor& self) {
    return PyreOp<Derived>::dispatch({self}, {});
  }

  static c10::DimVector inferShape(const OpContext& ctx) {
    return inferShapeIdentity(ctx);
  }
  static KernelSpec buildKernelSpec(
      const std::string& func_name, const OpContext& ctx) {
    return buildUnarySpec(
        Derived::torch_op, func_name, ctx,
        Derived::extraArgDecls(ctx),
        Derived::extraArgs(),
        Derived::extraArgTypes());
  }
  static std::string generateMlir(
      const std::string& func_name, const OpContext& ctx) {
    return buildUnaryMlir(
        Derived::torch_op, func_name, ctx,
        Derived::extraArgDecls(ctx),
        Derived::extraArgs(),
        Derived::extraArgTypes());
  }
  static std::string buildFuncName(const OpContext&) {
    return funcNameDefault(Derived::aten_name);
  }
};

// ---------------------------------------------------------------------------
// Concrete ops — binary
// ---------------------------------------------------------------------------

struct AddOp : AlphaBinaryOp<AddOp> {
  static constexpr const char* aten_name = "add.Tensor";
  static constexpr const char* aten_name_inplace = "add_.Tensor";
  static constexpr const char* torch_op = "add";
};
struct Atan2Op : RegularBinaryOp<Atan2Op> {
  static constexpr const char* aten_name = "atan2";
  static constexpr const char* aten_name_inplace = "atan2_";
  static constexpr const char* torch_op = "atan2";
};
struct BitwiseAndOp : RegularBinaryOp<BitwiseAndOp> {
  static constexpr const char* aten_name = "bitwise_and.Tensor";
  static constexpr const char* aten_name_inplace = "bitwise_and_.Tensor";
  static constexpr const char* torch_op = "bitwise_and";
};
struct BitwiseOrOp : RegularBinaryOp<BitwiseOrOp> {
  static constexpr const char* aten_name = "bitwise_or.Tensor";
  static constexpr const char* aten_name_inplace = "bitwise_or_.Tensor";
  static constexpr const char* torch_op = "bitwise_or";
};
struct BitwiseXorOp : RegularBinaryOp<BitwiseXorOp> {
  static constexpr const char* aten_name = "bitwise_xor.Tensor";
  static constexpr const char* aten_name_inplace = "bitwise_xor_.Tensor";
  static constexpr const char* torch_op = "bitwise_xor";
};
struct DivOp : RegularBinaryOp<DivOp> {
  static constexpr const char* aten_name = "div.Tensor";
  static constexpr const char* aten_name_inplace = "div_.Tensor";
  static constexpr const char* torch_op = "div";
};
struct FmodOp : RegularBinaryOp<FmodOp> {
  static constexpr const char* aten_name = "fmod.Tensor";
  static constexpr const char* aten_name_inplace = "fmod_.Tensor";
  static constexpr const char* torch_op = "fmod";
};
struct MaximumOp : RegularBinaryOp<MaximumOp> {
  static constexpr const char* aten_name = "maximum";
  static constexpr const char* torch_op = "maximum";
};
struct MinimumOp : RegularBinaryOp<MinimumOp> {
  static constexpr const char* aten_name = "minimum";
  static constexpr const char* torch_op = "minimum";
};
struct MulOp : RegularBinaryOp<MulOp> {
  static constexpr const char* aten_name = "mul.Tensor";
  static constexpr const char* aten_name_inplace = "mul_.Tensor";
  static constexpr const char* torch_op = "mul";
};
struct PowTensorOp : RegularBinaryOp<PowTensorOp> {
  static constexpr const char* aten_name = "pow.Tensor_Tensor";
  static constexpr const char* aten_name_inplace = "pow_.Tensor";
  static constexpr const char* torch_op = "pow";
};
struct RemainderOp : RegularBinaryOp<RemainderOp> {
  static constexpr const char* aten_name = "remainder.Tensor";
  static constexpr const char* aten_name_inplace = "remainder_.Tensor";
  static constexpr const char* torch_op = "remainder";
};
struct SubOp : AlphaBinaryOp<SubOp> {
  static constexpr const char* aten_name = "sub.Tensor";
  static constexpr const char* aten_name_inplace = "sub_.Tensor";
  static constexpr const char* torch_op = "sub";
};
struct MmOp : PyreOp<MmOp> {
  static constexpr const char* aten_name = "mm";
  static constexpr const char* torch_op = "mm";

  static at::Tensor impl(const at::Tensor& self, const at::Tensor& other) {
    TORCH_CHECK(self.dim() == 2 && other.dim() == 2,
        "pyre: mm requires 2D tensors");
    TORCH_CHECK(self.size(1) == other.size(0),
        "pyre: mm dimension mismatch");
    return dispatch({self, other}, {});
  }

  static c10::DimVector inferShape(const OpContext& ctx) {
    return {ctx.inputs[0].size(0), ctx.inputs[1].size(1)};
  }
  static KernelSpec buildKernelSpec(
      const std::string& func_name, const OpContext& ctx);
  static std::string generateMlir(
      const std::string& func_name, const OpContext& ctx);
  static std::string buildFuncName(const OpContext&) {
    return funcNameDefault(aten_name);
  }
};

// --- Unary ops ---

struct AbsOp : RegularUnaryOp<AbsOp> {
  static constexpr const char* aten_name = "abs";
  static constexpr const char* aten_name_inplace = "abs_";
  static constexpr const char* torch_op = "torch.aten.abs";
};
struct BitwiseNotOp : RegularUnaryOp<BitwiseNotOp> {
  static constexpr const char* aten_name = "bitwise_not";
  static constexpr const char* aten_name_inplace = "bitwise_not_";
  static constexpr const char* torch_op = "torch.aten.bitwise_not";
};
struct CeilOp : RegularUnaryOp<CeilOp> {
  static constexpr const char* aten_name = "ceil";
  static constexpr const char* aten_name_inplace = "ceil_";
  static constexpr const char* torch_op = "torch.aten.ceil";
};
struct CosOp : RegularUnaryOp<CosOp> {
  static constexpr const char* aten_name = "cos";
  static constexpr const char* aten_name_inplace = "cos_";
  static constexpr const char* torch_op = "torch.aten.cos";
};
struct ErfOp : RegularUnaryOp<ErfOp> {
  static constexpr const char* aten_name = "erf";
  static constexpr const char* aten_name_inplace = "erf_";
  static constexpr const char* torch_op = "torch.aten.erf";
};
struct ExpOp : RegularUnaryOp<ExpOp> {
  static constexpr const char* aten_name = "exp";
  static constexpr const char* aten_name_inplace = "exp_";
  static constexpr const char* torch_op = "torch.aten.exp";
};
struct FloorOp : RegularUnaryOp<FloorOp> {
  static constexpr const char* aten_name = "floor";
  static constexpr const char* aten_name_inplace = "floor_";
  static constexpr const char* torch_op = "torch.aten.floor";
};
struct LogOp : RegularUnaryOp<LogOp> {
  static constexpr const char* aten_name = "log";
  static constexpr const char* aten_name_inplace = "log_";
  static constexpr const char* torch_op = "torch.aten.log";
};
struct LogicalNotOp : RegularUnaryOp<LogicalNotOp> {
  static constexpr const char* aten_name = "logical_not";
  static constexpr const char* aten_name_inplace = "logical_not_";
  static constexpr const char* torch_op = "torch.aten.logical_not";
};
struct NegOp : RegularUnaryOp<NegOp> {
  static constexpr const char* aten_name = "neg";
  static constexpr const char* aten_name_inplace = "neg_";
  static constexpr const char* torch_op = "torch.aten.neg";
};
struct ReciprocalOp : RegularUnaryOp<ReciprocalOp> {
  static constexpr const char* aten_name = "reciprocal";
  static constexpr const char* aten_name_inplace = "reciprocal_";
  static constexpr const char* torch_op = "torch.aten.reciprocal";
};
struct ReluOp : RegularUnaryOp<ReluOp> {
  static constexpr const char* aten_name = "relu";
  static constexpr const char* aten_name_inplace = "relu_";
  static constexpr const char* torch_op = "torch.aten.relu";
};
struct RoundOp : RegularUnaryOp<RoundOp> {
  static constexpr const char* aten_name = "round";
  static constexpr const char* aten_name_inplace = "round_";
  static constexpr const char* torch_op = "torch.aten.round";
};
struct RsqrtOp : RegularUnaryOp<RsqrtOp> {
  static constexpr const char* aten_name = "rsqrt";
  static constexpr const char* aten_name_inplace = "rsqrt_";
  static constexpr const char* torch_op = "torch.aten.rsqrt";
};
struct SigmoidOp : RegularUnaryOp<SigmoidOp> {
  static constexpr const char* aten_name = "sigmoid";
  static constexpr const char* aten_name_inplace = "sigmoid_";
  static constexpr const char* torch_op = "torch.aten.sigmoid";
};
struct SignOp : RegularUnaryOp<SignOp> {
  static constexpr const char* aten_name = "sign";
  static constexpr const char* aten_name_inplace = "sign_";
  static constexpr const char* torch_op = "torch.aten.sign";
};
struct SiluOp : RegularUnaryOp<SiluOp> {
  static constexpr const char* aten_name = "silu";
  static constexpr const char* aten_name_inplace = "silu_";
  static constexpr const char* torch_op = "torch.aten.silu";
};
struct SinOp : RegularUnaryOp<SinOp> {
  static constexpr const char* aten_name = "sin";
  static constexpr const char* aten_name_inplace = "sin_";
  static constexpr const char* torch_op = "torch.aten.sin";
};
struct SqrtOp : RegularUnaryOp<SqrtOp> {
  static constexpr const char* aten_name = "sqrt";
  static constexpr const char* aten_name_inplace = "sqrt_";
  static constexpr const char* torch_op = "torch.aten.sqrt";
};
struct TanhOp : RegularUnaryOp<TanhOp> {
  static constexpr const char* aten_name = "tanh";
  static constexpr const char* aten_name_inplace = "tanh_";
  static constexpr const char* torch_op = "torch.aten.tanh";
};

// --- Parameterized unary ops ---

struct GeluOp : ParameterizedUnaryOp<GeluOp> {
  static constexpr const char* aten_name = "gelu";
  static constexpr const char* aten_name_inplace = "gelu_";
  static constexpr const char* torch_op = "torch.aten.gelu";
  static at::Tensor impl(const at::Tensor& self,
                          c10::string_view approximate = "none") {
    TORCH_CHECK(approximate == "none" || approximate == "tanh",
        "pyre: gelu approximate must be 'none' or 'tanh'");
    TORCH_CHECK(approximate == "none",
        "pyre: gelu approximate='tanh' not yet supported");
    return PyreOp<GeluOp>::dispatch({self}, {});
  }
  static at::Tensor& impl_inplace(at::Tensor& self,
                                    c10::string_view approximate = "none") {
    TORCH_CHECK(approximate == "none",
        "pyre: gelu_ approximate='tanh' not yet supported");
    return PyreOp<GeluOp>::dispatch_inplace(self, {}, {});
  }
  static std::string extraArgDecls(const OpContext&) {
    return "    %approx = torch.constant.str \"none\"";
  }
  static std::string extraArgs() { return ", %approx"; }
  static std::string extraArgTypes() { return ", !torch.str"; }
};

struct HardtanhOp : ParameterizedUnaryOp<HardtanhOp> {
  static constexpr const char* aten_name = "hardtanh";
  static constexpr const char* aten_name_inplace = "hardtanh_";
  static constexpr const char* torch_op = "torch.aten.hardtanh";
  static at::Tensor impl(const at::Tensor& self,
                          const at::Scalar& min_val = -1,
                          const at::Scalar& max_val = 1) {
    return PyreOp<HardtanhOp>::dispatch({self}, {min_val, max_val});
  }
  static at::Tensor& impl_inplace(at::Tensor& self,
                                    const at::Scalar& min_val = -1,
                                    const at::Scalar& max_val = 1) {
    return PyreOp<HardtanhOp>::dispatch_inplace(self, {}, {min_val, max_val});
  }
  static std::string extraArgDecls(const OpContext& ctx) {
    double mn = ctx.scalars[0].toDouble();
    double mx = ctx.scalars[1].toDouble();
    std::ostringstream ss;
    ss << std::fixed
       << "    %min = torch.constant.float " << mn << "\n"
       << "    %max = torch.constant.float " << mx;
    return ss.str();
  }
  static std::string extraArgs() { return ", %min, %max"; }
  static std::string extraArgTypes() { return ", !torch.float, !torch.float"; }
};

struct LeakyReluOp : ParameterizedUnaryOp<LeakyReluOp> {
  static constexpr const char* aten_name = "leaky_relu";
  static constexpr const char* aten_name_inplace = "leaky_relu_";
  static constexpr const char* torch_op = "torch.aten.leaky_relu";
  static at::Tensor impl(const at::Tensor& self,
                          const at::Scalar& negative_slope = 0.01) {
    return PyreOp<LeakyReluOp>::dispatch({self}, {negative_slope});
  }
  static at::Tensor& impl_inplace(at::Tensor& self,
                                    const at::Scalar& negative_slope = 0.01) {
    return PyreOp<LeakyReluOp>::dispatch_inplace(self, {}, {negative_slope});
  }
  static std::string extraArgDecls(const OpContext& ctx) {
    double slope = ctx.scalars[0].toDouble();
    std::ostringstream ss;
    ss << std::fixed << "    %neg_slope = torch.constant.float " << slope;
    return ss.str();
  }
  static std::string extraArgs() { return ", %neg_slope"; }
  static std::string extraArgTypes() { return ", !torch.float"; }
};

struct EluOp : ParameterizedUnaryOp<EluOp> {
  static constexpr const char* aten_name = "elu";
  static constexpr const char* aten_name_inplace = "elu_";
  static constexpr const char* torch_op = "torch.aten.elu";
  static at::Tensor impl(const at::Tensor& self,
                          const at::Scalar& alpha = 1,
                          const at::Scalar& scale = 1,
                          const at::Scalar& input_scale = 1) {
    return PyreOp<EluOp>::dispatch({self}, {alpha, scale, input_scale});
  }
  static at::Tensor& impl_inplace(at::Tensor& self,
                                    const at::Scalar& alpha = 1,
                                    const at::Scalar& scale = 1,
                                    const at::Scalar& input_scale = 1) {
    return PyreOp<EluOp>::dispatch_inplace(self, {}, {alpha, scale, input_scale});
  }
  static std::string extraArgDecls(const OpContext& ctx) {
    double a = ctx.scalars[0].toDouble();
    double s = ctx.scalars[1].toDouble();
    double is = ctx.scalars[2].toDouble();
    std::ostringstream ss;
    ss << std::fixed
       << "    %alpha = torch.constant.float " << a << "\n"
       << "    %scale = torch.constant.float " << s << "\n"
       << "    %input_scale = torch.constant.float " << is;
    return ss.str();
  }
  static std::string extraArgs() { return ", %alpha, %scale, %input_scale"; }
  static std::string extraArgTypes() {
    return ", !torch.float, !torch.float, !torch.float";
  }
};

// ---------------------------------------------------------------------------
// Intermediate: ScalarBinaryOp — one tensor + one scalar, identity shape
// ---------------------------------------------------------------------------

template <typename Derived>
struct ScalarBinaryOp : PyreOp<Derived> {
  static at::Tensor& impl_inplace(at::Tensor& self, const at::Scalar& other) {
    return PyreOp<Derived>::dispatch_inplace(self, {}, {other});
  }

  static c10::DimVector inferShape(const OpContext& ctx) {
    return inferShapeIdentity(ctx);
  }
  static KernelSpec buildKernelSpec(
      const std::string& func_name, const OpContext& ctx) {
    double scalar_val = ctx.scalars[0].toDouble();
    return buildScalarBinarySpec(
        Derived::torch_op, func_name, ctx, scalar_val,
        Derived::extraArgDecls(ctx),
        Derived::extraArgs(),
        Derived::extraArgTypes());
  }
  static std::string generateMlir(
      const std::string& func_name, const OpContext& ctx) {
    double scalar_val = ctx.scalars[0].toDouble();
    return buildScalarBinaryMlirHelper(
        Derived::torch_op, func_name, ctx, scalar_val,
        Derived::extraArgDecls(ctx),
        Derived::extraArgs(),
        Derived::extraArgTypes());
  }
  static std::string buildFuncName(const OpContext&) {
    return funcNameDefault(Derived::aten_name);
  }
  // Default: no extra args. Override in add.Scalar/sub.Scalar for alpha.
  static std::string extraArgDecls(const OpContext&) { return ""; }
  static std::string extraArgs() { return ""; }
  static std::string extraArgTypes() { return ""; }
};

// --- Scalar binary ops ---

struct AddScalarOp : ScalarBinaryOp<AddScalarOp> {
  static constexpr const char* aten_name = "add.Scalar";
  static constexpr const char* aten_name_inplace = "add_.Scalar";
  static constexpr const char* torch_op = "torch.aten.add.Scalar";
  static at::Tensor impl(const at::Tensor& self,
                          const at::Scalar& other,
                          const at::Scalar& alpha = 1) {
    double val = other.toDouble() * alpha.toDouble();
    return PyreOp<AddScalarOp>::dispatch({self}, {at::Scalar(val)});
  }
  static at::Tensor& impl_inplace(at::Tensor& self,
                                    const at::Scalar& other,
                                    const at::Scalar& alpha = 1) {
    double val = other.toDouble() * alpha.toDouble();
    return PyreOp<AddScalarOp>::dispatch_inplace(self, {}, {at::Scalar(val)});
  }
  // Alpha is always 1 in MLIR because impl pre-folds other*alpha above.
  static std::string extraArgDecls(const OpContext&) {
    return "    %alpha = torch.constant.int 1";
  }
  static std::string extraArgs() { return ", %alpha"; }
  static std::string extraArgTypes() { return ", !torch.int"; }
};

struct SubScalarOp : ScalarBinaryOp<SubScalarOp> {
  static constexpr const char* aten_name = "sub.Scalar";
  static constexpr const char* aten_name_inplace = "sub_.Scalar";
  static constexpr const char* torch_op = "torch.aten.sub.Scalar";
  static at::Tensor impl(const at::Tensor& self,
                          const at::Scalar& other,
                          const at::Scalar& alpha = 1) {
    double val = other.toDouble() * alpha.toDouble();
    return PyreOp<SubScalarOp>::dispatch({self}, {at::Scalar(val)});
  }
  static at::Tensor& impl_inplace(at::Tensor& self,
                                    const at::Scalar& other,
                                    const at::Scalar& alpha = 1) {
    double val = other.toDouble() * alpha.toDouble();
    return PyreOp<SubScalarOp>::dispatch_inplace(self, {}, {at::Scalar(val)});
  }
  // Alpha is always 1 in MLIR because impl pre-folds other*alpha above.
  static std::string extraArgDecls(const OpContext&) {
    return "    %alpha = torch.constant.int 1";
  }
  static std::string extraArgs() { return ", %alpha"; }
  static std::string extraArgTypes() { return ", !torch.int"; }
};

struct MulScalarOp : ScalarBinaryOp<MulScalarOp> {
  static constexpr const char* aten_name = "mul.Scalar";
  static constexpr const char* aten_name_inplace = "mul_.Scalar";
  static constexpr const char* torch_op = "torch.aten.mul.Scalar";
  static at::Tensor impl(const at::Tensor& self, const at::Scalar& other) {
    return PyreOp<MulScalarOp>::dispatch({self}, {other});
  }
};

struct DivScalarOp : ScalarBinaryOp<DivScalarOp> {
  static constexpr const char* aten_name = "div.Scalar";
  static constexpr const char* aten_name_inplace = "div_.Scalar";
  static constexpr const char* torch_op = "torch.aten.div.Scalar";
  static at::Tensor impl(const at::Tensor& self, const at::Scalar& other) {
    return PyreOp<DivScalarOp>::dispatch({self}, {other});
  }
};

struct PowScalarOp : ScalarBinaryOp<PowScalarOp> {
  static constexpr const char* aten_name = "pow.Tensor_Scalar";
  static constexpr const char* aten_name_inplace = "pow_.Scalar";
  static constexpr const char* torch_op = "torch.aten.pow.Tensor_Scalar";
  static at::Tensor impl(const at::Tensor& self, const at::Scalar& other) {
    return PyreOp<PowScalarOp>::dispatch({self}, {other});
  }
};

// ---------------------------------------------------------------------------
// Intermediate: ComparisonBinaryOp — two tensors → bool tensor
// ---------------------------------------------------------------------------

template <typename Derived>
struct ComparisonBinaryOp : PyreOp<Derived> {
  static at::Tensor impl(const at::Tensor& self, const at::Tensor& other) {
    return PyreOp<Derived>::dispatch({self, other}, {});
  }
  static c10::DimVector inferShape(const OpContext& ctx) {
    return inferShapeBroadcast(ctx);
  }
  static at::TensorOptions outputOptions(const OpContext& ctx) {
    return ctx.raw_inputs[0].options().dtype(at::kBool);
  }
  static KernelSpec buildKernelSpec(
      const std::string& func_name, const OpContext& ctx) {
    auto out_shape = inferShapeBroadcast(ctx);
    return buildComparisonKernelSpec(
        func_name, Derived::torch_op, ctx.dtype,
        ctx.inputs[0].sizes(), ctx.inputs[1].sizes(), out_shape);
  }
  static std::string generateMlir(
      const std::string& func_name, const OpContext& ctx) {
    auto out_shape = inferShapeBroadcast(ctx);
    return generateComparisonMlir(
        func_name, Derived::torch_op, ctx.dtype,
        ctx.inputs[0].sizes(), ctx.inputs[1].sizes(), out_shape);
  }
  static std::string buildFuncName(const OpContext&) {
    return funcNameDefault(Derived::aten_name);
  }
};

// ---------------------------------------------------------------------------
// Intermediate: ComparisonScalarOp — tensor + scalar → bool tensor
// ---------------------------------------------------------------------------

template <typename Derived>
struct ComparisonScalarOp : PyreOp<Derived> {
  static at::Tensor impl(const at::Tensor& self, const at::Scalar& other) {
    return PyreOp<Derived>::dispatch({self}, {other});
  }
  static c10::DimVector inferShape(const OpContext& ctx) {
    return inferShapeIdentity(ctx);
  }
  static at::TensorOptions outputOptions(const OpContext& ctx) {
    return ctx.raw_inputs[0].options().dtype(at::kBool);
  }
  static KernelSpec buildKernelSpec(
      const std::string& func_name, const OpContext& ctx) {
    double scalar_val = ctx.scalars[0].toDouble();
    return buildComparisonScalarKernelSpec(
        func_name, Derived::torch_op, ctx.dtype,
        ctx.inputs[0].sizes(), scalar_val);
  }
  static std::string generateMlir(
      const std::string& func_name, const OpContext& ctx) {
    double scalar_val = ctx.scalars[0].toDouble();
    return generateComparisonScalarMlir(
        func_name, Derived::torch_op, ctx.dtype,
        ctx.inputs[0].sizes(), scalar_val);
  }
  static std::string buildFuncName(const OpContext&) {
    return funcNameDefault(Derived::aten_name);
  }
};

// --- Comparison ops (tensor-tensor) ---

struct EqTensorOp : ComparisonBinaryOp<EqTensorOp> {
  static constexpr const char* aten_name = "eq.Tensor";
  static constexpr const char* torch_op = "torch.aten.eq.Tensor";
};
struct NeTensorOp : ComparisonBinaryOp<NeTensorOp> {
  static constexpr const char* aten_name = "ne.Tensor";
  static constexpr const char* torch_op = "torch.aten.ne.Tensor";
};
struct LtTensorOp : ComparisonBinaryOp<LtTensorOp> {
  static constexpr const char* aten_name = "lt.Tensor";
  static constexpr const char* torch_op = "torch.aten.lt.Tensor";
};
struct LeTensorOp : ComparisonBinaryOp<LeTensorOp> {
  static constexpr const char* aten_name = "le.Tensor";
  static constexpr const char* torch_op = "torch.aten.le.Tensor";
};
struct GtTensorOp : ComparisonBinaryOp<GtTensorOp> {
  static constexpr const char* aten_name = "gt.Tensor";
  static constexpr const char* torch_op = "torch.aten.gt.Tensor";
};
struct GeTensorOp : ComparisonBinaryOp<GeTensorOp> {
  static constexpr const char* aten_name = "ge.Tensor";
  static constexpr const char* torch_op = "torch.aten.ge.Tensor";
};

// --- Comparison ops (tensor-scalar) ---

struct EqScalarOp : ComparisonScalarOp<EqScalarOp> {
  static constexpr const char* aten_name = "eq.Scalar";
  static constexpr const char* torch_op = "torch.aten.eq.Scalar";
};
struct NeScalarOp : ComparisonScalarOp<NeScalarOp> {
  static constexpr const char* aten_name = "ne.Scalar";
  static constexpr const char* torch_op = "torch.aten.ne.Scalar";
};
struct LtScalarOp : ComparisonScalarOp<LtScalarOp> {
  static constexpr const char* aten_name = "lt.Scalar";
  static constexpr const char* torch_op = "torch.aten.lt.Scalar";
};
struct LeScalarOp : ComparisonScalarOp<LeScalarOp> {
  static constexpr const char* aten_name = "le.Scalar";
  static constexpr const char* torch_op = "torch.aten.le.Scalar";
};
struct GtScalarOp : ComparisonScalarOp<GtScalarOp> {
  static constexpr const char* aten_name = "gt.Scalar";
  static constexpr const char* torch_op = "torch.aten.gt.Scalar";
};
struct GeScalarOp : ComparisonScalarOp<GeScalarOp> {
  static constexpr const char* aten_name = "ge.Scalar";
  static constexpr const char* torch_op = "torch.aten.ge.Scalar";
};

// ---------------------------------------------------------------------------
// Reduction ops — custom dispatch (dim list + keepdim in spec/cache key)
// ---------------------------------------------------------------------------

c10::DimVector inferReducedShape(
    c10::IntArrayRef input_shape, c10::IntArrayRef dims, bool keepdim);

// Multi-dim reduction dispatch helper.
at::Tensor dispatchMultiDimReduction(
    const char* aten_name, const char* torch_op,
    const at::Tensor& self, c10::ArrayRef<int64_t> dims, bool keepdim,
    bool has_dtype_arg);

// ReductionOp: thin CRTP wrapper that normalizes dims then delegates.
template <typename Derived>
struct ReductionOp : PyreOp<Derived> {
  static at::Tensor impl_with_dtype(
      const at::Tensor& self, at::OptionalIntArrayRef dim,
      bool keepdim, std::optional<at::ScalarType>) {
    c10::SmallVector<int64_t, 6> dims;
    if (dim.has_value())
      for (auto d : *dim) dims.push_back(d);
    else
      for (int64_t i = 0; i < self.dim(); ++i) dims.push_back(i);
    return dispatchMultiDimReduction(
        Derived::aten_name, Derived::torch_op,
        self, dims, keepdim, Derived::has_dtype_arg);
  }

  static at::Tensor impl_no_dtype(
      const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
    c10::SmallVector<int64_t, 6> dims(dim.begin(), dim.end());
    if (dims.empty())
      for (int64_t i = 0; i < self.dim(); ++i) dims.push_back(i);
    return dispatchMultiDimReduction(
        Derived::aten_name, Derived::torch_op,
        self, dims, keepdim, Derived::has_dtype_arg);
  }
};

// --- Reduction ops ---

struct SumOp : ReductionOp<SumOp> {
  static constexpr const char* aten_name = "sum.dim_IntList";
  static constexpr const char* torch_op = "torch.aten.sum.dim_IntList";
  static constexpr bool has_dtype_arg = true;
  static at::Tensor impl(
      const at::Tensor& self, at::OptionalIntArrayRef dim,
      bool keepdim, std::optional<at::ScalarType> dtype) {
    return impl_with_dtype(self, dim, keepdim, dtype);
  }
};

struct MeanOp : ReductionOp<MeanOp> {
  static constexpr const char* aten_name = "mean.dim";
  static constexpr const char* torch_op = "torch.aten.mean.dim";
  static constexpr bool has_dtype_arg = true;
  static at::Tensor impl(
      const at::Tensor& self, at::OptionalIntArrayRef dim,
      bool keepdim, std::optional<at::ScalarType> dtype) {
    return impl_with_dtype(self, dim, keepdim, dtype);
  }
};

struct AmaxOp : ReductionOp<AmaxOp> {
  static constexpr const char* aten_name = "amax";
  static constexpr const char* torch_op = "torch.aten.amax";
  static constexpr bool has_dtype_arg = false;
  static at::Tensor impl(
      const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
    return impl_no_dtype(self, dim, keepdim);
  }
};

struct AminOp : ReductionOp<AminOp> {
  static constexpr const char* aten_name = "amin";
  static constexpr const char* torch_op = "torch.aten.amin";
  static constexpr bool has_dtype_arg = false;
  static at::Tensor impl(
      const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
    return impl_no_dtype(self, dim, keepdim);
  }
};

// Single-dim reduction dispatch helper (shared by ProdOp and future single-dim ops).
at::Tensor dispatchSingleDimReduction(
    const char* aten_name, const char* torch_op,
    const at::Tensor& self, int64_t dim, bool keepdim,
    const std::string& extra_decls = "",
    const std::string& extra_args = "",
    const std::string& extra_types = "");

struct ProdOp : PyreOp<ProdOp> {
  static constexpr const char* aten_name = "prod.dim_int";
  static constexpr const char* torch_op = "torch.aten.prod.dim_int";

  static at::Tensor impl(
      const at::Tensor& self, int64_t dim,
      bool keepdim, std::optional<at::ScalarType> /*dtype*/) {
    return dispatchSingleDimReduction(
        aten_name, torch_op, self, dim, keepdim,
        "    %none = torch.constant.none", ", %none", ", !torch.none");
  }
};

// --- EmbeddingOp (mixed-dtype: float weight + int64 indices) ---

struct EmbeddingOp : PyreOp<EmbeddingOp> {
  static constexpr const char* aten_name = "embedding";
  static at::Tensor impl(
      const at::Tensor& weight, const at::Tensor& indices,
      int64_t padding_idx, bool scale_grad_by_freq, bool sparse);
};

// --- IndexSelectOp ---

struct IndexSelectOp : PyreOp<IndexSelectOp> {
  static constexpr const char* aten_name = "index_select";
  static at::Tensor impl(
      const at::Tensor& self, int64_t dim, const at::Tensor& index);
};

// --- GatherOp ---

struct GatherOp : PyreOp<GatherOp> {
  static constexpr const char* aten_name = "gather";
  static at::Tensor impl(
      const at::Tensor& self, int64_t dim, const at::Tensor& index,
      bool sparse_grad);
};

// --- ScatterSrcOp ---

struct ScatterSrcOp : PyreOp<ScatterSrcOp> {
  static constexpr const char* aten_name = "scatter.src";
  static at::Tensor impl(
      const at::Tensor& self, int64_t dim,
      const at::Tensor& index, const at::Tensor& src);
  static at::Tensor& impl_inplace(
      at::Tensor& self, int64_t dim,
      const at::Tensor& index, const at::Tensor& src);
};

// --- ScatterAddOp ---

struct ScatterAddOp : PyreOp<ScatterAddOp> {
  static constexpr const char* aten_name = "scatter_add";
  static at::Tensor impl(
      const at::Tensor& self, int64_t dim,
      const at::Tensor& index, const at::Tensor& src);
  static at::Tensor& impl_inplace(
      at::Tensor& self, int64_t dim,
      const at::Tensor& index, const at::Tensor& src);
};

// --- IndexPutOp ---

struct IndexPutOp : PyreOp<IndexPutOp> {
  static constexpr const char* aten_name = "index_put";
  static at::Tensor impl(
      const at::Tensor& self,
      const c10::List<std::optional<at::Tensor>>& indices,
      const at::Tensor& values, bool accumulate);
  static at::Tensor& impl_inplace(
      at::Tensor& self,
      const c10::List<std::optional<at::Tensor>>& indices,
      const at::Tensor& values, bool accumulate);
};

// --- IndexTensorOp (advanced indexing with variable index list) ---

struct IndexTensorOp : PyreOp<IndexTensorOp> {
  static constexpr const char* aten_name = "index.Tensor";
  static at::Tensor impl(
      const at::Tensor& self,
      const c10::List<std::optional<at::Tensor>>& indices);
};

// --- ArangeOp (zero tensor inputs — scalar-only kernel) ---

struct ArangeOp : PyreOp<ArangeOp> {
  static constexpr const char* aten_name = "arange.start_step";
  static at::Tensor impl(
      const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
      std::optional<at::ScalarType> dtype,
      std::optional<at::Layout> layout,
      std::optional<at::Device> device,
      std::optional<bool> pin_memory);
};

// --- TypeCastOp (_to_copy with dtype) ---

struct TypeCastOp : PyreOp<TypeCastOp> {
  static constexpr const char* aten_name = "_to_copy";
  static at::Tensor impl(
      const at::Tensor& self,
      std::optional<at::ScalarType> dtype,
      std::optional<at::Layout> layout,
      std::optional<at::Device> device,
      std::optional<bool> pin_memory,
      bool non_blocking,
      std::optional<at::MemoryFormat> memory_format);
};

// --- BmmOp ---

struct BmmOp : PyreOp<BmmOp> {
  static constexpr const char* aten_name = "bmm";

  static at::Tensor impl(const at::Tensor& self, const at::Tensor& mat2) {
    TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3,
        "pyre: bmm requires 3D tensors");
    TORCH_CHECK(self.size(0) == mat2.size(0),
        "pyre: bmm batch size mismatch");
    TORCH_CHECK(self.size(2) == mat2.size(1),
        "pyre: bmm inner dimension mismatch");
    return dispatch({self, mat2}, {});
  }

  static c10::DimVector inferShape(const OpContext& ctx) {
    return {ctx.inputs[0].size(0), ctx.inputs[0].size(1), ctx.inputs[1].size(2)};
  }
  static KernelSpec buildKernelSpec(
      const std::string& func_name, const OpContext& ctx) {
    return buildBmmKernelSpec(func_name, ctx.dtype,
        ctx.inputs[0].sizes(), ctx.inputs[1].sizes(), inferShape(ctx));
  }
  static std::string generateMlir(
      const std::string& func_name, const OpContext& ctx) {
    return generateBmmMlir(func_name, ctx.dtype,
        ctx.inputs[0].sizes(), ctx.inputs[1].sizes(), inferShape(ctx));
  }
  static std::string buildFuncName(const OpContext&) {
    return funcNameDefault(aten_name);
  }
};

// --- WhereOp ---

struct WhereOp : PyreOp<WhereOp> {
  static constexpr const char* aten_name = "where.self";

  static at::Tensor impl(
      const at::Tensor& condition,
      const at::Tensor& self,
      const at::Tensor& other) {
    // dtype from self (not condition which is bool)
    return dispatch({condition, self, other}, {});
  }

  static c10::DimVector inferShape(const OpContext& ctx) {
    auto s = at::infer_size(ctx.raw_inputs[0].sizes(), ctx.raw_inputs[1].sizes());
    return c10::DimVector(at::infer_size(s, ctx.raw_inputs[2].sizes()));
  }

  static at::TensorOptions outputOptions(const OpContext& ctx) {
    // dtype from self (input[1]), not condition (input[0])
    return ctx.raw_inputs[1].options();
  }

  static KernelSpec buildKernelSpec(
      const std::string& func_name, const OpContext& ctx) {
    auto out_shape = inferShape(ctx);
    return buildWhereKernelSpec(func_name, ctx.raw_inputs[1].scalar_type(),
        ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
        ctx.inputs[2].sizes(), out_shape);
  }
  static std::string generateMlir(
      const std::string& func_name, const OpContext& ctx) {
    auto out_shape = inferShape(ctx);
    return generateWhereMlir(func_name, ctx.raw_inputs[1].scalar_type(),
        ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
        ctx.inputs[2].sizes(), out_shape);
  }
  static std::string buildFuncName(const OpContext&) {
    return funcNameDefault(aten_name);
  }
};

// --- CatOp (algorithmic MLIR generation) ---

struct CatOp : PyreOp<CatOp> {
  static constexpr const char* aten_name = "cat";

  static at::Tensor impl(const at::ITensorListRef& tensors, int64_t dim = 0);
};

// --- One-off: addmm ---

struct AddmmOp : PyreOp<AddmmOp> {
  static constexpr const char* aten_name = "addmm";
  static constexpr const char* torch_op = nullptr;

  static at::Tensor impl(
      const at::Tensor& bias, const at::Tensor& mat1,
      const at::Tensor& mat2,
      const at::Scalar& beta, const at::Scalar& alpha) {
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2,
        "pyre: addmm requires 2D matrices");
    return dispatch({bias, mat1, mat2}, {beta, alpha});
  }

  static c10::DimVector inferShape(const OpContext& ctx) {
    return {ctx.raw_inputs[1].size(0), ctx.raw_inputs[2].size(1)};
  }

  static KernelSpec buildKernelSpec(
      const std::string& func_name, const OpContext& ctx);
  static std::string generateMlir(
      const std::string& func_name, const OpContext& ctx);

  static std::string buildFuncName(const OpContext& ctx) {
    bool mat2_t = ctx.decision.arg_adapters.size() > 2
        && ctx.decision.arg_adapters[2].kind == ArgAdapter::kPermute;
    return mat2_t ? "pyre_addmm_t" : "pyre_addmm";
  }
};

// Registration function for all compiled ops.
void registerCompiledOps(torch::Library& m);

} // namespace at::pyre
