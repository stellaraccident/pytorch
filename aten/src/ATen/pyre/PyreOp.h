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
#ifdef AT_PER_OPERATOR_HEADERS
#include <ATen/ops/_to_copy_native.h>
#else
#include <ATen/NativeFunctions.h>
#endif
#include <c10/core/Scalar.h>
#include <c10/pyre/impl/PyreDevice.h>

#include <torch/library.h>

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

// ---------------------------------------------------------------------------
// CRTP base
// ---------------------------------------------------------------------------

template <typename Derived>
struct PyreOp {
  static void register_impl(torch::Library& m) {
    m.impl(Derived::aten_name, &Derived::impl);
  }

  static at::Tensor dispatch(
      c10::ArrayRef<at::Tensor> raw_inputs,
      c10::ArrayRef<at::Scalar> scalars) {
    // Promote 0-dim CPU scalar tensors to the target device (matches CUDA).
    c10::Device target_device = c10::kCPU;
    for (const auto& t : raw_inputs) {
      if (t.device().type() != c10::DeviceType::CPU) {
        target_device = t.device();
        break;
      }
    }
    c10::SmallVector<at::Tensor, 4> promoted_storage;
    auto effective_inputs = raw_inputs;
    if (target_device.type() != c10::DeviceType::CPU) {
      bool need_promote = false;
      for (const auto& t : raw_inputs) {
        if (t.dim() == 0 && t.device().type() == c10::DeviceType::CPU) {
          need_promote = true;
          break;
        }
      }
      if (need_promote) {
        // Find dtype from first device tensor for scalar casting
        auto target_dtype = effective_inputs[0].scalar_type();
        for (const auto& t : raw_inputs) {
          if (t.device() == target_device) {
            target_dtype = t.scalar_type();
            break;
          }
        }
        for (const auto& t : raw_inputs) {
          if (t.dim() == 0 && t.device().type() == c10::DeviceType::CPU)
            promoted_storage.push_back(
                t.to(target_device, target_dtype));
          else
            promoted_storage.push_back(t);
        }
        effective_inputs = promoted_storage;
      }
    }

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
// Concrete ops
// ---------------------------------------------------------------------------

struct AddOp : AlphaBinaryOp<AddOp> {
  static constexpr const char* aten_name = "add.Tensor";
  static constexpr const char* torch_op = "add";
};
struct SubOp : AlphaBinaryOp<SubOp> {
  static constexpr const char* aten_name = "sub.Tensor";
  static constexpr const char* torch_op = "sub";
};

struct MulOp : RegularBinaryOp<MulOp> {
  static constexpr const char* aten_name = "mul.Tensor";
  static constexpr const char* torch_op = "mul";
};
struct DivOp : RegularBinaryOp<DivOp> {
  static constexpr const char* aten_name = "div.Tensor";
  static constexpr const char* torch_op = "div";
};

// --- Binary ops (Epic 3) ---

struct PowTensorOp : RegularBinaryOp<PowTensorOp> {
  static constexpr const char* aten_name = "pow.Tensor_Tensor";
  static constexpr const char* torch_op = "pow";
};
struct MaximumOp : RegularBinaryOp<MaximumOp> {
  static constexpr const char* aten_name = "maximum";
  static constexpr const char* torch_op = "maximum";
};
struct MinimumOp : RegularBinaryOp<MinimumOp> {
  static constexpr const char* aten_name = "minimum";
  static constexpr const char* torch_op = "minimum";
};
struct RemainderOp : RegularBinaryOp<RemainderOp> {
  static constexpr const char* aten_name = "remainder.Tensor";
  static constexpr const char* torch_op = "remainder";
};
struct FmodOp : RegularBinaryOp<FmodOp> {
  static constexpr const char* aten_name = "fmod.Tensor";
  static constexpr const char* torch_op = "fmod";
};
struct BitwiseAndOp : RegularBinaryOp<BitwiseAndOp> {
  static constexpr const char* aten_name = "bitwise_and.Tensor";
  static constexpr const char* torch_op = "bitwise_and";
};
struct BitwiseOrOp : RegularBinaryOp<BitwiseOrOp> {
  static constexpr const char* aten_name = "bitwise_or.Tensor";
  static constexpr const char* torch_op = "bitwise_or";
};
struct BitwiseXorOp : RegularBinaryOp<BitwiseXorOp> {
  static constexpr const char* aten_name = "bitwise_xor.Tensor";
  static constexpr const char* torch_op = "bitwise_xor";
};
struct Atan2Op : RegularBinaryOp<Atan2Op> {
  static constexpr const char* aten_name = "atan2";
  static constexpr const char* torch_op = "atan2";
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

struct NegOp : RegularUnaryOp<NegOp> {
  static constexpr const char* aten_name = "neg";
  static constexpr const char* torch_op = "torch.aten.neg";
};
struct ReluOp : RegularUnaryOp<ReluOp> {
  static constexpr const char* aten_name = "relu";
  static constexpr const char* torch_op = "torch.aten.relu";
};
struct AbsOp : RegularUnaryOp<AbsOp> {
  static constexpr const char* aten_name = "abs";
  static constexpr const char* torch_op = "torch.aten.abs";
};

// --- Pure unary ops (Epic 3) ---

struct SiluOp : RegularUnaryOp<SiluOp> {
  static constexpr const char* aten_name = "silu";
  static constexpr const char* torch_op = "torch.aten.silu";
};
struct SigmoidOp : RegularUnaryOp<SigmoidOp> {
  static constexpr const char* aten_name = "sigmoid";
  static constexpr const char* torch_op = "torch.aten.sigmoid";
};
struct TanhOp : RegularUnaryOp<TanhOp> {
  static constexpr const char* aten_name = "tanh";
  static constexpr const char* torch_op = "torch.aten.tanh";
};
struct RsqrtOp : RegularUnaryOp<RsqrtOp> {
  static constexpr const char* aten_name = "rsqrt";
  static constexpr const char* torch_op = "torch.aten.rsqrt";
};
struct ExpOp : RegularUnaryOp<ExpOp> {
  static constexpr const char* aten_name = "exp";
  static constexpr const char* torch_op = "torch.aten.exp";
};
struct LogOp : RegularUnaryOp<LogOp> {
  static constexpr const char* aten_name = "log";
  static constexpr const char* torch_op = "torch.aten.log";
};
struct SqrtOp : RegularUnaryOp<SqrtOp> {
  static constexpr const char* aten_name = "sqrt";
  static constexpr const char* torch_op = "torch.aten.sqrt";
};
struct SinOp : RegularUnaryOp<SinOp> {
  static constexpr const char* aten_name = "sin";
  static constexpr const char* torch_op = "torch.aten.sin";
};
struct CosOp : RegularUnaryOp<CosOp> {
  static constexpr const char* aten_name = "cos";
  static constexpr const char* torch_op = "torch.aten.cos";
};
struct CeilOp : RegularUnaryOp<CeilOp> {
  static constexpr const char* aten_name = "ceil";
  static constexpr const char* torch_op = "torch.aten.ceil";
};
struct FloorOp : RegularUnaryOp<FloorOp> {
  static constexpr const char* aten_name = "floor";
  static constexpr const char* torch_op = "torch.aten.floor";
};
struct RoundOp : RegularUnaryOp<RoundOp> {
  static constexpr const char* aten_name = "round";
  static constexpr const char* torch_op = "torch.aten.round";
};
struct ReciprocalOp : RegularUnaryOp<ReciprocalOp> {
  static constexpr const char* aten_name = "reciprocal";
  static constexpr const char* torch_op = "torch.aten.reciprocal";
};
struct ErfOp : RegularUnaryOp<ErfOp> {
  static constexpr const char* aten_name = "erf";
  static constexpr const char* torch_op = "torch.aten.erf";
};
struct BitwiseNotOp : RegularUnaryOp<BitwiseNotOp> {
  static constexpr const char* aten_name = "bitwise_not";
  static constexpr const char* torch_op = "torch.aten.bitwise_not";
};
struct LogicalNotOp : RegularUnaryOp<LogicalNotOp> {
  static constexpr const char* aten_name = "logical_not";
  static constexpr const char* torch_op = "torch.aten.logical_not";
};
struct SignOp : RegularUnaryOp<SignOp> {
  static constexpr const char* aten_name = "sign";
  static constexpr const char* torch_op = "torch.aten.sign";
};

// --- Parameterized unary ops (Epic 3) ---

struct GeluOp : ParameterizedUnaryOp<GeluOp> {
  static constexpr const char* aten_name = "gelu";
  static constexpr const char* torch_op = "torch.aten.gelu";
  // ATen signature: gelu(Tensor self, *, str approximate="none") -> Tensor
  static at::Tensor impl(const at::Tensor& self, c10::string_view /*approximate*/ = "none") {
    return PyreOp<GeluOp>::dispatch({self}, {});
  }
  static std::string extraArgDecls(const OpContext&) {
    return "    %approx = torch.constant.str \"none\"";
  }
  static std::string extraArgs() { return ", %approx"; }
  static std::string extraArgTypes() { return ", !torch.str"; }
};

struct HardtanhOp : ParameterizedUnaryOp<HardtanhOp> {
  static constexpr const char* aten_name = "hardtanh";
  static constexpr const char* torch_op = "torch.aten.hardtanh";
  // ATen signature: hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1)
  static at::Tensor impl(const at::Tensor& self,
                          const at::Scalar& /*min_val*/ = -1,
                          const at::Scalar& /*max_val*/ = 1) {
    return PyreOp<HardtanhOp>::dispatch({self}, {});
  }
  static std::string extraArgDecls(const OpContext&) {
    return "    %min = torch.constant.float -1.000000e+00\n"
           "    %max = torch.constant.float 1.000000e+00";
  }
  static std::string extraArgs() { return ", %min, %max"; }
  static std::string extraArgTypes() { return ", !torch.float, !torch.float"; }
};

struct LeakyReluOp : ParameterizedUnaryOp<LeakyReluOp> {
  static constexpr const char* aten_name = "leaky_relu";
  static constexpr const char* torch_op = "torch.aten.leaky_relu";
  // ATen signature: leaky_relu(Tensor self, Scalar negative_slope=0.01)
  static at::Tensor impl(const at::Tensor& self,
                          const at::Scalar& /*negative_slope*/ = 0.01) {
    return PyreOp<LeakyReluOp>::dispatch({self}, {});
  }
  static std::string extraArgDecls(const OpContext&) {
    return "    %neg_slope = torch.constant.float 1.000000e-02";
  }
  static std::string extraArgs() { return ", %neg_slope"; }
  static std::string extraArgTypes() { return ", !torch.float"; }
};

struct EluOp : ParameterizedUnaryOp<EluOp> {
  static constexpr const char* aten_name = "elu";
  static constexpr const char* torch_op = "torch.aten.elu";
  // ATen signature: elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1)
  static at::Tensor impl(const at::Tensor& self,
                          const at::Scalar& /*alpha*/ = 1,
                          const at::Scalar& /*scale*/ = 1,
                          const at::Scalar& /*input_scale*/ = 1) {
    return PyreOp<EluOp>::dispatch({self}, {});
  }
  static std::string extraArgDecls(const OpContext&) {
    return "    %alpha = torch.constant.float 1.000000e+00\n"
           "    %scale = torch.constant.float 1.000000e+00\n"
           "    %input_scale = torch.constant.float 1.000000e+00";
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

// --- Scalar binary ops (Epic 3) ---

struct AddScalarOp : ScalarBinaryOp<AddScalarOp> {
  static constexpr const char* aten_name = "add.Scalar";
  static constexpr const char* torch_op = "torch.aten.add.Scalar";
  // ATen: add.Scalar(Tensor self, Scalar other, Scalar alpha=1)
  static at::Tensor impl(const at::Tensor& self,
                          const at::Scalar& other,
                          const at::Scalar& alpha = 1) {
    // Fold alpha*other into a single scalar for simplicity
    double val = other.toDouble() * alpha.toDouble();
    return PyreOp<AddScalarOp>::dispatch({self}, {at::Scalar(val)});
  }
  static std::string extraArgDecls(const OpContext& ctx) {
    return "    %alpha = torch.constant.int 1";
  }
  static std::string extraArgs() { return ", %alpha"; }
  static std::string extraArgTypes() { return ", !torch.int"; }
};

struct SubScalarOp : ScalarBinaryOp<SubScalarOp> {
  static constexpr const char* aten_name = "sub.Scalar";
  static constexpr const char* torch_op = "torch.aten.sub.Scalar";
  // ATen: sub.Scalar(Tensor self, Scalar other, Scalar alpha=1)
  static at::Tensor impl(const at::Tensor& self,
                          const at::Scalar& other,
                          const at::Scalar& alpha = 1) {
    double val = other.toDouble() * alpha.toDouble();
    return PyreOp<SubScalarOp>::dispatch({self}, {at::Scalar(val)});
  }
  static std::string extraArgDecls(const OpContext& ctx) {
    return "    %alpha = torch.constant.int 1";
  }
  static std::string extraArgs() { return ", %alpha"; }
  static std::string extraArgTypes() { return ", !torch.int"; }
};

struct MulScalarOp : ScalarBinaryOp<MulScalarOp> {
  static constexpr const char* aten_name = "mul.Scalar";
  static constexpr const char* torch_op = "torch.aten.mul.Scalar";
  static at::Tensor impl(const at::Tensor& self, const at::Scalar& other) {
    return PyreOp<MulScalarOp>::dispatch({self}, {other});
  }
};

struct DivScalarOp : ScalarBinaryOp<DivScalarOp> {
  static constexpr const char* aten_name = "div.Scalar";
  static constexpr const char* torch_op = "torch.aten.div.Scalar";
  static at::Tensor impl(const at::Tensor& self, const at::Scalar& other) {
    return PyreOp<DivScalarOp>::dispatch({self}, {other});
  }
};

struct PowScalarOp : ScalarBinaryOp<PowScalarOp> {
  static constexpr const char* aten_name = "pow.Tensor_Scalar";
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

// --- Comparison ops (tensor-tensor, Epic 3) ---

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

// --- Comparison ops (tensor-scalar, Epic 3) ---

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

// ReductionOp: sum.dim_IntList, mean.dim, amax, amin, prod.dim_int
// Derived must define: aten_name, torch_op, has_dtype_arg (bool)
template <typename Derived>
struct ReductionOp : PyreOp<Derived> {
  // sum/mean: (self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None)
  static at::Tensor impl_with_dtype(
      const at::Tensor& self,
      at::OptionalIntArrayRef dim,
      bool keepdim,
      std::optional<at::ScalarType> /*dtype*/) {
    c10::SmallVector<int64_t, 6> dims;
    if (dim.has_value()) {
      for (auto d : *dim) dims.push_back(d);
    } else {
      for (int64_t i = 0; i < self.dim(); ++i) dims.push_back(i);
    }
    return dispatchReduction(self, dims, keepdim);
  }

  // amax/amin: (self, int[1] dim=[], bool keepdim=False)
  static at::Tensor impl_no_dtype(
      const at::Tensor& self,
      at::IntArrayRef dim,
      bool keepdim) {
    c10::SmallVector<int64_t, 6> dims(dim.begin(), dim.end());
    if (dims.empty()) {
      for (int64_t i = 0; i < self.dim(); ++i) dims.push_back(i);
    }
    return dispatchReduction(self, dims, keepdim);
  }

  // prod.dim_int: (self, int dim, bool keepdim=False, *, ScalarType? dtype=None)
  static at::Tensor impl_single_dim(
      const at::Tensor& self,
      int64_t dim,
      bool keepdim,
      std::optional<at::ScalarType> /*dtype*/) {
    c10::SmallVector<int64_t, 6> dims = {dim};
    return dispatchReduction(self, dims, keepdim);
  }

  static at::Tensor dispatchReduction(
      const at::Tensor& self,
      c10::ArrayRef<int64_t> dims,
      bool keepdim) {
    TORCH_CHECK(hasPyreBuffer(self), "pyre: ", Derived::aten_name,
        " requires tensors with IREE buffers");

    auto dtype = self.scalar_type();
    auto decision = ArgShapeSpecializer().analyze(
        Derived::aten_name, dtype, {self},
        c10::pyre::PyreDevice::get(0)->capabilities());
    auto adapted = applyAdapters({self}, decision.arg_adapters);

    // Normalize negative dims
    c10::SmallVector<int64_t, 6> norm_dims;
    for (auto d : dims) {
      if (d < 0) d += self.dim();
      norm_dims.push_back(d);
    }

    auto out_shape = inferReducedShape(self.sizes(), norm_dims, keepdim);
    auto func_name = funcNameDefault(Derived::aten_name);

    // Extra args for sum/mean: dtype=None
    std::string extra_decls, extra_args, extra_types;
    if constexpr (Derived::has_dtype_arg) {
      extra_decls = "    %none = torch.constant.none";
      extra_args = ", %none";
      extra_types = ", !torch.none";
    }

    auto spec = buildReductionKernelSpec(
        func_name, Derived::torch_op, dtype,
        adapted[0].sizes(), out_shape, norm_dims, keepdim,
        extra_decls, extra_args, extra_types);

    auto cache_key = contentHashCacheKey(
        spec.template_sha1, spec.substitutions,
        c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags());

    auto& cache = PyreKernelCache::get();
    auto* kernel = cache.lookup(cache_key, func_name);
    if (!kernel) {
      auto mlir = generateReductionMlir(
          func_name, Derived::torch_op, dtype,
          adapted[0].sizes(), out_shape, norm_dims, keepdim,
          extra_decls, extra_args, extra_types);
      kernel = getOrCompile(cache_key, func_name, mlir);
    }

    return invokeKernel(kernel, adapted, out_shape, self.options());
  }
};

// --- Reduction ops (Epic 3) ---

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

struct ProdOp : PyreOp<ProdOp> {
  static constexpr const char* aten_name = "prod.dim_int";
  static constexpr const char* torch_op = "torch.aten.prod.dim_int";

  static at::Tensor impl(
      const at::Tensor& self, int64_t dim,
      bool keepdim, std::optional<at::ScalarType> /*dtype*/) {
    TORCH_CHECK(hasPyreBuffer(self), "pyre: prod requires IREE buffers");

    auto dtype = self.scalar_type();
    if (dim < 0) dim += self.dim();

    auto out_shape = inferReducedShape(self.sizes(), {dim}, keepdim);
    auto func_name = funcNameDefault(aten_name);

    std::string extra_decls = "    %none = torch.constant.none";
    std::string extra_args = ", %none";
    std::string extra_types = ", !torch.none";

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
      std::optional<at::MemoryFormat> memory_format) {
    // Only handle dtype-only conversions on pyre tensors. Fall through otherwise.
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
