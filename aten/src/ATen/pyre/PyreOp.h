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
    const OpContext& ctx);
std::string buildUnaryMlir(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx);

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
    for (const auto& t : raw_inputs)
      TORCH_CHECK(hasPyreBuffer(t), "pyre: ", Derived::aten_name,
          " requires tensors with IREE buffers");

    auto dtype = raw_inputs[0].scalar_type();

    auto decision = ArgShapeSpecializer().analyze(
        Derived::aten_name, dtype, raw_inputs,
        c10::pyre::PyreDevice::get(0)->capabilities());

    auto adapted = applyAdapters(
        {raw_inputs.begin(), raw_inputs.end()},
        decision.arg_adapters);

    OpContext ctx{adapted, raw_inputs, scalars, dtype, decision};

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
                        raw_inputs[0].options());
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
