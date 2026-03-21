#include <ATen/pyre/dispatch/PyreKernels.h>
#include <ATen/pyre/dispatch/PyreKernelAsmBuilder.h>
#include <ATen/pyre/dispatch/PyreKernelCache.h>
#include <ATen/pyre/dispatch/PyreKernelDispatch.h>
#include <ATen/pyre/PyreOps.h>
#include <ATen/pyre/PyreTensor.h>

#include <ATen/Dispatch.h>

#include <c10/pyre/impl/PyreStream.h>

#include <cstring>

namespace at::pyre {

namespace {

int64_t scalarToBitPattern(const at::Scalar& value, at::ScalarType dtype) {
  int64_t pattern = 0;
  AT_DISPATCH_ALL_TYPES_AND2(
      at::kHalf, at::kBFloat16, dtype, "pyre_fill_pattern", [&] {
        scalar_t val = value.to<scalar_t>();
        std::memcpy(&pattern, &val, sizeof(scalar_t));
      });
  return pattern;
}

// Single fragment for fill — linalg.fill is simple enough for one piece.
static constexpr std::string_view kFill = R"(!buf = tensor<$$N$$x$$TYPE$$>

module @module {
  util.func public @$$FUNC$$(%dst: !buf {iree.abi.output = 0 : index})
      -> !buf attributes {iree.abi.model = "coarse-fences"} {
    %fill = arith.constant $$PATTERN$$ : $$TYPE$$
    %result = linalg.fill ins(%fill : $$TYPE$$) outs(%dst : !buf) -> !buf
    util.return %result : !buf
  }
}
)";

PyreKernelAsmFragments& fillFragments() {
  static PyreKernelAsmFragments frags{kFill};
  return frags;
}

} // namespace

void executeCompiledFill(
    const at::Tensor& dst,
    const at::Scalar& value) {
  int64_t element_size = dst.element_size();
  int64_t fill_pattern = scalarToBitPattern(value, dst.scalar_type());
  int64_t numel = dst.numel();

  if (dst.is_contiguous()) {
    std::string func_name = "pyre_fill_" +
        std::to_string(element_size * 8) + "bit";

    auto recipe = [&](PyreKernelAsmBuilder& b) {
      b.appendFragment(0, {
          {"FUNC", func_name},
          {"N", std::to_string(numel)},
          {"TYPE", std::string(elementSizeToNativeInt(element_size))},
          {"PATTERN", std::to_string(fill_pattern)},
      });
    };

    auto& abi = AbiConfig::kNativeOpaque;
    auto cache_key = fillFragments().digest(abi.compilerFlags(), recipe);

    auto& cache = PyreKernelCache::get();
    auto* kernel = cache.lookup(cache_key, func_name, abi);
    if (!kernel) {
      auto mlir = fillFragments().generateMlir(recipe);
      kernel = getOrCompile(cache_key, func_name, mlir, abi);
    }

    auto dst_flat = dst.as_strided({numel}, {1}, dst.storage_offset());

    c10::pyre::PyreStream stream(c10::pyre::getCurrentHostStream(0));
    auto& ctx = stream.context();
    PyreKernelDispatch::invoke(kernel, {dst_flat}, dst_flat, ctx, abi);
  } else {
    // Non-contiguous fill: fill a contiguous temp, then strided copy.
    auto filled = at::empty(dst.sizes(), dst.options());
    executeCompiledFill(filled, value);

    auto plan = planCopy(
        dst.sizes(), filled.strides(), dst.strides(),
        0, dst.storage_offset(), element_size);

    if (plan.tier == CopyPlan::kCompiledKernel) {
      executeCompiledCopy(plan, filled, dst);
    } else {
      PyreTensor src_pt(filled);
      PyreTensor dst_pt(dst);
      auto* src_ctx = static_cast<c10::pyre::PyreBufferContext*>(
          filled.storage().data_ptr().get_context());
      auto* dst_ctx = static_cast<c10::pyre::PyreBufferContext*>(
          dst.storage().data_ptr().get_context());
      executeCopyPlan(plan, src_pt.buffer(), dst_pt.buffer(),
                      dst_pt.device(), src_ctx, dst_ctx);
    }
  }
}

} // namespace at::pyre
