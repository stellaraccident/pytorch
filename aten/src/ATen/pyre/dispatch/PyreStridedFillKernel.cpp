#include <ATen/pyre/dispatch/PyreStridedFillKernel.h>
#include <ATen/pyre/dispatch/PyreStridedCopyKernel.h>
#include <ATen/pyre/dispatch/PyreKernelAsmBuilder.h>
#include <ATen/pyre/dispatch/PyreKernelCache.h>
#include <ATen/pyre/dispatch/PyreKernelCompiler.h>
#include <ATen/pyre/dispatch/PyreKernelDispatch.h>
#include <ATen/pyre/PyreOp.h>
#include <ATen/pyre/PyreTensor.h>

#include <ATen/Dispatch.h>

#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreStream.h>

#include <cstring>
#include <sstream>

namespace at::pyre {

namespace {

const char* nativeIntType(int64_t element_size) {
  switch (element_size) {
    case 1: return "i8";
    case 2: return "i16";
    case 4: return "i32";
    case 8: return "i64";
    default:
      TORCH_CHECK(false, "pyre: unsupported element size ", element_size);
  }
}

int64_t scalarToBitPattern(const at::Scalar& value, at::ScalarType dtype) {
  int64_t pattern = 0;
  AT_DISPATCH_ALL_TYPES_AND2(
      at::kHalf, at::kBFloat16, dtype, "pyre_fill_pattern", [&] {
        scalar_t val = value.to<scalar_t>();
        std::memcpy(&pattern, &val, sizeof(scalar_t));
      });
  return pattern;
}

// Build compiler flags for native IREE input (not torch dialect).
std::vector<std::string> nativeCompilerFlags() {
  auto base = c10::pyre::PyreDevice::get(0)->capabilities().compilerFlags();
  std::vector<std::string> flags;
  for (const auto& f : base) {
    if (f.find("--iree-input-type=") != std::string::npos) continue;
    if (f.find("--iree-torch-") != std::string::npos) continue;
    flags.push_back(f);
  }
  flags.push_back("--iree-input-type=none");
  return flags;
}

} // namespace

std::string generateStridedFillMlir(
    const std::string& func_name,
    c10::ArrayRef<CoalescedDim> dims,
    int64_t dst_numel,
    int64_t element_size,
    int64_t fill_pattern) {
  TORCH_CHECK(!dims.empty(), "pyre: strided fill requires at least 1 dim");
  TORCH_CHECK(dims.size() <= 6,
      "pyre: strided fill supports up to rank 6, got ", dims.size());

  const char* itype = nativeIntType(element_size);
  // For contiguous fill: just linalg.fill the whole buffer.
  // The caller ensures this is only called for contiguous 8-byte.
  std::ostringstream os;
  os << "module @module {\n";
  os << "  util.func public @" << func_name << "(\n";
  os << "      %dst: tensor<" << dst_numel << "x" << itype
     << "> {iree.abi.output = 0 : index}\n";
  os << "  ) -> tensor<" << dst_numel << "x" << itype << ">"
     << " attributes {iree.abi.model = \"coarse-fences\"} {\n";
  os << "    %fill = arith.constant " << fill_pattern << " : " << itype << "\n";
  os << "    %result = linalg.fill ins(%fill : " << itype
     << ") outs(%dst : tensor<" << dst_numel << "x" << itype << ">)"
     << " -> tensor<" << dst_numel << "x" << itype << ">\n";
  os << "    util.return %result : tensor<" << dst_numel << "x" << itype << ">\n";
  os << "  }\n";
  os << "}\n";

  return os.str();
}

void executeCompiledFill(
    const at::Tensor& dst,
    const at::Scalar& value) {
  int64_t element_size = dst.element_size();
  int64_t fill_pattern = scalarToBitPattern(value, dst.scalar_type());
  int64_t numel = dst.numel();

  if (dst.is_contiguous()) {
    // Contiguous but element_size > 4 (the HAL pattern fill limit).
    std::string func_name = "pyre_fill_" +
        std::to_string(element_size * 8) + "bit";

    auto mlir = generateStridedFillMlir(
        func_name, {{numel, 1, 1}}, numel, element_size, fill_pattern);

    PYRE_LOG(DEBUG) << "fill MLIR:\n" << mlir << "\n";

    auto flags = nativeCompilerFlags();
    auto cache_key = contentHashCacheKey(
        "strided_fill", {{"mlir", mlir}}, flags);

    auto& cache = PyreKernelCache::get();
    auto* kernel = cache.lookup(cache_key, func_name, /*native_abi=*/true);
    if (!kernel) {
      PYRE_LOG(INFO) << "cache MISS: " << cache_key << ", compiling\n";
      auto vmfb = PyreKernelCompiler::compileSync(mlir, flags);
      PYRE_LOG(INFO) << "compiled " << vmfb->size() << " bytes\n";
      kernel = cache.store(cache_key, func_name, std::move(vmfb),
                           /*native_abi=*/true);
    }

    auto dst_flat = dst.as_strided({numel}, {1}, dst.storage_offset());

    c10::pyre::PyreStream stream(c10::pyre::getCurrentHostStream(0));
    auto& ctx = stream.context();
    PyreKernelDispatch::invokeNative(kernel, {dst_flat}, dst_flat, ctx);
  } else {
    // Non-contiguous fill: fill a contiguous temp, then strided copy.
    auto filled = at::empty(dst.sizes(), dst.options());
    executeCompiledFill(filled, value);  // recursive, contiguous path

    auto plan = planCopy(
        dst.sizes(),
        filled.strides(),
        dst.strides(),
        0,
        dst.storage_offset(),
        element_size);

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
