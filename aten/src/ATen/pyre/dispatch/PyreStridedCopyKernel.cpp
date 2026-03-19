#include <ATen/pyre/dispatch/PyreStridedCopyKernel.h>
#include <ATen/pyre/dispatch/PyreKernelAsmBuilder.h>
#include <ATen/pyre/dispatch/PyreKernelCache.h>
#include <ATen/pyre/dispatch/PyreKernelCompiler.h>
#include <ATen/pyre/dispatch/PyreKernelDispatch.h>
#include <ATen/pyre/PyreOp.h>
#include <ATen/pyre/PyreTensor.h>

#include <c10/pyre/impl/PyreDevice.h>
#include <c10/pyre/impl/PyreStream.h>

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

std::string generateStridedCopyMlir(
    const std::string& func_name,
    c10::ArrayRef<CoalescedDim> dims,
    int64_t src_numel,
    int64_t dst_numel,
    int64_t element_size) {
  TORCH_CHECK(!dims.empty(), "pyre: strided copy requires at least 1 dim");
  TORCH_CHECK(dims.size() <= 6,
      "pyre: strided copy supports up to rank 6, got ", dims.size());

  const char* itype = nativeIntType(element_size);
  int rank = static_cast<int>(dims.size());

  // Compute logical numel (product of coalesced sizes).
  int64_t numel = 1;
  for (const auto& d : dims) numel *= d.size;

  std::ostringstream os;
  os << "module @module {\n";
  os << "  util.func public @" << func_name << "(\n";
  os << "      %src: tensor<" << src_numel << "x" << itype << ">,\n";
  os << "      %dst: tensor<" << dst_numel << "x" << itype
     << "> {iree.abi.output = 0 : index}\n";
  os << "  ) -> tensor<" << dst_numel << "x" << itype << ">"
     << " attributes {iree.abi.model = \"coarse-fences\"} {\n";

  // Emit the copy as a linalg.generic over numel elements.
  // The iteration domain is 1D [0, numel). For each linear index,
  // we recover N-D coordinates via divmod, compute src and dst flat
  // offsets from strides, extract from src, insert into dst.

  // We use scf.for since linalg.generic with tensor.extract/insert
  // doesn't map cleanly to the gather/scatter pattern we need.

  // Constants.
  os << "    %c0 = arith.constant 0 : index\n";
  os << "    %c1 = arith.constant 1 : index\n";
  os << "    %numel = arith.constant " << numel << " : index\n";

  for (int d = 0; d < rank; ++d) {
    os << "    %size" << d << " = arith.constant "
       << dims[d].size << " : index\n";
    os << "    %src_stride" << d << " = arith.constant "
       << dims[d].src_stride << " : index\n";
    os << "    %dst_stride" << d << " = arith.constant "
       << dims[d].dst_stride << " : index\n";
  }

  // scf.for loop: iterate i = 0..numel-1, threading dst tensor.
  os << "    %result = scf.for %i = %c0 to %numel step %c1"
     << " iter_args(%out = %dst) -> (tensor<" << dst_numel << "x" << itype << ">) {\n";

  // Divmod chain to recover N-D indices.
  os << "      %carry_0 = arith.remui %i, %size0 : index\n";
  std::string src_off = "";
  std::string dst_off = "";

  for (int d = 0; d < rank; ++d) {
    std::string idx;
    if (d == 0) {
      idx = "%carry_0";
    } else {
      std::string prev_carry = "%div" + std::to_string(d - 1);
      idx = "%idx" + std::to_string(d);
      os << "      " << idx << " = arith.remui " << prev_carry
         << ", %size" << d << " : index\n";
    }

    if (d < rank - 1) {
      std::string divisor = (d == 0) ? "%i" : "%div" + std::to_string(d - 1);
      os << "      %div" << d << " = arith.divui " << divisor
         << ", %size" << d << " : index\n";
    }

    // src contribution
    std::string s_contrib = "%s_contrib" + std::to_string(d);
    os << "      " << s_contrib << " = arith.muli " << idx
       << ", %src_stride" << d << " : index\n";

    // dst contribution
    std::string d_contrib = "%d_contrib" + std::to_string(d);
    os << "      " << d_contrib << " = arith.muli " << idx
       << ", %dst_stride" << d << " : index\n";

    if (d == 0) {
      src_off = s_contrib;
      dst_off = d_contrib;
    } else {
      std::string new_src = "%src_off" + std::to_string(d);
      std::string new_dst = "%dst_off" + std::to_string(d);
      os << "      " << new_src << " = arith.addi "
         << src_off << ", " << s_contrib << " : index\n";
      os << "      " << new_dst << " = arith.addi "
         << dst_off << ", " << d_contrib << " : index\n";
      src_off = new_src;
      dst_off = new_dst;
    }
  }

  // Extract from src, insert into dst.
  os << "      %val = tensor.extract %src[" << src_off << "]"
     << " : tensor<" << src_numel << "x" << itype << ">\n";
  os << "      %updated = tensor.insert %val into %out[" << dst_off << "]"
     << " : tensor<" << dst_numel << "x" << itype << ">\n";
  os << "      scf.yield %updated : tensor<" << dst_numel << "x" << itype << ">\n";
  os << "    }\n";

  os << "    util.return %result : tensor<" << dst_numel << "x" << itype << ">\n";
  os << "  }\n";
  os << "}\n";

  return os.str();
}

void executeCompiledCopy(
    const CopyPlan& plan,
    const at::Tensor& src,
    const at::Tensor& dst) {
  TORCH_CHECK(plan.tier == CopyPlan::kCompiledKernel,
      "pyre: executeCompiledCopy requires Tier 2 plan");
  TORCH_CHECK(!plan.dims.empty(),
      "pyre: compiled copy plan has no dimensions");

  int64_t element_size = src.element_size();
  int rank = static_cast<int>(plan.dims.size());

  int64_t src_storage_numel = static_cast<int64_t>(
      src.storage().nbytes() / element_size);
  int64_t dst_storage_numel = static_cast<int64_t>(
      dst.storage().nbytes() / element_size);

  std::string func_name = "pyre_strided_copy_" +
      std::to_string(rank) + "d_" +
      std::to_string(element_size * 8) + "bit";

  auto mlir = generateStridedCopyMlir(
      func_name, plan.dims, src_storage_numel, dst_storage_numel,
      element_size);

  PYRE_LOG(DEBUG) << "strided copy MLIR:\n" << mlir << "\n";

  // Native IREE flags (not torch dialect).
  auto flags = nativeCompilerFlags();
  auto cache_key = contentHashCacheKey(
      "strided_copy",
      {{"mlir", mlir}},
      flags);

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name, /*native_abi=*/true);
  if (!kernel) {
    PYRE_LOG(INFO) << "cache MISS: " << cache_key << ", compiling\n";
    auto vmfb = PyreKernelCompiler::compileSync(mlir, flags);
    PYRE_LOG(INFO) << "compiled " << vmfb->size() << " bytes\n";
    kernel = cache.store(cache_key, func_name, std::move(vmfb),
                         /*native_abi=*/true);
  }

  // 1D flat views over raw storage.
  auto src_flat = src.as_strided({src_storage_numel}, {1}, 0);
  auto dst_flat = dst.as_strided({dst_storage_numel}, {1}, 0);

  c10::pyre::PyreStream stream(c10::pyre::getCurrentHostStream(0));
  auto& ctx = stream.context();
  PyreKernelDispatch::invokeNative(kernel, {src_flat, dst_flat}, dst_flat, ctx);
}

} // namespace at::pyre
