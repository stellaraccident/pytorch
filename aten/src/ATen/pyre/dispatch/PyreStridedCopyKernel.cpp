#include <ATen/pyre/dispatch/PyreKernels.h>
#include <ATen/pyre/dispatch/PyreKernelAsmBuilder.h>
#include <ATen/pyre/dispatch/PyreKernelCache.h>
#include <ATen/pyre/dispatch/PyreKernelDispatch.h>
#include <ATen/pyre/PyreOps.h>
#include <ATen/pyre/PyreTensor.h>

#include <c10/pyre/impl/PyreStream.h>

namespace at::pyre {

namespace {

// Fragment 0: type aliases, module, function sig, dispatch, loop start.
static constexpr std::string_view kHeader = R"(!src    = tensor<$$SRC_N$$x$$TYPE$$>
!dst    = tensor<$$DST_N$$x$$TYPE$$>
!src_dt = !iree_tensor_ext.dispatch.tensor<readonly:!src>
!dst_dt = !iree_tensor_ext.dispatch.tensor<readwrite:!dst>

module @module {
  util.func public @$$FUNC$$(%src: !src, %dst: !dst {iree.abi.output = 0 : index})
      -> !dst attributes {iree.abi.model = "coarse-fences"} {
    %numel = arith.constant $$NUMEL$$ : index
    %result = flow.dispatch.workgroups[%numel](%src, %dst, %numel)
        : (!src, !dst, index) -> (%dst) =
    (%src_t: !src_dt, %dst_t: !dst_dt, %n: index) {
      %src_v = iree_tensor_ext.dispatch.tensor.load %src_t,
          offsets=[0], sizes=[$$SRC_N$$], strides=[1] : !src_dt -> !src
      %dst_v = iree_tensor_ext.dispatch.tensor.load %dst_t,
          offsets=[0], sizes=[$$DST_N$$], strides=[1] : !dst_dt -> !dst
      %id = flow.dispatch.workgroup.id[0] : index
      %count = flow.dispatch.workgroup.count[0] : index
      %out = scf.for %i = %id to %n step %count iter_args(%acc = %dst_v) -> (!dst) {
)";

// Fragment 1: per-dimension constants + index extraction + stride multiply.
static constexpr std::string_view kDim = R"(
        %sz$$D$$ = arith.constant $$SZ$$ : index
        %ss$$D$$ = arith.constant $$SS$$ : index
        %ds$$D$$ = arith.constant $$DS$$ : index
        %ix$$D$$ = arith.remui $$CARRY$$, %sz$$D$$ : index
        %sc$$D$$ = arith.muli %ix$$D$$, %ss$$D$$ : index
        %dc$$D$$ = arith.muli %ix$$D$$, %ds$$D$$ : index)";

// Fragment 2: carry propagation between dimensions.
static constexpr std::string_view kCarry =
    "\n        %cr$$D$$ = arith.divui $$C$$, %sz$$D$$ : index";

// Fragment 3: offset accumulation for d > 0.
static constexpr std::string_view kAccum = R"(
        %so$$D$$ = arith.addi $$S$$, %sc$$D$$ : index
        %do$$D$$ = arith.addi $$X$$, %dc$$D$$ : index)";

// Fragment 4: base storage offset addition.
static constexpr std::string_view kBaseOff = R"(
        %$$TAG$$_base = arith.constant $$V$$ : index
        %$$TAG$$_final = arith.addi $$O$$, %$$TAG$$_base : index)";

// Fragment 5: extract, insert, yield, store, return.
static constexpr std::string_view kFooter = R"(
        %val = tensor.extract %src_v[$$SRC_OFF$$] : !src
        %upd = tensor.insert %val into %acc[$$DST_OFF$$] : !dst
        scf.yield %upd : !dst
      }
      iree_tensor_ext.dispatch.tensor.store %out, %dst_t,
          offsets=[0], sizes=[$$DST_N$$], strides=[1] : !dst -> !dst_dt
      flow.return
    }
    util.return %result : !dst
  }
}
)";

// Fragment indices.
enum : size_t { F_HEADER, F_DIM, F_CARRY, F_ACCUM, F_BASEOFF, F_FOOTER };

PyreKernelAsmFragments& copyFragments() {
  static PyreKernelAsmFragments frags{
      kHeader, kDim, kCarry, kAccum, kBaseOff, kFooter};
  return frags;
}

} // namespace

void executeCompiledCopy(
    const CopyPlan& plan,
    const at::Tensor& src,
    const at::Tensor& dst) {
  TORCH_CHECK(plan.tier == CopyPlan::kCompiledKernel,
      "pyre: executeCompiledCopy requires Tier 2 plan");
  TORCH_CHECK(!plan.dims.empty(),
      "pyre: compiled copy plan has no dimensions");
  TORCH_CHECK(plan.dims.size() <= 6,
      "pyre: strided copy supports up to rank 6, got ", plan.dims.size());

  int64_t element_size = src.element_size();
  int rank = static_cast<int>(plan.dims.size());

  int64_t src_storage_numel = static_cast<int64_t>(
      src.storage().nbytes() / element_size);
  int64_t dst_storage_numel = static_cast<int64_t>(
      dst.storage().nbytes() / element_size);

  std::string func_name = "pyre_strided_copy_" +
      std::to_string(rank) + "d_" +
      std::to_string(element_size * 8) + "bit";

  auto recipe = [&](PyreKernelAsmBuilder& b) {
    int64_t numel = 1;
    for (const auto& d : plan.dims) numel *= d.size;

    auto src_n = std::to_string(src_storage_numel);
    auto dst_n = std::to_string(dst_storage_numel);
    auto type = std::string(elementSizeToNativeInt(element_size));

    b.appendFragment(F_HEADER, {
        {"FUNC", func_name}, {"SRC_N", src_n}, {"DST_N", dst_n},
        {"TYPE", type}, {"NUMEL", std::to_string(numel)}});

    std::string carry = "%i", src_off, dst_off;
    for (int d = 0; d < rank; ++d) {
      auto D = std::to_string(d);
      b.appendFragment(F_DIM, {
          {"D", D},
          {"SZ", std::to_string(plan.dims[d].size)},
          {"SS", std::to_string(plan.dims[d].src_stride)},
          {"DS", std::to_string(plan.dims[d].dst_stride)},
          {"CARRY", carry}});
      if (d < rank - 1) {
        b.appendFragment(F_CARRY, {{"D", D}, {"C", carry}});
        carry = "%cr" + D;
      }
      if (d == 0) {
        src_off = "%sc0"; dst_off = "%dc0";
      } else {
        b.appendFragment(F_ACCUM, {{"D", D}, {"S", src_off}, {"X", dst_off}});
        src_off = "%so" + D; dst_off = "%do" + D;
      }
    }

    if (plan.src_base_offset != 0) {
      b.appendFragment(F_BASEOFF, {
          {"TAG", "src"}, {"V", std::to_string(plan.src_base_offset)},
          {"O", src_off}});
      src_off = "%src_final";
    }
    if (plan.dst_base_offset != 0) {
      b.appendFragment(F_BASEOFF, {
          {"TAG", "dst"}, {"V", std::to_string(plan.dst_base_offset)},
          {"O", dst_off}});
      dst_off = "%dst_final";
    }

    b.appendFragment(F_FOOTER, {
        {"SRC_OFF", src_off}, {"DST_OFF", dst_off}, {"DST_N", dst_n}});
  };

  auto& abi = AbiConfig::kNativeOpaque;
  auto cache_key = copyFragments().digest(abi.compilerFlags(), recipe);

  auto& cache = PyreKernelCache::get();
  auto* kernel = cache.lookup(cache_key, func_name, abi);
  if (!kernel) {
    auto mlir = copyFragments().generateMlir(recipe);
    kernel = getOrCompile(cache_key, func_name, mlir, abi);
  }

  auto src_flat = src.as_strided({src_storage_numel}, {1}, 0);
  auto dst_flat = dst.as_strided({dst_storage_numel}, {1}, 0);

  c10::pyre::PyreStream stream(c10::pyre::getCurrentHostStream(0));
  auto& ctx = stream.context();
  PyreKernelDispatch::invoke(kernel, {src_flat, dst_flat}, dst_flat, ctx, abi);
}

} // namespace at::pyre
