// Data movement kernels: compiled fill and strided copy.
//
// These use iree.abi.model = "coarse-fences" with flow.dispatch.workgroups
// (native IREE dialect, not torch). They own their entire dispatch pipeline:
// MLIR generation, compilation, arg packing, VM invocation, and timeline
// bookkeeping. No shared dispatch infrastructure (no AbiPacker, no
// invokeEnvelope, no PyreKernelDispatch).

#include <ATen/pyre/dispatch/PyreKernels.h>
#include <ATen/pyre/dispatch/PyreKernelAsmBuilder.h>
#include <ATen/pyre/dispatch/PyreKernelCache.h>
#include <ATen/pyre/dispatch/PyreTypeMapping.h>
#include <ATen/pyre/PyreOps.h>
#include <ATen/pyre/PyreTensor.h>

#include <ATen/Dispatch.h>

#include <c10/pyre/impl/PyreRuntime.h>
#include <c10/pyre/impl/PyreStorage.h>
#include <c10/pyre/impl/PyreStream.h>

#include <iree/hal/fence.h>
#include <iree/modules/hal/debugging.h>
#include <iree/modules/hal/module.h>
#include <iree/modules/hal/types.h>
#include <iree/vm/api.h>
#include <iree/vm/bytecode/module.h>

#include <cstring>

namespace at::pyre {

// ---------------------------------------------------------------------------
// Shared helpers for data movement dispatch
// ---------------------------------------------------------------------------

namespace {

static c10::ArrayRef<std::string> nativeCompilerFlags() {
  return AbiConfig::kNativeOpaque.compilerFlags();
}

static std::string resolveNativeFunction(const std::string& func_name) {
  return AbiConfig::kNativeOpaque.resolveFunction(func_name);
}

// Build an opaque buffer view (element type by bitwidth, for native dispatch).
static c10::pyre::hal_buffer_view_ptr makeOpaqueView(const at::Tensor& t) {
  auto* ctx = static_cast<c10::pyre::PyreBufferContext*>(
      t.storage().data_ptr().get_context());
  TORCH_CHECK(ctx && ctx->buffer, "pyre: tensor has no IREE buffer");

  c10::SmallVector<iree_hal_dim_t, 6> shape;
  for (int64_t i = 0; i < t.dim(); ++i)
    shape.push_back(static_cast<iree_hal_dim_t>(t.size(i)));

  iree_hal_element_type_t elt;
  switch (t.element_size()) {
    case 1: elt = IREE_HAL_ELEMENT_TYPE_SINT_8; break;
    case 2: elt = IREE_HAL_ELEMENT_TYPE_SINT_16; break;
    case 4: elt = IREE_HAL_ELEMENT_TYPE_SINT_32; break;
    case 8: elt = IREE_HAL_ELEMENT_TYPE_SINT_64; break;
    default:
      TORCH_CHECK(false, "pyre: unsupported element size: ", t.element_size());
  }

  iree_hal_buffer_view_t* view = nullptr;
  PYRE_CHECK_OK(iree_hal_buffer_view_create(
      ctx->buffer.get(), shape.size(), shape.data(),
      elt, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      c10::pyre::PyreRuntime::get().hostAllocator(), &view));
  return c10::pyre::hal_buffer_view_ptr::steal(view);
}

// Self-contained dispatch for native coarse-fences kernels.
// Args: [inputs...(buffer views), wait_fence, signal_fence] → [outputs...]
static void invokeNative(
    CachedKernel* kernel,
    c10::ArrayRef<at::Tensor> inputs,
    at::Tensor& output) {
  auto alloc = c10::pyre::PyreRuntime::get().hostAllocator();
  c10::pyre::PyreStream stream(c10::pyre::getCurrentHostStream(0));
  // Flush pending native ops before VM invoke.
  stream.flush();
  auto& stream_ctx = stream.context();

  c10::pyre::vm_list_ptr args;
  PYRE_CHECK_OK(iree_vm_list_create(
      iree_vm_make_undefined_type_def(),
      static_cast<iree_host_size_t>(inputs.size()) + 2, alloc,
      args.for_output()));

  for (const auto& t : inputs) {
    auto view = makeOpaqueView(t);
    iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(view.release());
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  }

  // Wait fence — always include stream's current timepoint to enforce
  // serial ordering on the timeline.
  {
    c10::pyre::hal_fence_ptr wait;
    PYRE_CHECK_OK(iree_hal_fence_create(
        static_cast<iree_host_size_t>(inputs.size() + 3), alloc,
        wait.for_output()));
    if (stream_ctx.timepoint > 0)
      PYRE_CHECK_OK(iree_hal_fence_insert(
          wait, stream_ctx.timeline.get(), stream_ctx.timepoint));
    for (const auto& t : inputs) {
      auto* ctx = static_cast<c10::pyre::PyreBufferContext*>(
          t.storage().data_ptr().get_context());
      if (ctx && ctx->mutation_sem && ctx->mutation_timepoint > 0)
        PYRE_CHECK_OK(iree_hal_fence_insert(
            wait, ctx->mutation_sem, ctx->mutation_timepoint));
    }
    {
      auto* ctx = static_cast<c10::pyre::PyreBufferContext*>(
          output.storage().data_ptr().get_context());
      if (ctx && ctx->mutation_sem && ctx->mutation_timepoint > 0)
        PYRE_CHECK_OK(iree_hal_fence_insert(
            wait, ctx->mutation_sem, ctx->mutation_timepoint));
    }
    iree_vm_ref_t ref = iree_hal_fence_move_ref(wait.release());
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  }

  // Signal fence — FenceGuard owns this ref and signals on exception.
  // A separate ref is retained into the args list for the dispatch engine.
  uint64_t signal_value = ++stream_ctx.timepoint;
  iree_hal_fence_t* signal = nullptr;
  PYRE_CHECK_OK(iree_hal_fence_create_at(
      stream_ctx.timeline.get(), signal_value, alloc, &signal));
  at::pyre::FenceGuard fence_guard(signal);
  {
    iree_hal_fence_retain(signal);  // args list gets its own ref
    iree_vm_ref_t ref = iree_hal_fence_move_ref(signal);
    PYRE_CHECK_OK(iree_vm_list_push_ref_move(args, &ref));
  }

  c10::pyre::vm_list_ptr rets;
  PYRE_CHECK_OK(iree_vm_list_create(
      iree_vm_make_undefined_type_def(), 1, alloc,
      rets.for_output()));

  PYRE_CHECK_OK(iree_vm_invoke(
      kernel->context.get(), kernel->function,
      IREE_VM_INVOCATION_FLAG_NONE, nullptr,
      args, rets, alloc));

  // Success — dispatch will signal the fence asynchronously.
  fence_guard.disarm();

  // Timeline bookkeeping.
  for (const auto& t : inputs) {
    auto* ctx = static_cast<c10::pyre::PyreBufferContext*>(
        t.storage().data_ptr().get_context());
    if (ctx) ctx->recordUse(stream_ctx.timeline.get(), signal_value);
  }
  auto* out_ctx = static_cast<c10::pyre::PyreBufferContext*>(
      output.storage().data_ptr().get_context());
  if (out_ctx) out_ctx->recordMutation(stream_ctx.timeline.get(), signal_value);
}

// Lookup or compile a native kernel via lookupOrClaim/fulfill/fail.
static CachedKernel* getOrCompileNative(
    const std::string& cache_key,
    const std::string& func_name,
    PyreKernelAsmFragments& frags,
    const std::function<void(PyreKernelAsmBuilder&)>& recipe) {
  auto& cache = PyreKernelCache::get();
  auto result = cache.lookupOrClaim(
      cache_key, func_name, AbiConfig::kNativeOpaque);

  if (result.is_compiler) {
    try {
      auto mlir = frags.generateMlir(recipe);
      PYRE_LOG(DEBUG) << "MLIR:\n" << mlir << "\n";
      auto vmfb = PyreKernelCompiler::compileSync(
          std::move(mlir), nativeCompilerFlags());
      PYRE_LOG(INFO) << "compiled " << vmfb->size() << " bytes\n";
      cache.fulfill(cache_key, std::move(vmfb), func_name,
                    AbiConfig::kNativeOpaque);
    } catch (...) {
      cache.fail(cache_key, std::current_exception());
      throw;
    }
  }

  return result.future.get();
}

// ---------------------------------------------------------------------------
// Fill kernel
// ---------------------------------------------------------------------------

int64_t scalarToBitPattern(const at::Scalar& value, at::ScalarType dtype) {
  int64_t pattern = 0;
  AT_DISPATCH_ALL_TYPES_AND2(
      at::kHalf, at::kBFloat16, dtype, "pyre_fill_pattern", [&] {
        scalar_t val = value.to<scalar_t>();
        std::memcpy(&pattern, &val, sizeof(scalar_t));
      });
  return pattern;
}

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

// ---------------------------------------------------------------------------
// Copy kernel
// ---------------------------------------------------------------------------

static constexpr std::string_view kCopyHeader = R"(!src    = tensor<$$SRC_N$$x$$TYPE$$>
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

static constexpr std::string_view kCopyDim = R"(
        %sz$$D$$ = arith.constant $$SZ$$ : index
        %ss$$D$$ = arith.constant $$SS$$ : index
        %ds$$D$$ = arith.constant $$DS$$ : index
        %ix$$D$$ = arith.remui $$CARRY$$, %sz$$D$$ : index
        %sc$$D$$ = arith.muli %ix$$D$$, %ss$$D$$ : index
        %dc$$D$$ = arith.muli %ix$$D$$, %ds$$D$$ : index)";

static constexpr std::string_view kCopyCarry =
    "\n        %cr$$D$$ = arith.divui $$C$$, %sz$$D$$ : index";

static constexpr std::string_view kCopyAccum = R"(
        %so$$D$$ = arith.addi $$S$$, %sc$$D$$ : index
        %do$$D$$ = arith.addi $$X$$, %dc$$D$$ : index)";

static constexpr std::string_view kCopyBaseOff = R"(
        %$$TAG$$_base = arith.constant $$V$$ : index
        %$$TAG$$_final = arith.addi $$O$$, %$$TAG$$_base : index)";

static constexpr std::string_view kCopyFooter = R"(
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

enum CopyFrag : size_t {
  CF_HEADER, CF_DIM, CF_CARRY, CF_ACCUM, CF_BASEOFF, CF_FOOTER
};

PyreKernelAsmFragments& copyFragments() {
  static PyreKernelAsmFragments frags{
      kCopyHeader, kCopyDim, kCopyCarry, kCopyAccum, kCopyBaseOff, kCopyFooter};
  return frags;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

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

    auto cache_key = fillFragments().digest(nativeCompilerFlags(), recipe);
    auto* kernel = getOrCompileNative(
        cache_key, func_name, fillFragments(), recipe);

    auto dst_flat = dst.as_strided({numel}, {1}, dst.storage_offset());
    invokeNative(kernel, {dst_flat}, dst_flat);
  } else {
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

void executeCompiledCopy(
    const CopyPlan& plan,
    const at::Tensor& src,
    const at::Tensor& dst) {
  TORCH_CHECK(plan.tier == CopyPlan::kCompiledKernel,
      "pyre: executeCompiledCopy requires Tier 2 plan");
  TORCH_CHECK(!plan.dims.empty() && plan.dims.size() <= 6,
      "pyre: compiled copy supports 1-6 dims, got ", plan.dims.size());

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

    b.appendFragment(CF_HEADER, {
        {"FUNC", func_name}, {"SRC_N", src_n}, {"DST_N", dst_n},
        {"TYPE", type}, {"NUMEL", std::to_string(numel)}});

    std::string carry = "%i", src_off, dst_off;
    for (int d = 0; d < rank; ++d) {
      auto D = std::to_string(d);
      b.appendFragment(CF_DIM, {
          {"D", D},
          {"SZ", std::to_string(plan.dims[d].size)},
          {"SS", std::to_string(plan.dims[d].src_stride)},
          {"DS", std::to_string(plan.dims[d].dst_stride)},
          {"CARRY", carry}});
      if (d < rank - 1) {
        b.appendFragment(CF_CARRY, {{"D", D}, {"C", carry}});
        carry = "%cr" + D;
      }
      if (d == 0) {
        src_off = "%sc0"; dst_off = "%dc0";
      } else {
        b.appendFragment(CF_ACCUM, {
            {"D", D}, {"S", src_off}, {"X", dst_off}});
        src_off = "%so" + D; dst_off = "%do" + D;
      }
    }

    if (plan.src_base_offset != 0) {
      b.appendFragment(CF_BASEOFF, {
          {"TAG", "src"}, {"V", std::to_string(plan.src_base_offset)},
          {"O", src_off}});
      src_off = "%src_final";
    }
    if (plan.dst_base_offset != 0) {
      b.appendFragment(CF_BASEOFF, {
          {"TAG", "dst"}, {"V", std::to_string(plan.dst_base_offset)},
          {"O", dst_off}});
      dst_off = "%dst_final";
    }

    b.appendFragment(CF_FOOTER, {
        {"SRC_OFF", src_off}, {"DST_OFF", dst_off}, {"DST_N", dst_n}});
  };

  auto cache_key = copyFragments().digest(nativeCompilerFlags(), recipe);
  auto* kernel = getOrCompileNative(
      cache_key, func_name, copyFragments(), recipe);

  auto src_flat = src.as_strided({src_storage_numel}, {1}, 0);
  auto dst_flat = dst.as_strided({dst_storage_numel}, {1}, 0);
  invokeNative(kernel, {src_flat, dst_flat}, dst_flat);
}

} // namespace at::pyre
