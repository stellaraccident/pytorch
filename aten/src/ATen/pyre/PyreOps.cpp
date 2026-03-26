#include <ATen/pyre/PyreOps.h>
#include <ATen/pyre/dispatch/PyreAlgorithmicKernels.h>
#include <c10/pyre/impl/PyreRuntime.h>
#include <c10/pyre/impl/PyreStream.h>
#include <c10/pyre/impl/PyreStorage.h>
#include <iree/hal/fence.h>
#include <iree/vm/api.h>

#ifdef AT_PER_OPERATOR_HEADERS
#include <ATen/ops/_to_copy_native.h>
#else
#include <ATen/NativeFunctions.h>
#endif

#include <cmath>
#include <set>
#include <unordered_map>

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

bool promoteScalarTensors(
    c10::ArrayRef<at::Tensor> raw_inputs,
    c10::SmallVector<at::Tensor, 4>& out,
    c10::ArrayRef<at::Tensor>& effective) {
  c10::Device target_device = c10::kCPU;
  for (const auto& t : raw_inputs) {
    if (t.device().type() != c10::DeviceType::CPU) {
      target_device = t.device();
      break;
    }
  }
  if (target_device.type() == c10::DeviceType::CPU) {
    effective = raw_inputs;
    return false;
  }
  bool need_promote = false;
  for (const auto& t : raw_inputs) {
    if (t.dim() == 0 && t.device().type() == c10::DeviceType::CPU) {
      need_promote = true;
      break;
    }
  }
  if (!need_promote) {
    effective = raw_inputs;
    return false;
  }
  auto target_dtype = raw_inputs[0].scalar_type();
  for (const auto& t : raw_inputs) {
    if (t.device() == target_device) {
      target_dtype = t.scalar_type();
      break;
    }
  }
  for (const auto& t : raw_inputs) {
    if (t.dim() == 0 && t.device().type() == c10::DeviceType::CPU)
      out.push_back(t.to(target_device, target_dtype));
    else
      out.push_back(t);
  }
  effective = out;
  return true;
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
        out_shape, c10::ArrayRef<ArgAdapter>{});
  }
  return buildBinaryKernelSpec(
      func_name, linalg_op, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
      out_shape, c10::ArrayRef<ArgAdapter>{});
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
        out_shape, c10::ArrayRef<ArgAdapter>{});
  }
  return generateBinaryMlir(
      func_name, linalg_op, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
      out_shape, c10::ArrayRef<ArgAdapter>{});
}

KernelSpec buildUnarySpec(
    const char* torch_op, const std::string& func_name,
    const OpContext& ctx,
    const std::string& extra_arg_decls,
    const std::string& extra_args,
    const std::string& extra_arg_types) {
  ArgAdapter adapter = c10::ArrayRef<ArgAdapter>{}.empty()
      ? ArgAdapter{ArgAdapter::kIdentity, {}}
      : c10::ArrayRef<ArgAdapter>{}[0];
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
  ArgAdapter adapter = c10::ArrayRef<ArgAdapter>{}.empty()
      ? ArgAdapter{ArgAdapter::kIdentity, {}}
      : c10::ArrayRef<ArgAdapter>{}[0];
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
    std::function<std::string()> mlir_generator,
    const AbiConfig& abi) {
  auto& cache = PyreKernelCache::get();
  auto result = cache.lookupOrClaim(cache_key, func_name, abi);

  if (result.is_compiler) {
    try {
      auto mlir = mlir_generator();
      PYRE_LOG(DEBUG) << "MLIR:\n" << mlir << "\n";
      auto vmfb = PyreKernelCompiler::compileSync(
          std::move(mlir), abi.compilerFlags());
      PYRE_LOG(INFO) << "compiled " << vmfb->size() << " bytes\n";
      cache.fulfill(cache_key, std::move(vmfb), func_name, abi);
    } catch (...) {
      cache.fail(cache_key, std::current_exception());
      throw;
    }
  }

  return result.future.get();
}



// ---------------------------------------------------------------------------
// Transient size query — TLS-cached per cache key
// ---------------------------------------------------------------------------

static iree_device_size_t queryTransientSize(
    CachedKernel* kernel,
    const AbiPacker& packer,
    iree_allocator_t alloc) {
  if (!kernel->has_transients_size) return 64;

  // The _transients_size companion has the same signature as the main
  // function. Pass null for transient/wait/signal — it's a pure query.
  c10::pyre::vm_list_ptr args;
  PYRE_CHECK_OK(iree_vm_list_create(
      iree_vm_make_undefined_type_def(), 32, alloc, args.for_output()));
  packer.packArgs(args.get(), /*transients=*/nullptr,
                  /*wait=*/nullptr, /*signal=*/nullptr);

  c10::pyre::vm_list_ptr rets;
  PYRE_CHECK_OK(iree_vm_list_create(
      iree_vm_make_undefined_type_def(), 1, alloc, rets.for_output()));

  PYRE_CHECK_OK(iree_vm_invoke(
      kernel->context.get(), kernel->transients_size_fn,
      IREE_VM_INVOCATION_FLAG_NONE, nullptr,
      args.get(), rets.get(), alloc));

  iree_vm_value_t size_val;
  PYRE_CHECK_OK(iree_vm_list_get_value(rets.get(), 0, &size_val));
  auto size = static_cast<iree_device_size_t>(size_val.i64);
  return size > 0 ? size : 64;  // minimum 64 bytes (non-null required)
}

static thread_local std::unordered_map<std::string, iree_device_size_t>
    tls_transient_size_cache;

static iree_device_size_t getTransientSize(
    CachedKernel* kernel,
    const AbiPacker& packer,
    const std::string& cache_key,
    iree_allocator_t alloc) {
  auto it = tls_transient_size_cache.find(cache_key);
  if (it != tls_transient_size_cache.end()) return it->second;

  iree_device_size_t size = queryTransientSize(kernel, packer, alloc);
  PYRE_LOG(INFO) << "transient size for " << cache_key << ": " << size << "\n";
  tls_transient_size_cache[cache_key] = size;
  return size;
}

// ---------------------------------------------------------------------------
// Envelope dispatch: AbiPacker-based invocation
// ---------------------------------------------------------------------------

void invokeEnvelope(
    CachedKernel* kernel,
    const AbiPacker& packer,
    c10::ArrayRef<at::Tensor> inputs,
    at::Tensor& output,
    const std::string& cache_key) {
  auto& runtime = c10::pyre::PyreRuntime::get();
  auto alloc = runtime.hostAllocator();

  c10::pyre::PyreStream stream(c10::pyre::getCurrentHostStream(0));
  auto& stream_ctx = stream.context();

  // Flush pending native ops (fill/copy) before VM invoke — the VM
  // manages its own command buffers internally, so we must submit any
  // buffered HAL commands first to maintain timeline ordering.
  stream.flush();

  // Allocate transient workspace first — queue_alloca advances the
  // timepoint, and the wait fence must include the post-alloca timepoint.
  // queue_alloca advances the timepoint; the signal fence must come after.
  auto* device = c10::pyre::PyreDevice::get(0);
  c10::pyre::hal_buffer_ptr transient_buf;
  {
    iree_device_size_t transient_size =
        getTransientSize(kernel, packer, cache_key, alloc);

    iree_hal_buffer_params_t params = {};
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL
                | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;

    uint64_t alloca_wait = stream_ctx.timepoint;
    auto* alloca_sem = stream_ctx.timeline.get();
    iree_hal_semaphore_list_t alloca_wait_list = {
        .count = (alloca_wait > 0) ? 1u : 0u,
        .semaphores = &alloca_sem,
        .payload_values = &alloca_wait};
    uint64_t alloca_signal = ++stream_ctx.timepoint;
    iree_hal_semaphore_list_t alloca_signal_list = {
        .count = 1, .semaphores = &alloca_sem,
        .payload_values = &alloca_signal};

    PYRE_CHECK_OK(iree_hal_device_queue_alloca(
        device->halDevice(), stream_ctx.affinity,
        alloca_wait_list, alloca_signal_list,
        IREE_HAL_ALLOCATOR_POOL_DEFAULT,
        params, transient_size,
        IREE_HAL_ALLOCA_FLAG_NONE,
        transient_buf.for_output()));
  }

  // Build wait fence. Include the stream's current timepoint (which now
  // includes the transient alloca signal) to enforce serial ordering.
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

  // Signal fence — FenceGuard owns this ref and signals on exception
  // so subsequent ops don't hang waiting on a timepoint never reached.
  uint64_t signal_value = ++stream_ctx.timepoint;
  iree_hal_fence_t* signal = nullptr;
  PYRE_CHECK_OK(iree_hal_fence_create_at(
      stream_ctx.timeline.get(), signal_value, alloc, &signal));
  FenceGuard fence_guard(signal);

  // Build args via packer.
  iree_host_size_t arg_count =
      static_cast<iree_host_size_t>(packer.numUniqueBuffers()) +
      static_cast<iree_host_size_t>(packer.dynamicDims().size()) +
      20;  // generous: offsets + output bufs + transients + fences
  c10::pyre::vm_list_ptr args;
  PYRE_CHECK_OK(iree_vm_list_create(
      iree_vm_make_undefined_type_def(), arg_count, alloc,
      args.for_output()));

  packer.packArgs(args, transient_buf, wait, signal);

  // Invoke.
  PYRE_CHECK_OK(iree_vm_invoke(
      kernel->context.get(), kernel->function,
      IREE_VM_INVOCATION_FLAG_NONE, nullptr,
      args, /*rets=*/nullptr, alloc));

  // Success — the dispatch itself will signal the fence asynchronously.
  fence_guard.disarm();

  // Timeline bookkeeping.
  for (const auto& t : inputs) {
    auto* ctx = static_cast<c10::pyre::PyreBufferContext*>(
        t.storage().data_ptr().get_context());
    if (ctx) ctx->recordUse(stream_ctx.timeline.get(), signal_value);
  }
  auto* out_ctx = static_cast<c10::pyre::PyreBufferContext*>(
      output.storage().data_ptr().get_context());
  if (out_ctx) out_ctx->recordMutation(
      stream_ctx.timeline.get(), signal_value);
}

// ---------------------------------------------------------------------------
// MmOp
// ---------------------------------------------------------------------------

KernelSpec MmOp::buildKernelSpec(
    const std::string& func_name, const OpContext& ctx) {
  auto out_shape = inferShape(ctx);
  return buildMmKernelSpec(
      func_name, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(), out_shape,
      c10::ArrayRef<ArgAdapter>{});
}

std::string MmOp::generateMlir(
    const std::string& func_name, const OpContext& ctx) {
  auto out_shape = inferShape(ctx);
  return generateMmMlir(
      func_name, ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(), out_shape,
      c10::ArrayRef<ArgAdapter>{});
}

// ---------------------------------------------------------------------------
// AddmmOp
// ---------------------------------------------------------------------------

static void addmmArgs(const OpContext& ctx, double& beta, double& alpha,
                      bool& mat2_permuted) {
  beta = ctx.scalars.size() >= 1 ? ctx.scalars[0].toDouble() : 1.0;
  alpha = ctx.scalars.size() >= 2 ? ctx.scalars[1].toDouble() : 1.0;
  mat2_permuted = c10::ArrayRef<ArgAdapter>{}.size() > 2
      && c10::ArrayRef<ArgAdapter>{}[2].kind == ArgAdapter::kPermute;
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

ComputeBody AddmmOp::generateComputeBody(
    const std::string& func_name, const OpContext& ctx) {
  double beta = ctx.scalars.size() >= 1 ? ctx.scalars[0].toDouble() : 1.0;
  double alpha = ctx.scalars.size() >= 2 ? ctx.scalars[1].toDouble() : 1.0;
  auto out_shape = inferShape(ctx);
  return generateAddmmComputeBody(ctx.dtype,
      ctx.inputs[0].sizes(), ctx.inputs[1].sizes(),
      ctx.inputs[2].sizes(), out_shape, beta, alpha);
}

// ---------------------------------------------------------------------------
// Multi-dim reduction dispatch helper
// ---------------------------------------------------------------------------

at::Tensor dispatchMultiDimReduction(
    const char* aten_name, const char* torch_op,
    const at::Tensor& self, c10::ArrayRef<int64_t> dims, bool keepdim,
    bool has_dtype_arg) {
  TORCH_CHECK(hasPyreBuffer(self), "pyre: ", aten_name,
      " requires IREE buffers");

  auto dtype = self.scalar_type();

  c10::SmallVector<int64_t, 6> norm_dims;
  for (auto d : dims) {
    if (d < 0) d += self.dim();
    norm_dims.push_back(d);
  }

  auto out_shape = inferReducedShape(self.sizes(), norm_dims, keepdim);
  auto func_name = funcNameDefault(aten_name);

  std::string extra_decls, extra_args, extra_types;
  if (has_dtype_arg) {
    extra_decls = "    %none = torch.constant.none";
    extra_args = ", %none";
    extra_types = ", !torch.none";
  }

  // Force contiguous for non-permutable strides.
  at::Tensor visit_self = self;
  {
    auto adapter = ArgAdapter::analyze(self);
    if (adapter.kind == ArgAdapter::kContiguous)
      visit_self = self.contiguous();
  }

  AbiPacker packer;
  packer.visitInput(visit_self);

  auto out = at::empty(out_shape, self.options());
  packer.visitOutput(out);

  auto spec = buildReductionKernelSpec(
      func_name, torch_op, dtype,
      visit_self.sizes(), out_shape, norm_dims, keepdim,
      extra_decls, extra_args, extra_types);

  auto cache_key = packer.cacheKey(
      spec.template_sha1, spec.substitutions,
      AbiConfig::kEnvelope.compilerFlags());

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    gen.visitInput(visit_self);
    gen.visitOutput(out);
    auto body = generateReductionComputeBody(
        torch_op, dtype, visit_self.sizes(), out_shape, norm_dims, keepdim,
        extra_decls, extra_args, extra_types);
    return gen.generateModule(func_name, body);
  });

  invokeEnvelope(kernel, packer, {visit_self}, out, cache_key);
  return out;
}

// ---------------------------------------------------------------------------
// Single-dim reduction dispatch helper
// ---------------------------------------------------------------------------

at::Tensor dispatchSingleDimReduction(
    const char* aten_name, const char* torch_op,
    const at::Tensor& self, int64_t dim, bool keepdim,
    const std::string& extra_decls,
    const std::string& extra_args,
    const std::string& extra_types) {
  TORCH_CHECK(hasPyreBuffer(self), "pyre: ", aten_name,
      " requires IREE buffers");

  auto dtype = self.scalar_type();
  if (dim < 0) dim += self.dim();

  auto out_shape = inferReducedShape(self.sizes(), {dim}, keepdim);
  auto func_name = funcNameDefault(aten_name);

  // Force contiguous for non-permutable strides.
  at::Tensor visit_self = self;
  {
    auto adapter = ArgAdapter::analyze(self);
    if (adapter.kind == ArgAdapter::kContiguous)
      visit_self = self.contiguous();
  }

  AbiPacker packer;
  packer.visitInput(visit_self);

  auto out = at::empty(out_shape, self.options());
  packer.visitOutput(out);

  auto spec = buildSingleDimReductionKernelSpec(
      func_name, torch_op, dtype,
      visit_self.sizes(), out_shape, dim, keepdim,
      extra_decls, extra_args, extra_types);

  auto cache_key = packer.cacheKey(
      spec.template_sha1, spec.substitutions,
      AbiConfig::kEnvelope.compilerFlags());

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    gen.visitInput(visit_self);
    gen.visitOutput(out);
    auto body = generateSingleDimReductionComputeBody(
        torch_op, dtype, visit_self.sizes(), out_shape, dim, keepdim,
        extra_decls, extra_args, extra_types);
    return gen.generateModule(func_name, body);
  });

  invokeEnvelope(kernel, packer, {visit_self}, out, cache_key);
  return out;
}

// ---------------------------------------------------------------------------
// EmbeddingOp
// ---------------------------------------------------------------------------

at::Tensor EmbeddingOp::impl(
    const at::Tensor& weight, const at::Tensor& indices,
    int64_t padding_idx, bool /*scale_grad_by_freq*/, bool /*sparse*/) {
  TORCH_CHECK(hasPyreBuffer(weight), "pyre: embedding requires IREE weight");
  TORCH_CHECK(hasPyreBuffer(indices), "pyre: embedding requires IREE indices");
  TORCH_CHECK(weight.dim() == 2, "pyre: embedding weight must be 2D");
  TORCH_CHECK(padding_idx == -1,
      "pyre: embedding with custom padding_idx not yet supported");

  auto dtype = weight.scalar_type();
  c10::DimVector out_shape(indices.sizes().begin(), indices.sizes().end());
  out_shape.push_back(weight.size(1));

  auto func_name = funcNameDefault(aten_name);

  // Force contiguous for non-permutable strides.
  at::Tensor visit_weight = weight;
  at::Tensor visit_indices = indices;
  {
    auto wa = ArgAdapter::analyze(weight);
    if (wa.kind == ArgAdapter::kContiguous) visit_weight = weight.contiguous();
    auto ia = ArgAdapter::analyze(indices);
    if (ia.kind == ArgAdapter::kContiguous) visit_indices = indices.contiguous();
  }

  AbiPacker packer;
  packer.visitInput(visit_weight);
  packer.visitInput(visit_indices);

  auto out = at::empty(out_shape, weight.options());
  packer.visitOutput(out);

  auto spec = buildEmbeddingKernelSpec(
      func_name, dtype, visit_weight.sizes(), visit_indices.sizes(), out_shape);
  auto cache_key = packer.cacheKey(
      spec.template_sha1, spec.substitutions,
      AbiConfig::kEnvelope.compilerFlags());

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    gen.visitInput(visit_weight);
    gen.visitInput(visit_indices);
    gen.visitOutput(out);
    auto body = generateEmbeddingComputeBody(
        dtype, visit_weight.sizes(), visit_indices.sizes(), out_shape);
    return gen.generateModule(func_name, body);
  });

  invokeEnvelope(kernel, packer, {visit_weight, visit_indices}, out, cache_key);
  return out;
}

// ---------------------------------------------------------------------------
// IndexSelectOp
// ---------------------------------------------------------------------------

at::Tensor IndexSelectOp::impl(
    const at::Tensor& self, int64_t dim, const at::Tensor& index) {
  TORCH_CHECK(hasPyreBuffer(self), "pyre: index_select requires IREE self");
  TORCH_CHECK(hasPyreBuffer(index), "pyre: index_select requires IREE index");

  if (dim < 0) dim += self.dim();
  auto dtype = self.scalar_type();

  c10::DimVector out_shape(self.sizes().begin(), self.sizes().end());
  out_shape[dim] = index.size(0);

  auto func_name = funcNameDefault(aten_name);

  at::Tensor visit_self = self;
  at::Tensor visit_index = index;
  {
    auto sa = ArgAdapter::analyze(self);
    if (sa.kind == ArgAdapter::kContiguous) visit_self = self.contiguous();
    auto ia = ArgAdapter::analyze(index);
    if (ia.kind == ArgAdapter::kContiguous) visit_index = index.contiguous();
  }

  AbiPacker packer;
  packer.visitInput(visit_self);
  packer.visitInput(visit_index);

  auto out = at::empty(out_shape, self.options());
  packer.visitOutput(out);

  auto spec = buildIndexSelectKernelSpec(
      func_name, dtype, visit_self.sizes(), visit_index.sizes(), out_shape, dim);
  auto cache_key = packer.cacheKey(
      spec.template_sha1, spec.substitutions,
      AbiConfig::kEnvelope.compilerFlags());

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    gen.visitInput(visit_self);
    gen.visitInput(visit_index);
    gen.visitOutput(out);
    auto body = generateIndexSelectComputeBody(
        dtype, visit_self.sizes(), visit_index.sizes(), out_shape, dim);
    return gen.generateModule(func_name, body);
  });

  invokeEnvelope(kernel, packer, {visit_self, visit_index}, out, cache_key);
  return out;
}

// ---------------------------------------------------------------------------
// GatherOp
// ---------------------------------------------------------------------------

at::Tensor GatherOp::impl(
    const at::Tensor& self, int64_t dim, const at::Tensor& index,
    bool /*sparse_grad*/) {
  TORCH_CHECK(hasPyreBuffer(self), "pyre: gather requires IREE self");
  TORCH_CHECK(hasPyreBuffer(index), "pyre: gather requires IREE index");

  if (dim < 0) dim += self.dim();
  auto dtype = self.scalar_type();
  c10::DimVector out_shape(index.sizes().begin(), index.sizes().end());

  auto func_name = funcNameDefault(aten_name);

  at::Tensor visit_self = self;
  at::Tensor visit_index = index;
  {
    auto sa = ArgAdapter::analyze(self);
    if (sa.kind == ArgAdapter::kContiguous) visit_self = self.contiguous();
    auto ia = ArgAdapter::analyze(index);
    if (ia.kind == ArgAdapter::kContiguous) visit_index = index.contiguous();
  }

  AbiPacker packer;
  packer.visitInput(visit_self);
  packer.visitInput(visit_index);

  auto out = at::empty(out_shape, self.options());
  packer.visitOutput(out);

  auto spec = buildGatherKernelSpec(
      func_name, dtype, visit_self.sizes(), visit_index.sizes(), out_shape, dim);
  auto cache_key = packer.cacheKey(
      spec.template_sha1, spec.substitutions,
      AbiConfig::kEnvelope.compilerFlags());

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    gen.visitInput(visit_self);
    gen.visitInput(visit_index);
    gen.visitOutput(out);
    auto body = generateGatherComputeBody(
        dtype, visit_self.sizes(), visit_index.sizes(), out_shape, dim);
    return gen.generateModule(func_name, body);
  });

  invokeEnvelope(kernel, packer, {visit_self, visit_index}, out, cache_key);
  return out;
}

// ---------------------------------------------------------------------------
// SoftmaxOp / LogSoftmaxOp — compiled kernel with first-class _softmax
// ---------------------------------------------------------------------------

static at::Tensor dispatchSoftmax(
    const char* aten_name, const std::string& softmax_op,
    const at::Tensor& self, int64_t dim) {
  TORCH_CHECK(hasPyreBuffer(self), "pyre: ", aten_name,
      " requires IREE buffers");
  if (dim < 0) dim += self.dim();
  auto dtype = self.scalar_type();
  auto func_name = funcNameDefault(aten_name);

  at::Tensor visit_self = self;
  {
    auto adapter = ArgAdapter::analyze(self);
    if (adapter.kind == ArgAdapter::kContiguous)
      visit_self = self.contiguous();
  }

  AbiPacker packer;
  packer.visitInput(visit_self);

  c10::DimVector out_shape(self.sizes().begin(), self.sizes().end());
  auto out = at::empty(out_shape, self.options());
  packer.visitOutput(out);

  auto spec = buildSoftmaxKernelSpec(func_name, softmax_op, dtype,
      visit_self.sizes(), dim);
  auto cache_key = packer.cacheKey(
      spec.template_sha1, spec.substitutions,
      AbiConfig::kEnvelope.compilerFlags());

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    gen.visitInput(visit_self);
    gen.visitOutput(out);
    auto body = generateSoftmaxComputeBody(
        softmax_op, dtype, visit_self.sizes(), dim);
    return gen.generateModule(func_name, body);
  });

  invokeEnvelope(kernel, packer, {visit_self}, out, cache_key);
  return out;
}

at::Tensor SoftmaxOp::impl(
    const at::Tensor& self, int64_t dim, bool /*half_to_float*/) {
  return dispatchSoftmax(aten_name, "_softmax", self, dim);
}

at::Tensor LogSoftmaxOp::impl(
    const at::Tensor& self, int64_t dim, bool /*half_to_float*/) {
  return dispatchSoftmax(aten_name, "_log_softmax", self, dim);
}

// ---------------------------------------------------------------------------
// ScatterSrcOp
// ---------------------------------------------------------------------------

static at::Tensor dispatchScatterSrc(
    const at::Tensor& self, int64_t dim,
    const at::Tensor& index, const at::Tensor& src) {
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(index) && hasPyreBuffer(src),
      "pyre: scatter.src requires IREE buffers");
  if (dim < 0) dim += self.dim();
  auto dtype = self.scalar_type();
  auto func_name = funcNameDefault("scatter_src");

  at::Tensor visit_self = self, visit_index = index, visit_src = src;
  {
    auto sa = ArgAdapter::analyze(self);
    if (sa.kind == ArgAdapter::kContiguous) visit_self = self.contiguous();
    auto ia = ArgAdapter::analyze(index);
    if (ia.kind == ArgAdapter::kContiguous) visit_index = index.contiguous();
    auto ra = ArgAdapter::analyze(src);
    if (ra.kind == ArgAdapter::kContiguous) visit_src = src.contiguous();
  }

  AbiPacker packer;
  packer.visitInput(visit_self);
  packer.visitInput(visit_index);
  packer.visitInput(visit_src);

  c10::DimVector out_shape(self.sizes().begin(), self.sizes().end());
  auto out = at::empty(out_shape, self.options());
  packer.visitOutput(out);

  auto spec = buildScatterSrcKernelSpec(
      func_name, dtype, visit_self.sizes(), visit_index.sizes(),
      visit_src.sizes(), dim);
  auto cache_key = packer.cacheKey(
      spec.template_sha1, spec.substitutions,
      AbiConfig::kEnvelope.compilerFlags());

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    gen.visitInput(visit_self);
    gen.visitInput(visit_index);
    gen.visitInput(visit_src);
    gen.visitOutput(out);
    auto body = generateScatterSrcComputeBody(
        dtype, visit_self.sizes(), visit_index.sizes(), visit_src.sizes(), dim);
    return gen.generateModule(func_name, body);
  });

  invokeEnvelope(kernel, packer, {visit_self, visit_index, visit_src}, out, cache_key);
  return out;
}

at::Tensor ScatterSrcOp::impl(
    const at::Tensor& self, int64_t dim,
    const at::Tensor& index, const at::Tensor& src) {
  return dispatchScatterSrc(self, dim, index, src);
}


at::Tensor& ScatterSrcOp::impl_inplace(
    at::Tensor& self, int64_t dim,
    const at::Tensor& index, const at::Tensor& src) {
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(index) && hasPyreBuffer(src),
      "pyre: scatter_.src requires IREE buffers");
  if (dim < 0) dim += self.dim();
  auto dtype = self.scalar_type();
  auto func_name = funcNameDefault("scatter_src_inplace");

  AbiPacker packer;
  packer.visitInput(self);
  packer.visitInput(index);
  packer.visitInput(src);
  packer.visitOutput(self);

  auto spec = buildScatterSrcInplaceKernelSpec(
      func_name, dtype, self.sizes(), index.sizes(), src.sizes(), dim);
  auto cache_key = packer.cacheKey(
      spec.template_sha1, spec.substitutions,
      AbiConfig::kEnvelope.compilerFlags());

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    gen.visitInput(self);
    gen.visitInput(index);
    gen.visitInput(src);
    gen.visitOutput(self);
    auto body = generateScatterSrcComputeBody(
        dtype, self.sizes(), index.sizes(), src.sizes(), dim);
    return gen.generateModule(func_name, body);
  });

  invokeEnvelope(kernel, packer, {self, index, src}, self, cache_key);
  return self;
}

// ---------------------------------------------------------------------------
// ScatterAddOp
// ---------------------------------------------------------------------------

static at::Tensor dispatchScatterAdd(
    const at::Tensor& self, int64_t dim,
    const at::Tensor& index, const at::Tensor& src) {
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(index) && hasPyreBuffer(src),
      "pyre: scatter_add requires IREE buffers");
  if (dim < 0) dim += self.dim();
  auto dtype = self.scalar_type();
  auto func_name = funcNameDefault("scatter_add");

  at::Tensor visit_self = self, visit_index = index, visit_src = src;
  {
    auto sa = ArgAdapter::analyze(self);
    if (sa.kind == ArgAdapter::kContiguous) visit_self = self.contiguous();
    auto ia = ArgAdapter::analyze(index);
    if (ia.kind == ArgAdapter::kContiguous) visit_index = index.contiguous();
    auto ra = ArgAdapter::analyze(src);
    if (ra.kind == ArgAdapter::kContiguous) visit_src = src.contiguous();
  }

  AbiPacker packer;
  packer.visitInput(visit_self);
  packer.visitInput(visit_index);
  packer.visitInput(visit_src);

  c10::DimVector out_shape(self.sizes().begin(), self.sizes().end());
  auto out = at::empty(out_shape, self.options());
  packer.visitOutput(out);

  auto spec = buildScatterAddKernelSpec(
      func_name, dtype, visit_self.sizes(), visit_index.sizes(),
      visit_src.sizes(), dim);
  auto cache_key = packer.cacheKey(
      spec.template_sha1, spec.substitutions,
      AbiConfig::kEnvelope.compilerFlags());

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    gen.visitInput(visit_self);
    gen.visitInput(visit_index);
    gen.visitInput(visit_src);
    gen.visitOutput(out);
    auto body = generateScatterAddComputeBody(
        dtype, visit_self.sizes(), visit_index.sizes(), visit_src.sizes(), dim);
    return gen.generateModule(func_name, body);
  });

  invokeEnvelope(kernel, packer, {visit_self, visit_index, visit_src}, out, cache_key);
  return out;
}

at::Tensor ScatterAddOp::impl(
    const at::Tensor& self, int64_t dim,
    const at::Tensor& index, const at::Tensor& src) {
  return dispatchScatterAdd(self, dim, index, src);
}


at::Tensor& ScatterAddOp::impl_inplace(
    at::Tensor& self, int64_t dim,
    const at::Tensor& index, const at::Tensor& src) {
  TORCH_CHECK(hasPyreBuffer(self) && hasPyreBuffer(index) && hasPyreBuffer(src),
      "pyre: scatter_add_ requires IREE buffers");
  if (dim < 0) dim += self.dim();
  auto dtype = self.scalar_type();
  auto func_name = funcNameDefault("scatter_add_inplace");

  AbiPacker packer;
  packer.visitInput(self);
  packer.visitInput(index);
  packer.visitInput(src);
  packer.visitOutput(self);

  auto spec = buildScatterAddInplaceKernelSpec(
      func_name, dtype, self.sizes(), index.sizes(), src.sizes(), dim);
  auto cache_key = packer.cacheKey(
      spec.template_sha1, spec.substitutions,
      AbiConfig::kEnvelope.compilerFlags());

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    gen.visitInput(self);
    gen.visitInput(index);
    gen.visitInput(src);
    gen.visitOutput(self);
    auto body = generateScatterAddComputeBody(
        dtype, self.sizes(), index.sizes(), src.sizes(), dim);
    return gen.generateModule(func_name, body);
  });

  invokeEnvelope(kernel, packer, {self, index, src}, self, cache_key);
  return self;
}

// ---------------------------------------------------------------------------
// Shape spec helpers for programmatic ComputeBody builders.
// ---------------------------------------------------------------------------
static c10::SmallVector<DimSpec, 6> broadcastAwareDimSpec(
    c10::ArrayRef<int64_t> sizes) {
  c10::SmallVector<DimSpec, 6> spec;
  for (int64_t s : sizes)
    spec.push_back(s == 1 ? DimSpec::fixed(1) : DimSpec::dynamic());
  return spec;
}

// ---------------------------------------------------------------------------
// IndexPutOp — envelope dispatch with programmatic ComputeBody
// ---------------------------------------------------------------------------

// Build a ComputeBody for index_put with variable-length index list.
static ComputeBody buildIndexPutBody(
    const std::string& elt,
    c10::IntArrayRef self_sizes, c10::IntArrayRef values_sizes,
    const c10::SmallVector<std::pair<int64_t, at::Tensor>, 4>& non_none,
    const c10::List<std::optional<at::Tensor>>& indices,
    bool accumulate) {
  auto dynS = [](c10::IntArrayRef s) {
    std::string r;
    for (size_t i = 0; i < s.size(); ++i) {
      if (i > 0) r += ",";
      r += (s[i] == 1) ? "1" : "?";
    }
    return r;
  };
  auto mkVt = [&](const std::string& sh, const std::string& e) {
    return "!torch.vtensor<[" + sh + "], " + e + ">";
  };
  auto mkBt = [&](const std::string& sh, const std::string& e) {
    std::string dims;
    for (char c : sh) dims += (c == ',') ? 'x' : c;
    std::string be = e;
    if (be.substr(0,2) == "si") be = "i" + be.substr(2);
    return dims.empty() ? "tensor<" + be + ">" : "tensor<" + dims + "x" + be + ">";
  };

  std::string self_s = dynS(self_sizes);
  std::string vals_s = dynS(values_sizes);

  ComputeBody body;
  // Inputs: self, idx0, idx1, ..., values
  body.input_names.push_back("self_v");
  body.input_vtensor_types.push_back(mkVt(self_s, elt));
  body.input_shape_specs.push_back(broadcastAwareDimSpec(self_sizes));
  body.input_tensor_types.push_back(mkBt(self_s, elt));

  for (size_t i = 0; i < non_none.size(); ++i) {
    body.input_names.push_back("idx" + std::to_string(i));
    std::string is = dynS(non_none[i].second.sizes());
    body.input_vtensor_types.push_back(mkVt(is, "si64"));
    body.input_shape_specs.push_back(broadcastAwareDimSpec(non_none[i].second.sizes()));
    body.input_tensor_types.push_back(mkBt(is, "si64"));
  }

  body.input_names.push_back("values");
  body.input_vtensor_types.push_back(mkVt(vals_s, elt));
  body.input_shape_specs.push_back(broadcastAwareDimSpec(values_sizes));
  body.input_tensor_types.push_back(mkBt(vals_s, elt));

  body.output_vtensor_type = mkVt(self_s, elt);
  body.output_tensor_type = mkBt(self_s, elt);
  body.output_shape_spec = broadcastAwareDimSpec(self_sizes);

  // Build torch ops
  std::string ops;
  // Derefine indices into optional list
  int tc = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
    if (indices[i].has_value() && indices[i]->defined()) {
      std::string is = dynS(non_none[tc].second.sizes());
      ops += "    %opt" + std::to_string(i) + " = torch.derefine %idx" +
             std::to_string(tc) + " : !torch.vtensor<[" + is +
             "], si64> to !torch.optional<vtensor>\n";
      tc++;
    } else {
      ops += "    %none" + std::to_string(i) + " = torch.constant.none\n";
      ops += "    %opt" + std::to_string(i) + " = torch.derefine %none" +
             std::to_string(i) +
             " : !torch.none to !torch.optional<vtensor>\n";
    }
  }
  // Build index list
  std::string opt_names, opt_types;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
    if (i > 0) { opt_names += ", "; opt_types += ", "; }
    opt_names += "%opt" + std::to_string(i);
    opt_types += "!torch.optional<vtensor>";
  }
  ops += "    %indices = torch.prim.ListConstruct " + opt_names +
         " : (" + opt_types + ") -> !torch.list<optional<vtensor>>\n";
  ops += "    %accum = torch.constant.bool " +
         std::string(accumulate ? "true" : "false") + "\n";
  ops += "    %result = torch.aten.index_put %self_v, %indices, %values, %accum : " +
         body.input_vtensor_types[0] + ", !torch.list<optional<vtensor>>, " +
         body.input_vtensor_types.back() + ", !torch.bool -> " +
         body.output_vtensor_type + "\n";
  body.mlir_ops = std::move(ops);
  return body;
}

// Collect non-None indices from the index list, promoting CPU to device.
static c10::SmallVector<std::pair<int64_t, at::Tensor>, 4> collectIndices(
    const c10::List<std::optional<at::Tensor>>& indices,
    c10::Device target_device) {
  c10::SmallVector<std::pair<int64_t, at::Tensor>, 4> non_none;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
    if (indices[i].has_value() && indices[i]->defined()) {
      auto idx = *indices[i];
      if (!hasPyreBuffer(idx))
        idx = idx.to(target_device);
      non_none.push_back({i, idx});
    }
  }
  return non_none;
}

static at::Tensor dispatchIndexPut(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values, bool accumulate) {
  TORCH_CHECK(hasPyreBuffer(self), "pyre: index_put requires IREE self");
  TORCH_CHECK(hasPyreBuffer(values), "pyre: index_put requires IREE values");

  auto non_none = collectIndices(indices, self.device());
  TORCH_CHECK(!non_none.empty(), "pyre: index_put requires at least one index tensor");

  auto dtype = self.scalar_type();
  std::string elt = scalarTypeToTorchMlir(dtype);
  auto func_name = funcNameDefault("index_put");

  // Force contiguous for non-permutable strides.
  at::Tensor v_self = self, v_values = values;
  if (ArgAdapter::analyze(self).kind == ArgAdapter::kContiguous)
    v_self = self.contiguous();
  if (ArgAdapter::analyze(values).kind == ArgAdapter::kContiguous)
    v_values = values.contiguous();

  // Visit all tensors
  AbiPacker packer;
  packer.visitInput(v_self);
  for (const auto& [dim, idx] : non_none) packer.visitInput(idx);
  packer.visitInput(v_values);

  auto out = at::empty(c10::DimVector(self.sizes()), self.options());
  packer.visitOutput(out);

  auto body = buildIndexPutBody(elt, self.sizes(), values.sizes(),
                                 non_none, indices, accumulate);

  // Cache key from op identity hash
  std::string identity = std::string("index_put\0", 10) + elt + "\0" +
      packer.bufTopology() + "\0" + packer.dimPattern();
  auto cache_key = c10::sha1(identity).str();

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    gen.visitInput(v_self);
    for (const auto& [dim, idx] : non_none) gen.visitInput(idx);
    gen.visitInput(v_values);
    gen.visitOutput(out);
    return gen.generateModule(func_name, body);
  });

  c10::SmallVector<at::Tensor, 8> inputs;
  inputs.push_back(v_self);
  for (const auto& [dim, idx] : non_none) inputs.push_back(idx);
  inputs.push_back(v_values);

  invokeEnvelope(kernel, packer, inputs, out, cache_key);
  return out;
}

at::Tensor IndexPutOp::impl(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values, bool accumulate) {
  return dispatchIndexPut(self, indices, values, accumulate);
}

at::Tensor& IndexPutOp::impl_inplace(
    at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values, bool accumulate) {
  TORCH_CHECK(hasPyreBuffer(self), "pyre: index_put_ requires IREE self");
  TORCH_CHECK(hasPyreBuffer(values), "pyre: index_put_ requires IREE values");

  auto non_none = collectIndices(indices, self.device());
  TORCH_CHECK(!non_none.empty(), "pyre: index_put_ requires at least one index tensor");

  auto dtype = self.scalar_type();
  std::string elt = scalarTypeToTorchMlir(dtype);
  auto func_name = funcNameDefault("index_put_inplace");

  at::Tensor v_values = values;
  if (ArgAdapter::analyze(values).kind == ArgAdapter::kContiguous)
    v_values = values.contiguous();

  // Inplace: self is both input (read) and output (write). The packer
  // deduplicates the storage — same buffer used for import and alias.
  AbiPacker packer;
  packer.visitInput(self);
  for (const auto& [dim, idx] : non_none) packer.visitInput(idx);
  packer.visitInput(v_values);
  packer.visitOutput(self);

  auto body = buildIndexPutBody(elt, self.sizes(), values.sizes(),
                                 non_none, indices, accumulate);

  std::string identity = std::string("index_put_ip\0", 13) + elt + "\0" +
      packer.bufTopology() + "\0" + packer.dimPattern();
  auto cache_key = c10::sha1(identity).str();

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    gen.visitInput(self);
    for (const auto& [dim, idx] : non_none) gen.visitInput(idx);
    gen.visitInput(v_values);
    gen.visitOutput(self);
    return gen.generateModule(func_name, body);
  });

  c10::SmallVector<at::Tensor, 8> inputs;
  inputs.push_back(self);
  for (const auto& [dim, idx] : non_none) inputs.push_back(idx);
  inputs.push_back(v_values);

  invokeEnvelope(kernel, packer, inputs, self, cache_key);
  return self;
}

// ---------------------------------------------------------------------------
// IndexTensorOp — envelope dispatch with programmatic ComputeBody
// ---------------------------------------------------------------------------

at::Tensor IndexTensorOp::impl(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices) {
  at::Tensor self_effective = self;
  if (!hasPyreBuffer(self) && self.is_cpu()) {
    c10::Device target = c10::kCPU;
    for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
      if (indices[i].has_value() && indices[i]->defined() &&
          indices[i]->is_privateuseone()) {
        target = indices[i]->device();
        break;
      }
    }
    if (target.type() != c10::DeviceType::CPU)
      self_effective = self.to(target);
  }
  TORCH_CHECK(hasPyreBuffer(self_effective),
      "pyre: index.Tensor requires IREE self");

  auto non_none = collectIndices(indices, self_effective.device());
  TORCH_CHECK(!non_none.empty(), "pyre: index.Tensor requires at least one index");

  // Fast path: single 1D index → decompose to index_select (already on envelope).
  if (non_none.size() == 1 && non_none[0].second.dim() == 1)
    return at::index_select(self_effective, non_none[0].first, non_none[0].second);

  auto dtype = self_effective.scalar_type();
  std::string elt = scalarTypeToTorchMlir(dtype);
  auto func_name = funcNameDefault(aten_name);

  auto dynS = [](c10::IntArrayRef s) {
    std::string r;
    for (size_t i = 0; i < s.size(); ++i) {
      if (i > 0) r += ",";
      r += (s[i] == 1) ? "1" : "?";
    }
    return r;
  };
  auto mkVt = [&](const std::string& sh, const std::string& e) {
    return "!torch.vtensor<[" + sh + "], " + e + ">";
  };
  auto mkBt = [&](const std::string& sh, const std::string& e) {
    std::string dims;
    for (char c : sh) dims += (c == ',') ? 'x' : c;
    std::string be = e;
    if (be.substr(0,2) == "si") be = "i" + be.substr(2);
    return dims.empty() ? "tensor<" + be + ">" : "tensor<" + dims + "x" + be + ">";
  };

  // Compute output shape.
  c10::DimVector bcast_shape;
  for (const auto& [dim, idx] : non_none) {
    if (bcast_shape.empty())
      bcast_shape.assign(idx.sizes().begin(), idx.sizes().end());
    else
      bcast_shape = c10::DimVector(at::infer_size(bcast_shape, idx.sizes()));
  }
  int64_t last_indexed = non_none.back().first;
  c10::DimVector out_shape(bcast_shape.begin(), bcast_shape.end());
  for (int64_t d = last_indexed + 1; d < self_effective.dim(); ++d)
    out_shape.push_back(self_effective.size(d));

  std::string self_s = dynS(self_effective.sizes());
  std::string out_s = dynS(out_shape);

  // Build ComputeBody
  ComputeBody body;
  body.input_names.push_back("self_v");
  body.input_vtensor_types.push_back(mkVt(self_s, elt));
  body.input_shape_specs.push_back(broadcastAwareDimSpec(self_effective.sizes()));
  body.input_tensor_types.push_back(mkBt(self_s, elt));
  for (size_t i = 0; i < non_none.size(); ++i) {
    body.input_names.push_back("idx" + std::to_string(i));
    std::string is = dynS(non_none[i].second.sizes());
    body.input_vtensor_types.push_back(mkVt(is, "si64"));
    body.input_shape_specs.push_back(broadcastAwareDimSpec(non_none[i].second.sizes()));
    body.input_tensor_types.push_back(mkBt(is, "si64"));
  }
  body.output_vtensor_type = mkVt(out_s, elt);
  body.output_tensor_type = mkBt(out_s, elt);
  body.output_shape_spec = broadcastAwareDimSpec(out_shape);

  std::string ops;
  int tc = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
    if (indices[i].has_value() && indices[i]->defined()) {
      std::string is = dynS(non_none[tc].second.sizes());
      ops += "    %opt" + std::to_string(i) + " = torch.derefine %idx" +
             std::to_string(tc) + " : " + mkVt(is, "si64") +
             " to !torch.optional<vtensor>\n";
      tc++;
    } else {
      ops += "    %none" + std::to_string(i) + " = torch.constant.none\n";
      ops += "    %opt" + std::to_string(i) + " = torch.derefine %none" +
             std::to_string(i) + " : !torch.none to !torch.optional<vtensor>\n";
    }
  }
  std::string onames, otypes;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
    if (i > 0) { onames += ", "; otypes += ", "; }
    onames += "%opt" + std::to_string(i);
    otypes += "!torch.optional<vtensor>";
  }
  ops += "    %indices = torch.prim.ListConstruct " + onames +
         " : (" + otypes + ") -> !torch.list<optional<vtensor>>\n";
  ops += "    %result = torch.aten.index.Tensor %self_v, %indices : " +
         body.input_vtensor_types[0] + ", !torch.list<optional<vtensor>> -> " +
         body.output_vtensor_type + "\n";
  body.mlir_ops = std::move(ops);

  // Envelope dispatch
  AbiPacker packer;
  packer.visitInput(self_effective);
  for (const auto& [dim, idx] : non_none) packer.visitInput(idx);
  auto out = at::empty(out_shape, self_effective.options());
  packer.visitOutput(out);

  std::string identity = std::string("index_tensor\0", 13) + elt + "\0" +
      packer.bufTopology() + "\0" + packer.dimPattern();
  auto cache_key = c10::sha1(identity).str();

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    gen.visitInput(self_effective);
    for (const auto& [dim, idx] : non_none) gen.visitInput(idx);
    gen.visitOutput(out);
    return gen.generateModule(func_name, body);
  });

  c10::SmallVector<at::Tensor, 8> inputs;
  inputs.push_back(self_effective);
  for (const auto& [dim, idx] : non_none) inputs.push_back(idx);

  invokeEnvelope(kernel, packer, inputs, out, cache_key);
  return out;
}

// ---------------------------------------------------------------------------
// ArangeOp
// ---------------------------------------------------------------------------

at::Tensor ArangeOp::impl(
    const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout>,
    std::optional<at::Device> device,
    std::optional<bool>) {
  double start_d = start.toDouble();
  double end_d = end.toDouble();
  double step_d = step.toDouble();
  TORCH_CHECK(step_d != 0, "pyre: arange step must be non-zero");
  TORCH_CHECK((end_d - start_d) / step_d >= 0,
      "pyre: arange upper bound and step mismatch");

  int64_t out_size = static_cast<int64_t>(
      std::ceil((end_d - start_d) / step_d));
  if (out_size < 0) out_size = 0;

  auto out_dtype = dtype.value_or(
      start.isFloatingPoint() || end.isFloatingPoint() || step.isFloatingPoint()
          ? at::kFloat : at::kLong);
  auto out_device = c10::device_or_default(device);

  auto func_name = funcNameDefault(aten_name);

  // Arange has zero tensor inputs — only an output.
  AbiPacker packer;
  auto out = at::empty({out_size},
      at::TensorOptions().dtype(out_dtype).device(out_device));
  packer.visitOutput(out);

  auto spec = buildArangeKernelSpec(
      func_name, out_dtype, out_size, start_d, end_d, step_d);
  auto cache_key = packer.cacheKey(
      spec.template_sha1, spec.substitutions,
      AbiConfig::kEnvelope.compilerFlags());

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    gen.visitOutput(out);
    auto body = generateArangeComputeBody(
        out_dtype, out_size, start_d, end_d, step_d);
    return gen.generateModule(func_name, body);
  });

  invokeEnvelope(kernel, packer, {}, out, cache_key);
  return out;
}

// ---------------------------------------------------------------------------
// TypeCastOp
// ---------------------------------------------------------------------------

at::Tensor TypeCastOp::impl(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<at::MemoryFormat> memory_format) {
  if (!hasPyreBuffer(self) || !dtype.has_value() || *dtype == self.scalar_type())
    return at::native::_to_copy(self, dtype, layout, device, pin_memory,
                                 non_blocking, memory_format);

  auto in_dtype = self.scalar_type();
  auto out_dtype = *dtype;
  auto func_name = funcNameDefault(aten_name);

  at::Tensor visit_self = self;
  {
    auto adapter = ArgAdapter::analyze(self);
    if (adapter.kind == ArgAdapter::kContiguous)
      visit_self = self.contiguous();
  }

  AbiPacker packer;
  packer.visitInput(visit_self);

  auto out = at::empty(
      c10::DimVector(self.sizes()), self.options().dtype(out_dtype));
  packer.visitOutput(out);

  auto spec = buildTypeCastKernelSpec(
      func_name, in_dtype, out_dtype, visit_self.sizes());
  auto cache_key = packer.cacheKey(
      spec.template_sha1, spec.substitutions,
      AbiConfig::kEnvelope.compilerFlags());

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    gen.visitInput(visit_self);
    gen.visitOutput(out);
    auto body = generateTypeCastComputeBody(
        in_dtype, out_dtype, visit_self.sizes());
    return gen.generateModule(func_name, body);
  });

  invokeEnvelope(kernel, packer, {visit_self}, out, cache_key);
  return out;
}

// ---------------------------------------------------------------------------
// CatOp — envelope dispatch with programmatic ComputeBody
// ---------------------------------------------------------------------------

at::Tensor CatOp::impl(const at::ITensorListRef& tensors, int64_t dim) {
  auto materialized = tensors.materialize();
  TORCH_CHECK(!materialized.empty(), "pyre: cat requires at least one tensor");

  for (const auto& t : materialized)
    TORCH_CHECK(hasPyreBuffer(t.get()), "pyre: cat requires IREE buffers");

  auto dtype = materialized[0].get().scalar_type();
  if (dim < 0) dim += materialized[0].get().dim();

  c10::DimVector out_shape(materialized[0].get().sizes().begin(),
                           materialized[0].get().sizes().end());
  out_shape[dim] = 0;
  for (const auto& t : materialized)
    out_shape[dim] += t.get().size(dim);

  auto func_name = funcNameDefault(aten_name);
  std::string elt = scalarTypeToTorchMlir(dtype);

  auto dynS = [](c10::IntArrayRef s) {
    std::string r;
    for (size_t i = 0; i < s.size(); ++i) {
      if (i > 0) r += ",";
      r += (s[i] == 1) ? "1" : "?";
    }
    return r;
  };
  auto mkVt = [&](const std::string& sh, const std::string& e) {
    return "!torch.vtensor<[" + sh + "], " + e + ">";
  };
  auto mkBt = [&](const std::string& sh, const std::string& e) {
    std::string dims;
    for (char c : sh) dims += (c == ',') ? 'x' : c;
    std::string be = e;
    if (be.substr(0,2) == "si") be = "i" + be.substr(2);
    return dims.empty() ? "tensor<" + be + ">" : "tensor<" + dims + "x" + be + ">";
  };

  std::string out_s = dynS(out_shape);

  // Build ComputeBody
  ComputeBody body;
  for (size_t i = 0; i < materialized.size(); ++i) {
    std::string is = dynS(materialized[i].get().sizes());
    body.input_names.push_back("input" + std::to_string(i));
    body.input_vtensor_types.push_back(mkVt(is, elt));
    body.input_shape_specs.push_back(broadcastAwareDimSpec(materialized[i].get().sizes()));
    body.input_tensor_types.push_back(mkBt(is, elt));
  }
  body.output_vtensor_type = mkVt(out_s, elt);
  body.output_tensor_type = mkBt(out_s, elt);
  body.output_shape_spec = broadcastAwareDimSpec(out_shape);

  std::string ops;
  std::string list_names, list_types;
  for (size_t i = 0; i < materialized.size(); ++i) {
    if (i > 0) { list_names += ", "; list_types += ", "; }
    list_names += "%input" + std::to_string(i);
    list_types += body.input_vtensor_types[i];
  }
  ops += "    %tensors = torch.prim.ListConstruct " + list_names +
         " : (" + list_types + ") -> !torch.list<vtensor>\n";
  ops += "    %dim = torch.constant.int " + std::to_string(dim) + "\n";
  ops += "    %result = torch.aten.cat %tensors, %dim : "
         "!torch.list<vtensor>, !torch.int -> " +
         body.output_vtensor_type + "\n";
  body.mlir_ops = std::move(ops);

  // Force contiguous for non-permutable strides.
  c10::SmallVector<at::Tensor, 8> visit_tensors;
  for (const auto& t : materialized) {
    if (ArgAdapter::analyze(t.get()).kind == ArgAdapter::kContiguous)
      visit_tensors.push_back(t.get().contiguous());
    else
      visit_tensors.push_back(t.get());
  }

  AbiPacker packer;
  for (const auto& t : visit_tensors) packer.visitInput(t);
  auto out = at::empty(out_shape, materialized[0].get().options());
  packer.visitOutput(out);

  std::string identity = std::string("cat\0", 4) + elt + "\0" +
      std::to_string(dim) + "\0" +
      packer.bufTopology() + "\0" + packer.dimPattern();
  auto cache_key = c10::sha1(identity).str();

  auto* kernel = getOrCompile(cache_key, func_name, [&]() {
    AbiGenerator gen;
    for (const auto& t : visit_tensors) gen.visitInput(t);
    gen.visitOutput(out);
    return gen.generateModule(func_name, body);
  });

  invokeEnvelope(kernel, packer, visit_tensors, out, cache_key);
  return out;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void registerCompiledOps(torch::Library& m) {
  // --- Binary ops (functional, sorted) ---
  AddOp::register_impl(m);
  AddmmOp::register_impl(m);
  Atan2Op::register_impl(m);
  BitwiseAndOp::register_impl(m);
  BitwiseOrOp::register_impl(m);
  BitwiseXorOp::register_impl(m);
  DivOp::register_impl(m);
  FmodOp::register_impl(m);
  MaximumOp::register_impl(m);
  MinimumOp::register_impl(m);
  MmOp::register_impl(m);
  MulOp::register_impl(m);
  PowTensorOp::register_impl(m);
  RemainderOp::register_impl(m);
  SubOp::register_impl(m);

  // --- Binary ops (in-place, sorted) ---
  AddOp::register_inplace(m);
  Atan2Op::register_inplace(m);
  BitwiseAndOp::register_inplace(m);
  BitwiseOrOp::register_inplace(m);
  BitwiseXorOp::register_inplace(m);
  DivOp::register_inplace(m);
  FmodOp::register_inplace(m);
  MulOp::register_inplace(m);
  PowTensorOp::register_inplace(m);
  RemainderOp::register_inplace(m);
  SubOp::register_inplace(m);

  // --- Unary ops (functional, sorted) ---
  AbsOp::register_impl(m);
  BitwiseNotOp::register_impl(m);
  CeilOp::register_impl(m);
  CosOp::register_impl(m);
  EluOp::register_impl(m);
  ErfOp::register_impl(m);
  ExpOp::register_impl(m);
  FloorOp::register_impl(m);
  GeluOp::register_impl(m);
  HardtanhOp::register_impl(m);
  LeakyReluOp::register_impl(m);
  LogOp::register_impl(m);
  LogicalNotOp::register_impl(m);
  NegOp::register_impl(m);
  ReciprocalOp::register_impl(m);
  ReluOp::register_impl(m);
  RoundOp::register_impl(m);
  RsqrtOp::register_impl(m);
  SigmoidOp::register_impl(m);
  SignOp::register_impl(m);
  SiluOp::register_impl(m);
  SinOp::register_impl(m);
  SqrtOp::register_impl(m);
  TanhOp::register_impl(m);

  // --- Unary ops (in-place, sorted) ---
  AbsOp::register_inplace(m);
  BitwiseNotOp::register_inplace(m);
  CeilOp::register_inplace(m);
  CosOp::register_inplace(m);
  EluOp::register_inplace(m);
  ErfOp::register_inplace(m);
  ExpOp::register_inplace(m);
  FloorOp::register_inplace(m);
  GeluOp::register_inplace(m);
  HardtanhOp::register_inplace(m);
  LeakyReluOp::register_inplace(m);
  LogOp::register_inplace(m);
  LogicalNotOp::register_inplace(m);
  NegOp::register_inplace(m);
  ReciprocalOp::register_inplace(m);
  ReluOp::register_inplace(m);
  RoundOp::register_inplace(m);
  RsqrtOp::register_inplace(m);
  SigmoidOp::register_inplace(m);
  SignOp::register_inplace(m);
  SiluOp::register_inplace(m);
  SinOp::register_inplace(m);
  SqrtOp::register_inplace(m);
  TanhOp::register_inplace(m);

  // --- Scalar binary ops (functional, sorted) ---
  AddScalarOp::register_impl(m);
  DivScalarOp::register_impl(m);
  MulScalarOp::register_impl(m);
  PowScalarOp::register_impl(m);
  SubScalarOp::register_impl(m);

  // --- Scalar binary ops (in-place, sorted) ---
  AddScalarOp::register_inplace(m);
  DivScalarOp::register_inplace(m);
  MulScalarOp::register_inplace(m);
  PowScalarOp::register_inplace(m);
  SubScalarOp::register_inplace(m);

  // --- Comparison ops (no in-place — output dtype differs) ---
  EqScalarOp::register_impl(m);
  EqTensorOp::register_impl(m);
  GeScalarOp::register_impl(m);
  GeTensorOp::register_impl(m);
  GtScalarOp::register_impl(m);
  GtTensorOp::register_impl(m);
  LeScalarOp::register_impl(m);
  LeTensorOp::register_impl(m);
  LtScalarOp::register_impl(m);
  LtTensorOp::register_impl(m);
  NeScalarOp::register_impl(m);
  NeTensorOp::register_impl(m);

  // --- Reduction ops (no in-place — output shape differs) ---
  AmaxOp::register_impl(m);
  AminOp::register_impl(m);
  MeanOp::register_impl(m);
  ProdOp::register_impl(m);
  SumOp::register_impl(m);

  // --- Custom ops ---
  ArangeOp::register_impl(m);
  BmmOp::register_impl(m);
  CatOp::register_impl(m);
  EmbeddingOp::register_impl(m);
  GatherOp::register_impl(m);
  IndexSelectOp::register_impl(m);
  IndexTensorOp::register_impl(m);
  LogSoftmaxOp::register_impl(m);
  SoftmaxOp::register_impl(m);
  m.impl("index_put", &IndexPutOp::impl);
  m.impl("index_put_", &IndexPutOp::impl_inplace);
  m.impl("scatter.src", &ScatterSrcOp::impl);
  m.impl("scatter_.src", &ScatterSrcOp::impl_inplace);
  m.impl("scatter_add", &ScatterAddOp::impl);
  m.impl("scatter_add_", &ScatterAddOp::impl_inplace);
  TypeCastOp::register_impl(m);
  WhereOp::register_impl(m);
}

} // namespace at::pyre
