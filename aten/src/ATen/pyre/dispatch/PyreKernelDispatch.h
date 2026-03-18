#pragma once

// PyreKernelDispatch: entry point for compiled kernel dispatch.
//
// Orchestrates: shape analysis → cache lookup → template expansion →
// compile → load → async VM invocation with coarse-fences.
//
// invokeAsync() builds wait/signal fences from tensor timelines
// and invokes the kernel's $async entry point. The local-task CPU
// driver is fully async — this is not a GPU optimization.
//
// See epic1_kernel_dispatch.md §4.10 and §4.12.

#include <ATen/Tensor.h>
#include <ATen/pyre/dispatch/PyreArgShapeSpecializer.h>
#include <ATen/pyre/dispatch/PyreKernelCache.h>
#include <ATen/pyre/dispatch/PyreVMContext.h>
#include <c10/pyre/impl/PyreStream.h>
#include <c10/util/ArrayRef.h>

#include <string>

namespace at::pyre {

class PyreKernelDispatch {
 public:
  // Async dispatch: build fences, invoke VM, record timeline bookkeeping.
  static void invokeAsync(
      CachedKernel* kernel,
      c10::ArrayRef<at::Tensor> inputs,
      at::Tensor& output,
      c10::pyre::PyreStreamContext& stream_ctx);

  // Full dispatch path: analyze → cache → compile → allocate → invoke.
  static at::Tensor dispatch(
      const std::string& op_name,
      const std::string& template_key,
      c10::ArrayRef<at::Tensor> inputs,
      const at::TensorOptions& output_opts);
};

} // namespace at::pyre
