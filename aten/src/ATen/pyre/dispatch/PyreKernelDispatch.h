#pragma once

// Kernel dispatch: invoke compiled kernels with proper arg packing.

#include <ATen/Tensor.h>
#include <ATen/pyre/dispatch/PyreVMContext.h>
#include <c10/pyre/impl/PyreStream.h>
#include <c10/util/ArrayRef.h>

namespace at::pyre {

class PyreKernelDispatch {
 public:
  // Invoke a compiled kernel (torch-mlir calling convention).
  // Args: [output, inputs..., workspace, wait_fence, signal_fence] → void.
  // If opaque_types is true, buffer views use integer types matching
  // the element bitwidth (for data movement kernels using si8/si16/si32/si64).
  static void invoke(
      CachedKernel* kernel,
      c10::ArrayRef<at::Tensor> inputs,
      at::Tensor& output,
      c10::pyre::PyreStreamContext& stream_ctx,
      bool opaque_types = false);

  // Invoke a compiled kernel (native IREE calling convention).
  // Args: [inputs..., wait_fence, signal_fence] → [outputs...].
  // Buffer views always use opaque integer types by bitwidth.
  static void invokeNative(
      CachedKernel* kernel,
      c10::ArrayRef<at::Tensor> inputs,
      at::Tensor& output,
      c10::pyre::PyreStreamContext& stream_ctx);
};

} // namespace at::pyre
