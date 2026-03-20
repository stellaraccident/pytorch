#pragma once

// Kernel dispatch: invoke compiled kernels with proper arg packing.

#include <ATen/Tensor.h>
#include <ATen/pyre/dispatch/PyreAbiConfig.h>
#include <ATen/pyre/dispatch/PyreVMContext.h>
#include <c10/pyre/impl/PyreStream.h>
#include <c10/util/ArrayRef.h>

namespace at::pyre {

class PyreKernelDispatch {
 public:
  // Invoke a compiled kernel. AbiConfig controls arg packing order,
  // buffer view construction, and function resolution.
  static void invoke(
      CachedKernel* kernel,
      c10::ArrayRef<at::Tensor> inputs,
      at::Tensor& output,
      c10::pyre::PyreStreamContext& stream_ctx,
      const AbiConfig& abi = AbiConfig::kTorchTyped);
};

} // namespace at::pyre
