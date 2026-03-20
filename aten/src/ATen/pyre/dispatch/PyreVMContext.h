#pragma once

// VM dispatch context: load compiled kernels and resolve entry points.

#include <ATen/pyre/dispatch/PyreAbiConfig.h>
#include <ATen/pyre/dispatch/PyreKernelCompiler.h>
#include <c10/pyre/impl/PyreHelpers.h>

#include <memory>
#include <string>

namespace at::pyre {

struct CachedKernel {
  std::shared_ptr<CompilerOutput> vmfb;  // owns VMFB data (module refs it)
  c10::pyre::vm_module_ptr module;
  c10::pyre::vm_context_ptr context;
  iree_vm_function_t function{};
  iree_vm_function_t transients_size_fn{};
  bool has_transients_size = false;
};

// Load a compiled VMFB into a ready-to-dispatch CachedKernel.
CachedKernel loadKernel(
    std::shared_ptr<CompilerOutput> vmfb,
    const std::string& func_name,
    const AbiConfig& abi = AbiConfig::kTorchTyped);

} // namespace at::pyre
