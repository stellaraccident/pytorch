#pragma once

// VM dispatch context management: load VMFB modules, create VM contexts,
// and resolve entry points for kernel invocation.
//
// CachedKernel holds a loaded module, a VM context linking it with the
// HAL module, and the resolved $async entry point function.
//
// See epic1_kernel_dispatch.md §4.8 tier 1 and §4.10.

#include <c10/pyre/impl/PyreHelpers.h>

#include <cstdint>
#include <string>
#include <vector>

namespace at::pyre {

struct CachedKernel {
  c10::pyre::vm_module_ptr module;
  c10::pyre::vm_context_ptr context;
  iree_vm_function_t function;  // resolved entry point
};

// Load a VMFB byte buffer into a VM module and create a ready-to-dispatch
// CachedKernel with context and resolved entry point.
//
// func_name is the base function name (without $async suffix).
// The context links the kernel module with the HAL module from the VM instance.
CachedKernel loadKernel(
    const std::vector<uint8_t>& vmfb,
    const std::string& func_name);

} // namespace at::pyre
