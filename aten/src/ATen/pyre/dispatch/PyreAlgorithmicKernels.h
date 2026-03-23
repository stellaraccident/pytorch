#pragma once

// Fragment-based algorithmic kernels: ops whose MLIR is generated
// procedurally (variable structure) rather than from fixed templates.

#include <ATen/pyre/dispatch/PyreKernelAsmBuilder.h>

namespace at::pyre {

PyreKernelAsmFragments& catFragments();
PyreKernelAsmFragments& indexFragments();
PyreKernelAsmFragments& indexPutFragments();
PyreKernelAsmFragments& indexPutInplaceFragments();

} // namespace at::pyre
