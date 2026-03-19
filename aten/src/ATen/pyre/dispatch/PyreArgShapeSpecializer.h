#pragma once

// Shape analysis for kernel specialization.

#include <ATen/Tensor.h>
#include <ATen/pyre/dispatch/PyreArgAdapter.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/util/ArrayRef.h>

#include <string>
#include <vector>

namespace at::pyre {

struct SpecDecision {
  std::vector<ArgAdapter> arg_adapters;
};

class ArgShapeSpecializer {
 public:
  SpecDecision analyze(
      const std::string& op_name,
      c10::ScalarType dtype,
      c10::ArrayRef<at::Tensor> inputs,
      const c10::pyre::DeviceCapabilities& caps);
};

} // namespace at::pyre
