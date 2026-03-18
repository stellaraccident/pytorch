#pragma once

// Shape analysis for kernel specialization.

#include <ATen/Tensor.h>
#include <ATen/pyre/dispatch/PyreArgAdapter.h>
#include <ATen/pyre/dispatch/PyreSpecKey.h>
#include <c10/pyre/impl/PyreDevice.h>
#include <c10/util/ArrayRef.h>

#include <string>
#include <vector>

namespace at::pyre {

struct SpecDecision {
  PyreSpecKey spec_key;
  std::vector<ArgShapeInfo> arg_shapes;
  std::vector<ArgAdapter> arg_adapters;
};

class ArgShapeSpecializer {
 public:
  SpecDecision analyze(
      const std::string& op_name,
      c10::ScalarType dtype,
      c10::ArrayRef<at::Tensor> inputs,
      const c10::pyre::DeviceCapabilities& caps);

 private:
  static std::vector<BroadcastEntry> detectBroadcast(
      const at::Tensor& lhs, const at::Tensor& rhs);
  static std::vector<DimSpec> buildDimSpecs(
      const at::Tensor& tensor);
};

} // namespace at::pyre
