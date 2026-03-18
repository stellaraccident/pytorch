#pragma once

// ArgShapeSpecializer: analyzes concrete tensor metadata and produces
// a prioritized specialization decision.
//
// Epic 1 scope: broadcast detection + contiguous-or-bail.
// Infrastructure for divisibility and stride adapters is in place.
// See epic1_kernel_dispatch.md §4.1.

#include <ATen/Tensor.h>
#include <ATen/pyre/dispatch/PyreArgAdapter.h>
#include <ATen/pyre/dispatch/PyreSpecKey.h>
#include <c10/pyre/impl/PyreDeviceCapabilities.h>
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
  // Analyze inputs and produce a specialization decision.
  SpecDecision analyze(
      const std::string& op_name,
      c10::ScalarType dtype,
      c10::ArrayRef<at::Tensor> inputs,
      const c10::pyre::PyreDeviceCapabilities& device_caps);

 private:
  // Detect broadcast dimensions between two tensors (binary ops).
  static std::vector<BroadcastEntry> detectBroadcast(
      const at::Tensor& lhs, const at::Tensor& rhs);

  // Build dim specs for a single tensor.
  static std::vector<DimSpec> buildDimSpecs(
      const at::Tensor& tensor,
      const c10::pyre::PyreDeviceCapabilities& device_caps,
      const std::string& op_name);
};

} // namespace at::pyre
