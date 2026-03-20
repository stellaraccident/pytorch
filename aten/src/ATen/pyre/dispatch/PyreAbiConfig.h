#pragma once

// AbiConfig: compile-time selection of IREE calling convention.
//
// Two conventions coexist:
// - kTorchTyped: torch-mlir dialect, typed buffer views, $async entry points.
//   Used by arithmetic ops (add, mul, mm, etc.)
// - kNativeOpaque: native IREE dialect, opaque integer buffer views,
//   bare entry points. Used by data movement (copy, fill).
//
// AbiConfig controls compiler flags, function resolution, buffer view
// construction, and arg packing order. New configs are only added as
// named static constants — no public constructor.

#include <ATen/core/Tensor.h>
#include <c10/pyre/impl/PyreHelpers.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>

#include <string>

namespace at::pyre {

class AbiConfig {
 public:
  enum class Convention { kTorch, kNative };
  enum class DtypeMapping { kTyped, kOpaque };

  Convention convention() const { return convention_; }
  DtypeMapping dtypeMapping() const { return dtype_mapping_; }

  // Compiler flags for this ABI. Cached after first call.
  c10::ArrayRef<std::string> compilerFlags() const;

  // Resolve exported function name (appends $async for torch convention).
  std::string resolveFunction(const std::string& func_name) const;

  // Build buffer view with appropriate element type mapping.
  c10::pyre::hal_buffer_view_ptr buildView(const at::Tensor& t) const;

  // Whether loadKernel should use native_abi resolution.
  bool isNative() const { return convention_ == Convention::kNative; }

  // Torch-mlir convention with typed buffer views.
  static const AbiConfig kTorchTyped;

  // Native IREE convention with opaque integer buffer views.
  static const AbiConfig kNativeOpaque;

 private:
  AbiConfig(Convention c, DtypeMapping d)
      : convention_(c), dtype_mapping_(d) {}
  Convention convention_;
  DtypeMapping dtype_mapping_;
  mutable c10::SmallVector<std::string, 8> cached_flags_;
  mutable bool flags_cached_ = false;
};

} // namespace at::pyre
