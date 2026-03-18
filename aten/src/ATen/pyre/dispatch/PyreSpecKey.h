#pragma once

// Spec key: uniquely identifies a compiled kernel specialization.
//
// Format: "{op}::{dtype}::{rank}({dim_specs})[{broadcast_mask}]"
//
// dim_specs per dim:
//   "?"        — fully dynamic
//   "1"        — known size-1 (broadcast candidate)
//   "%256"     — divisible by N
//   "128"      — static (opt-in, rarely used)
//
// Broadcast mask: arg-dim pairs where arg has 1-dim paired with non-1-dim
// in the other arg (binary ops only).
//
// See epic1_kernel_dispatch.md §4.1.

#include <c10/core/ScalarType.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace at::pyre {

// Per-dimension specialization info.
struct DimSpec {
  enum Kind { kDynamic, kBroadcast, kDivisible, kStatic };
  Kind kind = kDynamic;
  int64_t value = 0;  // divisor or static size

  std::string toString() const;
};

// Per-argument shape info for specialization.
struct ArgShapeInfo {
  int64_t rank = 0;
  std::vector<DimSpec> dims;
  std::vector<int64_t> sizes;  // concrete sizes from the tensor
};

// Broadcast mask entry: arg index + dim index.
struct BroadcastEntry {
  int arg;
  int dim;
  bool operator==(const BroadcastEntry& o) const {
    return arg == o.arg && dim == o.dim;
  }
};

class PyreSpecKey {
 public:
  PyreSpecKey(
      std::string op_name,
      c10::ScalarType dtype,
      int64_t rank,
      std::vector<DimSpec> dim_specs,
      std::vector<BroadcastEntry> broadcast_mask = {});

  // String representation (cache key format).
  const std::string& toString() const { return key_str_; }

  // Filesystem-safe version (replaces special chars).
  std::string toFilename() const;

  bool operator==(const PyreSpecKey& other) const {
    return key_str_ == other.key_str_;
  }

  struct Hash {
    size_t operator()(const PyreSpecKey& k) const {
      return std::hash<std::string>{}(k.key_str_);
    }
  };

  const std::string& opName() const { return op_name_; }
  c10::ScalarType dtype() const { return dtype_; }
  int64_t rank() const { return rank_; }
  const std::vector<DimSpec>& dimSpecs() const { return dim_specs_; }
  const std::vector<BroadcastEntry>& broadcastMask() const {
    return broadcast_mask_;
  }

 private:
  std::string op_name_;
  c10::ScalarType dtype_;
  int64_t rank_;
  std::vector<DimSpec> dim_specs_;
  std::vector<BroadcastEntry> broadcast_mask_;
  std::string key_str_;

  void buildKeyString();
};

// Map PyTorch ScalarType to MLIR type string.
const char* scalarTypeToMlir(c10::ScalarType dtype);

// Map PyTorch ScalarType to IREE element type string suffix (f, i).
bool pyreIsFloatingType(c10::ScalarType dtype);

} // namespace at::pyre
