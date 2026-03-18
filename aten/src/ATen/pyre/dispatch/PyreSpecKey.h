#pragma once

// Spec key: uniquely identifies a compiled kernel specialization.
// Format: "{op}::{dtype}::{rank}({dim_specs})[{broadcast_mask}]"

#include <ATen/pyre/dispatch/PyreTypeMapping.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace at::pyre {

struct DimSpec {
  enum Kind { kDynamic, kBroadcast, kDivisible, kStatic };
  Kind kind = kDynamic;
  int64_t value = 0;
  std::string toString() const;
};

struct ArgShapeInfo {
  int64_t rank = 0;
  std::vector<DimSpec> dims;
  std::vector<int64_t> sizes;
};

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
      std::string op_name, c10::ScalarType dtype, int64_t rank,
      std::vector<DimSpec> dim_specs,
      std::vector<BroadcastEntry> broadcast_mask = {});

  const std::string& toString() const { return key_str_; }
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

 private:
  std::string op_name_;
  c10::ScalarType dtype_;
  int64_t rank_;
  std::vector<DimSpec> dim_specs_;
  std::vector<BroadcastEntry> broadcast_mask_;
  std::string key_str_;

  void buildKeyString();
};

} // namespace at::pyre
