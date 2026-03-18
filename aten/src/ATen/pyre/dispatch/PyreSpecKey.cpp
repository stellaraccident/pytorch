#include <ATen/pyre/dispatch/PyreSpecKey.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <sstream>

namespace at::pyre {

std::string DimSpec::toString() const {
  switch (kind) {
    case kDynamic:
      return "?";
    case kBroadcast:
      return "1";
    case kDivisible:
      return "%" + std::to_string(value);
    case kStatic:
      return std::to_string(value);
  }
  return "?";
}

PyreSpecKey::PyreSpecKey(
    std::string op_name,
    c10::ScalarType dtype,
    int64_t rank,
    std::vector<DimSpec> dim_specs,
    std::vector<BroadcastEntry> broadcast_mask)
    : op_name_(std::move(op_name)),
      dtype_(dtype),
      rank_(rank),
      dim_specs_(std::move(dim_specs)),
      broadcast_mask_(std::move(broadcast_mask)) {
  buildKeyString();
}

void PyreSpecKey::buildKeyString() {
  std::ostringstream ss;
  ss << op_name_ << "::" << scalarTypeToMlir(dtype_) << "::" << rank_;

  ss << "(";
  for (size_t i = 0; i < dim_specs_.size(); ++i) {
    if (i > 0) ss << ",";
    ss << dim_specs_[i].toString();
  }
  ss << ")";

  if (!broadcast_mask_.empty()) {
    ss << "[";
    for (const auto& e : broadcast_mask_) {
      ss << e.arg << e.dim;
    }
    ss << "]";
  }

  key_str_ = ss.str();
}

std::string PyreSpecKey::toFilename() const {
  std::string result = key_str_;
  // Replace characters not safe in filenames.
  for (char& c : result) {
    if (c == ':' || c == '(' || c == ')' || c == '[' || c == ']' ||
        c == ',' || c == '%') {
      c = '_';
    }
  }
  return result;
}

const char* scalarTypeToMlir(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float:
      return "f32";
    case c10::ScalarType::Double:
      return "f64";
    case c10::ScalarType::Half:
      return "f16";
    case c10::ScalarType::BFloat16:
      return "bf16";
    case c10::ScalarType::Int:
      return "i32";
    case c10::ScalarType::Long:
      return "i64";
    case c10::ScalarType::Short:
      return "i16";
    case c10::ScalarType::Byte:
      return "i8";
    case c10::ScalarType::Char:
      return "si8";
    case c10::ScalarType::Bool:
      return "i1";
    default:
      TORCH_CHECK(
          false, "pyre: unsupported dtype for kernel compilation: ",
          c10::toString(dtype));
  }
}

bool pyreIsFloatingType(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float:
    case c10::ScalarType::Double:
    case c10::ScalarType::Half:
    case c10::ScalarType::BFloat16:
      return true;
    default:
      return false;
  }
}

} // namespace at::pyre
