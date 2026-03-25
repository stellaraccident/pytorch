#pragma once

// Free-standing utilities shared across pyre dispatch infrastructure.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

namespace at::pyre {

// Format a double as an MLIR float literal. Special values (inf, nan)
// use hex notation since bare "inf"/"nan" aren't valid MLIR syntax.
inline std::string mlirFloatLiteral(double value) {
  if (std::isfinite(value)) {
    std::ostringstream ss;
    ss << std::fixed << value;
    return ss.str();
  }
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  std::ostringstream ss;
  ss << "0x" << std::hex << std::uppercase << bits;
  return ss.str();
}

inline bool isIntegralScalar(double value) {
  return std::isfinite(value) && value == static_cast<int64_t>(value);
}

} // namespace at::pyre
