#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace at::pyre {

// Replaces all occurrences of $$key$$ in tmpl with the corresponding value
// from vars. No escaping needed — $$ has zero overlap with MLIR syntax.
// No control flow, no conditionals — intentionally simple.
//
// Unmatched placeholders are left as-is (makes debugging obvious).
// Unknown keys in vars that don't appear in tmpl are silently ignored.
inline std::string pyreSplice(
    std::string_view tmpl,
    const std::vector<std::pair<std::string, std::string>>& vars) {
  std::string result(tmpl);
  for (const auto& [key, value] : vars) {
    std::string placeholder = "$$" + key + "$$";
    size_t pos = 0;
    while ((pos = result.find(placeholder, pos)) != std::string::npos) {
      result.replace(pos, placeholder.size(), value);
      pos += value.size();
    }
  }
  return result;
}

} // namespace at::pyre
