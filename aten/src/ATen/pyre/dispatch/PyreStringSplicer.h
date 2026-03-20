#pragma once

#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

namespace at::pyre {

// Replaces all occurrences of $$key$$ in tmpl with the corresponding value.
// No escaping needed — $$ has zero overlap with MLIR syntax.
// No control flow, no conditionals — intentionally simple.
//
// Unmatched placeholders are left as-is (makes debugging obvious).
// Unknown keys in vars that don't appear in tmpl are silently ignored.

namespace detail {
inline void spliceOne(std::string& result, std::string_view key,
                      std::string_view value) {
  std::string placeholder = "$$";
  placeholder.append(key.data(), key.size());
  placeholder += "$$";
  size_t pos = 0;
  while ((pos = result.find(placeholder, pos)) != std::string::npos) {
    result.replace(pos, placeholder.size(), value.data(), value.size());
    pos += value.size();
  }
}
} // namespace detail

// For inline brace-initialized pairs (used in appendFragment, emitStrideMath).
inline std::string pyreSplice(
    std::string_view tmpl,
    std::initializer_list<std::pair<std::string_view, std::string_view>> vars) {
  std::string result(tmpl);
  for (const auto& [key, value] : vars)
    detail::spliceOne(result, key, value);
  return result;
}

// For container of pair<string,string> (SubstPairs, vector — used by template ops).
template <typename PairRange>
inline std::string pyreSpliceRange(std::string_view tmpl, const PairRange& vars) {
  std::string result(tmpl);
  for (const auto& [key, value] : vars)
    detail::spliceOne(result, key, value);
  return result;
}

} // namespace at::pyre
