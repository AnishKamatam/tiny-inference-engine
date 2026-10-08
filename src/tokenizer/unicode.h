#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace tie {

inline constexpr uint32_t kReplacementChar = 0xFFFD;

bool is_letter(uint32_t cpt);      // \p{L}
bool is_number(uint32_t cpt);      // \p{N}
bool is_whitespace(uint32_t cpt);  // \s, the Unicode White_Space property

// Decodes the UTF-8 sequence starting at `offset` (which must be < s.size()) and
// advances past it. Invalid or truncated sequences decode to U+FFFD and advance
// exactly one byte, so every byte of the input is consumed exactly once.
uint32_t utf8_decode(std::string_view s, size_t& offset);
std::string utf8_encode(uint32_t cpt);

}  // namespace tie
