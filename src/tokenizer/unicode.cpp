#include "tokenizer/unicode.h"

#include <algorithm>
#include <span>

#include "tokenizer/unicode_tables.h"

namespace tie {

namespace {

bool in_ranges(std::span<const CodepointRange> ranges, uint32_t cpt) {
  // First range whose last codepoint is >= cpt; cpt is inside it iff first <= cpt.
  const auto it = std::lower_bound(ranges.begin(), ranges.end(), cpt,
                                   [](const CodepointRange& r, uint32_t c) { return r.last < c; });
  return it != ranges.end() && it->first <= cpt;
}

bool is_continuation(std::string_view s, size_t i) {
  return i < s.size() && (static_cast<uint8_t>(s[i]) & 0xC0) == 0x80;
}

}  // namespace

bool is_letter(uint32_t cpt) {
  if (cpt < 0x80) return (cpt | 0x20) >= 'a' && (cpt | 0x20) <= 'z';
  return in_ranges({kLetterRanges, kLetterRangeCount}, cpt);
}

bool is_number(uint32_t cpt) {
  if (cpt < 0x80) return cpt >= '0' && cpt <= '9';
  return in_ranges({kNumberRanges, kNumberRangeCount}, cpt);
}

bool is_whitespace(uint32_t cpt) { return in_ranges({kWhitespaceRanges, kWhitespaceRangeCount}, cpt); }

uint32_t utf8_decode(std::string_view s, size_t& offset) {
  const auto byte = [&](size_t i) { return static_cast<uint32_t>(static_cast<uint8_t>(s[i])); };
  const uint32_t b0 = byte(offset);
  if (b0 < 0x80) {
    offset += 1;
    return b0;
  }
  if ((b0 & 0xE0) == 0xC0 && is_continuation(s, offset + 1)) {
    const uint32_t cpt = ((b0 & 0x1F) << 6) | (byte(offset + 1) & 0x3F);
    offset += 2;
    return cpt;
  }
  if ((b0 & 0xF0) == 0xE0 && is_continuation(s, offset + 1) && is_continuation(s, offset + 2)) {
    const uint32_t cpt = ((b0 & 0x0F) << 12) | ((byte(offset + 1) & 0x3F) << 6) | (byte(offset + 2) & 0x3F);
    offset += 3;
    return cpt;
  }
  if ((b0 & 0xF8) == 0xF0 && is_continuation(s, offset + 1) && is_continuation(s, offset + 2) &&
      is_continuation(s, offset + 3)) {
    const uint32_t cpt = ((b0 & 0x07) << 18) | ((byte(offset + 1) & 0x3F) << 12) | ((byte(offset + 2) & 0x3F) << 6) |
                         (byte(offset + 3) & 0x3F);
    offset += 4;
    return cpt;
  }
  offset += 1;
  return kReplacementChar;
}

std::string utf8_encode(uint32_t cpt) {
  std::string out;
  if (cpt <= 0x7F) {
    out.push_back(static_cast<char>(cpt));
  } else if (cpt <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (cpt >> 6)));
    out.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  } else if (cpt <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (cpt >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | ((cpt >> 18) & 0x07)));
    out.push_back(static_cast<char>(0x80 | ((cpt >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  }
  return out;
}

}  // namespace tie
