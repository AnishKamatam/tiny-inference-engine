#pragma once

#include <cstddef>
#include <cstdint>

namespace tie {

// Inclusive codepoint range. Tables are sorted and non-overlapping.
struct CodepointRange {
  uint32_t first;
  uint32_t last;
};

// Defined in the generated unicode_tables.cpp (tools/gen_unicode_tables.py).
extern const CodepointRange kLetterRanges[];      // \p{L}
extern const size_t kLetterRangeCount;
extern const CodepointRange kNumberRanges[];      // \p{N}
extern const size_t kNumberRangeCount;
extern const CodepointRange kWhitespaceRanges[];  // \s (White_Space)
extern const size_t kWhitespaceRangeCount;

}  // namespace tie
