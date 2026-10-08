#include "tokenizer/pretokenize.h"

#include <cstdint>

#include "tokenizer/unicode.h"

namespace tie {

namespace {

// The text as codepoints, with each codepoint's starting byte, so matches found
// in codepoint space can be sliced out of the original bytes.
struct Codepoints {
  std::vector<uint32_t> cpts;
  std::vector<size_t> starts;  // one extra entry: text.size()

  explicit Codepoints(std::string_view text) {
    for (size_t off = 0; off < text.size();) {
      starts.push_back(off);
      cpts.push_back(utf8_decode(text, off));
    }
    starts.push_back(text.size());
  }
  size_t size() const { return cpts.size(); }
};

bool is_newline(uint32_t c) { return c == '\r' || c == '\n'; }
// [^\s\p{L}\p{N}]: punctuation, symbols and everything else that is not a word character.
bool is_other(uint32_t c) { return !is_whitespace(c) && !is_letter(c) && !is_number(c); }
// ASCII-only lowercase; the contraction alternatives only contain ASCII letters.
uint32_t lower(uint32_t c) { return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c; }

// Length in codepoints of the match starting at `pos`, trying the pattern's
// alternatives in order, as a backtracking regex engine does. Always >= 1.
size_t match_at(const Codepoints& t, size_t pos) {
  const size_t n = t.size();
  const auto at = [&](size_t i) { return t.cpts[i]; };

  // (?i:'s|'t|'re|'ve|'m|'ll|'d)
  if (at(pos) == '\'' && pos + 1 < n) {
    const uint32_t a = lower(at(pos + 1));
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if (pos + 2 < n) {
      const uint32_t b = lower(at(pos + 2));
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    }
  }

  // [^\r\n\p{L}\p{N}]?\p{L}+
  size_t letters_from = pos;
  if (!is_letter(at(pos)) && !is_newline(at(pos)) && !is_number(at(pos)) && pos + 1 < n && is_letter(at(pos + 1))) {
    letters_from = pos + 1;
  }
  if (is_letter(at(letters_from))) {
    size_t end = letters_from;
    while (end < n && is_letter(at(end))) ++end;
    return end - pos;
  }

  // \p{N}
  if (is_number(at(pos))) return 1;

  // ' '?[^\s\p{L}\p{N}]+[\r\n]*
  const size_t other_from = at(pos) == ' ' ? pos + 1 : pos;
  if (other_from < n && is_other(at(other_from))) {
    size_t end = other_from;
    while (end < n && is_other(at(end))) ++end;
    while (end < n && is_newline(at(end))) ++end;
    return end - pos;
  }

  // The remaining alternatives all start with whitespace.
  size_t ws_end = pos;
  size_t after_last_newline = 0;
  while (ws_end < n && is_whitespace(at(ws_end))) {
    if (is_newline(at(ws_end))) after_last_newline = ws_end + 1;
    ++ws_end;
  }
  // \s*[\r\n]+ : backtracks to end just after the run's last newline.
  if (after_last_newline > 0) return after_last_newline - pos;
  const size_t run = ws_end - pos;
  // \s+(?!\S) : leave the last space to prefix the following word.
  if (run > 1 && ws_end < n) return run - 1;
  // \s+
  if (run > 0) return run;

  return 1;  // nothing matched: emit the codepoint on its own
}

}  // namespace

std::vector<std::string_view> pretokenize_qwen2(std::string_view text) {
  const Codepoints t(text);
  std::vector<std::string_view> pieces;
  for (size_t pos = 0; pos < t.size();) {
    const size_t end = pos + match_at(t, pos);
    pieces.push_back(text.substr(t.starts[pos], t.starts[end] - t.starts[pos]));
    pos = end;
  }
  return pieces;
}

}  // namespace tie
