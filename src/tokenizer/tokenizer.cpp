#include "tokenizer/tokenizer.h"

#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <limits>

#include "core/error.h"
#include "tokenizer/pretokenize.h"
#include "tokenizer/unicode.h"

namespace tie {

namespace {

// Bytes 0x21-0x7E, 0xA1-0xAC and 0xAE-0xFF stand for themselves; every other
// byte maps to 256 + n, in byte order.
std::array<std::string, 256> build_byte_alphabet() {
  std::array<std::string, 256> out;
  std::array<bool, 256> printable{};
  for (int b = 0x21; b <= 0x7E; ++b) printable[static_cast<size_t>(b)] = true;
  for (int b = 0xA1; b <= 0xAC; ++b) printable[static_cast<size_t>(b)] = true;
  for (int b = 0xAE; b <= 0xFF; ++b) printable[static_cast<size_t>(b)] = true;
  uint32_t next = 256;
  for (int b = 0; b < 256; ++b) {
    const auto i = static_cast<size_t>(b);
    out[i] = utf8_encode(printable[i] ? static_cast<uint32_t>(b) : next++);
  }
  return out;
}

// Length of the longest prefix of `s` that ends on a complete UTF-8 character.
size_t complete_utf8_prefix(std::string_view s) {
  size_t i = s.size();
  for (int back = 0; i > 0 && back < 4; ++back) {
    --i;
    const auto b = static_cast<uint8_t>(s[i]);
    if ((b & 0xC0) == 0x80) continue;  // continuation byte: keep walking back to the lead
    const size_t need = b < 0x80 ? 1 : (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : (b & 0xF8) == 0xF0 ? 4 : 1;
    return s.size() - i >= need ? s.size() : i;
  }
  return s.size();  // only continuation bytes: not valid UTF-8, pass through
}

}  // namespace

const std::array<std::string, 256>& gpt2_byte_alphabet() {
  static const std::array<std::string, 256> alphabet = build_byte_alphabet();
  return alphabet;
}

std::string nfc_normalize(std::string_view text) {
  if (std::all_of(text.begin(), text.end(), [](char c) { return static_cast<unsigned char>(c) < 0x80; })) {
    return std::string(text);  // ASCII is already NFC
  }
  CFStringRef src = CFStringCreateWithBytes(nullptr, reinterpret_cast<const UInt8*>(text.data()),
                                            static_cast<CFIndex>(text.size()), kCFStringEncodingUTF8, false);
  if (src == nullptr) return std::string(text);  // invalid UTF-8
  CFMutableStringRef s = CFStringCreateMutableCopy(nullptr, 0, src);
  CFRelease(src);
  CFStringNormalize(s, kCFStringNormalizationFormC);

  const CFIndex len = CFStringGetLength(s);
  const CFIndex max = CFStringGetMaximumSizeForEncoding(len, kCFStringEncodingUTF8);
  std::string out(static_cast<size_t>(max), '\0');
  CFIndex used = 0;
  CFStringGetBytes(s, CFRangeMake(0, len), kCFStringEncodingUTF8, 0, false, reinterpret_cast<UInt8*>(out.data()), max,
                   &used);
  CFRelease(s);
  out.resize(static_cast<size_t>(used));
  return out;
}

Tokenizer::Tokenizer(const TokenizerData& data) {
  const auto& alphabet = gpt2_byte_alphabet();
  std::unordered_map<uint32_t, char> alphabet_to_byte;
  for (size_t b = 0; b < 256; ++b) {
    size_t off = 0;
    alphabet_to_byte.emplace(utf8_decode(alphabet[b], off), static_cast<char>(b));
  }

  std::vector<bool> is_special(data.tokens.size(), false);
  for (int32_t id : data.special_ids) {
    if (id < 0 || static_cast<size_t>(id) >= data.tokens.size()) fail<LoadError>("special token id {} out of range", id);
    if (data.tokens[static_cast<size_t>(id)].empty()) fail<LoadError>("special token id {} has empty text", id);
    is_special[static_cast<size_t>(id)] = true;
    specials_.emplace_back(data.tokens[static_cast<size_t>(id)], id);
  }
  std::sort(specials_.begin(), specials_.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

  token_bytes_.resize(data.tokens.size());
  token_ids_.reserve(data.tokens.size());
  for (size_t id = 0; id < data.tokens.size(); ++id) {
    const std::string& token = data.tokens[id];
    if (token.empty()) continue;  // unused id
    token_ids_.emplace(token, static_cast<int32_t>(id));
    if (is_special[id]) {
      token_bytes_[id] = token;
      continue;
    }
    for (size_t off = 0; off < token.size();) {
      const uint32_t cpt = utf8_decode(token, off);
      const auto it = alphabet_to_byte.find(cpt);
      token_bytes_[id] += it != alphabet_to_byte.end() ? std::string(1, it->second) : utf8_encode(cpt);
    }
  }

  merge_ranks_.reserve(data.merges.size());
  for (size_t rank = 0; rank < data.merges.size(); ++rank) {
    const auto& [left, right] = data.merges[rank];
    merge_ranks_.emplace(left + ' ' + right, static_cast<int32_t>(rank));
  }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
  std::vector<int32_t> ids;
  size_t pos = 0;
  while (pos < text.size()) {
    // Earliest special token at or after pos; the longest one wins at equal positions.
    size_t best_pos = std::string_view::npos;
    const std::pair<std::string, int32_t>* best = nullptr;
    for (const auto& special : specials_) {
      const size_t found = text.find(special.first, pos);
      if (found < best_pos) {
        best_pos = found;
        best = &special;
      }
    }
    const size_t segment_end = best != nullptr ? best_pos : text.size();
    if (segment_end > pos) encode_ordinary(text.substr(pos, segment_end - pos), ids);
    if (best == nullptr) break;
    ids.push_back(best->second);
    pos = best_pos + best->first.size();
  }
  return ids;
}

void Tokenizer::encode_ordinary(std::string_view text, std::vector<int32_t>& out) const {
  const std::string normalized = nfc_normalize(text);
  for (std::string_view piece : pretokenize_qwen2(normalized)) encode_piece(piece, out);
}

void Tokenizer::encode_piece(std::string_view piece, std::vector<int32_t>& out) const {
  const auto& alphabet = gpt2_byte_alphabet();
  std::vector<std::string> symbols;
  symbols.reserve(piece.size());
  for (char c : piece) symbols.push_back(alphabet[static_cast<uint8_t>(c)]);

  // No whole-word shortcut: Qwen's tokenizer.json sets ignore_merges=false, so a
  // piece is always built by merges even when it is itself a vocabulary entry.
  // Repeatedly merge the adjacent pair with the lowest rank (all of its
  // occurrences, left to right) until no adjacent pair has a merge rule.
  std::string key;
  while (symbols.size() > 1) {
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    size_t best = 0;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      key.assign(symbols[i]).append(" ").append(symbols[i + 1]);
      if (const auto it = merge_ranks_.find(key); it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best = i;
      }
    }
    if (best_rank == std::numeric_limits<int32_t>::max()) break;

    const std::string left = symbols[best];
    const std::string right = symbols[best + 1];
    std::vector<std::string> merged;
    merged.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size();) {
      if (i + 1 < symbols.size() && symbols[i] == left && symbols[i + 1] == right) {
        merged.push_back(left + right);
        i += 2;
      } else {
        merged.push_back(std::move(symbols[i]));
        ++i;
      }
    }
    symbols = std::move(merged);
  }

  for (const std::string& s : symbols) {
    const auto it = token_ids_.find(s);
    if (it == token_ids_.end()) fail<LoadError>("BPE produced '{}', which is not in the vocabulary", s);
    out.push_back(it->second);
  }
}

std::string Tokenizer::decode(std::span<const int32_t> ids) const {
  std::string out;
  for (int32_t id : ids) out += token_bytes(id);
  return out;
}

const std::string& Tokenizer::token_bytes(int32_t id) const {
  if (id < 0 || id >= vocab_size()) fail<InvalidArgument>("token id {} outside vocabulary of {}", id, vocab_size());
  return token_bytes_[static_cast<size_t>(id)];
}

std::optional<int32_t> Tokenizer::find_token(std::string_view token) const {
  const auto it = token_ids_.find(token);
  if (it == token_ids_.end()) return std::nullopt;
  return it->second;
}

std::string StreamDecoder::push(int32_t id) {
  pending_ += tokenizer_.token_bytes(id);
  const size_t complete = complete_utf8_prefix(pending_);
  std::string out = pending_.substr(0, complete);
  pending_.erase(0, complete);
  return out;
}

std::string StreamDecoder::flush() { return std::exchange(pending_, {}); }

}  // namespace tie
