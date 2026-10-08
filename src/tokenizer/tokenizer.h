#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "model/loader.h"

namespace tie {

// GPT-2's reversible byte -> printable-codepoint map that byte-level BPE vocabularies are written in.
const std::array<std::string, 256>& gpt2_byte_alphabet();

// NFC-normalizes UTF-8 text. Invalid UTF-8 is returned unchanged.
std::string nfc_normalize(std::string_view text);

// Qwen's byte-level BPE. Special tokens are matched verbatim first; the text in
// between is NFC-normalized, split by the Qwen2 pre-tokenizer, mapped to the
// GPT-2 byte alphabet and merged by rank.
class Tokenizer {
 public:
  explicit Tokenizer(const TokenizerData& data);

  std::vector<int32_t> encode(std::string_view text) const;
  std::string decode(std::span<const int32_t> ids) const;

  // Raw bytes of one token; special tokens decode to their literal text.
  const std::string& token_bytes(int32_t id) const;
  std::optional<int32_t> find_token(std::string_view token) const;
  int32_t vocab_size() const { return static_cast<int32_t>(token_bytes_.size()); }

 private:
  struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const { return std::hash<std::string_view>{}(s); }
  };
  using StringMap = std::unordered_map<std::string, int32_t, StringHash, std::equal_to<>>;

  void encode_ordinary(std::string_view text, std::vector<int32_t>& out) const;
  void encode_piece(std::string_view piece, std::vector<int32_t>& out) const;

  StringMap token_ids_;    // vocabulary entry (byte alphabet) -> id
  StringMap merge_ranks_;  // "left right" -> rank; tokens never contain a literal space
  std::vector<std::pair<std::string, int32_t>> specials_;  // longest first
  std::vector<std::string> token_bytes_;                    // id -> raw bytes
};

// Incremental decoding for streamed output: holds bytes back until they form
// complete UTF-8 characters, so a character split across tokens is never
// printed half-written.
class StreamDecoder {
 public:
  explicit StreamDecoder(const Tokenizer& tokenizer) : tokenizer_(tokenizer) {}

  std::string push(int32_t id);  // text completed by this token; may be empty
  std::string flush();           // everything still held back

 private:
  const Tokenizer& tokenizer_;
  std::string pending_;
};

}  // namespace tie
