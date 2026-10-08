#pragma once

#include <string_view>
#include <vector>

namespace tie {

// Splits text into the pieces BPE runs on, exactly as Qwen2's pre-tokenizer regex does:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Pieces are views into `text` and concatenate back to it exactly.
std::vector<std::string_view> pretokenize_qwen2(std::string_view text);

}  // namespace tie
