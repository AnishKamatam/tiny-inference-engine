#pragma once

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tie {

// One request of a benchmark workload.
struct BenchRequest {
  std::vector<int32_t> prompt;
  int32_t max_tokens = 0;
  double arrival = 0.0;  // seconds after the run starts
};

struct LengthRange {
  int32_t lo = 1;
  int32_t hi = 1;
};

// One line of a JSONL workload: {"prompt": "...", "max_tokens": N, "chat": bool}.
struct PromptLine {
  std::string text;
  std::optional<int32_t> max_tokens;
  bool chat = false;
};

// Blank lines are skipped. Throws InvalidArgument naming `source:line` for a malformed line,
// and when there are no prompts at all.
std::vector<PromptLine> parse_prompts_jsonl(std::istream& in, std::string_view source);

// `n` requests of random token ids in [0, vocab_size) with lengths drawn uniformly from the ranges.
std::vector<BenchRequest> synthetic_requests(int32_t n, LengthRange prompt_len, LengthRange output_len,
                                             int32_t vocab_size, uint64_t seed);

// Poisson arrivals at `rate` requests per second, in order; an infinite rate puts every request at 0.
void assign_arrivals(std::vector<BenchRequest>& requests, double rate, uint64_t seed);

}  // namespace tie
