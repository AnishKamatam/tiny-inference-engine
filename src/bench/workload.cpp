#include "bench/workload.h"

#include <cmath>
#include <istream>
#include <nlohmann/json.hpp>
#include <random>

#include "core/error.h"

namespace tie {

std::vector<PromptLine> parse_prompts_jsonl(std::istream& in, std::string_view source) {
  std::vector<PromptLine> lines;
  std::string text;
  for (int number = 1; std::getline(in, text); ++number) {
    if (text.find_first_not_of(" \t\r") == std::string::npos) continue;
    nlohmann::json j;
    try {
      j = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& e) {
      fail<InvalidArgument>("{}:{}: not valid JSON: {}", source, number, e.what());
    }
    if (!j.is_object() || !j.contains("prompt") || !j["prompt"].is_string()) {
      fail<InvalidArgument>("{}:{}: expected an object with a string \"prompt\"", source, number);
    }
    PromptLine line;
    line.text = j["prompt"].get<std::string>();
    if (line.text.empty()) fail<InvalidArgument>("{}:{}: \"prompt\" must not be empty", source, number);
    if (j.contains("max_tokens")) {
      if (!j["max_tokens"].is_number_integer() || j["max_tokens"].get<int64_t>() < 1 ||
          j["max_tokens"].get<int64_t>() > INT32_MAX) {
        fail<InvalidArgument>("{}:{}: \"max_tokens\" must be a positive integer", source, number);
      }
      line.max_tokens = j["max_tokens"].get<int32_t>();
    }
    if (j.contains("chat")) {
      if (!j["chat"].is_boolean()) fail<InvalidArgument>("{}:{}: \"chat\" must be true or false", source, number);
      line.chat = j["chat"].get<bool>();
    }
    lines.push_back(std::move(line));
  }
  if (lines.empty()) fail<InvalidArgument>("{}: no prompts", source);
  return lines;
}

std::vector<BenchRequest> synthetic_requests(int32_t n, LengthRange prompt_len, LengthRange output_len,
                                             int32_t vocab_size, uint64_t seed) {
  if (n < 1) fail<InvalidArgument>("synthetic request count must be at least 1, got {}", n);
  if (vocab_size < 1) fail<InvalidArgument>("vocab_size must be positive, got {}", vocab_size);
  if (prompt_len.lo < 1 || prompt_len.hi < prompt_len.lo) {
    fail<InvalidArgument>("prompt length range needs 1 <= lo <= hi, got {}:{}", prompt_len.lo, prompt_len.hi);
  }
  if (output_len.lo < 1 || output_len.hi < output_len.lo) {
    fail<InvalidArgument>("output length range needs 1 <= lo <= hi, got {}:{}", output_len.lo, output_len.hi);
  }
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int32_t> token(0, vocab_size - 1);
  std::uniform_int_distribution<int32_t> plen(prompt_len.lo, prompt_len.hi);
  std::uniform_int_distribution<int32_t> olen(output_len.lo, output_len.hi);
  std::vector<BenchRequest> requests(static_cast<size_t>(n));
  for (BenchRequest& r : requests) {
    r.prompt.resize(static_cast<size_t>(plen(rng)));
    for (int32_t& t : r.prompt) t = token(rng);
    r.max_tokens = olen(rng);
  }
  return requests;
}

void assign_arrivals(std::vector<BenchRequest>& requests, double rate, uint64_t seed) {
  if (!(rate > 0)) fail<InvalidArgument>("arrival rate must be positive, got {}", rate);
  if (std::isinf(rate)) {
    for (BenchRequest& r : requests) r.arrival = 0.0;
    return;
  }
  std::mt19937_64 rng(seed);
  std::exponential_distribution<double> gap(rate);
  double t = 0.0;
  for (BenchRequest& r : requests) {
    t += gap(rng);
    r.arrival = t;
  }
}

}  // namespace tie
