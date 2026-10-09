#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace tie {

struct SamplingParams {
  float temperature = 0.0f;  // 0 = greedy
  int32_t top_k = 0;         // 0 = disabled
  float top_p = 1.0f;        // 1 = disabled
  uint64_t seed = 0;
  int32_t max_tokens = 256;  // generated tokens, including a final stop token
};

// Throws InvalidArgument naming the first impossible field.
void validate(const SamplingParams& params);

class Sampler {
 public:
  // Greedy at temperature 0 (or so small that 1/temperature overflows). Otherwise scales by 1/temperature, keeps the top_k
  // most likely tokens, then the smallest prefix whose probability reaches
  // top_p, and draws from what is left using `rng`.
  //
  // Throws InvalidArgument on empty logits, any NaN logit, or logits that are all -inf.
  // A +inf logit (including one produced by scaling) is treated as a certain token:
  // the lowest-id +inf token is returned without consulting `rng`.
  int32_t sample(std::span<const float> logits, const SamplingParams& params, std::mt19937_64& rng);

 private:
  std::vector<std::pair<float, int32_t>> candidates_;  // scratch reused across calls
};

}  // namespace tie
