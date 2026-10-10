#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "kv/paged_kv_cache.h"
#include "sampling/sampler.h"

namespace tie {

enum class SeqState { Waiting, Prefilling, Decoding, Finished };
enum class FinishReason { None, Stop, Length, Error };

struct Sequence {
  int32_t id = 0;
  std::vector<int32_t> tokens;  // prompt followed by generated tokens
  int32_t prompt_len = 0;
  int32_t num_computed = 0;  // tokens whose K/V are already in the cache
  BlockTable block_table;
  SamplingParams params;
  std::mt19937_64 rng;
  SeqState state = SeqState::Waiting;
  FinishReason finish_reason = FinishReason::None;

  int32_t num_generated() const { return static_cast<int32_t>(tokens.size()) - prompt_len; }
  int32_t num_pending() const { return static_cast<int32_t>(tokens.size()) - num_computed; }
};

}  // namespace tie
