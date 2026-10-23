#pragma once

#include <cstdint>

#include "model/qwen3.h"

namespace tie {

// The shape of one forward step, shared by the model (activation sizes), the scheduler
// (what a step may hold) and the engine (what it checks every step). Built once and
// passed to all three so they cannot disagree.
struct ServingConfig {
  int32_t max_tokens_per_step = 512;  // tokens in one step, prefill and decode together
  int32_t max_seqs = 64;              // sequences in one step
};

// Throws InvalidArgument unless both are positive and max_seqs <= max_tokens_per_step:
// every running sequence must be able to decode one token in the same step.
void validate(const ServingConfig& serving);

// The model limits this serving shape needs: one logit row per sequence.
ModelLimits model_limits(const ServingConfig& serving, int32_t max_blocks_per_seq);

}  // namespace tie
