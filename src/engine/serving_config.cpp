#include "engine/serving_config.h"

#include "core/error.h"

namespace tie {

void validate(const ServingConfig& serving) {
  if (serving.max_tokens_per_step <= 0) {
    fail<InvalidArgument>("max_tokens_per_step must be positive, got {}", serving.max_tokens_per_step);
  }
  if (serving.max_seqs <= 0) fail<InvalidArgument>("max_seqs must be positive, got {}", serving.max_seqs);
  if (serving.max_seqs > serving.max_tokens_per_step) {
    fail<InvalidArgument>("max_seqs ({}) cannot exceed max_tokens_per_step ({}): every sequence must decode each step",
                          serving.max_seqs, serving.max_tokens_per_step);
  }
}

ModelLimits model_limits(const ServingConfig& serving, int32_t max_blocks_per_seq) {
  ModelLimits limits;
  limits.max_tokens_per_step = serving.max_tokens_per_step;
  limits.max_seqs = serving.max_seqs;
  limits.max_logit_rows = serving.max_seqs;
  limits.max_blocks_per_seq = max_blocks_per_seq;
  return limits;
}

}  // namespace tie
