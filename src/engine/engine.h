#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "engine/scheduler.h"
#include "engine/sequence.h"
#include "kv/paged_kv_cache.h"
#include "model/qwen3.h"
#include "sampling/sampler.h"

namespace tie {

struct EngineConfig {
  int32_t max_tokens_per_step = 512;
  std::vector<int32_t> stop_tokens;  // generation ends after emitting any of these
};

struct EngineStats {
  int64_t prefill_tokens = 0;
  double prefill_seconds = 0.0;
  int64_t decode_tokens = 0;
  double decode_seconds = 0.0;
};

using TokenCallback = std::function<void(int32_t seq_id, int32_t token)>;

// Drives generation one step at a time: the scheduler picks the work, the
// engine builds the batch and its KV slots, runs the model and samples.
class Engine {
 public:
  Engine(Qwen3Model& model, PagedKVCache& cache, std::unique_ptr<Scheduler> scheduler, EngineConfig config);

  // Queues a request. Only a prompt that cannot run at all (no room for even one
  // generated token in the model context, the KV pool or a block-table row) is
  // rejected, with CapacityError, before any compute. Otherwise the request is
  // accepted and its max_tokens is clamped to the most that fits all three, so it
  // runs until it finishes or reaches the capacity it can be given (FinishReason::Length),
  // and cannot fail for capacity reasons mid-generation.
  int32_t add_request(std::vector<int32_t> prompt, const SamplingParams& params);
  void step();
  bool has_work() const { return scheduler_->has_work(); }

  const Sequence& sequence(int32_t id) const;
  void release(int32_t id);  // forget a finished sequence

  // Runs one request to completion and returns its generated tokens, including
  // the stop token that ended it, if any.
  std::vector<int32_t> generate(std::vector<int32_t> prompt, const SamplingParams& params);

  void set_token_callback(TokenCallback callback) { on_token_ = std::move(callback); }
  const EngineStats& stats() const { return stats_; }

 private:
  void finish(Sequence& seq, FinishReason reason);

  Qwen3Model& model_;
  PagedKVCache& cache_;
  std::unique_ptr<Scheduler> scheduler_;
  EngineConfig config_;
  Sampler sampler_;
  TokenCallback on_token_;
  EngineStats stats_;
  std::unordered_map<int32_t, std::unique_ptr<Sequence>> sequences_;
  int32_t next_id_ = 0;

  // Step scratch, reused so steady-state decoding does not allocate.
  std::vector<int32_t> tokens_, positions_, slots_, query_start_, context_lens_, block_tables_, logit_rows_;
  std::vector<Sequence*> sampled_;
  std::vector<float> logits_;
};

}  // namespace tie
