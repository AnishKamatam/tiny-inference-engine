#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "engine/scheduler.h"
#include "engine/serving_config.h"
#include "engine/sequence.h"
#include "kv/paged_kv_cache.h"
#include "model/qwen3.h"
#include "sampling/sampler.h"

namespace tie {

struct EngineConfig {
  std::vector<int32_t> stop_tokens;  // generation ends after emitting any of these
};

struct EngineStats {
  int64_t steps = 0;
  int64_t seqs_scheduled = 0;  // summed over steps; divided by steps it is the mean batch size
  int64_t prefill_tokens = 0;  // prompt tokens, and tokens recomputed after preemption
  int64_t decode_tokens = 0;
  double step_seconds = 0.0;   // time in the model's forward pass
  int64_t preemptions = 0;
};

using TokenCallback = std::function<void(int32_t seq_id, int32_t token)>;

// Drives generation one step at a time: the scheduler picks the work, the
// engine builds the batch and its KV slots, runs the model and samples.
class Engine {
 public:
  Engine(Qwen3Model& model, PagedKVCache& cache, std::unique_ptr<Scheduler> scheduler, ServingConfig serving,
         EngineConfig config = {});

  // Queues a request. Only a prompt that cannot run at all (no room for even one
  // generated token in the model context, the KV pool or a block-table row) is
  // rejected, with CapacityError, before any compute. Otherwise the request is
  // accepted and its max_tokens is clamped to the most that fits all three, so it
  // runs until it finishes or reaches the capacity it can be given (FinishReason::Length).
  // It cannot fail for capacity reasons mid-generation: the scheduler reserves blocks before
  // each step and, when several sequences run, preempts newer ones so the oldest can always grow.
  int32_t add_request(std::vector<int32_t> prompt, const SamplingParams& params);
  void step();
  bool has_work() const { return scheduler_->has_work(); }

  const Sequence& sequence(int32_t id) const;
  void release(int32_t id);  // forget a finished sequence; not allowed during step() (e.g. from a token callback)

  // Finishes a waiting, running or preempted request with FinishReason::Cancelled and frees its
  // blocks; a no-op for a finished one. Safe from a token callback.
  void cancel(int32_t id);

  // Runs one request to completion and returns its generated tokens, including
  // the stop token that ended it, if any.
  std::vector<int32_t> generate(std::vector<int32_t> prompt, const SamplingParams& params);

  void set_token_callback(TokenCallback callback) { on_token_ = std::move(callback); }
  EngineStats stats() const;

 private:
  void check_batch(const ScheduledBatch& batch) const;
  void finish(Sequence& seq, FinishReason reason);
  Sequence& get(int32_t id);

  Qwen3Model& model_;
  PagedKVCache& cache_;
  std::unique_ptr<Scheduler> scheduler_;
  ServingConfig serving_;
  EngineConfig config_;
  Sampler sampler_;
  TokenCallback on_token_;
  EngineStats stats_;
  std::unordered_map<int32_t, std::unique_ptr<Sequence>> sequences_;
  int32_t next_id_ = 0;
  bool stepping_ = false;

  // Step scratch, reused so steady-state decoding does not allocate.
  std::vector<int32_t> tokens_, positions_, slots_, query_start_, context_lens_, block_tables_, logit_rows_;
  std::vector<Sequence*> sampled_;
  std::vector<float> logits_;
};

}  // namespace tie
