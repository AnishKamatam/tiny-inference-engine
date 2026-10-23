#pragma once

#include <cstdint>
#include <deque>
#include <vector>

#include "engine/sequence.h"
#include "kv/paged_kv_cache.h"

namespace tie {

struct ScheduledSeq {
  Sequence* seq;
  int32_t num_tokens;  // pending tokens of `seq` to run this step
};

struct ScheduledBatch {
  uint64_t id = 0;
  std::vector<ScheduledSeq> seqs;
  bool empty() const { return seqs.empty(); }
};

// Decides which sequences run in each engine step and how many of their pending tokens,
// and owns their KV blocks while they run: every sequence schedule() returns already has a
// block table covering num_computed + num_tokens. The engine owns the sequences.
//
// Deciding a step (schedule) is separate from finishing it (complete / abandon). A sequence
// in a batch that is neither completed nor abandoned is not scheduled again, so a pipelined
// engine can keep several batches in flight.
class Scheduler {
 public:
  virtual ~Scheduler() = default;
  virtual void add(Sequence* seq) = 0;
  virtual ScheduledBatch schedule() = 0;
  virtual void complete(const ScheduledBatch& batch) = 0;  // the engine advanced num_computed
  virtual void abandon(const ScheduledBatch& batch) = 0;   // the step failed; tables trimmed to num_computed
  virtual void finish(Sequence* seq) = 0;                  // done or cancelled: forget it
  virtual bool has_work() const = 0;
  virtual int64_t num_preemptions() const { return 0; }
};

// One running sequence at a time, first come first served. Prefill is chunked
// to max_tokens_per_step; decode is one token per step.
class FcfsScheduler final : public Scheduler {
 public:
  FcfsScheduler(PagedKVCache& cache, int32_t max_tokens_per_step);

  void add(Sequence* seq) override { waiting_.push_back(seq); }
  ScheduledBatch schedule() override;
  void complete(const ScheduledBatch&) override { in_flight_ = false; }
  void abandon(const ScheduledBatch&) override { in_flight_ = false; }
  void finish(Sequence* seq) override;
  bool has_work() const override { return running_ != nullptr || !waiting_.empty(); }

 private:
  PagedKVCache& cache_;
  int32_t max_tokens_per_step_;
  std::deque<Sequence*> waiting_;
  Sequence* running_ = nullptr;
  bool in_flight_ = false;
  uint64_t next_batch_id_ = 1;
};

}  // namespace tie
