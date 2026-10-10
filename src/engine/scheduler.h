#pragma once

#include <cstdint>
#include <deque>
#include <vector>

#include "engine/sequence.h"

namespace tie {

struct ScheduledSeq {
  Sequence* seq;
  int32_t num_tokens;  // pending tokens of `seq` to run this step
};

struct ScheduledBatch {
  std::vector<ScheduledSeq> seqs;
  bool empty() const { return seqs.empty(); }
};

// Decides which sequences run in each engine step and how many of their
// pending tokens. The engine owns the sequences; schedulers only point at them.
class Scheduler {
 public:
  virtual ~Scheduler() = default;
  virtual void add(Sequence* seq) = 0;
  virtual ScheduledBatch schedule() = 0;
  virtual void finish(Sequence* seq) = 0;  // done or cancelled: forget it
  virtual bool has_work() const = 0;
};

// One running sequence at a time, first come first served. Prefill is chunked
// to max_tokens_per_step; decode is one token per step. Continuous batching
// replaces this class without touching the engine or the model.
class FcfsScheduler final : public Scheduler {
 public:
  explicit FcfsScheduler(int32_t max_tokens_per_step);

  void add(Sequence* seq) override { waiting_.push_back(seq); }
  ScheduledBatch schedule() override;
  void finish(Sequence* seq) override;
  bool has_work() const override { return running_ != nullptr || !waiting_.empty(); }

 private:
  int32_t max_tokens_per_step_;
  std::deque<Sequence*> waiting_;
  Sequence* running_ = nullptr;
};

}  // namespace tie
