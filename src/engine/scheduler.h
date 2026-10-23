#pragma once

#include <cstdint>
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

}  // namespace tie
