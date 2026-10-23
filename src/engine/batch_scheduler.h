#pragma once

#include <cstdint>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "engine/scheduler.h"
#include "engine/serving_config.h"
#include "kv/paged_kv_cache.h"

namespace tie {

// Order in which waiting sequences are admitted.
enum class OrderingPolicy {
  Fcfs,       // by arrival (or by preemption, for a preempted sequence)
  SrptAging,  // least remaining prefill first, minus `aging` tokens per step waited
};

OrderingPolicy parse_policy(std::string_view name);  // "fcfs" or "srpt-aging"

struct BatchSchedulerConfig {
  ServingConfig serving;
  OrderingPolicy policy = OrderingPolicy::SrptAging;
  int32_t aging = 16;             // srpt-aging: priority (in tokens) gained per step waited; 0 = pure SRPT
  int32_t watermark_blocks = -1;  // blocks admission leaves free for running sequences; -1 = max(1, 1% of the pool)
};

// Continuous batching. Each step, under the token budget, the sequence cap and the free KV blocks:
//   1. every decoding sequence advances one token, oldest first;
//   2. partially prefilled sequences continue with as much as fits;
//   3. waiting sequences are admitted in policy order.
// When a decode needs a block and none is free, the newest admission is preempted: its blocks
// are freed and it waits to recompute its tokens. Preempting newest-first means an older
// sequence is never preempted for a newer one, so the oldest always makes progress.
class BatchScheduler final : public Scheduler {
 public:
  BatchScheduler(PagedKVCache& cache, BatchSchedulerConfig config);

  void add(Sequence* seq) override;
  ScheduledBatch schedule() override;
  void complete(const ScheduledBatch& batch) override;
  void abandon(const ScheduledBatch& batch) override;
  void finish(Sequence* seq) override;
  bool has_work() const override { return !waiting_.empty() || !running_.empty(); }
  int64_t num_preemptions() const override { return preemptions_; }

  int32_t watermark_blocks() const { return watermark_; }
  const std::vector<Sequence*>& running() const { return running_; }  // admission order

 private:
  bool busy(const Sequence* seq) const { return in_flight_.contains(seq); }
  bool before(const Sequence* a, const Sequence* b) const;
  int32_t fit(const Sequence& seq, int32_t want) const;
  Sequence* newest_after(size_t index) const;
  bool newer_busy(size_t index) const;
  void preempt(Sequence* seq);

  PagedKVCache& cache_;
  BatchSchedulerConfig config_;
  int32_t watermark_;
  std::vector<Sequence*> waiting_;
  std::vector<Sequence*> running_;  // admission order, oldest first
  std::unordered_set<const Sequence*> in_flight_;
  int64_t step_ = 0;
  uint64_t next_batch_id_ = 1;
  int64_t preemptions_ = 0;
};

}  // namespace tie
