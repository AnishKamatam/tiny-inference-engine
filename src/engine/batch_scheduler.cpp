#include "engine/batch_scheduler.h"

#include <algorithm>

#include "core/error.h"

namespace tie {

OrderingPolicy parse_policy(std::string_view name) {
  if (name == "fcfs") return OrderingPolicy::Fcfs;
  if (name == "srpt-aging") return OrderingPolicy::SrptAging;
  fail<InvalidArgument>("scheduling policy must be fcfs or srpt-aging, got '{}'", name);
}

BatchScheduler::BatchScheduler(PagedKVCache& cache, BatchSchedulerConfig config) : cache_(cache), config_(config) {
  validate(config_.serving);
  if (config_.aging < 0) fail<InvalidArgument>("aging must be non-negative, got {}", config_.aging);
  if (config_.watermark_blocks < -1 || config_.watermark_blocks >= cache_.num_blocks()) {
    fail<InvalidArgument>("watermark of {} blocks must be -1 (auto) or below the pool's {} blocks",
                          config_.watermark_blocks, cache_.num_blocks());
  }
  watermark_ = config_.watermark_blocks >= 0 ? config_.watermark_blocks : std::max(1, cache_.num_blocks() / 100);
}

void BatchScheduler::add(Sequence* seq) {
  seq->state = SeqState::Waiting;
  seq->waiting_since_step = step_;
  waiting_.push_back(seq);
}

bool BatchScheduler::before(const Sequence* a, const Sequence* b) const {
  if (config_.policy == OrderingPolicy::SrptAging) {
    const auto score = [this](const Sequence* s) {
      return int64_t{s->num_pending()} - int64_t{config_.aging} * (step_ - s->waiting_since_step);
    };
    if (score(a) != score(b)) return score(a) < score(b);
  }
  if (a->waiting_since_step != b->waiting_since_step) return a->waiting_since_step < b->waiting_since_step;
  return a->id < b->id;
}

// The most of `want` tokens `seq` can run next with the blocks it holds plus the free ones.
// Between steps a table holds exactly the blocks its computed tokens need.
int32_t BatchScheduler::fit(const Sequence& seq, int32_t want) const {
  const int64_t room =
      (static_cast<int64_t>(seq.block_table.size()) + cache_.num_free_blocks()) * cache_.block_size() - seq.num_computed;
  return static_cast<int32_t>(std::clamp<int64_t>(room, 0, want));
}

// The newest running sequence admitted after running_[index] that can be preempted: not in flight
// and holding blocks (preempting one without blocks frees nothing), if any.
Sequence* BatchScheduler::newest_after(size_t index) const {
  for (size_t j = running_.size(); j-- > index + 1;) {
    if (!busy(running_[j]) && !running_[j]->block_table.empty()) return running_[j];
  }
  return nullptr;
}

// Whether a sequence admitted after running_[index] is in flight, so its blocks come back when it completes.
bool BatchScheduler::newer_busy(size_t index) const {
  for (size_t j = index + 1; j < running_.size(); ++j) {
    if (busy(running_[j])) return true;
  }
  return false;
}

void BatchScheduler::preempt(Sequence* seq) {
  cache_.release(seq->block_table);
  seq->num_computed = 0;  // its tokens (prompt and generated) stay and are recomputed on re-admission
  seq->state = SeqState::Waiting;
  seq->waiting_since_step = step_;
  ++seq->num_preemptions;
  ++preemptions_;
  std::erase(running_, seq);
  waiting_.push_back(seq);
}

ScheduledBatch BatchScheduler::schedule() {
  ++step_;
  ScheduledBatch batch;
  batch.id = next_batch_id_++;
  int32_t budget = config_.serving.max_tokens_per_step;
  const int32_t max_seqs = config_.serving.max_seqs;
  const auto full = [&] { return budget == 0 || static_cast<int32_t>(batch.seqs.size()) == max_seqs; };
  const auto take = [&](Sequence* seq, int32_t n) {
    if (!cache_.reserve(seq->block_table, int64_t{seq->num_computed} + n)) {
      fail<InvalidArgument>("scheduler bug: {} tokens of sequence {} do not fit the blocks it counted", n, seq->id);
    }
    batch.seqs.push_back({seq, n});
    budget -= n;
  };

  // 1. Decodes, oldest first. Running sequences never exceed max_seqs <= max_tokens_per_step, so all fit.
  //    One that needs a block when none is free preempts the newest sequence admitted after it
  //    that is idle and holds blocks. If there is none but a newer one is in flight, it waits for
  //    that batch to complete (an older sequence is never preempted for a newer one). It preempts
  //    itself only when no newer sequence is in flight or holds blocks.
  for (size_t i = 0; i < running_.size() && !full();) {
    Sequence* seq = running_[i];
    if (busy(seq) || seq->num_pending() != 1) {
      ++i;
      continue;
    }
    bool preempted_self = false;
    bool deferred = false;
    while (fit(*seq, 1) == 0) {
      Sequence* victim = newest_after(i);
      if (victim == nullptr && newer_busy(i)) {
        deferred = true;
        break;
      }
      preempt(victim != nullptr ? victim : seq);
      if (victim == nullptr) {
        preempted_self = true;
        break;
      }
    }
    if (deferred) {
      ++i;
      continue;
    }
    if (preempted_self) continue;  // running_[i] is now the next sequence
    take(seq, 1);
    ++i;
  }

  // 2. Partial prefills continue in admission order with as much as the budget and free blocks allow.
  for (Sequence* seq : running_) {
    if (full()) break;
    if (busy(seq) || seq->num_pending() <= 1) continue;
    if (const int32_t n = fit(*seq, std::min(seq->num_pending(), budget)); n > 0) take(seq, n);
  }

  // 3. Admission in policy order. The watermark keeps blocks free for running sequences to grow
  //    into. Admission stops at the first sequence that does not fit, so a large request is not
  //    overtaken forever by small ones. With nothing running the whole pool is free and
  //    add_request's clamp guarantees the head fits, so the watermark does not apply.
  //    A sequence preempted during this step waits at least one step, so it cannot thrash.
  std::ranges::sort(waiting_, [this](const Sequence* a, const Sequence* b) { return before(a, b); });
  size_t admitted = 0;
  for (Sequence* seq : waiting_) {
    if (full() || static_cast<int32_t>(running_.size()) >= max_seqs) break;
    if (seq->num_preemptions > 0 && seq->waiting_since_step == step_) continue;
    const int32_t n = std::min(seq->num_pending(), budget);
    const int32_t keep_free = running_.empty() ? 0 : watermark_;
    if (cache_.num_free_blocks() - PagedKVCache::blocks_for(n, cache_.block_size()) < keep_free) break;
    seq->state = SeqState::Prefilling;
    running_.push_back(seq);
    take(seq, n);
    seq->waiting_since_step = -1;  // marks it admitted for the erase below
    ++admitted;
  }
  if (admitted > 0) std::erase_if(waiting_, [](const Sequence* s) { return s->waiting_since_step == -1; });

  // 4. Never stall: if nothing could run (running sequences all mid-prefill on a full pool), the
  //    oldest running sequence preempts newer ones until it can take a chunk.
  if (batch.empty()) {
    for (size_t i = 0; i < running_.size(); ++i) {
      Sequence* seq = running_[i];
      if (busy(seq)) continue;
      const int32_t want = std::min(seq->num_pending(), budget);
      while (fit(*seq, want) == 0) {
        Sequence* victim = newest_after(i);
        if (victim == nullptr) break;
        preempt(victim);
      }
      if (const int32_t n = fit(*seq, want); n > 0) take(seq, n);
      break;
    }
  }

  for (const ScheduledSeq& s : batch.seqs) in_flight_.insert(s.seq);
  return batch;
}

void BatchScheduler::complete(const ScheduledBatch& batch) {
  for (const ScheduledSeq& s : batch.seqs) in_flight_.erase(s.seq);
}

void BatchScheduler::abandon(const ScheduledBatch& batch) {
  for (const ScheduledSeq& s : batch.seqs) in_flight_.erase(s.seq);
}

void BatchScheduler::finish(Sequence* seq) {
  std::erase(waiting_, seq);
  std::erase(running_, seq);
  in_flight_.erase(seq);
}

}  // namespace tie
