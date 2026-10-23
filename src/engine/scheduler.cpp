#include "engine/scheduler.h"

#include <algorithm>

#include "core/error.h"

namespace tie {

FcfsScheduler::FcfsScheduler(PagedKVCache& cache, int32_t max_tokens_per_step)
    : cache_(cache), max_tokens_per_step_(max_tokens_per_step) {
  if (max_tokens_per_step_ <= 0) fail<InvalidArgument>("max_tokens_per_step must be positive");
}

ScheduledBatch FcfsScheduler::schedule() {
  if (in_flight_) return {};
  if (running_ == nullptr && !waiting_.empty()) {
    running_ = waiting_.front();
    waiting_.pop_front();
    running_->state = SeqState::Prefilling;
  }
  if (running_ == nullptr) return {};
  const int32_t n = std::min(running_->num_pending(), max_tokens_per_step_);
  // add_request clamps every request to fit the whole pool, and only one sequence runs.
  if (!cache_.reserve(running_->block_table, int64_t{running_->num_computed} + n)) {
    fail<CapacityError>("KV cache cannot hold sequence {}'s next {} tokens", running_->id, n);
  }
  in_flight_ = true;
  return {next_batch_id_++, {{running_, n}}};
}

void FcfsScheduler::finish(Sequence* seq) {
  if (running_ == seq) {
    running_ = nullptr;
    in_flight_ = false;
  } else {
    std::erase(waiting_, seq);
  }
}

}  // namespace tie
