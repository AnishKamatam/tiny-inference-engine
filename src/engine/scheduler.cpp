#include "engine/scheduler.h"

#include <algorithm>

#include "core/error.h"

namespace tie {

FcfsScheduler::FcfsScheduler(int32_t max_tokens_per_step) : max_tokens_per_step_(max_tokens_per_step) {
  if (max_tokens_per_step_ <= 0) fail<InvalidArgument>("max_tokens_per_step must be positive");
}

ScheduledBatch FcfsScheduler::schedule() {
  if (running_ == nullptr && !waiting_.empty()) {
    running_ = waiting_.front();
    waiting_.pop_front();
    running_->state = SeqState::Prefilling;
  }
  if (running_ == nullptr) return {};
  return {{{running_, std::min(running_->num_pending(), max_tokens_per_step_)}}};
}

void FcfsScheduler::finish(Sequence* seq) {
  if (running_ == seq) {
    running_ = nullptr;
  } else {
    std::erase(waiting_, seq);
  }
}

}  // namespace tie
