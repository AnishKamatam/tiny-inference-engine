#include "core/thread_pool.h"

#include <sys/sysctl.h>

#include <algorithm>

namespace tie {

namespace {

int performance_core_count() {
  int count = 0;
  size_t size = sizeof(count);
  if (sysctlbyname("hw.perflevel0.physicalcpu", &count, &size, nullptr, 0) == 0 && count > 0) return count;
  return std::max(1u, std::thread::hardware_concurrency());
}

}  // namespace

ThreadPool::ThreadPool(int num_threads) : num_threads_(num_threads > 0 ? num_threads : performance_core_count()) {
  for (int i = 1; i < num_threads_; ++i) {
    workers_.emplace_back([this, i] { worker_loop(i); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard lock(mu_);
    stop_ = true;
  }
  work_cv_.notify_all();
  for (auto& w : workers_) w.join();
}

void ThreadPool::run_chunk(int chunk) const {
  const int64_t per = (job_n_ + job_chunks_ - 1) / job_chunks_;
  const int64_t begin = chunk * per;
  const int64_t end = std::min(job_n_, begin + per);
  if (begin < end) (*job_)(begin, end);
}

void ThreadPool::parallel_for(int64_t n, const RangeFn& fn) {
  if (n <= 0) return;
  const int chunks = static_cast<int>(std::min<int64_t>(num_threads_, n));
  if (chunks == 1) {
    fn(0, n);
    return;
  }
  {
    std::lock_guard lock(mu_);
    job_ = &fn;
    job_n_ = n;
    job_chunks_ = chunks;
    pending_ = chunks - 1;
    ++generation_;
  }
  work_cv_.notify_all();
  run_chunk(0);

  std::unique_lock lock(mu_);
  done_cv_.wait(lock, [this] { return pending_ == 0; });
  job_ = nullptr;
}

void ThreadPool::worker_loop(int worker_id) {
  uint64_t seen = 0;
  std::unique_lock lock(mu_);
  for (;;) {
    work_cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
    if (stop_) return;
    seen = generation_;
    if (worker_id >= job_chunks_) continue;
    lock.unlock();
    run_chunk(worker_id);
    lock.lock();
    if (--pending_ == 0) done_cv_.notify_one();
  }
}

}  // namespace tie
