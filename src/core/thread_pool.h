#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace tie {

// Fixed set of worker threads for data-parallel kernels. The calling thread
// participates, so a pool of size N has N - 1 background workers.
class ThreadPool {
 public:
  using RangeFn = std::function<void(int64_t begin, int64_t end)>;

  // 0 selects the number of performance cores.
  explicit ThreadPool(int num_threads = 0);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return num_threads_; }

  // Splits [0, n) into at most size() contiguous chunks and runs `fn` on each
  // in parallel. Blocks until every chunk is done. Not reentrant.
  void parallel_for(int64_t n, const RangeFn& fn);

 private:
  void worker_loop(int worker_id);
  void run_chunk(int chunk) const;

  int num_threads_;
  std::vector<std::thread> workers_;

  std::mutex mu_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  uint64_t generation_ = 0;
  bool stop_ = false;
  const RangeFn* job_ = nullptr;
  int64_t job_n_ = 0;
  int job_chunks_ = 0;
  int pending_ = 0;
};

}  // namespace tie
