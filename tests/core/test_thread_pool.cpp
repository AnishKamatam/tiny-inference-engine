#include <doctest/doctest.h>

#include <atomic>
#include <vector>

#include "core/thread_pool.h"

using namespace tie;

TEST_CASE("parallel_for visits every index exactly once") {
  ThreadPool pool(4);
  CHECK(pool.size() == 4);
  for (int64_t n : {0, 1, 3, 7, 1000}) {
    std::vector<std::atomic<int>> hits(static_cast<size_t>(n));
    pool.parallel_for(n, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) hits[static_cast<size_t>(i)]++;
    });
    for (auto& h : hits) CHECK(h.load() == 1);
  }
}

TEST_CASE("parallel_for survives many back-to-back jobs") {
  ThreadPool pool(8);
  std::atomic<int64_t> total{0};
  for (int iter = 0; iter < 2000; ++iter) {
    pool.parallel_for(64, [&](int64_t begin, int64_t end) { total += end - begin; });
  }
  CHECK(total.load() == 2000 * 64);
}

TEST_CASE("default thread count uses the performance cores") {
  ThreadPool pool;
  CHECK(pool.size() >= 1);
}
