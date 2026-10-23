#include <doctest/doctest.h>

#include <algorithm>
#include <deque>
#include <random>

#include "backend/cpu/cpu_backend.h"
#include "engine/batch_scheduler.h"
#include "kv/paged_kv_cache.h"

using namespace tie;

namespace {

struct Request {
  int32_t arrival_step;
  int32_t prompt_len;
  int32_t max_tokens;
};

}  // namespace

TEST_CASE("randomized: requests all finish, blocks are never shared or leaked, the oldest is never preempted") {
  int64_t preemptions = 0;
  for (uint64_t seed = 0; seed < 300; ++seed) {
    CAPTURE(seed);
    std::mt19937_64 rng(seed);
    const auto pick = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };

    const int32_t blocks = pick(3, 16);
    const int32_t pool_tokens = blocks * 4;
    KVCacheConfig kc;
    kc.num_layers = 1;
    kc.num_kv_heads = 1;
    kc.head_dim = 8;
    kc.block_size = 4;
    kc.dtype = DType::F32;
    kc.budget_bytes = size_t(blocks) * (2 * 4 * 8 * 4);
    CpuBackend be(1);
    PagedKVCache cache(be, kc);

    BatchSchedulerConfig cfg;
    cfg.serving.max_tokens_per_step = pick(4, 32);
    cfg.serving.max_seqs = pick(1, std::min(8, cfg.serving.max_tokens_per_step));
    cfg.policy = seed % 2 ? OrderingPolicy::Fcfs : OrderingPolicy::SrptAging;
    cfg.aging = pick(0, 8);
    cfg.watermark_blocks = pick(-1, blocks - 1);
    BatchScheduler sched(cache, cfg);

    // Like Engine::add_request: every request fits the whole pool on its own.
    std::vector<Request> requests(size_t(pick(1, 20)));
    for (Request& r : requests) {
      r.arrival_step = pick(0, 40);
      r.prompt_len = pick(1, pool_tokens - 1);
      r.max_tokens = pick(1, pool_tokens - r.prompt_len);
    }
    std::ranges::sort(requests, {}, &Request::arrival_step);

    std::deque<Sequence> seqs;
    std::vector<Sequence*> live;
    size_t next = 0;
    int empty_streak = 0;
    const auto finish = [&](Sequence* s) {
      cache.release(s->block_table);
      s->state = SeqState::Finished;
      sched.finish(s);
      std::erase(live, s);
    };

    int step = 0;
    for (; step < 20000 && (next < requests.size() || !live.empty()); ++step) {
      for (; next < requests.size() && requests[next].arrival_step <= step; ++next) {
        Sequence& s = seqs.emplace_back();
        s.id = int32_t(seqs.size()) - 1;
        s.tokens.assign(size_t(requests[next].prompt_len), 1);
        s.prompt_len = requests[next].prompt_len;
        s.params.max_tokens = requests[next].max_tokens;
        sched.add(&s);
        live.push_back(&s);
      }
      if (!live.empty() && pick(0, 49) == 0) finish(live[size_t(pick(0, int(live.size()) - 1))]);  // cancel

      Sequence* oldest = sched.running().empty() ? nullptr : sched.running().front();
      const int32_t oldest_preemptions = oldest ? oldest->num_preemptions : 0;

      const ScheduledBatch batch = sched.schedule();

      if (oldest != nullptr) CHECK(oldest->num_preemptions == oldest_preemptions);
      int32_t total = 0;
      for (const ScheduledSeq& s : batch.seqs) {
        total += s.num_tokens;
        CHECK(s.num_tokens > 0);
        CHECK(s.num_tokens <= s.seq->num_pending());
        CHECK(int64_t(s.seq->block_table.size()) >=
              PagedKVCache::blocks_for(s.seq->num_computed + s.num_tokens, 4));
      }
      CHECK(total <= cfg.serving.max_tokens_per_step);
      CHECK(int32_t(batch.seqs.size()) <= cfg.serving.max_seqs);
      // An empty step is allowed only right after a preemption made the waiter sit out one step.
      empty_streak = batch.empty() && sched.has_work() ? empty_streak + 1 : 0;
      CHECK(empty_streak <= 1);

      for (const ScheduledSeq& s : batch.seqs) s.seq->num_computed += s.num_tokens;
      sched.complete(batch);
      for (const ScheduledSeq& s : batch.seqs) {
        if (s.seq->num_pending() != 0) continue;
        s.seq->tokens.push_back(3);
        s.seq->state = SeqState::Decoding;
        if (s.seq->num_generated() >= s.seq->params.max_tokens) finish(s.seq);
      }

      // Between steps: every block has one owner, and each table holds exactly its computed tokens.
      int64_t owned = 0;
      for (const Sequence* s : live) {
        owned += int64_t(s->block_table.size());
        CHECK(int64_t(s->block_table.size()) == PagedKVCache::blocks_for(s->num_computed, 4));
        for (int32_t b : s->block_table) CHECK(cache.pool().ref_count(b) == 1);
      }
      CHECK(owned + cache.num_free_blocks() == blocks);
    }
    CHECK(live.empty());
    CHECK(next == requests.size());
    CHECK(cache.num_free_blocks() == blocks);
    CHECK_FALSE(sched.has_work());
    preemptions += sched.num_preemptions();
  }
  CHECK(preemptions > 0);  // the preemption invariants above are not vacuous
}
