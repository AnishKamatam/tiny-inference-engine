#include <doctest/doctest.h>

#include "backend/cpu/cpu_backend.h"
#include "engine/scheduler.h"
#include "kv/paged_kv_cache.h"

using namespace tie;

namespace {

KVCacheConfig kv_config(size_t blocks) {
  KVCacheConfig c;
  c.num_layers = 1;
  c.num_kv_heads = 1;
  c.head_dim = 8;
  c.block_size = 4;
  c.dtype = DType::F32;
  c.budget_bytes = blocks * (2 * 1 * 1 * 4 * 8 * 4);
  return c;
}

Sequence make_seq(int32_t id, int n_prompt) {
  Sequence s;
  s.id = id;
  s.tokens.assign(size_t(n_prompt), 1);
  s.prompt_len = n_prompt;
  return s;
}

}  // namespace

TEST_CASE("FcfsScheduler chunks prefill, reserves blocks, then decodes one token per step") {
  CpuBackend be(1);
  PagedKVCache cache(be, kv_config(16));
  FcfsScheduler sched(cache, 4);
  Sequence a = make_seq(0, 10);
  sched.add(&a);
  CHECK(sched.has_work());

  for (int expected : {4, 4, 2}) {
    const ScheduledBatch b = sched.schedule();
    REQUIRE(b.seqs.size() == 1);
    CHECK(b.seqs[0].seq == &a);
    CHECK(b.seqs[0].num_tokens == expected);
    CHECK(a.block_table.size() == size_t(PagedKVCache::blocks_for(a.num_computed + expected, 4)));
    CHECK(sched.schedule().empty());  // in flight until completed
    a.num_computed += expected;
    sched.complete(b);
  }
  a.tokens.push_back(5);  // the engine appends the sampled token
  const ScheduledBatch d = sched.schedule();
  CHECK(d.seqs[0].num_tokens == 1);
  CHECK(a.block_table.size() == 3);  // position 10 starts the third block
}

TEST_CASE("FcfsScheduler reschedules an abandoned batch and runs requests in arrival order") {
  CpuBackend be(1);
  PagedKVCache cache(be, kv_config(16));
  FcfsScheduler sched(cache, 16);
  Sequence a = make_seq(0, 3), b = make_seq(1, 3), c = make_seq(2, 3);
  sched.add(&a);
  sched.add(&b);
  sched.add(&c);
  const ScheduledBatch first = sched.schedule();
  CHECK(first.seqs[0].seq == &a);
  cache.truncate(a.block_table, a.num_computed);  // what the engine does when forward fails
  sched.abandon(first);
  const ScheduledBatch again = sched.schedule();
  CHECK(again.seqs[0].seq == &a);
  CHECK(again.id != first.id);
  sched.complete(again);
  sched.finish(&c);  // cancelling a waiting request drops it
  cache.release(a.block_table);
  sched.finish(&a);
  CHECK(sched.schedule().seqs[0].seq == &b);
}
