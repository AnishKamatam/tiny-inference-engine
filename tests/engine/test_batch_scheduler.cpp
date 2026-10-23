#include <doctest/doctest.h>

#include <deque>

#include "backend/cpu/cpu_backend.h"
#include "core/error.h"
#include "engine/batch_scheduler.h"
#include "kv/paged_kv_cache.h"

using namespace tie;

namespace {

KVCacheConfig kv_config(int32_t blocks) {  // 4-token blocks
  KVCacheConfig c;
  c.num_layers = 1;
  c.num_kv_heads = 1;
  c.head_dim = 8;
  c.block_size = 4;
  c.dtype = DType::F32;
  c.budget_bytes = size_t(blocks) * (2 * 1 * 1 * 4 * 8 * 4);
  return c;
}

BatchSchedulerConfig config(int32_t max_tokens_per_step, int32_t max_seqs, OrderingPolicy policy = OrderingPolicy::Fcfs,
                            int32_t watermark = 0, int32_t aging = 16) {
  BatchSchedulerConfig c;
  c.serving = {max_tokens_per_step, max_seqs};
  c.policy = policy;
  c.watermark_blocks = watermark;
  c.aging = aging;
  return c;
}

// A scheduler over a real block pool, playing the engine's part between steps.
struct Fixture {
  CpuBackend backend{1};
  PagedKVCache cache;
  BatchScheduler sched;
  std::deque<Sequence> seqs;  // stable addresses

  Fixture(int32_t blocks, BatchSchedulerConfig c) : cache(backend, kv_config(blocks)), sched(cache, c) {}

  Sequence& add(int prompt_len) {
    Sequence& s = seqs.emplace_back();
    s.id = int32_t(seqs.size()) - 1;
    s.tokens.assign(size_t(prompt_len), 1);
    s.prompt_len = prompt_len;
    sched.add(&s);
    return s;
  }
  // One engine step: schedule, record progress, complete, and append a sampled token for every
  // sequence whose input is now all computed.
  ScheduledBatch run() {
    ScheduledBatch b = sched.schedule();
    for (const ScheduledSeq& s : b.seqs) s.seq->num_computed += s.num_tokens;
    sched.complete(b);
    for (const ScheduledSeq& s : b.seqs) {
      if (s.seq->num_pending() == 0) {
        s.seq->tokens.push_back(2);
        s.seq->state = SeqState::Decoding;
      }
    }
    return b;
  }
  void finish(Sequence& s) {
    cache.release(s.block_table);
    s.state = SeqState::Finished;
    sched.finish(&s);
  }
};

int32_t tokens_of(const ScheduledBatch& b, const Sequence& s) {
  for (const ScheduledSeq& x : b.seqs) {
    if (x.seq == &s) return x.num_tokens;
  }
  return 0;
}

int32_t total(const ScheduledBatch& b) {
  int32_t n = 0;
  for (const ScheduledSeq& x : b.seqs) n += x.num_tokens;
  return n;
}

}  // namespace

TEST_CASE("BatchScheduler decodes every running sequence first and fills the budget with prefill") {
  Fixture f(64, config(8, 4));
  Sequence& a = f.add(3);
  Sequence& b = f.add(3);
  ScheduledBatch s1 = f.run();
  CHECK(tokens_of(s1, a) == 3);
  CHECK(tokens_of(s1, b) == 3);
  Sequence& c = f.add(20);
  for (int step = 0; step < 3; ++step) {
    const ScheduledBatch s = f.run();
    REQUIRE(s.seqs.size() == 3);
    CHECK(s.seqs[0].seq == &a);
    CHECK(s.seqs[0].num_tokens == 1);
    CHECK(s.seqs[1].seq == &b);
    CHECK(s.seqs[1].num_tokens == 1);
    CHECK(s.seqs[2].seq == &c);
    CHECK(s.seqs[2].num_tokens == 6);
    CHECK(total(s) <= 8);
  }
  for (const ScheduledSeq& x : f.run().seqs) {
    CHECK(x.seq->block_table.size() == size_t(PagedKVCache::blocks_for(x.seq->num_computed, 4)));
  }
}

TEST_CASE("BatchScheduler caps running sequences at max_seqs; the rest wait, not rejected") {
  Fixture f(64, config(16, 2));
  Sequence& a = f.add(2);
  Sequence& b = f.add(2);
  Sequence& c = f.add(2);
  ScheduledBatch s = f.run();
  CHECK(tokens_of(s, a) == 2);
  CHECK(tokens_of(s, b) == 2);
  CHECK(tokens_of(s, c) == 0);
  CHECK(c.state == SeqState::Waiting);
  CHECK(f.sched.running().size() == 2);
  f.finish(a);
  s = f.run();
  CHECK(tokens_of(s, c) == 2);
  CHECK(tokens_of(s, b) == 1);
}

TEST_CASE("BatchScheduler admits a one-token prompt and samples it") {
  Fixture f(16, config(8, 2));
  Sequence& a = f.add(1);
  const ScheduledBatch s = f.run();
  CHECK(tokens_of(s, a) == 1);
  CHECK(a.num_generated() == 1);
  CHECK(tokens_of(f.run(), a) == 1);  // then decodes
}

TEST_CASE("BatchScheduler shrinks a prefill chunk to the free blocks") {
  Fixture f(4, config(8, 2));
  Sequence& x = f.add(4);
  Sequence& a = f.add(16);
  ScheduledBatch s = f.run();
  CHECK(tokens_of(s, x) == 4);  // 1 block
  CHECK(tokens_of(s, a) == 4);  // budget-limited first chunk, 1 block
  s = f.run();                  // x decodes into a second block, leaving 1 free
  CHECK(tokens_of(s, x) == 1);
  CHECK(tokens_of(s, a) == 4);  // 7 fit the budget; 1 block more holds only 4
  CHECK(f.cache.num_free_blocks() == 0);
}

TEST_CASE("BatchScheduler keeps the watermark free and stops admission at a head that does not fit") {
  Fixture f(8, config(64, 8, OrderingPolicy::Fcfs, /*watermark=*/2));
  Sequence& a = f.add(16);
  CHECK(tokens_of(f.run(), a) == 16);  // 4 blocks, 4 free
  Sequence& b = f.add(12);             // 3 blocks
  Sequence& c = f.add(2);              // 1 block: would fit, but b is ahead of it
  const ScheduledBatch s = f.run();    // a's decode takes a 5th block, 3 free
  CHECK(tokens_of(s, a) == 1);
  CHECK(tokens_of(s, b) == 0);
  CHECK(tokens_of(s, c) == 0);
  CHECK(b.state == SeqState::Waiting);
  CHECK(c.state == SeqState::Waiting);
}

TEST_CASE("BatchScheduler admits the head past the watermark when nothing is running") {
  Fixture f(4, config(64, 4, OrderingPolicy::Fcfs, /*watermark=*/3));
  Sequence& a = f.add(12);  // 3 blocks leave 1 free, below the watermark
  CHECK(tokens_of(f.run(), a) == 12);
}

TEST_CASE("srpt-aging admits the shortest work first, and aging lets a long waiter in") {
  for (OrderingPolicy policy : {OrderingPolicy::SrptAging, OrderingPolicy::Fcfs}) {
    CAPTURE(int(policy));
    Fixture f(256, config(8, 1, policy, 0, /*aging=*/2));
    Sequence& lng = f.add(40);
    int admitted_at = -1;
    for (int step = 1; step <= 40 && admitted_at < 0; ++step) {
      f.add(6);  // a fresh short request arrives every step
      f.run();
      if (lng.state != SeqState::Waiting) admitted_at = step;
      for (Sequence& s : f.seqs) {
        if (&s != &lng && s.state == SeqState::Decoding) f.finish(s);  // shorts leave after one token
      }
    }
    if (policy == OrderingPolicy::Fcfs) {
      CHECK(admitted_at == 1);
    } else {
      // Score = pending - aging * steps waited: 40 - 2t catches up with a fresh short (6 - 2) by t = 18.
      // Exactly t = 18: lng's score 40 - 2t ties the fresh short's 6 - 2 = 4, and lng has waited since step 0.
      CHECK(admitted_at == 18);
    }
  }
}

TEST_CASE("a decode that needs a block preempts the newest admission, which keeps its tokens") {
  Fixture f(4, config(64, 4));
  Sequence& a = f.add(8);  // 2 blocks
  Sequence& b = f.add(7);  // 2 blocks
  f.run();                 // both prefill and sample: a has 9 tokens, b has 8
  REQUIRE(f.cache.num_free_blocks() == 0);
  ScheduledBatch s = f.run();  // a's 9th token needs a 3rd block
  CHECK(tokens_of(s, a) == 1);
  CHECK(tokens_of(s, b) == 0);
  CHECK(b.state == SeqState::Waiting);
  CHECK(b.num_computed == 0);
  CHECK(b.block_table.empty());
  CHECK(b.tokens.size() == 8);  // prompt + its generated token, recomputed later
  CHECK(b.num_preemptions == 1);
  CHECK(f.sched.num_preemptions() == 1);
  f.finish(a);
  s = f.run();
  CHECK(tokens_of(s, b) == 8);  // recompute, then sample the next token
  CHECK(b.num_generated() == 2);
}

TEST_CASE("a decode with nobody newer to preempt preempts itself") {
  Fixture f(3, config(64, 4));
  Sequence& a = f.add(3);  // 4 tokens after sampling: still 1 block
  Sequence& b = f.add(8);  // 2 blocks; its 9th token needs a 3rd
  f.run();
  REQUIRE(f.cache.num_free_blocks() == 0);
  const ScheduledBatch s = f.run();
  CHECK(tokens_of(s, a) == 1);
  CHECK(tokens_of(s, b) == 0);
  CHECK(b.num_preemptions == 1);
  CHECK(a.num_preemptions == 0);
}

TEST_CASE("in-flight sequences are skipped; an abandoned batch is offered again") {
  Fixture f(16, config(8, 2));
  Sequence& a = f.add(4);
  const ScheduledBatch b1 = f.sched.schedule();
  CHECK(tokens_of(b1, a) == 4);
  CHECK(f.sched.schedule().empty());
  f.cache.truncate(a.block_table, a.num_computed);  // what the engine does when forward fails
  f.sched.abandon(b1);
  CHECK(tokens_of(f.sched.schedule(), a) == 4);
}

TEST_CASE("the oldest of several stalled prefills preempts newer ones to make progress") {
  Fixture f(4, config(8, 2));
  Sequence& a = f.add(12);
  Sequence& b = f.add(12);
  const ScheduledBatch b1 = f.sched.schedule();  // a: 8 tokens, 2 blocks
  const ScheduledBatch b2 = f.sched.schedule();  // a in flight; b admitted: 8 tokens, 2 blocks
  CHECK(tokens_of(b1, a) == 8);
  CHECK(tokens_of(b2, b) == 8);
  for (const ScheduledBatch* batch : {&b1, &b2}) {
    for (const ScheduledSeq& s : batch->seqs) s.seq->num_computed += s.num_tokens;
    f.sched.complete(*batch);
  }
  REQUIRE(f.cache.num_free_blocks() == 0);  // both mid-prefill on a full pool
  const ScheduledBatch s = f.sched.schedule();
  CHECK(tokens_of(s, a) == 4);
  CHECK(b.num_preemptions == 1);
}

TEST_CASE("finish forgets a sequence whether waiting, running or preempted") {
  Fixture f(16, config(16, 1));
  Sequence& a = f.add(3);
  Sequence& b = f.add(3);
  f.run();
  f.finish(b);  // waiting
  CHECK(f.sched.has_work());
  f.finish(a);  // running
  CHECK_FALSE(f.sched.has_work());
  CHECK(f.cache.num_free_blocks() == 16);

  // A preempted sequence is waiting again; finishing it forgets it too.
  Fixture g(4, config(64, 4));
  Sequence& p = g.add(8);
  Sequence& q = g.add(7);
  g.run();
  g.run();  // p's decode needs a 3rd block and preempts q
  REQUIRE(q.num_preemptions == 1);
  g.finish(q);
  CHECK(tokens_of(g.run(), q) == 0);
  g.finish(p);
  CHECK_FALSE(g.sched.has_work());
  CHECK(g.cache.num_free_blocks() == 4);
}

TEST_CASE("BatchScheduler validates its config and parses policy names") {
  CpuBackend be(1);
  PagedKVCache cache(be, kv_config(8));
  CHECK(parse_policy("fcfs") == OrderingPolicy::Fcfs);
  CHECK(parse_policy("srpt-aging") == OrderingPolicy::SrptAging);
  CHECK_THROWS_WITH_AS(parse_policy("lifo"), doctest::Contains("lifo"), InvalidArgument);
  CHECK(BatchScheduler(cache, config(8, 2, OrderingPolicy::Fcfs, -1)).watermark_blocks() == 1);
  CHECK_THROWS_AS(BatchScheduler(cache, config(8, 9)), InvalidArgument);
  CHECK_THROWS_AS(BatchScheduler(cache, config(8, 2, OrderingPolicy::Fcfs, 8)), InvalidArgument);
  CHECK_THROWS_AS(BatchScheduler(cache, config(8, 2, OrderingPolicy::Fcfs, -2)), InvalidArgument);
  CHECK_THROWS_AS(BatchScheduler(cache, config(8, 2, OrderingPolicy::SrptAging, 0, -1)), InvalidArgument);
}

TEST_CASE("a decode defers instead of preempting itself while a newer sequence is in flight") {
  Fixture f(3, config(64, 4));
  Sequence& a = f.add(4);  // 1 block
  Sequence& b = f.add(6);  // 2 blocks: the pool is full
  const ScheduledBatch b1 = f.sched.schedule();
  REQUIRE(tokens_of(b1, a) == 4);
  REQUIRE(tokens_of(b1, b) == 6);
  for (const ScheduledSeq& x : b1.seqs) x.seq->num_computed += x.num_tokens;
  f.sched.complete(b1);
  b.tokens.push_back(2);  // b sampled and decodes; a has not sampled yet, so it is not schedulable
  b.state = SeqState::Decoding;
  REQUIRE(f.cache.num_free_blocks() == 0);
  const ScheduledBatch b2 = f.sched.schedule();
  REQUIRE(b2.seqs.size() == 1);
  REQUIRE(b2.seqs[0].seq == &b);
  a.tokens.push_back(2);  // now a's decode needs a 2nd block, and only the in-flight b is newer
  a.state = SeqState::Decoding;
  const ScheduledBatch b3 = f.sched.schedule();
  CHECK(tokens_of(b3, a) == 0);
  CHECK(a.num_computed == 4);
  CHECK(a.num_preemptions == 0);
  CHECK(b.num_preemptions == 0);
  b.num_computed += 1;
  f.sched.complete(b2);
  const ScheduledBatch b4 = f.sched.schedule();
  CHECK(tokens_of(b4, a) == 1);
  CHECK(b.num_preemptions == 1);
  CHECK(a.num_preemptions == 0);
}

TEST_CASE("a decode does not count a preemption for a newer sequence that holds no blocks") {
  Fixture f(2, config(64, 4));
  Sequence& a = f.add(4);
  Sequence& c = f.add(4);
  f.run();  // one block each; both sample
  f.cache.release(c.block_table);  // c's forward failed and its table was truncated to nothing
  c.num_computed = 0;
  while (a.num_computed < 8) f.run();  // a grows into the pool; c cannot get a block
  REQUIRE(a.block_table.size() == 2);
  const ScheduledBatch s = f.run();  // a's 9th token needs a 3rd block that does not exist
  CHECK(tokens_of(s, a) == 0);
  CHECK(c.num_preemptions == 0);
  CHECK(a.num_preemptions == 1);
  CHECK(f.sched.num_preemptions() == 1);
}

TEST_CASE("a sequence preempted this step is not readmitted in the same step, even if it would fit") {
  Fixture f(3, config(5, 2));
  Sequence& a = f.add(3);
  Sequence& b = f.add(6);
  f.run();  // a: 3 tokens, b: first 2 of its prompt
  f.run();  // a decodes; b finishes its prompt and samples: the pool is full
  REQUIRE(f.cache.num_free_blocks() == 0);
  REQUIRE(b.block_table.size() == 2);
  const ScheduledBatch s = f.run();  // a needs a block: b is preempted, freeing two
  CHECK(tokens_of(s, a) == 1);
  CHECK(tokens_of(s, b) == 0);
  CHECK(b.state == SeqState::Waiting);
  CHECK(b.num_preemptions == 1);
  CHECK(tokens_of(f.run(), b) == 4);  // readmitted the step after
}
