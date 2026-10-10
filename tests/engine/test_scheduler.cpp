#include <doctest/doctest.h>

#include "engine/scheduler.h"

using namespace tie;

namespace {

Sequence make_seq(int32_t id, int n_prompt) {
  Sequence s;
  s.id = id;
  s.tokens.assign(size_t(n_prompt), 1);
  s.prompt_len = n_prompt;
  return s;
}

}  // namespace

TEST_CASE("FcfsScheduler chunks prefill, then decodes one token per step") {
  FcfsScheduler sched(4);
  Sequence a = make_seq(0, 10);
  sched.add(&a);
  CHECK(sched.has_work());

  for (int expected : {4, 4, 2}) {
    const ScheduledBatch b = sched.schedule();
    REQUIRE(b.seqs.size() == 1);
    CHECK(b.seqs[0].seq == &a);
    CHECK(b.seqs[0].num_tokens == expected);
    CHECK(a.state == SeqState::Prefilling);
    a.num_computed += expected;
  }
  a.tokens.push_back(5);  // the engine appends the sampled token
  CHECK(sched.schedule().seqs[0].num_tokens == 1);
}

TEST_CASE("FcfsScheduler runs requests one at a time in arrival order") {
  FcfsScheduler sched(16);
  Sequence a = make_seq(0, 3), b = make_seq(1, 3), c = make_seq(2, 3);
  sched.add(&a);
  sched.add(&b);
  sched.add(&c);
  CHECK(sched.schedule().seqs[0].seq == &a);
  CHECK(sched.schedule().seqs[0].seq == &a);  // still running
  sched.finish(&c);                           // cancelling a waiting request drops it
  sched.finish(&a);
  CHECK(sched.schedule().seqs[0].seq == &b);
  sched.finish(&b);
  CHECK(sched.schedule().empty());
  CHECK_FALSE(sched.has_work());
}
