#include <doctest/doctest.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "backend/metal/metal_backend.h"
#include "core/error.h"
#include "engine/engine.h"
#include "kv/paged_kv_cache.h"
#include "model/loader.h"
#include "model/qwen3.h"
#include "support/engine_rig.h"
#include "support/test_data.h"
#include "support/tiny_model.h"

using namespace tie;
using test::Rig;
using test::prompt;
using test::greedy;

namespace {

// Test-only: runs every queued sequence in one step, each with all its pending tokens
// (or a fixed, possibly invalid, token count), which FcfsScheduler never does.
class AllAtOnceScheduler final : public Scheduler {
 public:
  AllAtOnceScheduler(PagedKVCache& cache, int32_t forced_num_tokens = -1, bool reserve = true, bool duplicate = false)
      : cache_(cache), forced_num_tokens_(forced_num_tokens), reserve_(reserve), duplicate_(duplicate) {}
  void add(Sequence* seq) override { seqs_.push_back(seq); }
  ScheduledBatch schedule() override {
    ScheduledBatch batch;
    batch.id = ++next_id_;
    for (Sequence* seq : seqs_) {
      const int32_t n = forced_num_tokens_ >= 0 ? forced_num_tokens_ : seq->num_pending();
      if (reserve_) REQUIRE(cache_.reserve(seq->block_table, int64_t(seq->num_computed) + n));
      batch.seqs.push_back({seq, n});
      if (duplicate_) batch.seqs.push_back({seq, n});
    }
    return batch;
  }
  void complete(const ScheduledBatch&) override {}
  void abandon(const ScheduledBatch&) override {}
  void finish(Sequence* seq) override { std::erase(seqs_, seq); }
  bool has_work() const override { return !seqs_.empty(); }

 private:
  PagedKVCache& cache_;
  int32_t forced_num_tokens_;
  bool reserve_, duplicate_;
  uint64_t next_id_ = 0;
  std::vector<Sequence*> seqs_;
};

test::SchedulerFactory all_at_once(int32_t forced = -1, bool reserve = true, bool duplicate = false) {
  return [=](PagedKVCache& cache) { return std::make_unique<AllAtOnceScheduler>(cache, forced, reserve, duplicate); };
}

}  // namespace

TEST_CASE("generate stops at max_tokens and returns every KV block") {
  Rig rig(test::make_tiny_model(1), 64);
  const int32_t free_before = rig.cache->num_free_blocks();
  const int32_t id = rig.engine->add_request(prompt(10), greedy(5));
  while (rig.engine->has_work()) rig.engine->step();
  const Sequence& seq = rig.engine->sequence(id);
  CHECK(seq.num_generated() == 5);
  CHECK(seq.finish_reason == FinishReason::Length);
  CHECK(rig.cache->num_free_blocks() == free_before);
  rig.engine->release(id);
  CHECK_THROWS_AS(rig.engine->sequence(id), InvalidArgument);
}

TEST_CASE("generate stops on a stop token and includes it") {
  const std::vector<int32_t> free_run = Rig(test::make_tiny_model(2), 64).engine->generate(prompt(10), greedy(6));
  REQUIRE(free_run.size() == 6);
  const int32_t stop = free_run[2];
  Rig rig(test::make_tiny_model(2), 64, {stop});
  const std::vector<int32_t> stopped = rig.engine->generate(prompt(10), greedy(6));
  // Generation ends at the first occurrence of the stop token, which it includes.
  const auto first = std::find(free_run.begin(), free_run.end(), stop);
  CHECK(stopped == std::vector<int32_t>(free_run.begin(), first + 1));
  CHECK(rig.engine->stats().decode_tokens + 1 == int64_t(stopped.size()));
}

TEST_CASE("chunked prefill generates exactly what unchunked prefill does") {
  const auto whole = Rig(test::make_tiny_model(3), 64).engine->generate(prompt(23), greedy(8));
  const auto chunked = Rig(test::make_tiny_model(3), 5).engine->generate(prompt(23), greedy(8));
  CHECK(whole == chunked);
}

TEST_CASE("queued requests all complete and report their tokens") {
  Rig rig(test::make_tiny_model(4), 64);
  std::vector<std::pair<int32_t, int32_t>> seen;
  rig.engine->set_token_callback([&](int32_t seq, int32_t tok) { seen.emplace_back(seq, tok); });
  const int32_t a = rig.engine->add_request(prompt(4), greedy(3));
  const int32_t b = rig.engine->add_request(prompt(6), greedy(2));
  while (rig.engine->has_work()) rig.engine->step();
  CHECK(seen.size() == 5);
  CHECK(rig.engine->sequence(a).state == SeqState::Finished);
  CHECK(rig.engine->sequence(b).state == SeqState::Finished);
  CHECK(rig.engine->stats().prefill_tokens == 10);
  CHECK(rig.engine->stats().decode_tokens == 3);  // a decodes 2 tokens after its first, b decodes 1
}

// Runs a request to completion and returns its generated token count, checking it ends by length and frees its blocks.
int32_t run_clamped(Rig& rig, int prompt_len, int32_t max_tokens) {
  const int32_t free_before = rig.cache->num_free_blocks();
  const int32_t id = rig.engine->add_request(prompt(prompt_len), greedy(max_tokens));
  while (rig.engine->has_work()) rig.engine->step();
  const Sequence& seq = rig.engine->sequence(id);
  CHECK(seq.finish_reason == FinishReason::Length);
  CHECK(rig.cache->num_free_blocks() == free_before);
  return seq.num_generated();
}

TEST_CASE("add_request rejects prompts that cannot run and clamps max_tokens otherwise") {
  SUBCASE("context length") {
    Rig rig(test::make_tiny_model(5), 64);  // context 256
    CHECK_THROWS_WITH_AS(rig.engine->add_request(prompt(256), greedy(1)), doctest::Contains("context"), CapacityError);
    CHECK(run_clamped(rig, 250, 100) == 6);
  }
  SUBCASE("KV pool") {
    Rig rig(test::make_tiny_model(5), 64, {}, /*kv_blocks=*/4);  // 16 tokens
    CHECK_THROWS_WITH_AS(rig.engine->add_request(prompt(16), greedy(1)), doctest::Contains("--kv-mem"), CapacityError);
    CHECK(rig.cache->num_free_blocks() == 4);
    CHECK(run_clamped(rig, 10, 10) == 6);
    CHECK(run_clamped(rig, 15, 5) == 1);
  }
  SUBCASE("block-table row") {
    Rig rig(test::make_tiny_model(5), 64, {}, 256, DType::F32, /*max_blocks_per_seq=*/3);  // 12 tokens
    CHECK_THROWS_WITH_AS(rig.engine->add_request(prompt(12), greedy(1)), doctest::Contains("per sequence"), CapacityError);
    CHECK(run_clamped(rig, 10, 10) == 2);
    CHECK(run_clamped(rig, 6, 3) == 3);  // fits as asked
  }
  SUBCASE("bad prompts") {
    Rig rig(test::make_tiny_model(5), 64);
    CHECK_THROWS_AS(rig.engine->add_request({}, greedy(1)), InvalidArgument);
    CHECK_THROWS_AS(rig.engine->add_request({64}, greedy(1)), InvalidArgument);
    CHECK_THROWS_AS(rig.engine->add_request(prompt(3), greedy(0)), InvalidArgument);
  }
}

TEST_CASE("greedy generation on CPU matches HuggingFace token for token") {
  const auto dir = test::models_dir() / "Qwen3-0.6B";
  const auto fixture = test::fixtures_dir() / "greedy.json";
  TIE_REQUIRE_DATA(dir);
  TIE_REQUIRE_DATA(fixture);
  const auto ref = nlohmann::json::parse(std::ifstream(fixture));
  const auto prompt_ids = ref.at("prompt_ids").get<std::vector<int32_t>>();
  const auto want = ref.at("output_ids").get<std::vector<int32_t>>();

  Rig rig(load_model(dir), 64, {151645, 151643}, /*kv_blocks=*/64);
  CHECK(rig.engine->generate(prompt_ids, greedy(32)) == want);
}

TEST_CASE("greedy generation on Metal starts exactly like HuggingFace") {
  const auto dir = test::models_dir() / "Qwen3-0.6B";
  const auto fixture = test::fixtures_dir() / "greedy.json";
  TIE_REQUIRE_DATA(dir);
  TIE_REQUIRE_DATA(fixture);
  const auto ref = nlohmann::json::parse(std::ifstream(fixture));
  const auto prompt_ids = ref.at("prompt_ids").get<std::vector<int32_t>>();
  const auto want = ref.at("output_ids").get<std::vector<int32_t>>();

  // The production configuration: Metal, F16 KV cache, 16-token blocks.
  MetalBackend metal;
  const LoadedModel weights = load_model(dir);
  const ModelConfig& c = weights.config;
  KVCacheConfig kc;
  kc.num_layers = int32_t(c.num_layers);
  kc.num_kv_heads = int32_t(c.num_kv_heads);
  kc.head_dim = int32_t(c.head_dim);
  kc.budget_bytes = size_t{256} << 20;
  PagedKVCache cache(metal, kc);
  const ServingConfig serving{64, 1};
  Qwen3Model model(metal, weights,
                   model_limits(serving, PagedKVCache::blocks_for(c.max_position_embeddings, kc.block_size)));
  EngineConfig ec;
  ec.stop_tokens = {151645, 151643};
  Engine engine(model, cache, std::make_unique<FcfsScheduler>(cache, 64), serving, ec);

  const std::vector<int32_t> got = engine.generate(prompt_ids, greedy(32));
  // "Three primary colors are" — half-precision prefill tiles may change a later near-tie.
  REQUIRE(got.size() >= 5);
  CHECK(std::vector<int32_t>(got.begin(), got.begin() + 5) == std::vector<int32_t>(want.begin(), want.begin() + 5));
  MESSAGE("Metal greedy matches HF for " << std::mismatch(got.begin(), got.end(), want.begin(), want.end()).first - got.begin()
                                          << " of " << want.size() << " tokens");
}

TEST_CASE("a failing token callback finishes only that sequence and leaves the engine usable") {
  Rig rig(test::make_tiny_model(6), 64);
  const int32_t free_before = rig.cache->num_free_blocks();
  bool fail_first = true;
  rig.engine->set_token_callback([&](int32_t, int32_t) {
    if (fail_first) {
      fail_first = false;
      throw std::runtime_error("callback failed");
    }
  });
  const int32_t a = rig.engine->add_request(prompt(6), greedy(4));
  const int32_t b = rig.engine->add_request(prompt(5), greedy(3));
  CHECK_THROWS_AS(rig.engine->step(), std::runtime_error);
  CHECK(rig.engine->sequence(a).state == SeqState::Finished);
  CHECK(rig.engine->sequence(a).finish_reason == FinishReason::Error);
  CHECK(rig.cache->num_free_blocks() == free_before);
  REQUIRE(rig.engine->has_work());  // b is still queued
  while (rig.engine->has_work()) rig.engine->step();
  CHECK(rig.engine->sequence(b).finish_reason == FinishReason::Length);
  CHECK(rig.engine->sequence(b).num_generated() == 3);
  CHECK(rig.cache->num_free_blocks() == free_before);
}

TEST_CASE("a step whose forward fails rolls back its blocks and progress") {
  Rig rig(test::make_tiny_model(7), 64);
  // A cache whose geometry does not match the model makes forward() throw after blocks were reserved.
  KVCacheConfig kc = rig.cache->config();
  kc.num_layers += 1;
  PagedKVCache wrong(rig.backend, kc);
  Engine engine(*rig.model, wrong, std::make_unique<FcfsScheduler>(wrong, 64), ServingConfig{64, 1});
  const int32_t free_before = wrong.num_free_blocks();
  const int32_t id = engine.add_request(prompt(10), greedy(3));
  CHECK_THROWS_AS(engine.step(), InvalidArgument);
  const Sequence& seq = engine.sequence(id);
  CHECK(seq.num_computed == 0);
  CHECK(seq.block_table.empty());
  CHECK(wrong.num_free_blocks() == free_before);
}

TEST_CASE("a failing token callback in a two-sequence step finishes both sequences") {
  Rig rig(test::make_tiny_model(8), 64, {}, 256, DType::F32, 0, /*max_seqs=*/2, all_at_once());
  const int32_t total = rig.cache->num_free_blocks();
  rig.engine->set_token_callback([](int32_t, int32_t) { throw std::runtime_error("callback failed"); });
  const int32_t a = rig.engine->add_request(prompt(6), greedy(4));
  const int32_t b = rig.engine->add_request(prompt(5), greedy(3));
  CHECK_THROWS_AS(rig.engine->step(), std::runtime_error);
  for (int32_t id : {a, b}) {
    CHECK(rig.engine->sequence(id).state == SeqState::Finished);
    CHECK(rig.engine->sequence(id).finish_reason == FinishReason::Error);
  }
  CHECK(rig.cache->num_free_blocks() == total);
  CHECK_FALSE(rig.engine->has_work());
}

TEST_CASE("step rejects a scheduled token count the engine cannot run, before touching state") {
  for (int32_t bad : {0, 65}) {
    CAPTURE(bad);
    Rig rig(test::make_tiny_model(9), 64, {}, 256, DType::F32, 0, 1, all_at_once(bad));
    const int32_t total = rig.cache->num_free_blocks();
    const int32_t id = rig.engine->add_request(prompt(70), greedy(2));
    CHECK_THROWS_AS(rig.engine->step(), InvalidArgument);
    CHECK(rig.engine->sequence(id).num_computed == 0);
    CHECK(rig.engine->sequence(id).block_table.empty());
    CHECK(rig.cache->num_free_blocks() == total);
  }
}

TEST_CASE("generate cleans up its sequence when forward fails") {
  Rig rig(test::make_tiny_model(7), 64);
  KVCacheConfig kc = rig.cache->config();
  kc.num_layers += 1;
  PagedKVCache wrong(rig.backend, kc);
  Engine engine(*rig.model, wrong, std::make_unique<FcfsScheduler>(wrong, 64), ServingConfig{64, 1});
  const int32_t total = wrong.num_free_blocks();
  CHECK_THROWS_AS(engine.generate(prompt(10), greedy(3)), InvalidArgument);
  CHECK_FALSE(engine.has_work());
  CHECK(wrong.num_free_blocks() == total);
  CHECK_THROWS_AS(engine.sequence(0), InvalidArgument);  // the failed request is no longer registered
}

TEST_CASE("the engine rejects a serving config the model was not sized for") {
  Rig rig(test::make_tiny_model(1), 64);  // model sized for 64 tokens and 1 sequence per step
  const auto build = [&](ServingConfig serving) {
    Engine engine(*rig.model, *rig.cache, std::make_unique<FcfsScheduler>(*rig.cache, serving.max_tokens_per_step), serving);
  };
  CHECK_NOTHROW(build(ServingConfig{32, 1}));
  CHECK_THROWS_WITH_AS(build(ServingConfig{128, 1}), doctest::Contains("exceeds the model"), InvalidArgument);
  CHECK_THROWS_WITH_AS(build(ServingConfig{64, 2}), doctest::Contains("exceeds the model"), InvalidArgument);
}

TEST_CASE("step rejects a batch whose KV blocks were not reserved, or that repeats a sequence") {
  for (bool duplicate : {false, true}) {
    CAPTURE(duplicate);
    Rig rig(test::make_tiny_model(10), 64, {}, 256, DType::F32, 0, /*max_seqs=*/2,
            all_at_once(-1, /*reserve=*/duplicate, duplicate));
    const int32_t total = rig.cache->num_free_blocks();
    const int32_t id = rig.engine->add_request(prompt(9), greedy(2));
    CHECK_THROWS_WITH_AS(rig.engine->step(), doctest::Contains(duplicate ? "twice" : "KV blocks"), InvalidArgument);
    CHECK(rig.engine->sequence(id).num_computed == 0);
    CHECK(rig.cache->num_free_blocks() == total);
  }
}

TEST_CASE("a forward failure abandons the batch so the scheduler offers it again") {
  Rig rig(test::make_tiny_model(7), 64);
  KVCacheConfig kc = rig.cache->config();
  kc.num_layers += 1;
  PagedKVCache wrong(rig.backend, kc);
  Engine engine(*rig.model, wrong, std::make_unique<FcfsScheduler>(wrong, 64), ServingConfig{64, 1});
  engine.add_request(prompt(10), greedy(3));
  CHECK_THROWS_AS(engine.step(), InvalidArgument);
  CHECK_THROWS_AS(engine.step(), InvalidArgument);  // rescheduled, not silently dropped
}

TEST_CASE("step rejects a batch with more sequences than the serving config allows") {
  Rig rig(test::make_tiny_model(11), 64, {}, 256, DType::F32, 0, /*max_seqs=*/1, all_at_once());
  const int32_t total = rig.cache->num_free_blocks();
  const int32_t a = rig.engine->add_request(prompt(6), greedy(2));
  const int32_t b = rig.engine->add_request(prompt(5), greedy(2));
  CHECK_THROWS_WITH_AS(rig.engine->step(), doctest::Contains("a step allows at most"), InvalidArgument);
  CHECK(rig.cache->num_free_blocks() == total);
  CHECK(rig.engine->sequence(a).block_table.empty());
  CHECK(rig.engine->sequence(b).block_table.empty());
}
