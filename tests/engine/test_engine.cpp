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
#include "support/test_data.h"
#include "support/tiny_model.h"

using namespace tie;

namespace {

// A tiny model, its cache and an engine, wired the way the CLI wires the real one.
struct Rig {
  CpuBackend backend{2};
  LoadedModel weights;
  std::unique_ptr<Qwen3Model> model;
  std::unique_ptr<PagedKVCache> cache;
  std::unique_ptr<Engine> engine;

  Rig(LoadedModel w, int32_t max_tokens_per_step, std::vector<int32_t> stop_tokens = {},
      size_t kv_blocks = 256, DType kv_dtype = DType::F32, int32_t max_blocks_per_seq = 0)
      : weights(std::move(w)) {
    const ModelConfig& c = weights.config;
    KVCacheConfig kc;
    kc.num_layers = int32_t(c.num_layers);
    kc.num_kv_heads = int32_t(c.num_kv_heads);
    kc.head_dim = int32_t(c.head_dim);
    kc.block_size = 4;
    kc.dtype = kv_dtype;
    kc.budget_bytes = kv_blocks * 2 * size_t(c.num_layers * c.num_kv_heads * c.head_dim) * 4 * bytes_for(kv_dtype, 1);
    cache = std::make_unique<PagedKVCache>(backend, kc);

    ModelLimits limits;
    limits.max_tokens_per_step = max_tokens_per_step;
    limits.max_blocks_per_seq =
        max_blocks_per_seq > 0 ? max_blocks_per_seq : PagedKVCache::blocks_for(c.max_position_embeddings, kc.block_size);
    model = std::make_unique<Qwen3Model>(backend, weights, limits);

    EngineConfig ec;
    ec.max_tokens_per_step = max_tokens_per_step;
    ec.stop_tokens = std::move(stop_tokens);
    engine = std::make_unique<Engine>(*model, *cache, std::make_unique<FcfsScheduler>(max_tokens_per_step), ec);
  }
};

std::vector<int32_t> prompt(int n) {
  std::vector<int32_t> ids;
  for (int i = 0; i < n; ++i) ids.push_back((i * 5 + 1) % 64);
  return ids;
}

SamplingParams greedy(int32_t max_tokens) {
  SamplingParams p;
  p.max_tokens = max_tokens;
  return p;
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
  ModelLimits limits;
  limits.max_tokens_per_step = 64;
  limits.max_blocks_per_seq = PagedKVCache::blocks_for(c.max_position_embeddings, kc.block_size);
  Qwen3Model model(metal, weights, limits);
  EngineConfig ec;
  ec.max_tokens_per_step = 64;
  ec.stop_tokens = {151645, 151643};
  Engine engine(model, cache, std::make_unique<FcfsScheduler>(64), ec);

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
  EngineConfig ec;
  ec.max_tokens_per_step = 64;
  Engine engine(*rig.model, wrong, std::make_unique<FcfsScheduler>(64), ec);
  const int32_t free_before = wrong.num_free_blocks();
  const int32_t id = engine.add_request(prompt(10), greedy(3));
  CHECK_THROWS_AS(engine.step(), InvalidArgument);
  const Sequence& seq = engine.sequence(id);
  CHECK(seq.num_computed == 0);
  CHECK(seq.block_table.empty());
  CHECK(wrong.num_free_blocks() == free_before);
}
