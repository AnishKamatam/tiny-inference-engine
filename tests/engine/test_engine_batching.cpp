#include <doctest/doctest.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

#include "backend/metal/metal_backend.h"
#include "engine/batch_scheduler.h"
#include "model/loader.h"
#include "support/engine_rig.h"
#include "support/test_data.h"
#include "support/tiny_model.h"

using namespace tie;
using test::Rig;

namespace {

struct Req {
  std::vector<int32_t> prompt;
  SamplingParams params;
};

std::vector<Req> mixed_requests(int32_t vocab, bool sampled) {
  const std::vector<int> lens = {3, 17, 9, 30, 1, 12, 25, 6};
  const std::vector<int32_t> outs = {10, 4, 12, 8, 6, 9, 5, 11};
  std::vector<Req> reqs;
  for (size_t i = 0; i < lens.size(); ++i) {
    Req r;
    for (int t = 0; t < lens[i]; ++t) r.prompt.push_back(int32_t((t * 7 + int(i) * 13 + 1) % vocab));
    r.params.max_tokens = outs[i];
    if (sampled) {
      r.params.temperature = 0.8f;
      r.params.top_k = 10;
      r.params.seed = 100 + i;
    }
    reqs.push_back(r);
  }
  return reqs;
}

// Runs all requests through one engine at once and returns each one's generated tokens.
std::vector<std::vector<int32_t>> run_together(Engine& engine, const std::vector<Req>& reqs) {
  std::vector<int32_t> ids;
  for (const Req& r : reqs) ids.push_back(engine.add_request(r.prompt, r.params));
  while (engine.has_work()) engine.step();
  std::vector<std::vector<int32_t>> out;
  for (int32_t id : ids) {
    const Sequence& s = engine.sequence(id);
    out.emplace_back(s.tokens.begin() + s.prompt_len, s.tokens.end());
  }
  return out;
}

}  // namespace

TEST_CASE("on CPU, requests batched together generate exactly what each generates alone") {
  for (bool sampled : {false, true}) {
    for (size_t kv_blocks : {size_t{256}, size_t{12}}) {  // 12 blocks of 4 tokens forces preemption
      CAPTURE(sampled);
      CAPTURE(kv_blocks);
      const std::vector<Req> reqs = mixed_requests(64, sampled);
      Rig together(test::make_tiny_model(20), 16, {}, kv_blocks, DType::F32, 0, /*max_seqs=*/8);
      const auto got = run_together(*together.engine, reqs);
      if (kv_blocks == 12) CHECK(together.engine->stats().preemptions > 0);
      CHECK(together.engine->stats().seqs_scheduled > together.engine->stats().steps);  // really batched
      for (size_t i = 0; i < reqs.size(); ++i) {
        CAPTURE(i);
        Rig alone(test::make_tiny_model(20), 16);
        CHECK(alone.engine->generate(reqs[i].prompt, reqs[i].params) == got[i]);
      }
    }
  }
}

TEST_CASE("a forward failure in a batched step leaves every request able to finish correctly") {
  const std::vector<Req> reqs = mixed_requests(64, false);
  Rig rig(test::make_tiny_model(21), 16, {}, 256, DType::F32, 0, /*max_seqs=*/8);
  int calls = 0;
  int running_at_failure = 0;
  std::vector<int32_t> ids;
  rig.model->set_layer_hook([&](int layer, const Tensor&) {
    if (layer != 1 || ++calls != 3) return;  // third step, mid-model
    for (int32_t id : ids) {
      const SeqState st = rig.engine->sequence(id).state;
      running_at_failure += st == SeqState::Prefilling || st == SeqState::Decoding;
    }
    throw std::runtime_error("injected failure");
  });
  for (const Req& r : reqs) ids.push_back(rig.engine->add_request(r.prompt, r.params));
  int failures = 0;
  for (int steps = 0; rig.engine->has_work(); ++steps) {
    REQUIRE(steps < 1000);
    try {
      rig.engine->step();
    } catch (const std::runtime_error&) {
      ++failures;
    }
  }
  CHECK(failures == 1);
  CHECK(running_at_failure > 1);  // the failure hit a step with several sequences in flight
  for (size_t i = 0; i < reqs.size(); ++i) {
    CAPTURE(i);
    const Sequence& s = rig.engine->sequence(ids[i]);
    Rig alone(test::make_tiny_model(21), 16);
    CHECK(std::vector<int32_t>(s.tokens.begin() + s.prompt_len, s.tokens.end()) ==
          alone.engine->generate(reqs[i].prompt, reqs[i].params));
  }
}

TEST_CASE("Qwen3-0.6B on CPU: batched greedy output equals each request alone") {
  const auto dir = test::models_dir() / "Qwen3-0.6B";
  const auto fixture = test::fixtures_dir() / "greedy.json";
  TIE_REQUIRE_DATA(dir);
  TIE_REQUIRE_DATA(fixture);
  const auto base = nlohmann::json::parse(std::ifstream(fixture)).at("prompt_ids").get<std::vector<int32_t>>();
  std::vector<Req> reqs;
  for (size_t cut : {base.size(), base.size() - 3, base.size() / 2, size_t{2}}) {
    Req r;
    r.prompt.assign(base.begin(), base.begin() + std::ptrdiff_t(cut));
    r.params.max_tokens = 12;
    reqs.push_back(r);
  }
  Rig together(load_model(dir), 64, {}, 64, DType::F32, 0, /*max_seqs=*/4);
  const auto got = run_together(*together.engine, reqs);
  for (size_t i = 0; i < reqs.size(); ++i) {
    CAPTURE(i);
    Rig alone(load_model(dir), 64, {}, 64);
    CHECK(alone.engine->generate(reqs[i].prompt, reqs[i].params) == got[i]);
  }
}

namespace {

// The production shape on Metal: F16 KV cache, 16-token blocks.
std::vector<std::vector<int32_t>> metal_run(const LoadedModel& weights, const std::vector<Req>& reqs, int32_t max_seqs) {
  MetalBackend metal;
  const ModelConfig& c = weights.config;
  KVCacheConfig kc;
  kc.num_layers = int32_t(c.num_layers);
  kc.num_kv_heads = int32_t(c.num_kv_heads);
  kc.head_dim = int32_t(c.head_dim);
  kc.budget_bytes = size_t{64} << 20;
  PagedKVCache cache(metal, kc);
  const ServingConfig serving{64, max_seqs};
  Qwen3Model model(metal, weights,
                   model_limits(serving, PagedKVCache::blocks_for(c.max_position_embeddings, kc.block_size)));
  Engine engine(model, cache, std::make_unique<BatchScheduler>(cache, test::fcfs(serving)), serving);
  return run_together(engine, reqs);
}

void check_metal_batched_prefix(const LoadedModel& weights, const std::vector<Req>& reqs, size_t prefix) {
  const auto together = metal_run(weights, reqs, int32_t(reqs.size()));
  for (size_t i = 0; i < reqs.size(); ++i) {
    CAPTURE(i);
    const auto alone = metal_run(weights, {reqs[i]}, 1)[0];
    REQUIRE(together[i].size() >= prefix);
    REQUIRE(alone.size() >= prefix);
    // Batching changes Metal's matmul path and reduction order, so only the first tokens must agree.
    CHECK(std::vector<int32_t>(together[i].begin(), together[i].begin() + std::ptrdiff_t(prefix)) ==
          std::vector<int32_t>(alone.begin(), alone.begin() + std::ptrdiff_t(prefix)));
  }
}

}  // namespace

TEST_CASE("on Metal, batched greedy generation starts like each request alone (tiny model)") {
  const LoadedModel weights = test::make_tiny_model(22);
  std::vector<Req> reqs = mixed_requests(64, false);
  reqs.resize(4);
  check_metal_batched_prefix(weights, reqs, 3);
}

TEST_CASE("on Metal, batched greedy generation starts like each request alone (Qwen3-0.6B)") {
  const auto dir = test::models_dir() / "Qwen3-0.6B";
  const auto fixture = test::fixtures_dir() / "greedy.json";
  TIE_REQUIRE_DATA(dir);
  TIE_REQUIRE_DATA(fixture);
  const auto base = nlohmann::json::parse(std::ifstream(fixture)).at("prompt_ids").get<std::vector<int32_t>>();
  std::vector<Req> reqs;
  for (size_t cut : {base.size(), base.size() - 3, base.size() / 2}) {
    Req r;
    r.prompt.assign(base.begin(), base.begin() + std::ptrdiff_t(cut));
    r.params.max_tokens = 8;
    reqs.push_back(r);
  }
  check_metal_batched_prefix(load_model(dir), reqs, 3);
}
