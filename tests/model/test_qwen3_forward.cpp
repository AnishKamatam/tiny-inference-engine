#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <string>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "backend/metal/metal_backend.h"
#include "core/error.h"
#include "kv/paged_kv_cache.h"
#include "model/loader.h"
#include "model/qwen3.h"
#include "support/npy.h"
#include "support/sequence_runner.h"
#include "support/test_data.h"
#include "support/tiny_model.h"

using namespace tie;

namespace {

KVCacheConfig cache_config(const ModelConfig& c, DType dtype = DType::F32) {
  KVCacheConfig k;
  k.num_layers = static_cast<int32_t>(c.num_layers);
  k.num_kv_heads = static_cast<int32_t>(c.num_kv_heads);
  k.head_dim = static_cast<int32_t>(c.head_dim);
  k.block_size = 4;
  k.dtype = dtype;
  k.budget_bytes = size_t{64} << 20;
  return k;
}

ModelLimits limits(int32_t max_tokens, int32_t max_seqs, int32_t max_logit_rows) {
  ModelLimits l;
  l.max_tokens_per_step = max_tokens;
  l.max_seqs = max_seqs;
  l.max_blocks_per_seq = 64;
  l.max_logit_rows = max_logit_rows;
  return l;
}

std::vector<int32_t> prompt(int n) {
  std::vector<int32_t> ids;
  for (int i = 0; i < n; ++i) ids.push_back((i * 7 + 3) % 64);
  return ids;
}

float max_abs_diff(std::span<const float> a, std::span<const float> b) {
  float m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

int argmax(std::span<const float> v) { return int(std::max_element(v.begin(), v.end()) - v.begin()); }

}  // namespace

TEST_CASE("one pass, chunked prefill and token-by-token decode agree") {
  CpuBackend be(4);
  const LoadedModel m = test::make_tiny_model(7);
  Qwen3Model model(be, m, limits(32, 1, 32));
  PagedKVCache cache(be, cache_config(m.config));
  const std::vector<int32_t> ids = prompt(20);
  const int64_t V = m.config.vocab_size;

  std::vector<float> all;
  {
    test::SequenceRunner run(model, cache);
    all = run.step(ids, /*all_logits=*/true);
  }
  std::vector<float> chunked;
  {
    test::SequenceRunner run(model, cache);
    for (size_t i = 0; i < ids.size(); i += 7) {
      chunked = run.step(std::span(ids).subspan(i, std::min<size_t>(7, ids.size() - i)));
    }
  }
  test::SequenceRunner run(model, cache);
  for (size_t i = 0; i < ids.size(); ++i) {
    const std::vector<float> one = run.step(std::span(ids).subspan(i, 1));
    CAPTURE(i);
    CHECK(max_abs_diff(one, std::span(all).subspan(i * size_t(V), size_t(V))) < 1e-4f);
  }
  CHECK(max_abs_diff(chunked, std::span(all).subspan((ids.size() - 1) * size_t(V), size_t(V))) < 1e-4f);
}

TEST_CASE("two sequences in one batch equal each sequence run alone") {
  CpuBackend be(4);
  const LoadedModel m = test::make_tiny_model(8);
  Qwen3Model model(be, m, limits(32, 2, 2));
  PagedKVCache cache(be, cache_config(m.config));
  const std::vector<int32_t> a = prompt(6), b = prompt(9);

  std::vector<float> alone_a, alone_b;
  {
    test::SequenceRunner run(model, cache);
    alone_a = run.step(a);
  }
  {
    test::SequenceRunner run(model, cache);
    alone_b = run.step(b);
  }

  BlockTable ta, tb;
  std::vector<int32_t> slots;
  cache.append_slots(ta, 0, 6, slots);
  cache.append_slots(tb, 0, 9, slots);
  std::vector<int32_t> tokens = a, positions;
  tokens.insert(tokens.end(), b.begin(), b.end());
  for (int i = 0; i < 6; ++i) positions.push_back(i);
  for (int i = 0; i < 9; ++i) positions.push_back(i);
  std::vector<int32_t> tables(2 * 64, 0);
  std::copy(ta.begin(), ta.end(), tables.begin());
  std::copy(tb.begin(), tb.end(), tables.begin() + 64);
  const std::vector<int32_t> query_start = {0, 6, 15}, context_lens = {6, 9}, rows = {5, 14};

  std::vector<float> logits;
  model.forward(BatchInput{tokens, positions, slots, query_start, context_lens, tables, rows}, cache, logits);
  const size_t V = size_t(m.config.vocab_size);
  CHECK(max_abs_diff(std::span(logits).first(V), alone_a) < 1e-4f);
  CHECK(max_abs_diff(std::span(logits).subspan(V, V), alone_b) < 1e-4f);
  cache.release(ta);
  cache.release(tb);
}

TEST_CASE("forward rejects steps beyond the model limits") {
  CpuBackend be(1);
  const LoadedModel m = test::make_tiny_model(9);
  Qwen3Model model(be, m, limits(8, 1, 1));
  PagedKVCache cache(be, cache_config(m.config));
  test::SequenceRunner run(model, cache);
  CHECK_THROWS_AS(run.step(prompt(9)), CapacityError);

  KVCacheConfig wrong = cache_config(m.config);
  wrong.num_layers = 3;
  PagedKVCache other(be, wrong);
  test::SequenceRunner bad(model, other);
  CHECK_THROWS_AS(bad.step(prompt(2)), InvalidArgument);
}

TEST_CASE("Qwen3-0.6B on CPU matches HuggingFace layer by layer") {
  const auto dir = test::models_dir() / "Qwen3-0.6B";
  const auto fixtures = test::fixtures_dir();
  TIE_REQUIRE_DATA(dir);
  TIE_REQUIRE_DATA(fixtures / "parity_logits.npy");

  CpuBackend be;
  const LoadedModel m = load_model(dir);
  Qwen3Model model(be, m, limits(64, 1, 32));
  PagedKVCache cache(be, cache_config(m.config));

  const test::NpyArray ids_npy = test::load_npy(fixtures / "parity_ids.npy");
  const std::vector<int32_t> ids(ids_npy.as<int32_t>(), ids_npy.as<int32_t>() + ids_npy.numel());
  std::vector<std::vector<float>> hidden(size_t(m.config.num_layers));
  model.set_layer_hook([&](int layer, const Tensor& h) {
    hidden[size_t(layer)].assign(h.data<float>(), h.data<float>() + h.numel());
  });

  test::SequenceRunner run(model, cache);
  const std::vector<float> logits = run.step(ids, /*all_logits=*/true);

  for (int64_t l = 0; l < m.config.num_layers; ++l) {
    const test::NpyArray ref = test::load_npy(fixtures / std::format("parity_hidden_{}.npy", l));
    const std::span<const float> want(ref.as<float>(), size_t(ref.numel()));
    REQUIRE(hidden[size_t(l)].size() == want.size());
    float ref_max = 0;
    for (float x : want) ref_max = std::max(ref_max, std::abs(x));
    const float diff = max_abs_diff(hidden[size_t(l)], want);
    CAPTURE(l);
    MESSAGE("layer " << l << ": max|diff| = " << diff << ", max|ref| = " << ref_max);
    CHECK(diff <= 1e-3f * ref_max);
  }

  const test::NpyArray ref_logits = test::load_npy(fixtures / "parity_logits.npy");
  const std::span<const float> want(ref_logits.as<float>(), size_t(ref_logits.numel()));
  REQUIRE(logits.size() == want.size());
  const float logit_diff = max_abs_diff(logits, want);
  MESSAGE("logits: max|diff| = " << logit_diff);
  CHECK(logit_diff <= 1e-2f);
  const size_t V = size_t(m.config.vocab_size);
  for (size_t t = 0; t < ids.size(); ++t) {
    CAPTURE(t);
    CHECK(argmax(std::span(logits).subspan(t * V, V)) == argmax(want.subspan(t * V, V)));
  }
}

TEST_CASE("GGUF BF16 and safetensors give the same logits; Q8_0 keeps the same top-1") {
  const auto fixtures = test::fixtures_dir();
  const auto st = test::models_dir() / "Qwen3-0.6B";
  const auto bf16 = test::models_dir() / "Qwen3-0.6B-BF16.gguf";
  const auto q8 = test::models_dir() / "Qwen3-0.6B-Q8_0.gguf";
  TIE_REQUIRE_DATA(fixtures / "parity_ids.npy");
  TIE_REQUIRE_DATA(st);
  TIE_REQUIRE_DATA(bf16);
  TIE_REQUIRE_DATA(q8);

  const test::NpyArray ids_npy = test::load_npy(fixtures / "parity_ids.npy");
  const std::vector<int32_t> ids(ids_npy.as<int32_t>(), ids_npy.as<int32_t>() + ids_npy.numel());
  CpuBackend be;
  const auto run_file = [&](const std::filesystem::path& path) {
    const LoadedModel m = load_model(path);
    Qwen3Model model(be, m, limits(64, 1, 32));
    PagedKVCache cache(be, cache_config(m.config));
    test::SequenceRunner run(model, cache);
    return run.step(ids, /*all_logits=*/true);
  };

  const std::vector<float> from_st = run_file(st);
  CHECK(max_abs_diff(run_file(bf16), from_st) <= 1e-5f);

  const std::vector<float> from_q8 = run_file(q8);
  const size_t V = from_st.size() / ids.size();
  size_t agree = 0;
  for (size_t t = 0; t < ids.size(); ++t) {
    agree += argmax(std::span(from_q8).subspan(t * V, V)) == argmax(std::span(from_st).subspan(t * V, V));
  }
  MESSAGE("Q8_0 top-1 agreement: " << agree << "/" << ids.size());
  CHECK(agree * 5 >= ids.size() * 4);  // at least 80%
}

TEST_CASE("the tiny model gives the same logits on Metal and CPU") {
  CpuBackend cpu(4);
  MetalBackend metal;
  const LoadedModel m = test::make_tiny_model(10);
  const std::vector<int32_t> ids = prompt(20);
  const auto run_on = [&](Backend& be) {
    Qwen3Model model(be, m, limits(32, 1, 32));
    PagedKVCache cache(be, cache_config(m.config));
    test::SequenceRunner run(model, cache);
    std::vector<float> logits;
    for (size_t i = 0; i < ids.size(); i += 8) {  // steps of at most 8 tokens
      const std::vector<float> part = run.step(std::span(ids).subspan(i, std::min<size_t>(8, ids.size() - i)), true);
      logits.insert(logits.end(), part.begin(), part.end());
    }
    return logits;
  };
  // Steps of <= 8 tokens keep every Metal matmul on the F32 matvec path, so results agree tightly.
  CHECK(max_abs_diff(run_on(metal), run_on(cpu)) < 1e-4f);
}

TEST_CASE("Qwen3-0.6B on Metal agrees with the CPU reference") {
  const auto fixtures = test::fixtures_dir();
  TIE_REQUIRE_DATA(fixtures / "parity_ids.npy");
  const test::NpyArray ids_npy = test::load_npy(fixtures / "parity_ids.npy");
  const std::vector<int32_t> ids(ids_npy.as<int32_t>(), ids_npy.as<int32_t>() + ids_npy.numel());
  CpuBackend cpu;
  MetalBackend metal;

  for (const std::string file : {"Qwen3-0.6B", "Qwen3-0.6B-Q8_0.gguf"}) {
    const auto path = test::models_dir() / file;
    TIE_REQUIRE_DATA(path);
    CAPTURE(file);
    const LoadedModel m = load_model(path);
    const auto run_on = [&](Backend& be, DType kv_dtype) {
      Qwen3Model model(be, m, limits(64, 1, 32));
      PagedKVCache cache(be, cache_config(m.config, kv_dtype));
      test::SequenceRunner run(model, cache);
      return run.step(ids, /*all_logits=*/true);
    };
    const std::vector<float> want = run_on(cpu, DType::F32);
    const std::vector<float> got = run_on(metal, DType::F16);  // the production configuration
    const size_t V = want.size() / ids.size();
    size_t agree = 0;
    for (size_t t = 0; t < ids.size(); ++t) {
      agree += argmax(std::span(got).subspan(t * V, V)) == argmax(std::span(want).subspan(t * V, V));
    }
    MESSAGE(file << ": Metal top-1 agreement " << agree << "/" << ids.size() << ", max |logit diff| "
                 << max_abs_diff(got, want));
    CHECK(agree + 1 >= ids.size());  // half-precision prefill tiles may flip at most one near-tie
  }
}
