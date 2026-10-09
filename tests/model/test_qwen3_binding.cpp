#include <doctest/doctest.h>

#include <random>

#include "backend/cpu/cpu_backend.h"
#include "core/error.h"
#include "model/loader.h"
#include "model/qwen3.h"
#include "support/test_data.h"
#include "support/tiny_model.h"

using namespace tie;

namespace {

ModelLimits small_limits() {
  ModelLimits l;
  l.max_tokens_per_step = 32;
  l.max_seqs = 2;
  l.max_blocks_per_seq = 16;
  l.max_logit_rows = 32;
  return l;
}

}  // namespace

TEST_CASE("Qwen3Model binds a complete checkpoint") {
  CpuBackend be(1);
  const LoadedModel m = test::make_tiny_model(1);
  const Qwen3Model model(be, m, small_limits());
  CHECK(model.config().num_layers == 2);
}

TEST_CASE("Qwen3Model binding is strict") {
  CpuBackend be(1);
  LoadedModel m = test::make_tiny_model(2);
  std::mt19937_64 rng(3);

  SUBCASE("missing tensor") {
    test::without_tensor(m, "model.layers.1.self_attn.k_norm.weight");
    CHECK_THROWS_WITH_AS(Qwen3Model(be, m, small_limits()), doctest::Contains("model.layers.1.self_attn.k_norm.weight"),
                         LoadError);
  }
  SUBCASE("unexpected tensor") {
    test::with_tensor(m, "model.layers.0.self_attn.q_proj.bias", DType::F32, Shape{64}, rng);
    CHECK_THROWS_WITH_AS(Qwen3Model(be, m, small_limits()), doctest::Contains("model.layers.0.self_attn.q_proj.bias"),
                         LoadError);
  }
  SUBCASE("transposed projection") {
    test::with_tensor(m, "model.layers.0.self_attn.q_proj.weight", DType::F32, Shape{32, 128}, rng);
    CHECK_THROWS_WITH_AS(Qwen3Model(be, m, small_limits()), doctest::Contains("q_proj"), LoadError);
  }
  SUBCASE("tied checkpoints may carry a duplicate lm_head") {
    test::with_tensor(m, "lm_head.weight", DType::F32, Shape{64, 32}, rng);
    CHECK_NOTHROW(Qwen3Model(be, m, small_limits()));
  }
  SUBCASE("untied checkpoints need lm_head") {
    m.config.tie_word_embeddings = false;
    CHECK_THROWS_WITH_AS(Qwen3Model(be, m, small_limits()), doctest::Contains("lm_head.weight"), LoadError);
  }
}

TEST_CASE("Qwen3Model binds Qwen3-0.6B from every supported file") {
  CpuBackend be(1);
  for (const char* file : {"Qwen3-0.6B", "Qwen3-0.6B-BF16.gguf", "Qwen3-0.6B-Q8_0.gguf"}) {
    const auto path = test::models_dir() / file;
    TIE_REQUIRE_DATA(path);
    CAPTURE(file);
    const LoadedModel m = load_model(path);
    CHECK_NOTHROW(Qwen3Model(be, m, small_limits()));
  }
}
