#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include "core/error.h"
#include "model/config.h"

using namespace tie;

namespace {

nlohmann::json qwen3_json() {
  return nlohmann::json::parse(R"({
    "model_type": "qwen3", "vocab_size": 151936, "hidden_size": 1024, "intermediate_size": 3072,
    "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
    "max_position_embeddings": 40960, "rms_norm_eps": 1e-06, "rope_theta": 1000000,
    "tie_word_embeddings": true, "use_sliding_window": false, "rope_scaling": null, "attention_bias": false
  })");
}

}  // namespace

TEST_CASE("config_from_hf_json reads Qwen3-0.6B") {
  const ModelConfig c = config_from_hf_json(qwen3_json());
  CHECK(c.architecture == "qwen3");
  CHECK(c.hidden_size == 1024);
  CHECK(c.num_layers == 28);
  CHECK(c.head_dim == 128);
  CHECK(c.q_dim() == 2048);
  CHECK(c.kv_dim() == 1024);
  CHECK(c.rope_theta == 1e6f);
  CHECK(c.rms_norm_eps == 1e-6f);
  CHECK(c.tie_word_embeddings);
}

TEST_CASE("config_from_hf_json derives head_dim when absent") {
  nlohmann::json j = qwen3_json();
  j.erase("head_dim");
  CHECK(config_from_hf_json(j).head_dim == 1024 / 16);
}

TEST_CASE("config_from_hf_json rejects what tie cannot run") {
  nlohmann::json j = qwen3_json();
  SUBCASE("other architecture") {
    j["model_type"] = "llama";
    CHECK_THROWS_WITH_AS(config_from_hf_json(j), doctest::Contains("llama"), UnsupportedError);
  }
  SUBCASE("sliding window") {
    j["use_sliding_window"] = true;
    CHECK_THROWS_AS(config_from_hf_json(j), UnsupportedError);
  }
  SUBCASE("rope scaling") {
    j["rope_scaling"] = {{"type", "yarn"}};
    CHECK_THROWS_AS(config_from_hf_json(j), UnsupportedError);
  }
  SUBCASE("missing field") {
    j.erase("hidden_size");
    CHECK_THROWS_WITH_AS(config_from_hf_json(j), doctest::Contains("hidden_size"), LoadError);
  }
  SUBCASE("heads not a multiple of kv heads") {
    j["num_key_value_heads"] = 5;
    CHECK_THROWS_AS(config_from_hf_json(j), LoadError);
  }
}
