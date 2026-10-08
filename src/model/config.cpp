#include "model/config.h"

#include <nlohmann/json.hpp>
#include <utility>

#include "core/error.h"

namespace tie {

void ModelConfig::validate() const {
  if (architecture != "qwen3") {
    fail<UnsupportedError>("architecture '{}' is not supported (supported: qwen3)", architecture);
  }
  const std::pair<const char*, int64_t> dims[] = {
      {"vocab_size", vocab_size},     {"hidden_size", hidden_size}, {"intermediate_size", intermediate_size},
      {"num_layers", num_layers},     {"num_heads", num_heads},     {"num_kv_heads", num_kv_heads},
      {"head_dim", head_dim},         {"max_position_embeddings", max_position_embeddings},
  };
  for (const auto& [field, value] : dims) {
    if (value <= 0) fail<LoadError>("config {} must be positive, got {}", field, value);
  }
  if (num_heads % num_kv_heads != 0) {
    fail<LoadError>("num_heads {} is not a multiple of num_kv_heads {}", num_heads, num_kv_heads);
  }
  if (head_dim % 2 != 0) fail<LoadError>("head_dim {} must be even for rotary embeddings", head_dim);
  if (!(rms_norm_eps > 0.0f) || !(rope_theta > 0.0f)) {
    fail<LoadError>("rms_norm_eps ({}) and rope_theta ({}) must be positive", rms_norm_eps, rope_theta);
  }
}

ModelConfig config_from_hf_json(const nlohmann::json& j) {
  auto required = [&](const char* key) -> const nlohmann::json& {
    if (!j.contains(key)) fail<LoadError>("config.json has no '{}'", key);
    return j.at(key);
  };

  try {
    ModelConfig c;
    c.architecture = required("model_type").get<std::string>();
    if (c.architecture != "qwen3") {
      fail<UnsupportedError>("architecture '{}' is not supported (supported: qwen3)", c.architecture);
    }
    if (j.value("use_sliding_window", false)) {
      fail<UnsupportedError>("use_sliding_window=true is not supported; tie computes full attention");
    }
    if (j.contains("rope_scaling") && !j.at("rope_scaling").is_null()) {
      fail<UnsupportedError>("rope_scaling {} is not supported", j.at("rope_scaling").dump());
    }
    if (j.value("attention_bias", false)) fail<UnsupportedError>("attention_bias=true is not supported");

    c.vocab_size = required("vocab_size").get<int64_t>();
    c.hidden_size = required("hidden_size").get<int64_t>();
    c.intermediate_size = required("intermediate_size").get<int64_t>();
    c.num_layers = required("num_hidden_layers").get<int64_t>();
    c.num_heads = required("num_attention_heads").get<int64_t>();
    c.num_kv_heads = j.value("num_key_value_heads", c.num_heads);
    c.head_dim = j.contains("head_dim") ? j.at("head_dim").get<int64_t>() : c.hidden_size / c.num_heads;
    c.max_position_embeddings = required("max_position_embeddings").get<int64_t>();
    c.rms_norm_eps = required("rms_norm_eps").get<float>();
    c.rope_theta = j.value("rope_theta", 10000.0f);
    c.tie_word_embeddings = j.value("tie_word_embeddings", false);
    c.validate();
    return c;
  } catch (const nlohmann::json::exception& e) {
    fail<LoadError>("config.json: {}", e.what());
  }
}

}  // namespace tie
