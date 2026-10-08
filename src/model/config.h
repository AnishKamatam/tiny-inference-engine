#pragma once

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>

namespace tie {

struct ModelConfig {
  std::string architecture;  // only "qwen3" is supported
  int64_t vocab_size = 0;
  int64_t hidden_size = 0;
  int64_t intermediate_size = 0;
  int64_t num_layers = 0;
  int64_t num_heads = 0;
  int64_t num_kv_heads = 0;
  int64_t head_dim = 0;
  int64_t max_position_embeddings = 0;
  float rms_norm_eps = 0.0f;
  float rope_theta = 0.0f;
  bool tie_word_embeddings = false;

  int64_t q_dim() const { return num_heads * head_dim; }
  int64_t kv_dim() const { return num_kv_heads * head_dim; }

  // Throws UnsupportedError for architectures tie cannot run and LoadError for
  // values that are inconsistent.
  void validate() const;

  bool operator==(const ModelConfig&) const = default;
};

// Parses a HuggingFace config.json.
ModelConfig config_from_hf_json(const nlohmann::json& j);

}  // namespace tie
