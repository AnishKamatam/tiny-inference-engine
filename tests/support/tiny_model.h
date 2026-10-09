#pragma once

#include <format>
#include <memory>
#include <random>
#include <span>
#include <string>

#include "core/buffer.h"
#include "core/half.h"
#include "model/loader.h"

namespace tie::test {

// Qwen3-shaped but tiny: q_dim (128) != hidden (32), like Qwen3-0.6B's non-square projections.
// head_dim is 32, the smallest the Metal attention kernel supports.
inline ModelConfig tiny_config() {
  ModelConfig c;
  c.architecture = "qwen3";
  c.vocab_size = 64;
  c.hidden_size = 32;
  c.intermediate_size = 64;
  c.num_layers = 2;
  c.num_heads = 4;
  c.num_kv_heads = 2;
  c.head_dim = 32;
  c.max_position_embeddings = 256;
  c.rms_norm_eps = 1e-6f;
  c.rope_theta = 10000.0f;
  c.tie_word_embeddings = true;
  return c;
}

// Adds (or replaces) a tensor backed by its own host buffer.
inline void with_tensor(LoadedModel& m, const std::string& name, DType dtype, Shape shape, std::mt19937_64& rng,
                        float mean = 0.0f, float stddev = 0.1f) {
  auto buffer = std::make_shared<HostBuffer>(bytes_for(dtype, shape.numel()));
  std::normal_distribution<float> dist(mean, stddev);
  for (int64_t i = 0; i < shape.numel(); ++i) {
    const float v = dist(rng);
    if (dtype == DType::F32) static_cast<float*>(buffer->data())[i] = v;
    if (dtype == DType::F16) static_cast<uint16_t*>(buffer->data())[i] = f32_to_f16(v);
    if (dtype == DType::BF16) static_cast<uint16_t*>(buffer->data())[i] = f32_to_bf16(v);
  }
  const auto* data = static_cast<const uint8_t*>(buffer->data());
  m.regions.emplace_back(data, buffer->size());
  m.owners.push_back(buffer);

  TensorTable rebuilt;
  for (const std::string& n : m.tensors.names()) {
    if (n != name) rebuilt.add(n, m.tensors.at(n));
  }
  rebuilt.add(name, TensorInfo{dtype, shape, m.regions.size() - 1, 0, buffer->size(), data});
  m.tensors = std::move(rebuilt);
}

inline void without_tensor(LoadedModel& m, const std::string& name) {
  TensorTable rebuilt;
  for (const std::string& n : m.tensors.names()) {
    if (n != name) rebuilt.add(n, m.tensors.at(n));
  }
  m.tensors = std::move(rebuilt);
}

// A randomly initialized tiny Qwen3 with HF tensor names.
inline LoadedModel make_tiny_model(uint64_t seed, DType matrix_dtype = DType::F32) {
  LoadedModel m;
  m.config = tiny_config();
  const ModelConfig& c = m.config;
  std::mt19937_64 rng(seed);
  const auto matrix = [&](const std::string& name, int64_t rows, int64_t cols) {
    with_tensor(m, name, matrix_dtype, Shape{rows, cols}, rng);
  };
  const auto norm = [&](const std::string& name, int64_t dim) { with_tensor(m, name, DType::F32, Shape{dim}, rng, 1.0f, 0.1f); };

  matrix("model.embed_tokens.weight", c.vocab_size, c.hidden_size);
  for (int64_t l = 0; l < c.num_layers; ++l) {
    const std::string p = std::format("model.layers.{}.", l);
    norm(p + "input_layernorm.weight", c.hidden_size);
    matrix(p + "self_attn.q_proj.weight", c.q_dim(), c.hidden_size);
    matrix(p + "self_attn.k_proj.weight", c.kv_dim(), c.hidden_size);
    matrix(p + "self_attn.v_proj.weight", c.kv_dim(), c.hidden_size);
    matrix(p + "self_attn.o_proj.weight", c.hidden_size, c.q_dim());
    norm(p + "self_attn.q_norm.weight", c.head_dim);
    norm(p + "self_attn.k_norm.weight", c.head_dim);
    norm(p + "post_attention_layernorm.weight", c.hidden_size);
    matrix(p + "mlp.gate_proj.weight", c.intermediate_size, c.hidden_size);
    matrix(p + "mlp.up_proj.weight", c.intermediate_size, c.hidden_size);
    matrix(p + "mlp.down_proj.weight", c.hidden_size, c.intermediate_size);
  }
  norm("model.norm.weight", c.hidden_size);
  return m;
}

}  // namespace tie::test
