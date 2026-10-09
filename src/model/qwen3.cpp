#include "model/qwen3.h"

#include <cmath>
#include <cstring>
#include <format>
#include <set>
#include <string>

#include "core/error.h"
#include "kernels/cpu/vec.h"

namespace tie {

namespace {

size_t align64(size_t n) { return (n + 63) / 64 * 64; }

std::string join(const std::vector<std::string>& names) {
  std::string out;
  for (const std::string& n : names) out += (out.empty() ? "" : ", ") + n;
  return out;
}

}  // namespace

Qwen3Model::Qwen3Model(Backend& backend, const LoadedModel& weights, const ModelLimits& limits)
    : backend_(backend), config_(weights.config), limits_(limits) {
  config_.validate();
  if (limits_.max_tokens_per_step <= 0 || limits_.max_seqs <= 0 || limits_.max_blocks_per_seq <= 0 ||
      limits_.max_logit_rows <= 0) {
    fail<InvalidArgument>("model limits must all be positive");
  }
  bind_weights(weights);
  allocate_activations();
}

void Qwen3Model::bind_weights(const LoadedModel& weights) {
  for (const auto region : weights.regions) regions_.push_back(backend_.wrap(region.data(), region.size()));

  const int64_t D = config_.hidden_size;
  const int64_t Dh = config_.head_dim;
  const int64_t L = config_.num_layers;
  norms_ = backend_.alloc(sizeof(float) * static_cast<size_t>(L * (2 * D + 2 * Dh) + D));
  size_t norm_offset = 0;
  std::set<std::string, std::less<>> bound;

  const auto matrix = [&](const std::string& name, int64_t rows, int64_t cols) {
    const TensorInfo& info = weights.tensors.at(name);
    if (!(info.shape == Shape{rows, cols})) {
      fail<LoadError>("tensor '{}' has shape {}, expected [{}, {}]", name, info.shape.str(), rows, cols);
    }
    if (info.dtype == DType::I32) fail<LoadError>("tensor '{}' has unsupported dtype I32", name);
    if (info.dtype == DType::Q8_0 && cols % kQ8_0BlockElems != 0) {
      fail<LoadError>("Q8_0 tensor '{}' has {} columns, not a multiple of 32", name, cols);
    }
    bound.insert(name);
    return Tensor(regions_[info.region].get(), info.offset, info.dtype, info.shape);
  };

  const auto norm = [&](const std::string& name, int64_t dim) {
    const TensorInfo& info = weights.tensors.at(name);
    if (!(info.shape == Shape{dim})) fail<LoadError>("norm '{}' has shape {}, expected [{}]", name, info.shape.str(), dim);
    if (info.dtype != DType::F32 && info.dtype != DType::F16 && info.dtype != DType::BF16) {
      fail<LoadError>("norm '{}' must be F32, F16 or BF16, got {}", name, tie::name(info.dtype));
    }
    bound.insert(name);
    Tensor t(norms_.get(), norm_offset, DType::F32, Shape{dim});
    norm_offset += t.nbytes();
    to_f32(info.dtype, info.data, t.data<float>(), dim);
    return t;
  };

  const int64_t V = config_.vocab_size;
  const int64_t I = config_.intermediate_size;
  embed_tokens_ = matrix("model.embed_tokens.weight", V, D);
  layers_.reserve(static_cast<size_t>(L));
  for (int64_t l = 0; l < L; ++l) {
    const std::string p = std::format("model.layers.{}.", l);
    Layer& w = layers_.emplace_back();
    w.input_norm = norm(p + "input_layernorm.weight", D);
    w.q_proj = matrix(p + "self_attn.q_proj.weight", config_.q_dim(), D);
    w.k_proj = matrix(p + "self_attn.k_proj.weight", config_.kv_dim(), D);
    w.v_proj = matrix(p + "self_attn.v_proj.weight", config_.kv_dim(), D);
    w.o_proj = matrix(p + "self_attn.o_proj.weight", D, config_.q_dim());
    w.q_norm = norm(p + "self_attn.q_norm.weight", Dh);
    w.k_norm = norm(p + "self_attn.k_norm.weight", Dh);
    w.post_attention_norm = norm(p + "post_attention_layernorm.weight", D);
    w.gate_proj = matrix(p + "mlp.gate_proj.weight", I, D);
    w.up_proj = matrix(p + "mlp.up_proj.weight", I, D);
    w.down_proj = matrix(p + "mlp.down_proj.weight", D, I);
  }
  final_norm_ = norm("model.norm.weight", D);

  if (config_.tie_word_embeddings) {
    lm_head_ = embed_tokens_;
    // HF checkpoints of tied models still store a duplicate head; accept and ignore it.
    if (weights.tensors.find("lm_head.weight") != nullptr) matrix("lm_head.weight", V, D);
  } else {
    lm_head_ = matrix("lm_head.weight", V, D);
  }

  std::vector<std::string> leftover;
  for (const std::string& name : weights.tensors.names()) {
    if (!bound.contains(name)) leftover.push_back(name);
  }
  if (!leftover.empty()) fail<LoadError>("tensors not used by qwen3: {}", join(leftover));
}

void Qwen3Model::allocate_activations() {
  const int64_t T = limits_.max_tokens_per_step;
  const int64_t S = limits_.max_seqs;
  const int64_t R = limits_.max_logit_rows;
  const int64_t D = config_.hidden_size;
  struct Slot {
    Tensor* tensor;
    DType dtype;
    Shape shape;
  };
  const Slot slots[] = {
      {&x_, DType::F32, {T, D}},
      {&h_, DType::F32, {T, D}},
      {&q_, DType::F32, {T, config_.q_dim()}},
      {&k_, DType::F32, {T, config_.kv_dim()}},
      {&v_, DType::F32, {T, config_.kv_dim()}},
      {&attn_, DType::F32, {T, config_.q_dim()}},
      {&gate_, DType::F32, {T, config_.intermediate_size}},
      {&up_, DType::F32, {T, config_.intermediate_size}},
      {&last_, DType::F32, {R, D}},
      {&logits_, DType::F32, {R, config_.vocab_size}},
      {&tokens_, DType::I32, {T}},
      {&positions_, DType::I32, {T}},
      {&slots_, DType::I32, {T}},
      {&query_start_, DType::I32, {S + 1}},
      {&context_lens_, DType::I32, {S}},
      {&block_tables_, DType::I32, {S, limits_.max_blocks_per_seq}},
      {&logit_rows_, DType::I32, {R}},
  };
  size_t total = 0;
  for (const Slot& s : slots) total += align64(bytes_for(s.dtype, s.shape.numel()));
  arena_ = backend_.alloc(total);
  size_t offset = 0;
  for (const Slot& s : slots) {
    *s.tensor = Tensor(arena_.get(), offset, s.dtype, s.shape);
    offset += align64(bytes_for(s.dtype, s.shape.numel()));
  }
}

}  // namespace tie
