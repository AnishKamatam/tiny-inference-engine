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

namespace {

// The first n elements of a 1-D tensor.
Tensor prefix(const Tensor& t, int64_t n) { return Tensor(t.buffer(), t.offset(), t.dtype(), Shape{n}); }

void copy_into(const Tensor& dst, std::span<const int32_t> src) {
  std::memcpy(dst.raw(), src.data(), src.size_bytes());
}

}  // namespace

void Qwen3Model::forward(const BatchInput& in, PagedKVCache& cache, std::vector<float>& logits) {
  const auto T = static_cast<int64_t>(in.tokens.size());
  const auto S = static_cast<int64_t>(in.context_lens.size());
  const auto R = static_cast<int64_t>(in.logit_rows.size());
  if (T < 1 || T > limits_.max_tokens_per_step) {
    fail<CapacityError>("step has {} tokens; the model allows 1 to {}", T, limits_.max_tokens_per_step);
  }
  if (S < 1 || S > limits_.max_seqs) fail<CapacityError>("step has {} sequences; the model allows 1 to {}", S, limits_.max_seqs);
  if (R < 1 || R > limits_.max_logit_rows) {
    fail<CapacityError>("step requests {} logit rows; the model allows 1 to {}", R, limits_.max_logit_rows);
  }
  if (static_cast<int64_t>(in.positions.size()) != T || static_cast<int64_t>(in.slot_mapping.size()) != T ||
      static_cast<int64_t>(in.query_start.size()) != S + 1 ||
      static_cast<int64_t>(in.block_tables.size()) != S * limits_.max_blocks_per_seq) {
    fail<InvalidArgument>("BatchInput sizes are inconsistent with {} tokens and {} sequences", T, S);
  }
  const KVCacheConfig& kv = cache.config();
  if (kv.num_layers != config_.num_layers || kv.num_kv_heads != config_.num_kv_heads || kv.head_dim != config_.head_dim) {
    fail<InvalidArgument>("KV cache geometry ({} layers, {} kv heads, head_dim {}) does not match the model", kv.num_layers,
                          kv.num_kv_heads, kv.head_dim);
  }

  copy_into(tokens_, in.tokens);
  copy_into(positions_, in.positions);
  copy_into(slots_, in.slot_mapping);
  copy_into(query_start_, in.query_start);
  copy_into(context_lens_, in.context_lens);
  copy_into(block_tables_, in.block_tables);
  copy_into(logit_rows_, in.logit_rows);

  const int heads = static_cast<int>(config_.num_heads);
  const int kv_heads = static_cast<int>(config_.num_kv_heads);
  const int head_dim = static_cast<int>(config_.head_dim);
  const float eps = config_.rms_norm_eps;
  const float theta = config_.rope_theta;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  const Tensor tokens = prefix(tokens_, T);
  const Tensor positions = prefix(positions_, T);
  const Tensor slots = prefix(slots_, T);
  Tensor x = x_.slice_rows(0, T);
  Tensor h = h_.slice_rows(0, T);
  Tensor q = q_.slice_rows(0, T);
  Tensor k = k_.slice_rows(0, T);
  Tensor v = v_.slice_rows(0, T);
  Tensor attn = attn_.slice_rows(0, T);
  Tensor gate = gate_.slice_rows(0, T);
  Tensor up = up_.slice_rows(0, T);
  Tensor q_heads = q.reshape(Shape{T * heads, head_dim});
  Tensor k_heads = k.reshape(Shape{T * kv_heads, head_dim});
  const AttentionMetadata meta{static_cast<int32_t>(S), query_start_, context_lens_, block_tables_};

  backend_.begin_step();
  backend_.embed(embed_tokens_, tokens, x);
  for (size_t l = 0; l < layers_.size(); ++l) {
    const Layer& w = layers_[l];

    // Attention block. QK-norm runs per head and before RoPE, as in HF Qwen3.
    backend_.rms_norm(x, w.input_norm, eps, h);
    backend_.matmul(h, w.q_proj, q);
    backend_.matmul(h, w.k_proj, k);
    backend_.matmul(h, w.v_proj, v);
    backend_.rms_norm(q_heads, w.q_norm, eps, q_heads);
    backend_.rms_norm(k_heads, w.k_norm, eps, k_heads);
    backend_.rope_neox(q, positions, heads, head_dim, theta);
    backend_.rope_neox(k, positions, kv_heads, head_dim, theta);
    Tensor k_cache = cache.k_cache(static_cast<int32_t>(l));
    Tensor v_cache = cache.v_cache(static_cast<int32_t>(l));
    backend_.kv_write(k, v, slots, k_cache, v_cache);
    backend_.paged_attention(q, k_cache, v_cache, meta, heads, scale, attn);
    backend_.matmul(attn, w.o_proj, h);
    backend_.add(x, h, x);

    // SwiGLU MLP block.
    backend_.rms_norm(x, w.post_attention_norm, eps, h);
    backend_.matmul(h, w.gate_proj, gate);
    backend_.matmul(h, w.up_proj, up);
    backend_.silu_mul(gate, up, gate);
    backend_.matmul(gate, w.down_proj, h);
    backend_.add(x, h, x);

    if (layer_hook_) {
      backend_.end_step();  // make the residual stream readable on the host
      layer_hook_(static_cast<int>(l), x);
      backend_.begin_step();
    }
  }

  // Only the requested rows reach the final norm and the vocabulary projection.
  Tensor last = last_.slice_rows(0, R);
  Tensor out = logits_.slice_rows(0, R);
  backend_.gather_rows(x, prefix(logit_rows_, R), last);
  backend_.rms_norm(last, final_norm_, eps, last);
  backend_.matmul(last, lm_head_, out);
  backend_.end_step();

  logits.assign(out.data<float>(), out.data<float>() + out.numel());
}

}  // namespace tie
