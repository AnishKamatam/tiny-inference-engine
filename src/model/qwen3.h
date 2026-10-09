#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "backend/backend.h"
#include "kv/paged_kv_cache.h"
#include "model/loader.h"

namespace tie {

// Upper bounds the model sizes its activation arena and input tensors for.
struct ModelLimits {
  int32_t max_tokens_per_step = 512;
  int32_t max_seqs = 1;
  int32_t max_blocks_per_seq = 0;  // length of one block-table row
  int32_t max_logit_rows = 1;      // logits are only computed for requested rows
};

// One forward step over a flat, unpadded batch of sequences.
struct BatchInput {
  std::span<const int32_t> tokens;        // [T]
  std::span<const int32_t> positions;     // [T]
  std::span<const int32_t> slot_mapping;  // [T] cache slot receiving each token's K/V
  std::span<const int32_t> query_start;   // [S + 1] batch rows of each sequence
  std::span<const int32_t> context_lens;  // [S] cached tokens per sequence after this step
  std::span<const int32_t> block_tables;  // [S * limits.max_blocks_per_seq], row-major
  std::span<const int32_t> logit_rows;    // batch rows whose logits are returned
};

// Qwen3 decoder. Weights are used in place (mmapped files wrapped, not copied),
// except the tiny norm vectors, which are converted to F32 once at load.
class Qwen3Model {
 public:
  // Binds every tensor of `weights` strictly: each expected tensor must exist with
  // its exact shape and an allowed dtype, and nothing may be left over.
  // `weights` must outlive the model.
  Qwen3Model(Backend& backend, const LoadedModel& weights, const ModelLimits& limits);

  const ModelConfig& config() const { return config_; }
  const ModelLimits& limits() const { return limits_; }

  // Runs one step: writes this step's K/V into `cache` and returns the logits of
  // `input.logit_rows` as a row-major [rows, vocab_size] matrix in `logits`.
  void forward(const BatchInput& input, PagedKVCache& cache, std::vector<float>& logits);

  // Test/debug hook observing the residual stream [T, hidden] after each layer.
  using LayerHook = std::function<void(int layer, const Tensor& hidden)>;
  void set_layer_hook(LayerHook hook) { layer_hook_ = std::move(hook); }

 private:
  struct Layer {
    Tensor input_norm, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm;
    Tensor post_attention_norm, gate_proj, up_proj, down_proj;
  };

  void bind_weights(const LoadedModel& weights);
  void allocate_activations();

  Backend& backend_;
  ModelConfig config_;
  ModelLimits limits_;
  LayerHook layer_hook_;

  std::vector<std::unique_ptr<Buffer>> regions_;  // wrapped weight files
  std::unique_ptr<Buffer> norms_;                 // all norm weights, converted to F32
  std::unique_ptr<Buffer> arena_;                 // activations and step inputs

  Tensor embed_tokens_, final_norm_, lm_head_;
  std::vector<Layer> layers_;

  // Activations, sized for limits_. Each step uses a prefix of the rows.
  Tensor x_, h_, q_, k_, v_, attn_, gate_, up_, last_, logits_;
  // Step inputs copied from BatchInput.
  Tensor tokens_, positions_, slots_, query_start_, context_lens_, block_tables_, logit_rows_;
};

}  // namespace tie
