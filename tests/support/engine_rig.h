#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "engine/engine.h"
#include "engine/serving_config.h"
#include "kv/paged_kv_cache.h"
#include "model/loader.h"
#include "model/qwen3.h"

namespace tie::test {

using SchedulerFactory = std::function<std::unique_ptr<Scheduler>(PagedKVCache&)>;

// A model, its cache and an engine on the CPU, wired the way the CLI wires the real one.
// KV blocks hold 4 tokens.
struct Rig {
  CpuBackend backend{2};
  LoadedModel weights;
  std::unique_ptr<Qwen3Model> model;
  std::unique_ptr<PagedKVCache> cache;
  std::unique_ptr<Engine> engine;

  Rig(LoadedModel w, int32_t max_tokens_per_step, std::vector<int32_t> stop_tokens = {}, size_t kv_blocks = 256,
      DType kv_dtype = DType::F32, int32_t max_blocks_per_seq = 0, int32_t max_seqs = 1,
      SchedulerFactory make_scheduler = nullptr)
      : weights(std::move(w)) {
    const ModelConfig& c = weights.config;
    KVCacheConfig kc;
    kc.num_layers = int32_t(c.num_layers);
    kc.num_kv_heads = int32_t(c.num_kv_heads);
    kc.head_dim = int32_t(c.head_dim);
    kc.block_size = 4;
    kc.dtype = kv_dtype;
    kc.budget_bytes = kv_blocks * 2 * size_t(c.num_layers * c.num_kv_heads * c.head_dim) * 4 * bytes_for(kv_dtype, 1);
    cache = std::make_unique<PagedKVCache>(backend, kc);

    const ServingConfig serving{max_tokens_per_step, max_seqs};
    model = std::make_unique<Qwen3Model>(
        backend, weights,
        model_limits(serving, max_blocks_per_seq > 0 ? max_blocks_per_seq
                                                     : PagedKVCache::blocks_for(c.max_position_embeddings, kc.block_size)));

    EngineConfig ec;
    ec.stop_tokens = std::move(stop_tokens);
    std::unique_ptr<Scheduler> scheduler = make_scheduler
                                               ? make_scheduler(*cache)
                                               : std::make_unique<FcfsScheduler>(*cache, max_tokens_per_step);
    engine = std::make_unique<Engine>(*model, *cache, std::move(scheduler), serving, ec);
  }
};

inline std::vector<int32_t> prompt(int n) {
  std::vector<int32_t> ids;
  for (int i = 0; i < n; ++i) ids.push_back((i * 5 + 1) % 64);
  return ids;
}

inline SamplingParams greedy(int32_t max_tokens) {
  SamplingParams p;
  p.max_tokens = max_tokens;
  return p;
}

}  // namespace tie::test
