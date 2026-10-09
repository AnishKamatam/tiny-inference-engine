#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "backend/backend.h"
#include "core/tensor.h"
#include "kv/block_pool.h"

namespace tie {

// The blocks holding one sequence's tokens, in order: token j lives in block
// table[j / block_size] at offset j % block_size.
using BlockTable = std::vector<int32_t>;

struct KVCacheConfig {
  int32_t num_layers = 0;
  int32_t num_kv_heads = 0;
  int32_t head_dim = 0;
  int32_t block_size = 16;
  DType dtype = DType::F16;  // F16 or F32
  size_t budget_bytes = 0;   // the pool takes as many whole blocks as fit
};

// The KV cache: one allocation laid out as
//   [num_layers][2 (K, V)][num_blocks][num_kv_heads][block_size][head_dim]
// A block id addresses the same slots in every layer, so allocating one block
// reserves a token range's K and V across the whole model.
class PagedKVCache {
 public:
  PagedKVCache(Backend& backend, const KVCacheConfig& config);

  const KVCacheConfig& config() const { return config_; }
  int32_t block_size() const { return config_.block_size; }
  int32_t num_blocks() const { return pool_.num_blocks(); }
  int32_t num_free_blocks() const { return pool_.num_free(); }
  size_t bytes_per_block() const { return bytes_per_block_; }
  BlockPool& pool() { return pool_; }

  static int32_t blocks_for(int64_t tokens, int32_t block_size) {
    return static_cast<int32_t>((tokens + block_size - 1) / block_size);
  }

  // Grows `table` to cover `new_len` tokens and appends the cache slot
  // (block * block_size + offset) of every position in [old_len, new_len) to
  // `slots`. All-or-nothing: on CapacityError neither `table` nor the pool changes.
  void append_slots(BlockTable& table, int64_t old_len, int64_t new_len, std::vector<int32_t>& slots);
  // Releases every block of `table` and clears it.
  void release(BlockTable& table);

  // [num_blocks, num_kv_heads, block_size, head_dim] views of one layer.
  Tensor k_cache(int32_t layer) const { return plane(layer, 0); }
  Tensor v_cache(int32_t layer) const { return plane(layer, 1); }

 private:
  Tensor plane(int32_t layer, int which) const;

  KVCacheConfig config_;
  size_t bytes_per_block_;
  BlockPool pool_;
  std::unique_ptr<Buffer> buffer_;
};

}  // namespace tie
