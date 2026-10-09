#include "kv/paged_kv_cache.h"

#include <algorithm>
#include <limits>

#include "core/error.h"

namespace tie {

namespace {

size_t checked_block_bytes(const KVCacheConfig& c) {
  if (c.dtype != DType::F16 && c.dtype != DType::F32) {
    fail<InvalidArgument>("KV cache dtype must be F16 or F32, got {}", name(c.dtype));
  }
  if (c.num_layers <= 0 || c.num_kv_heads <= 0 || c.head_dim <= 0 || c.block_size <= 0) {
    fail<InvalidArgument>("KV cache dimensions must be positive");
  }
  return 2 * static_cast<size_t>(c.num_layers) * static_cast<size_t>(c.num_kv_heads) *
         static_cast<size_t>(c.block_size) * bytes_for(c.dtype, c.head_dim);
}

int32_t blocks_in_budget(const KVCacheConfig& c) {
  const size_t per_block = checked_block_bytes(c);
  const size_t blocks = c.budget_bytes / per_block;
  if (blocks == 0) {
    fail<CapacityError>("KV budget of {} bytes is smaller than one {}-byte block", c.budget_bytes, per_block);
  }
  // Slots (block * block_size + offset) must fit in int32.
  const size_t max_blocks = static_cast<size_t>(std::numeric_limits<int32_t>::max()) / static_cast<size_t>(c.block_size);
  return static_cast<int32_t>(std::min(blocks, max_blocks));
}

}  // namespace

PagedKVCache::PagedKVCache(Backend& backend, const KVCacheConfig& config)
    : config_(config), bytes_per_block_(checked_block_bytes(config)), pool_(blocks_in_budget(config)) {
  buffer_ = backend.alloc(bytes_per_block_ * static_cast<size_t>(pool_.num_blocks()));
}

void PagedKVCache::append_slots(BlockTable& table, int64_t old_len, int64_t new_len, std::vector<int32_t>& slots) {
  const int32_t bs = config_.block_size;
  if (old_len < 0 || new_len < old_len) fail<InvalidArgument>("bad token range [{}, {})", old_len, new_len);
  if (static_cast<int64_t>(table.size()) != blocks_for(old_len, bs)) {
    fail<InvalidArgument>("block table has {} blocks but {} cached tokens need {}", table.size(), old_len,
                          blocks_for(old_len, bs));
  }
  const int32_t needed = blocks_for(new_len, bs) - static_cast<int32_t>(table.size());
  if (needed > pool_.num_free()) {
    fail<CapacityError>("KV cache needs {} more blocks to hold {} tokens but only {} of {} are free", needed, new_len,
                        pool_.num_free(), pool_.num_blocks());
  }
  for (int32_t i = 0; i < needed; ++i) table.push_back(pool_.allocate());
  for (int64_t pos = old_len; pos < new_len; ++pos) {
    slots.push_back(table[static_cast<size_t>(pos / bs)] * bs + static_cast<int32_t>(pos % bs));
  }
}

void PagedKVCache::release(BlockTable& table) {
  for (int32_t block : table) pool_.release(block);
  table.clear();
}

Tensor PagedKVCache::plane(int32_t layer, int which) const {
  if (layer < 0 || layer >= config_.num_layers) {
    fail<InvalidArgument>("layer {} outside [0, {})", layer, config_.num_layers);
  }
  const Shape shape{num_blocks(), config_.num_kv_heads, config_.block_size, config_.head_dim};
  const size_t plane_bytes = bytes_for(config_.dtype, shape.numel());
  const size_t offset = (static_cast<size_t>(layer) * 2 + static_cast<size_t>(which)) * plane_bytes;
  return Tensor(buffer_.get(), offset, config_.dtype, shape);
}

}  // namespace tie
