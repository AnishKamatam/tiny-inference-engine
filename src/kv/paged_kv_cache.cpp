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

bool PagedKVCache::reserve(BlockTable& table, int64_t new_len) {
  if (new_len < 0) fail<InvalidArgument>("cannot reserve KV blocks for {} tokens", new_len);
  const int64_t needed = int64_t{blocks_for(new_len, config_.block_size)} - static_cast<int64_t>(table.size());
  if (needed <= 0) return true;
  if (needed > pool_.num_free()) return false;
  table.reserve(table.size() + static_cast<size_t>(needed));  // so push_back cannot throw after a block is taken
  for (int64_t i = 0; i < needed; ++i) table.push_back(pool_.allocate());
  return true;
}

void PagedKVCache::slots(const BlockTable& table, int64_t old_len, int64_t new_len, std::vector<int32_t>& out) const {
  const int32_t bs = config_.block_size;
  if (old_len < 0 || new_len < old_len) fail<InvalidArgument>("bad token range [{}, {})", old_len, new_len);
  if (blocks_for(new_len, bs) > static_cast<int64_t>(table.size())) {
    fail<InvalidArgument>("block table of {} blocks cannot hold {} tokens", table.size(), new_len);
  }
  for (int64_t pos = old_len; pos < new_len; ++pos) {
    out.push_back(table[static_cast<size_t>(pos / bs)] * bs + static_cast<int32_t>(pos % bs));
  }
}

void PagedKVCache::truncate(BlockTable& table, int64_t len) {
  if (len < 0) fail<InvalidArgument>("cannot truncate a block table to {} tokens", len);
  const auto keep = static_cast<size_t>(blocks_for(len, config_.block_size));
  while (table.size() > keep) {
    pool_.release(table.back());
    table.pop_back();
  }
}

void PagedKVCache::append_slots(BlockTable& table, int64_t old_len, int64_t new_len, std::vector<int32_t>& slots_out) {
  const int32_t bs = config_.block_size;
  if (old_len < 0 || new_len < old_len) fail<InvalidArgument>("bad token range [{}, {})", old_len, new_len);
  if (static_cast<int64_t>(table.size()) != blocks_for(old_len, bs)) {
    fail<InvalidArgument>("block table has {} blocks but {} cached tokens need {}", table.size(), old_len,
                          blocks_for(old_len, bs));
  }
  if (!reserve(table, new_len)) {
    fail<CapacityError>("KV cache needs {} more blocks to hold {} tokens but only {} of {} are free",
                        blocks_for(new_len, bs) - static_cast<int32_t>(table.size()), new_len, pool_.num_free(),
                        pool_.num_blocks());
  }
  slots(table, old_len, new_len, slots_out);
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
