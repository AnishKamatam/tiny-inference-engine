#include "kv/block_pool.h"

#include "core/error.h"

namespace tie {

BlockPool::BlockPool(int32_t num_blocks) {
  if (num_blocks <= 0) fail<InvalidArgument>("a block pool needs at least one block, got {}", num_blocks);
  ref_counts_.assign(static_cast<size_t>(num_blocks), 0);
  free_.reserve(static_cast<size_t>(num_blocks));
  for (int32_t b = num_blocks - 1; b >= 0; --b) free_.push_back(b);
}

void BlockPool::check(int32_t block) const {
  if (block < 0 || block >= num_blocks()) fail<InvalidArgument>("block id {} outside [0, {})", block, num_blocks());
}

int32_t BlockPool::ref_count(int32_t block) const {
  check(block);
  return ref_counts_[static_cast<size_t>(block)];
}

int32_t BlockPool::allocate() {
  if (free_.empty()) fail<CapacityError>("KV cache is full: all {} blocks are in use", num_blocks());
  const int32_t block = free_.back();
  free_.pop_back();
  ref_counts_[static_cast<size_t>(block)] = 1;
  return block;
}

void BlockPool::retain(int32_t block) {
  check(block);
  int32_t& refs = ref_counts_[static_cast<size_t>(block)];
  if (refs == 0) fail<InvalidArgument>("cannot retain free block {}", block);
  ++refs;
}

void BlockPool::release(int32_t block) {
  check(block);
  int32_t& refs = ref_counts_[static_cast<size_t>(block)];
  if (refs == 0) fail<InvalidArgument>("block {} released while already free", block);
  if (--refs == 0) free_.push_back(block);
}

}  // namespace tie
