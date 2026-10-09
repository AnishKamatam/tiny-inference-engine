#pragma once

#include <cstdint>
#include <vector>

namespace tie {

// A fixed set of KV-cache blocks with reference counts; a block is free when
// its count is zero. Counts above one arrive with prefix sharing (radix cache).
class BlockPool {
 public:
  explicit BlockPool(int32_t num_blocks);

  int32_t num_blocks() const { return static_cast<int32_t>(ref_counts_.size()); }
  int32_t num_free() const { return static_cast<int32_t>(free_.size()); }
  int32_t ref_count(int32_t block) const;

  int32_t allocate();           // CapacityError when no block is free
  void retain(int32_t block);   // add an owner to a live block
  void release(int32_t block);  // drop an owner; the block is freed at zero

 private:
  void check(int32_t block) const;

  std::vector<int32_t> free_;  // stack, lowest id on top for a fresh pool
  std::vector<int32_t> ref_counts_;
};

}  // namespace tie
