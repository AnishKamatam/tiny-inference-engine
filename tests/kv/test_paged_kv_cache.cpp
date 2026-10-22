#include <doctest/doctest.h>

#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "core/error.h"
#include "kv/paged_kv_cache.h"

using namespace tie;

namespace {

KVCacheConfig small_config(size_t blocks) {
  KVCacheConfig c;
  c.num_layers = 2;
  c.num_kv_heads = 2;
  c.head_dim = 8;
  c.block_size = 4;
  c.dtype = DType::F32;
  c.budget_bytes = blocks * (2 * 2 * 2 * 4 * 8 * 4);  // blocks * bytes_per_block
  return c;
}

}  // namespace

TEST_CASE("PagedKVCache sizes the pool from the byte budget") {
  CpuBackend be(1);
  KVCacheConfig qwen;
  qwen.num_layers = 28;
  qwen.num_kv_heads = 8;
  qwen.head_dim = 128;
  qwen.budget_bytes = size_t{4} << 30;
  const PagedKVCache cache(be, qwen);
  CHECK(cache.bytes_per_block() == 1835008);  // 1.75 MiB: 2 * 28 * 8 * 16 * 128 * 2 bytes
  CHECK(cache.num_blocks() == 2340);

  KVCacheConfig tiny = small_config(1);
  tiny.budget_bytes -= 1;
  CHECK_THROWS_AS(PagedKVCache(be, tiny), CapacityError);
  KVCacheConfig bad_dtype = small_config(4);
  bad_dtype.dtype = DType::I32;
  CHECK_THROWS_AS(PagedKVCache(be, bad_dtype), InvalidArgument);
}

TEST_CASE("append_slots grows the block table and maps positions to slots") {
  CpuBackend be(1);
  PagedKVCache cache(be, small_config(8));
  BlockTable table;
  std::vector<int32_t> slots;

  cache.append_slots(table, 0, 6, slots);
  REQUIRE(table.size() == 2);
  REQUIRE(slots.size() == 6);
  for (int pos = 0; pos < 6; ++pos) CHECK(slots[size_t(pos)] == table[size_t(pos / 4)] * 4 + pos % 4);

  slots.clear();
  cache.append_slots(table, 6, 7, slots);  // fits in the second block
  CHECK(table.size() == 2);
  CHECK(slots == std::vector<int32_t>{table[1] * 4 + 2});

  CHECK_THROWS_AS(cache.append_slots(table, 3, 9, slots), InvalidArgument);  // table holds 7 tokens, not 3
  cache.release(table);
  CHECK(table.empty());
  CHECK(cache.num_free_blocks() == 8);
}

TEST_CASE("append_slots is all-or-nothing when the pool runs out") {
  CpuBackend be(1);
  PagedKVCache cache(be, small_config(2));
  BlockTable table;
  std::vector<int32_t> slots;
  cache.append_slots(table, 0, 8, slots);
  CHECK(cache.num_free_blocks() == 0);
  slots.clear();
  CHECK_THROWS_AS(cache.append_slots(table, 8, 9, slots), CapacityError);
  CHECK(table.size() == 2);
  CHECK(slots.empty());
}

TEST_CASE("reserve grows a table all-or-nothing and reports a shortfall") {
  CpuBackend be(1);
  PagedKVCache cache(be, small_config(3));
  BlockTable table;
  CHECK(cache.reserve(table, 5));  // 2 blocks
  CHECK(table.size() == 2);
  CHECK(cache.reserve(table, 8));  // still 2 blocks
  CHECK(table.size() == 2);
  CHECK_FALSE(cache.reserve(table, 13));  // needs 4 blocks; the pool has 3
  CHECK(table.size() == 2);
  CHECK(cache.num_free_blocks() == 1);
  CHECK(cache.reserve(table, 12));
  CHECK(cache.num_free_blocks() == 0);
  CHECK(cache.reserve(table, 0));  // never shrinks
  CHECK(table.size() == 3);
  CHECK_THROWS_AS(cache.reserve(table, -1), InvalidArgument);
}

TEST_CASE("slots maps positions without allocating and rejects ranges the table does not cover") {
  CpuBackend be(1);
  PagedKVCache cache(be, small_config(4));
  BlockTable table;
  REQUIRE(cache.reserve(table, 6));
  const int32_t free_before = cache.num_free_blocks();
  std::vector<int32_t> out;
  cache.slots(table, 0, 6, out);
  REQUIRE(out.size() == 6);
  for (int pos = 0; pos < 6; ++pos) CHECK(out[size_t(pos)] == table[size_t(pos / 4)] * 4 + pos % 4);
  CHECK(cache.num_free_blocks() == free_before);
  CHECK_THROWS_AS(cache.slots(table, 6, 9, out), InvalidArgument);  // 9 tokens need 3 blocks
  CHECK_THROWS_AS(cache.slots(table, 4, 2, out), InvalidArgument);
  CHECK_THROWS_AS(cache.slots(table, -1, 2, out), InvalidArgument);
}

TEST_CASE("truncate frees the blocks beyond a length") {
  CpuBackend be(1);
  PagedKVCache cache(be, small_config(4));
  BlockTable table;
  REQUIRE(cache.reserve(table, 12));
  cache.truncate(table, 5);
  CHECK(table.size() == 2);
  CHECK(cache.num_free_blocks() == 2);
  cache.truncate(table, 5);  // idempotent
  CHECK(table.size() == 2);
  cache.truncate(table, 0);
  CHECK(table.empty());
  CHECK(cache.num_free_blocks() == 4);
  CHECK_THROWS_AS(cache.truncate(table, -1), InvalidArgument);
}

TEST_CASE("k_cache and v_cache are disjoint per-layer views") {
  CpuBackend be(1);
  const PagedKVCache cache(be, small_config(3));
  const Tensor k0 = cache.k_cache(0);
  CHECK((k0.shape() == Shape{3, 2, 4, 8}));
  const size_t plane = k0.nbytes();
  CHECK(cache.v_cache(0).offset() == k0.offset() + plane);
  CHECK(cache.k_cache(1).offset() == k0.offset() + 2 * plane);
  CHECK(cache.v_cache(1).offset() + plane == k0.buffer()->size());
  CHECK_THROWS_AS(cache.k_cache(2), InvalidArgument);
}
