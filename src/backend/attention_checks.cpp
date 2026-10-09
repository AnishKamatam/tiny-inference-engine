#include "backend/attention_checks.h"

#include <algorithm>

#include "backend/checks.h"
#include "core/error.h"

namespace tie {

CacheGeometry check_kv_cache(const Tensor& k_cache, const Tensor& v_cache) {
  if (k_cache.shape().ndim() != 4 || !(k_cache.shape() == v_cache.shape()) || k_cache.dtype() != v_cache.dtype()) {
    fail<InvalidArgument>("K/V caches must share one [blocks, kv_heads, block_size, head_dim] shape, got {} and {}",
                          k_cache.shape().str(), v_cache.shape().str());
  }
  if (k_cache.dtype() != DType::F32 && k_cache.dtype() != DType::F16) {
    fail<InvalidArgument>("KV cache dtype {} unsupported", name(k_cache.dtype()));
  }
  return {k_cache.dim(0), k_cache.dim(1), k_cache.dim(2), k_cache.dim(3)};
}

void check_kv_write(const Tensor& k, const Tensor& v, const Tensor& slot_mapping, const CacheGeometry& g) {
  expect_dtype(slot_mapping, DType::I32, "slot_mapping");
  expect_dtype(k, DType::F32, "kv_write keys");
  expect_dtype(v, DType::F32, "kv_write values");
  const int64_t T = slot_mapping.numel();
  expect_shape(k, Shape{T, g.kv_heads * g.head_dim}, "kv_write keys");
  expect_shape(v, Shape{T, g.kv_heads * g.head_dim}, "kv_write values");
  const int32_t* slots = slot_mapping.data<int32_t>();
  const int64_t num_slots = g.num_blocks * g.block_size;
  for (int64_t t = 0; t < T; ++t) {
    if (slots[t] < 0 || slots[t] >= num_slots) fail<InvalidArgument>("slot {} outside cache of {} slots", slots[t], num_slots);
  }
}

int64_t check_paged_attention(const Tensor& q, const CacheGeometry& g, const AttentionMetadata& meta, int num_heads,
                              const Tensor& out) {
  const int64_t H = num_heads;
  const int64_t D = g.head_dim;
  const int64_t T = q.rows();
  if (D > kMaxHeadDim) fail<InvalidArgument>("head_dim {} exceeds {}", D, kMaxHeadDim);
  if (H <= 0 || H % g.kv_heads != 0) fail<InvalidArgument>("{} query heads are not a multiple of {} KV heads", H, g.kv_heads);
  expect_dtype(q, DType::F32, "attention queries");
  expect_shape(q, Shape{T, H * D}, "attention queries");
  expect_shape(out, q.shape(), "attention output");

  const int32_t S = meta.num_seqs;
  expect_dtype(meta.query_start, DType::I32, "query_start");
  expect_dtype(meta.context_lens, DType::I32, "context_lens");
  expect_dtype(meta.block_tables, DType::I32, "block_tables");
  if (S <= 0 || meta.query_start.numel() < S + 1 || meta.context_lens.numel() < S || meta.block_tables.rows() < S) {
    fail<InvalidArgument>("attention metadata does not describe {} sequences", S);
  }
  const int32_t* qs = meta.query_start.data<int32_t>();
  const int32_t* ctx = meta.context_lens.data<int32_t>();
  const int32_t* tables = meta.block_tables.data<int32_t>();
  const int64_t max_blocks = meta.block_tables.cols();
  if (qs[0] != 0 || qs[S] != T) fail<InvalidArgument>("query_start must span [0, {}), got [{}, {})", T, qs[0], qs[S]);

  int64_t max_ctx = 0;
  for (int32_t s = 0; s < S; ++s) {
    const int64_t n_new = qs[s + 1] - qs[s];
    if (n_new < 0 || ctx[s] < n_new) fail<InvalidArgument>("sequence {} has {} queries but a context of {}", s, n_new, ctx[s]);
    const int64_t used_blocks = (ctx[s] + g.block_size - 1) / g.block_size;
    if (used_blocks > max_blocks) fail<InvalidArgument>("sequence {} needs {} blocks, table holds {}", s, used_blocks, max_blocks);
    for (int64_t b = 0; b < used_blocks; ++b) {
      const int32_t block = tables[s * max_blocks + b];
      if (block < 0 || block >= g.num_blocks) fail<InvalidArgument>("sequence {} references block {}", s, block);
    }
    max_ctx = std::max<int64_t>(max_ctx, ctx[s]);
  }
  return max_ctx;
}

}  // namespace tie
