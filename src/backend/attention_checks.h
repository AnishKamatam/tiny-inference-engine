#pragma once

#include <cstdint>

#include "backend/backend.h"

namespace tie {

struct CacheGeometry {
  int64_t num_blocks, kv_heads, block_size, head_dim;
};

// K and V caches must share one [num_blocks, kv_heads, block_size, head_dim] F32 or F16 shape,
// every dimension positive.
CacheGeometry check_kv_cache(const Tensor& k_cache, const Tensor& v_cache);

// k, v [T, kv_heads * head_dim] F32 and slot_mapping I32 [T] with every slot inside the cache.
void check_kv_write(const Tensor& k, const Tensor& v, const Tensor& slot_mapping, const CacheGeometry& cache);

// q, out [T, num_heads * head_dim] F32; metadata consistent with T; every block id a
// sequence reads inside the pool. Returns the longest context in the batch.
int64_t check_paged_attention(const Tensor& q, const CacheGeometry& cache, const AttentionMetadata& meta,
                              int num_heads, const Tensor& out);

}  // namespace tie
