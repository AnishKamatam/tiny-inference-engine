#pragma once
// Argument structs for tie's own Metal kernels. Shared by the C++ host and the
// shaders: keep layouts identical.
#ifndef __METAL_VERSION__
#include <cstdint>
#endif

typedef struct {
  int32_t num_tokens;
  int32_t kv_heads;
  int32_t head_dim;
  int32_t block_size;
} tie_kv_write_args;

typedef struct {
  int32_t num_seqs;
  int32_t num_heads;
  int32_t kv_heads;
  int32_t head_dim;
  int32_t block_size;
  int32_t max_blocks;  // block_tables row length
  float scale;
} tie_attention_args;
