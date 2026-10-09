#pragma once

#include <cstdint>

#include "core/dtype.h"

namespace tie {

// GGUF Q8_0: 32 values sharing one F16 scale, value[i] = d * qs[i].
struct BlockQ8_0 {
  uint16_t d;
  int8_t qs[kQ8_0BlockElems];
};
static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 blocks must match the GGUF layout");

// `n` is an element count and must be a multiple of 32.
void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t n);
void dequantize_row_q8_0(const BlockQ8_0* x, float* y, int64_t n);
// Dot product of F32 activations with Q8_0 weights; F32 accumulation.
float dot_f32_q8_0(const float* x, const BlockQ8_0* w, int64_t n);

}  // namespace tie
