#pragma once

#include <cstdint>

#include "core/dtype.h"

namespace tie {

// Converts n contiguous elements of `src` to F32.
void to_f32(DType dtype, const void* src, float* dst, int64_t n);

// Dot products with F32 accumulation. `w` holds raw F16 or BF16 bits.
float dot_f32(const float* x, const float* y, int64_t n);
float dot_f32_f16(const float* x, const uint16_t* w, int64_t n);
float dot_f32_bf16(const float* x, const uint16_t* w, int64_t n);

}  // namespace tie
