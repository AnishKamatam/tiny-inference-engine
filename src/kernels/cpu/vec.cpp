#include "kernels/cpu/vec.h"

#include <arm_neon.h>

#include <cstring>

#include "core/error.h"
#include "core/half.h"
#include "kernels/cpu/q8_0.h"

namespace tie {

namespace {

inline float32x4_t bf16_lo(uint16x8_t b) { return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b), 16)); }
inline float32x4_t bf16_hi(uint16x8_t b) { return vreinterpretq_f32_u32(vshll_high_n_u16(b, 16)); }

}  // namespace

void to_f32(DType dtype, const void* src, float* dst, int64_t n) {
  switch (dtype) {
    case DType::F32:
      std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
      return;
    case DType::F16: {
      const auto* s = static_cast<const uint16_t*>(src);
      int64_t i = 0;
      for (; i + 8 <= n; i += 8) {
        const float16x8_t h = vld1q_f16(reinterpret_cast<const float16_t*>(s + i));
        vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst + i + 4, vcvt_high_f32_f16(h));
      }
      for (; i < n; ++i) dst[i] = f16_to_f32(s[i]);
      return;
    }
    case DType::BF16: {
      const auto* s = static_cast<const uint16_t*>(src);
      int64_t i = 0;
      for (; i + 8 <= n; i += 8) {
        const uint16x8_t b = vld1q_u16(s + i);
        vst1q_f32(dst + i, bf16_lo(b));
        vst1q_f32(dst + i + 4, bf16_hi(b));
      }
      for (; i < n; ++i) dst[i] = bf16_to_f32(s[i]);
      return;
    }
    case DType::Q8_0:
      dequantize_row_q8_0(static_cast<const BlockQ8_0*>(src), dst, n);
      return;
    default:
      fail<InvalidArgument>("cannot convert {} to F32", name(dtype));
  }
}

float dot_f32(const float* x, const float* y, int64_t n) {
  float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0), acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(x + i), vld1q_f32(y + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(x + i + 4), vld1q_f32(y + i + 4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(x + i + 8), vld1q_f32(y + i + 8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(x + i + 12), vld1q_f32(y + i + 12));
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
  for (; i < n; ++i) sum += x[i] * y[i];
  return sum;
}

float dot_f32_f16(const float* x, const uint16_t* w, int64_t n) {
  float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const float16x8_t h = vld1q_f16(reinterpret_cast<const float16_t*>(w + i));
    acc0 = vfmaq_f32(acc0, vld1q_f32(x + i), vcvt_f32_f16(vget_low_f16(h)));
    acc1 = vfmaq_f32(acc1, vld1q_f32(x + i + 4), vcvt_high_f32_f16(h));
  }
  float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; i < n; ++i) sum += x[i] * f16_to_f32(w[i]);
  return sum;
}

float dot_f32_bf16(const float* x, const uint16_t* w, int64_t n) {
  float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const uint16x8_t b = vld1q_u16(w + i);
    acc0 = vfmaq_f32(acc0, vld1q_f32(x + i), bf16_lo(b));
    acc1 = vfmaq_f32(acc1, vld1q_f32(x + i + 4), bf16_hi(b));
  }
  float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; i < n; ++i) sum += x[i] * bf16_to_f32(w[i]);
  return sum;
}

}  // namespace tie
