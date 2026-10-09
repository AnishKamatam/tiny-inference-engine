#include "kernels/cpu/q8_0.h"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>

#include "core/half.h"

namespace tie {

void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t n) {
  for (int64_t b = 0; b < n / kQ8_0BlockElems; ++b) {
    const float* xb = x + b * kQ8_0BlockElems;
    float amax = 0.0f;
    for (int i = 0; i < kQ8_0BlockElems; ++i) amax = std::max(amax, std::abs(xb[i]));
    // Quantize against the scale as stored (F16-rounded). Its relative error can exceed
    // 2^-8 in the F16 subnormal range, so q is clamped; amax/127 > 65504 overflows the scale.
    y[b].d = f32_to_f16(amax / 127.0f);
    // If rounding made the scale too small to cover amax, step to the next F16 up
    // (matters in the subnormal range, where spacing is coarse relative to the value).
    while (127.0f * f16_to_f32(y[b].d) < amax && y[b].d < 0x7BFF) ++y[b].d;
    const float d = f16_to_f32(y[b].d);
    const float inv = d != 0.0f ? 1.0f / d : 0.0f;
    for (int i = 0; i < kQ8_0BlockElems; ++i) y[b].qs[i] = static_cast<int8_t>(std::clamp<long>(std::lround(xb[i] * inv), -127, 127));
  }
}

void dequantize_row_q8_0(const BlockQ8_0* x, float* y, int64_t n) {
  for (int64_t b = 0; b < n / kQ8_0BlockElems; ++b) {
    const float d = f16_to_f32(x[b].d);
    for (int i = 0; i < kQ8_0BlockElems; ++i) y[b * kQ8_0BlockElems + i] = d * static_cast<float>(x[b].qs[i]);
  }
}

float dot_f32_q8_0(const float* x, const BlockQ8_0* w, int64_t n) {
  float sum = 0.0f;
  for (int64_t b = 0; b < n / kQ8_0BlockElems; ++b) {
    const float* xb = x + b * kQ8_0BlockElems;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int i = 0; i < kQ8_0BlockElems; i += 8) {
      // Widen 8 int8 quants to two float32x4 and multiply-accumulate against x.
      const int16x8_t q16 = vmovl_s8(vld1_s8(w[b].qs + i));
      acc = vfmaq_f32(acc, vld1q_f32(xb + i), vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16))));
      acc = vfmaq_f32(acc, vld1q_f32(xb + i + 4), vcvtq_f32_s32(vmovl_high_s16(q16)));
    }
    sum += f16_to_f32(w[b].d) * vaddvq_f32(acc);
  }
  return sum;
}

}  // namespace tie
