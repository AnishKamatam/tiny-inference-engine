#pragma once

#include <cmath>

namespace tie {

// RoPE inverse frequencies, HF formula in F32: out[i] = 1 / theta^(2i / head_dim) for i < head_dim / 2.
// Shared by every backend so rotation angles match bit for bit.
inline void rope_inv_freq(float theta, int head_dim, float* out) {
  for (int i = 0; i < head_dim / 2; ++i) {
    out[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
  }
}

}  // namespace tie
