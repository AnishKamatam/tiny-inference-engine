#pragma once

#include <bit>
#include <cstdint>

namespace tie {

inline float f16_to_f32(uint16_t h) { return static_cast<float>(std::bit_cast<_Float16>(h)); }

inline uint16_t f32_to_f16(float f) { return std::bit_cast<uint16_t>(static_cast<_Float16>(f)); }

inline float bf16_to_f32(uint16_t h) { return std::bit_cast<float>(static_cast<uint32_t>(h) << 16); }

// Round-to-nearest-even; NaNs stay NaN (quieted).
inline uint16_t f32_to_bf16(float f) {
  uint32_t bits = std::bit_cast<uint32_t>(f);
  if ((bits & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((bits >> 16) | 0x40u);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

}  // namespace tie
