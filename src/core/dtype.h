#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace tie {

enum class DType : uint8_t { F32, F16, BF16, Q8_0, I32 };

// Block-quantized types pack `block_elems` values into `block_bytes`;
// plain types are blocks of one element.
struct DTypeTraits {
  std::string_view name;
  int64_t block_elems;
  int64_t block_bytes;
};

inline constexpr int64_t kQ8_0BlockElems = 32;

constexpr DTypeTraits traits(DType t) {
  switch (t) {
    case DType::F32: return {"F32", 1, 4};
    case DType::F16: return {"F16", 1, 2};
    case DType::BF16: return {"BF16", 1, 2};
    case DType::Q8_0: return {"Q8_0", kQ8_0BlockElems, 2 + kQ8_0BlockElems};
    case DType::I32: return {"I32", 1, 4};
  }
  return {"?", 1, 0};
}

constexpr std::string_view name(DType t) { return traits(t).name; }

// Bytes occupied by `n` contiguous elements. Throws InvalidArgument when `n`
// does not fill a whole number of quantization blocks.
size_t bytes_for(DType t, int64_t n);

}  // namespace tie
