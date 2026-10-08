#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "backend/backend.h"
#include "core/buffer.h"
#include "core/error.h"
#include "core/half.h"
#include "core/tensor.h"

namespace tie::test {

// A tensor that owns its memory, for tests: plain host memory, or memory from
// a backend (Metal kernels can only run on Metal-allocated buffers).
struct OwnedTensor {
  std::unique_ptr<Buffer> buffer;
  Tensor t;

  OwnedTensor(DType dtype, Shape shape)
      : buffer(std::make_unique<HostBuffer>(bytes_for(dtype, shape.numel()))), t(buffer.get(), 0, dtype, shape) {}

  OwnedTensor(Backend& backend, DType dtype, Shape shape)
      : buffer(backend.alloc(bytes_for(dtype, shape.numel()))), t(buffer.get(), 0, dtype, shape) {}

  template <typename T>
  T* data() {
    return t.data<T>();
  }
};

// A copy of `src` in memory allocated by `backend`.
inline OwnedTensor on(Backend& backend, const OwnedTensor& src) {
  OwnedTensor out(backend, src.t.dtype(), src.t.shape());
  std::memcpy(out.t.raw(), src.t.raw(), src.t.nbytes());
  return out;
}

inline OwnedTensor random_f32(Shape shape, uint64_t seed, float scale = 1.0f) {
  OwnedTensor out(DType::F32, shape);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.0f, scale);
  for (int64_t i = 0; i < shape.numel(); ++i) out.data<float>()[i] = dist(rng);
  return out;
}

inline OwnedTensor i32_tensor(const std::vector<int32_t>& values) {
  OwnedTensor out(DType::I32, Shape{static_cast<int64_t>(values.size())});
  for (size_t i = 0; i < values.size(); ++i) out.data<int32_t>()[i] = values[i];
  return out;
}

// Converts an F32 tensor to F32, F16 or BF16 (same shape).
inline OwnedTensor convert(const OwnedTensor& src, DType dtype) {
  OwnedTensor out(dtype, src.t.shape());
  const float* s = src.t.data<float>();
  for (int64_t i = 0; i < src.t.numel(); ++i) {
    switch (dtype) {
      case DType::F32: out.data<float>()[i] = s[i]; break;
      case DType::F16: out.data<uint16_t>()[i] = f32_to_f16(s[i]); break;
      case DType::BF16: out.data<uint16_t>()[i] = f32_to_bf16(s[i]); break;
      default: fail<InvalidArgument>("convert() supports F32, F16 and BF16");
    }
  }
  return out;
}

}  // namespace tie::test
