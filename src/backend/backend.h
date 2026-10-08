#pragma once

#include <cstddef>
#include <memory>
#include <string_view>

#include "core/buffer.h"
#include "core/tensor.h"

namespace tie {

inline constexpr int kMaxHeadDim = 512;

// The compute operations models are written against. Activations are F32 and
// row-major; weight operands may be F32, F16, BF16 or Q8_0. An output may alias
// an input only where noted.
class Backend {
 public:
  virtual ~Backend() = default;
  virtual std::string_view name() const = 0;

  virtual std::unique_ptr<Buffer> alloc(size_t bytes) = 0;
  // Wraps memory owned elsewhere (an mmapped model file) without copying.
  virtual std::unique_ptr<Buffer> wrap(const void* data, size_t bytes) = 0;

  // A step brackets one forward pass. GPU backends record every op in between and
  // submit once in end_step(), which returns when results are readable on the host.
  virtual void begin_step() {}
  virtual void end_step() {}

  // out[t] = table[ids[t]]. table [V, D] any weight dtype; ids I32 [T]; out [T, D].
  virtual void embed(const Tensor& table, const Tensor& ids, Tensor& out) = 0;
  // Row-wise RMSNorm with an F32 weight [D]. x, out [R, D]; out may alias x.
  virtual void rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) = 0;
  // In-place NEOX (rotate-half) rotary embedding. x [T, num_heads * head_dim]; positions I32 [T].
  virtual void rope_neox(Tensor& x, const Tensor& positions, int num_heads, int head_dim, float theta) = 0;
  // out = silu(gate) * up elementwise; out may alias either input.
  virtual void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out) = 0;
  // out = a + b elementwise; out may alias either input.
  virtual void add(const Tensor& a, const Tensor& b, Tensor& out) = 0;
  // out[r] = x[rows[r]]. x [T, D]; rows I32 [R]; out [R, D].
  virtual void gather_rows(const Tensor& x, const Tensor& rows, Tensor& out) = 0;
};

}  // namespace tie
