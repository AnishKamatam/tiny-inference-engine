#pragma once

#include <memory>

#include "backend/backend.h"

namespace tie {

// Apple GPU backend. Shaders are compiled at startup from source embedded in
// the binary. All ops between begin_step() and end_step() are encoded into one
// command buffer and submitted once; an op called outside a step runs in its
// own implicit step and has completed when it returns. Tensors must live in
// buffers this backend allocated or wrapped. An op that throws discards the current
// step (implicit or explicit); begin a new one.
class MetalBackend final : public Backend {
 public:
  MetalBackend();  // UnsupportedError when no Metal device is available
  ~MetalBackend() override;

  std::string_view name() const override { return "metal"; }
  std::unique_ptr<Buffer> alloc(size_t bytes) override;
  std::unique_ptr<Buffer> wrap(const void* data, size_t bytes) override;
  void begin_step() override;
  void end_step() override;

  void embed(const Tensor& table, const Tensor& ids, Tensor& out) override;
  // Same checks as the CPU, plus K % 4 == 0; with T <= 8 or K < 64 (the matvec kernels,
  // which read weight rows in pairs) N must also be even.
  void matmul(const Tensor& x, const Tensor& w, Tensor& out) override;
  void rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) override;
  void rope_neox(Tensor& x, const Tensor& positions, int num_heads, int head_dim, float theta) override;
  void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out) override;
  void add(const Tensor& a, const Tensor& b, Tensor& out) override;
  void gather_rows(const Tensor& x, const Tensor& rows, Tensor& out) override;

 private:
  struct Impl;
  class OpScope;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tie
