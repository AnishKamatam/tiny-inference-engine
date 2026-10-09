#pragma once

#include <cstdint>
#include <vector>

#include "backend/backend.h"
#include "core/thread_pool.h"

namespace tie {

// Reference backend: straightforward NEON kernels parallelized over rows.
class CpuBackend final : public Backend {
 public:
  explicit CpuBackend(int num_threads = 0);

  std::string_view name() const override { return "cpu"; }
  std::unique_ptr<Buffer> alloc(size_t bytes) override;
  std::unique_ptr<Buffer> wrap(const void* data, size_t bytes) override;

  void embed(const Tensor& table, const Tensor& ids, Tensor& out) override;
  void matmul(const Tensor& x, const Tensor& w, Tensor& out) override;
  void rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) override;
  void rope_neox(Tensor& x, const Tensor& positions, int num_heads, int head_dim, float theta) override;
  void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out) override;
  void add(const Tensor& a, const Tensor& b, Tensor& out) override;
  void gather_rows(const Tensor& x, const Tensor& rows, Tensor& out) override;
  void kv_write(const Tensor& k, const Tensor& v, const Tensor& slot_mapping, Tensor& k_cache,
                Tensor& v_cache) override;
  void paged_attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache, const AttentionMetadata& meta,
                       int num_heads, float scale, Tensor& out) override;

 private:
  // Runs fn over [0, n) on the pool when n * cost_per_item is large enough to
  // pay for waking the workers; otherwise inline on the calling thread.
  void parallel(int64_t n, int64_t cost_per_item, const ThreadPool::RangeFn& fn);

  // Computes out[t * N + n] = dot(t, n) for all t < T, n < N, parallel over n.
  template <typename RowDot>
  void run_matmul(int64_t T, int64_t N, int64_t K, float* out, RowDot dot);

  ThreadPool pool_;
  std::vector<int32_t> token_seq_;  // paged_attention scratch: batch row -> sequence index
};

}  // namespace tie
