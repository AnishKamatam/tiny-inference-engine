#include "backend/cpu/cpu_backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "core/error.h"
#include "kernels/cpu/q8_0.h"
#include "kernels/cpu/vec.h"

namespace tie {

namespace {

constexpr int64_t kMinParallelWork = 1 << 15;

void expect_dtype(const Tensor& t, DType dtype, const char* what) {
  if (t.dtype() != dtype) fail<InvalidArgument>("{} must be {}, got {}", what, name(dtype), name(t.dtype()));
}

void expect_shape(const Tensor& t, const Shape& shape, const char* what) {
  if (!(t.shape() == shape)) fail<InvalidArgument>("{} must be {}, got {}", what, shape.str(), t.shape().str());
}

}  // namespace

CpuBackend::CpuBackend(int num_threads) : pool_(num_threads) {}

std::unique_ptr<Buffer> CpuBackend::alloc(size_t bytes) { return std::make_unique<HostBuffer>(bytes); }

std::unique_ptr<Buffer> CpuBackend::wrap(const void* data, size_t bytes) {
  return std::make_unique<ExternalBuffer>(data, bytes);
}

void CpuBackend::parallel(int64_t n, int64_t cost_per_item, const ThreadPool::RangeFn& fn) {
  if (n <= 0) return;
  if (n * cost_per_item < kMinParallelWork) {
    fn(0, n);
  } else {
    pool_.parallel_for(n, fn);
  }
}

void CpuBackend::embed(const Tensor& table, const Tensor& ids, Tensor& out) {
  expect_dtype(ids, DType::I32, "embed ids");
  // The allowed set must match what to_f32 supports.
  if (table.dtype() != DType::F32 && table.dtype() != DType::F16 && table.dtype() != DType::BF16 &&
      table.dtype() != DType::Q8_0) {
    fail<InvalidArgument>("embed table must be F32, F16, BF16 or Q8_0, got {}", tie::name(table.dtype()));
  }
  if (table.dtype() == DType::Q8_0 && table.cols() % kQ8_0BlockElems != 0) {
    fail<InvalidArgument>("embed: Q8_0 table needs columns divisible by {}, got {}", kQ8_0BlockElems, table.cols());
  }
  expect_dtype(out, DType::F32, "embed output");
  const int64_t vocab = table.rows();
  const int64_t dim = table.cols();
  expect_shape(out, Shape{ids.numel(), dim}, "embed output");
  const int32_t* id = ids.data<int32_t>();
  for (int64_t t = 0; t < ids.numel(); ++t) {
    if (id[t] < 0 || id[t] >= vocab) fail<InvalidArgument>("token id {} outside vocabulary of {}", id[t], vocab);
  }
  const auto* rows = static_cast<const char*>(table.raw());
  float* o = out.data<float>();
  parallel(ids.numel(), dim, [&](int64_t begin, int64_t end) {
    for (int64_t t = begin; t < end; ++t) {
      to_f32(table.dtype(), rows + static_cast<size_t>(id[t]) * table.row_bytes(), o + t * dim, dim);
    }
  });
}

template <typename RowDot>
void CpuBackend::run_matmul(int64_t T, int64_t N, int64_t K, float* out, RowDot dot) {
  // Weight rows are processed in small blocks so each activation row is reused
  // from L1 across the block instead of being re-streamed for every weight row.
  constexpr int64_t kRowBlock = 8;
  parallel(N, T * K, [&](int64_t begin, int64_t end) {
    for (int64_t n0 = begin; n0 < end; n0 += kRowBlock) {
      const int64_t n1 = std::min(end, n0 + kRowBlock);
      for (int64_t t = 0; t < T; ++t) {
        for (int64_t n = n0; n < n1; ++n) out[t * N + n] = dot(t, n);
      }
    }
  });
}

void CpuBackend::matmul(const Tensor& x, const Tensor& w, Tensor& out) {
  expect_dtype(x, DType::F32, "matmul input");
  expect_dtype(out, DType::F32, "matmul output");
  // Validate everything up front: nothing may throw inside a pool worker.
  const DType wd = w.dtype();
  if (wd != DType::F32 && wd != DType::F16 && wd != DType::BF16 && wd != DType::Q8_0) {
    fail<InvalidArgument>("matmul weight dtype {} unsupported", tie::name(wd));
  }
  const int64_t T = x.rows();
  const int64_t K = x.cols();
  const int64_t N = w.rows();
  if (w.cols() != K) fail<InvalidArgument>("matmul: input {} does not match weight {}", x.shape().str(), w.shape().str());
  if (wd == DType::Q8_0 && K % kQ8_0BlockElems != 0) {
    fail<InvalidArgument>("matmul: Q8_0 weight needs K divisible by {}, got {}", kQ8_0BlockElems, K);
  }
  expect_shape(out, Shape{T, N}, "matmul output");

  const float* xp = x.data<float>();
  const auto* wp = static_cast<const char*>(w.raw());
  const size_t wrow = w.row_bytes();
  float* op = out.data<float>();

  switch (wd) {
    case DType::F32:
      run_matmul(T, N, K, op, [&](int64_t t, int64_t n) {
        return dot_f32(xp + t * K, reinterpret_cast<const float*>(wp + n * wrow), K);
      });
      return;
    case DType::F16:
      run_matmul(T, N, K, op, [&](int64_t t, int64_t n) {
        return dot_f32_f16(xp + t * K, reinterpret_cast<const uint16_t*>(wp + n * wrow), K);
      });
      return;
    case DType::BF16:
      run_matmul(T, N, K, op, [&](int64_t t, int64_t n) {
        return dot_f32_bf16(xp + t * K, reinterpret_cast<const uint16_t*>(wp + n * wrow), K);
      });
      return;
    case DType::Q8_0:
      run_matmul(T, N, K, op, [&](int64_t t, int64_t n) {
        return dot_f32_q8_0(xp + t * K, reinterpret_cast<const BlockQ8_0*>(wp + n * wrow), K);
      });
      return;
    default:
      fail<InvalidArgument>("matmul weight dtype {} unsupported", tie::name(wd));
  }
}

void CpuBackend::rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) {
  expect_dtype(x, DType::F32, "rms_norm input");
  expect_dtype(weight, DType::F32, "rms_norm weight");
  expect_shape(weight, Shape{x.cols()}, "rms_norm weight");
  expect_dtype(out, DType::F32, "rms_norm output");
  expect_shape(out, x.shape(), "rms_norm output");
  const int64_t dim = x.cols();
  const float* xp = x.data<float>();
  const float* w = weight.data<float>();
  float* o = out.data<float>();
  parallel(x.rows(), dim, [&](int64_t begin, int64_t end) {
    for (int64_t r = begin; r < end; ++r) {
      const float* xr = xp + r * dim;
      float* orow = o + r * dim;
      const float inv = 1.0f / std::sqrt(dot_f32(xr, xr, dim) / static_cast<float>(dim) + eps);
      for (int64_t i = 0; i < dim; ++i) orow[i] = (xr[i] * inv) * w[i];
    }
  });
}

void CpuBackend::rope_neox(Tensor& x, const Tensor& positions, int num_heads, int head_dim, float theta) {
  expect_dtype(x, DType::F32, "rope input");
  expect_dtype(positions, DType::I32, "rope positions");
  if (head_dim % 2 != 0 || head_dim > kMaxHeadDim) fail<InvalidArgument>("rope head_dim {} unsupported", head_dim);
  if (num_heads <= 0 || head_dim <= 0) fail<InvalidArgument>("rope needs positive num_heads and head_dim, got {} and {}", num_heads, head_dim);
  expect_shape(positions, Shape{x.rows()}, "rope positions");
  expect_shape(x, Shape{positions.numel(), static_cast<int64_t>(num_heads) * head_dim}, "rope input");

  // Matches HF: inv_freq = 1 / theta^(2i/d) and angle = pos * inv_freq, all in F32.
  const int half = head_dim / 2;
  std::array<float, kMaxHeadDim / 2> inv_freq{};
  for (int i = 0; i < half; ++i) {
    inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
  }
  const int32_t* pos = positions.data<int32_t>();
  float* xp = x.data<float>();
  const int64_t row = static_cast<int64_t>(num_heads) * head_dim;
  parallel(positions.numel(), row, [&](int64_t begin, int64_t end) {
    for (int64_t t = begin; t < end; ++t) {
      float* xr = xp + t * row;
      for (int i = 0; i < half; ++i) {
        const float angle = static_cast<float>(pos[t]) * inv_freq[static_cast<size_t>(i)];
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        for (int h = 0; h < num_heads; ++h) {
          float* v = xr + h * head_dim;
          const float x1 = v[i];
          const float x2 = v[i + half];
          v[i] = x1 * c - x2 * s;
          v[i + half] = x2 * c + x1 * s;
        }
      }
    }
  });
}

void CpuBackend::silu_mul(const Tensor& gate, const Tensor& up, Tensor& out) {
  expect_dtype(gate, DType::F32, "silu_mul gate");
  expect_dtype(up, DType::F32, "silu_mul up");
  expect_dtype(out, DType::F32, "silu_mul output");
  expect_shape(up, gate.shape(), "silu_mul up");
  expect_shape(out, gate.shape(), "silu_mul output");
  const float* g = gate.data<float>();
  const float* u = up.data<float>();
  float* o = out.data<float>();
  parallel(gate.numel(), 4, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) o[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
  });
}

void CpuBackend::add(const Tensor& a, const Tensor& b, Tensor& out) {
  expect_dtype(a, DType::F32, "add input");
  expect_dtype(b, DType::F32, "add input");
  expect_dtype(out, DType::F32, "add output");
  expect_shape(b, a.shape(), "add input");
  expect_shape(out, a.shape(), "add output");
  const float* ap = a.data<float>();
  const float* bp = b.data<float>();
  float* o = out.data<float>();
  parallel(a.numel(), 1, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) o[i] = ap[i] + bp[i];
  });
}

void CpuBackend::gather_rows(const Tensor& x, const Tensor& rows, Tensor& out) {
  expect_dtype(x, DType::F32, "gather_rows input");
  expect_dtype(rows, DType::I32, "gather_rows indices");
  expect_dtype(out, DType::F32, "gather_rows output");
  expect_shape(out, Shape{rows.numel(), x.cols()}, "gather_rows output");
  const int32_t* r = rows.data<int32_t>();
  for (int64_t i = 0; i < rows.numel(); ++i) {
    if (r[i] < 0 || r[i] >= x.rows()) fail<InvalidArgument>("gather row {} outside [0, {})", r[i], x.rows());
    std::memcpy(out.data<float>() + i * x.cols(), x.data<float>() + r[i] * x.cols(), x.row_bytes());
  }
}

}  // namespace tie
