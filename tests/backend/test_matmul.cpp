#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "core/error.h"
#include "core/half.h"
#include "kernels/cpu/q8_0.h"
#include "kernels/cpu/vec.h"
#include "support/tensors.h"

using namespace tie;
using test::OwnedTensor;

namespace {

// out[t][n] = sum_k x[t][k] * w[n][k] in double, with w given as F32 values.
std::vector<double> reference(const float* x, const float* w, int64_t T, int64_t N, int64_t K) {
  std::vector<double> out(static_cast<size_t>(T * N));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t n = 0; n < N; ++n) {
      double s = 0;
      for (int64_t k = 0; k < K; ++k) s += double(x[t * K + k]) * w[n * K + k];
      out[static_cast<size_t>(t * N + n)] = s;
    }
  }
  return out;
}

OwnedTensor quantize(const OwnedTensor& f32) {
  OwnedTensor q(DType::Q8_0, f32.t.shape());
  for (int64_t r = 0; r < f32.t.rows(); ++r) {
    quantize_row_q8_0(f32.t.data<float>() + r * f32.t.cols(),
                      reinterpret_cast<BlockQ8_0*>(static_cast<char*>(q.t.raw()) + r * q.t.row_bytes()), f32.t.cols());
  }
  return q;
}

std::vector<float> dequantize(const OwnedTensor& q) {
  std::vector<float> out(static_cast<size_t>(q.t.numel()));
  to_f32(DType::Q8_0, q.t.raw(), out.data(), q.t.numel());
  return out;
}

}  // namespace

TEST_CASE("matmul matches a double-precision reference for float weights") {
  CpuBackend be(4);
  const int64_t T = 3, N = 37, K = 100;  // K not a multiple of 16 exercises the scalar tails
  const OwnedTensor x = test::random_f32(Shape{T, K}, 11);
  const OwnedTensor w32 = test::random_f32(Shape{N, K}, 12);
  for (DType dtype : {DType::F32, DType::F16, DType::BF16}) {
    CAPTURE(name(dtype));
    const OwnedTensor w = test::convert(w32, dtype);
    std::vector<float> wf(static_cast<size_t>(N * K));
    to_f32(dtype, w.t.raw(), wf.data(), N * K);
    const auto want = reference(x.t.data<float>(), wf.data(), T, N, K);

    OwnedTensor out(DType::F32, Shape{T, N});
    be.matmul(x.t, w.t, out.t);
    for (size_t i = 0; i < want.size(); ++i) CHECK(out.data<float>()[i] == doctest::Approx(want[i]).epsilon(1e-4));
  }
}

TEST_CASE("Q8_0 quantization stores the max-magnitude scale and rounds to the nearest step") {
  const OwnedTensor x = test::random_f32(Shape{1, 64}, 13, 2.0f);
  const OwnedTensor q = quantize(x);
  const auto back = dequantize(q);
  const auto* blocks = reinterpret_cast<const BlockQ8_0*>(q.t.raw());
  for (int b = 0; b < 2; ++b) {
    float amax = 0;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::abs(x.t.data<float>()[b * 32 + i]));
    CHECK(f16_to_f32(blocks[b].d) == doctest::Approx(amax / 127.0f).epsilon(1e-3));
  }
  for (int i = 0; i < 64; ++i) {
    const float step = f16_to_f32(blocks[i / 32].d);
    CHECK(std::abs(back[static_cast<size_t>(i)] - x.t.data<float>()[i]) <= step * 0.5f + 1e-6f);
  }
  // An all-zero block gets scale 0 and zero quants rather than NaNs.
  OwnedTensor zeros(DType::F32, Shape{1, 32});
  for (int i = 0; i < 32; ++i) zeros.data<float>()[i] = 0.0f;
  for (float v : dequantize(quantize(zeros))) CHECK(v == 0.0f);
}

TEST_CASE("matmul with Q8_0 weights equals F32 activations times the dequantized weights") {
  CpuBackend be(4);
  const int64_t T = 2, N = 1000, K = 128;  // N large enough to take the parallel path
  const OwnedTensor x = test::random_f32(Shape{T, K}, 14);
  const OwnedTensor wq = quantize(test::random_f32(Shape{N, K}, 15));
  const auto want = reference(x.t.data<float>(), dequantize(wq).data(), T, N, K);
  OwnedTensor out(DType::F32, Shape{T, N});
  be.matmul(x.t, wq.t, out.t);
  for (size_t i = 0; i < want.size(); ++i) CHECK(out.data<float>()[i] == doctest::Approx(want[i]).epsilon(1e-4));
}

TEST_CASE("embed accepts a Q8_0 table") {
  CpuBackend be(1);
  const OwnedTensor table = quantize(test::random_f32(Shape{4, 32}, 16));
  const OwnedTensor ids = test::i32_tensor({2});
  OwnedTensor out(DType::F32, Shape{1, 32});
  be.embed(table.t, ids.t, out.t);
  const auto all = dequantize(table);
  for (int i = 0; i < 32; ++i) CHECK(out.data<float>()[i] == all[size_t(2 * 32 + i)]);
}

TEST_CASE("matmul rejects mismatched shapes") {
  CpuBackend be(1);
  const OwnedTensor x = test::random_f32(Shape{2, 8}, 17);
  const OwnedTensor w = test::random_f32(Shape{4, 9}, 18);
  OwnedTensor out(DType::F32, Shape{2, 4});
  CHECK_THROWS_AS(be.matmul(x.t, w.t, out.t), InvalidArgument);
}

TEST_CASE("Q8_0 quantization clamps in the F16 subnormal-scale range") {
  for (float amax : {1e-4f, 5e-6f}) {
    CAPTURE(amax);
    OwnedTensor x(DType::F32, Shape{1, 32});
    for (int i = 0; i < 32; ++i) x.data<float>()[i] = amax * (0.1f + 0.9f * float(i) / 31.0f);
    x.data<float>()[7] = -amax;  // largest magnitude, negative
    const OwnedTensor q = quantize(x);
    const auto* blk = reinterpret_cast<const BlockQ8_0*>(q.t.raw());
    const auto back = dequantize(q);
    const float step = f16_to_f32(blk->d);
    for (int i = 0; i < 32; ++i) {
      CHECK(blk->qs[i] >= -127);
      CHECK(std::abs(back[size_t(i)] - x.data<float>()[i]) <= step + 1e-12f);
    }
    CHECK(blk->qs[7] < 0);
  }
}

TEST_CASE("Q8_0 operands need a column count divisible by 32") {
  CpuBackend be(1);
  const OwnedTensor table(DType::Q8_0, Shape{2, 48});
  const OwnedTensor ids = test::i32_tensor({0});
  OwnedTensor eout(DType::F32, Shape{1, 48});
  CHECK_THROWS_AS(be.embed(table.t, ids.t, eout.t), InvalidArgument);

  const OwnedTensor x = test::random_f32(Shape{1, 48}, 19);
  OwnedTensor out(DType::F32, Shape{1, 2});
  CHECK_THROWS_AS(be.matmul(x.t, table.t, out.t), InvalidArgument);
}
