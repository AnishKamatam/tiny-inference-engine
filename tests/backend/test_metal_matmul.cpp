#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "backend/metal/metal_backend.h"
#include "core/error.h"
#include "kernels/cpu/q8_0.h"
#include "kernels/cpu/vec.h"
#include "support/tensors.h"

using namespace tie;
using test::OwnedTensor;

namespace {

OwnedTensor make_weights(DType dtype, int64_t N, int64_t K, uint64_t seed) {
  const OwnedTensor f32 = test::random_f32(Shape{N, K}, seed);
  if (dtype != DType::Q8_0) return test::convert(f32, dtype);
  OwnedTensor q(DType::Q8_0, Shape{N, K});
  for (int64_t r = 0; r < N; ++r) {
    quantize_row_q8_0(f32.t.data<float>() + r * K,
                      reinterpret_cast<BlockQ8_0*>(static_cast<char*>(q.t.raw()) + r * q.t.row_bytes()), K);
  }
  return q;
}

// ||got - ref|| / ||ref|| against a double-precision product with the exact stored weights.
double relative_error(const OwnedTensor& got, const OwnedTensor& x, const OwnedTensor& w) {
  const int64_t T = x.t.rows(), K = x.t.cols(), N = w.t.rows();
  std::vector<float> wf(static_cast<size_t>(N * K));
  to_f32(w.t.dtype(), w.t.raw(), wf.data(), N * K);
  double err = 0, norm = 0;
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t n = 0; n < N; ++n) {
      double ref = 0;
      for (int64_t k = 0; k < K; ++k) ref += double(x.t.data<float>()[t * K + k]) * wf[size_t(n * K + k)];
      const double d = got.t.data<float>()[t * N + n] - ref;
      err += d * d;
      norm += ref * ref;
    }
  }
  return std::sqrt(err / norm);
}

}  // namespace

TEST_CASE("Metal matmul matches a double reference on the matvec and matmul paths") {
  MetalBackend metal;
  struct Case {
    int64_t T, K, N;
  };
  // T = 1 and 5 take the matvec kernels; T = 40 takes the matmul kernel, with
  // N and T off the 64/32 tile grid (bounds-checked output) and, for K = 100,
  // K off the 32-wide k-step (bounds-checked input).
  for (const Case c : {Case{1, 256, 100}, Case{5, 256, 100}, Case{40, 256, 130}, Case{40, 100, 64}}) {
    for (DType dtype : {DType::F32, DType::F16, DType::BF16, DType::Q8_0}) {
      if (dtype == DType::Q8_0 && c.K % 32 != 0) continue;
      CAPTURE(c.T);
      CAPTURE(c.K);
      CAPTURE(c.N);
      CAPTURE(name(dtype));
      const OwnedTensor x = test::on(metal, test::random_f32(Shape{c.T, c.K}, 7));
      const OwnedTensor w = test::on(metal, make_weights(dtype, c.N, c.K, 8));
      OwnedTensor out(metal, DType::F32, Shape{c.T, c.N});
      metal.matmul(x.t, w.t, out.t);
      const double err = relative_error(out, x, w);
      // Matvec keeps F32 activations; matmul stages tiles as half precision.
      CHECK(err < (c.T > 8 ? 2e-3 : 1e-5));
    }
  }
}

TEST_CASE("Metal matmul rejects shapes its kernels cannot read") {
  MetalBackend metal;
  const OwnedTensor x = test::on(metal, test::random_f32(Shape{2, 6}, 9));
  const OwnedTensor w = test::on(metal, test::random_f32(Shape{4, 6}, 10));
  OwnedTensor out(metal, DType::F32, Shape{2, 4});
  CHECK_THROWS_AS(metal.matmul(x.t, w.t, out.t), InvalidArgument);  // K = 6 is not a multiple of 4
}

TEST_CASE("Metal matmul rejects an odd row count on the matvec path") {
  MetalBackend metal;
  const OwnedTensor x = test::on(metal, test::random_f32(Shape{1, 256}, 11));
  const OwnedTensor w = test::on(metal, test::random_f32(Shape{99, 256}, 12));
  OwnedTensor out(metal, DType::F32, Shape{1, 99});
  // The matvec kernels read weight rows in pairs: N = 99 would read a 100th row.
  CHECK_THROWS_WITH_AS(metal.matmul(x.t, w.t, out.t), doctest::Contains("pairs"), InvalidArgument);
}
