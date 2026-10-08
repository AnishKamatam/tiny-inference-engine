#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "core/error.h"
#include "core/half.h"
#include "kernels/cpu/vec.h"
#include "support/tensors.h"

using namespace tie;
using test::OwnedTensor;

TEST_CASE("embed gathers and converts rows of every float dtype") {
  CpuBackend be(4);
  const OwnedTensor table = test::random_f32(Shape{10, 8}, 1);
  const OwnedTensor ids = test::i32_tensor({3, 0, 9});
  for (DType dtype : {DType::F32, DType::F16, DType::BF16}) {
    CAPTURE(name(dtype));
    const OwnedTensor typed = test::convert(table, dtype);
    OwnedTensor out(DType::F32, Shape{3, 8});
    be.embed(typed.t, ids.t, out.t);
    std::vector<float> want(8);
    to_f32(dtype, static_cast<const char*>(typed.t.raw()) + 9 * typed.t.row_bytes(), want.data(), 8);
    for (int i = 0; i < 8; ++i) CHECK(out.t.data<float>()[2 * 8 + i] == want[static_cast<size_t>(i)]);
  }
  OwnedTensor out(DType::F32, Shape{1, 8});
  const OwnedTensor bad = test::i32_tensor({10});
  CHECK_THROWS_AS(be.embed(table.t, bad.t, out.t), InvalidArgument);
}

TEST_CASE("rms_norm matches the formula and works in place") {
  CpuBackend be(4);
  OwnedTensor x = test::random_f32(Shape{5, 64}, 2, 3.0f);
  const OwnedTensor w = test::random_f32(Shape{64}, 3);
  const float eps = 1e-6f;

  std::vector<double> want(5 * 64);
  for (int r = 0; r < 5; ++r) {
    double ss = 0;
    for (int i = 0; i < 64; ++i) ss += double(x.data<float>()[r * 64 + i]) * x.data<float>()[r * 64 + i];
    const double inv = 1.0 / std::sqrt(ss / 64 + eps);
    for (int i = 0; i < 64; ++i) want[size_t(r * 64 + i)] = x.data<float>()[r * 64 + i] * inv * w.t.data<float>()[i];
  }
  be.rms_norm(x.t, w.t, eps, x.t);  // in place
  for (size_t i = 0; i < want.size(); ++i) CHECK(x.data<float>()[i] == doctest::Approx(want[i]).epsilon(1e-5));
}

TEST_CASE("rope_neox rotates pairs (i, i + d/2) and encodes relative position") {
  CpuBackend be(1);
  const int heads = 2, dim = 8;
  const float theta = 10000.0f;

  SUBCASE("matches the formula") {
    OwnedTensor x = test::random_f32(Shape{2, heads * dim}, 4);
    const std::vector<float> before(x.data<float>(), x.data<float>() + 2 * heads * dim);
    const OwnedTensor pos = test::i32_tensor({0, 7});
    be.rope_neox(x.t, pos.t, heads, dim, theta);
    for (int i = 0; i < heads * dim; ++i) CHECK(x.data<float>()[i] == before[size_t(i)]);  // position 0 is identity
    for (int h = 0; h < heads; ++h) {
      for (int i = 0; i < dim / 2; ++i) {
        const double angle = 7.0 * std::pow(double(theta), -2.0 * i / dim);
        const double x1 = before[size_t(heads * dim + h * dim + i)];
        const double x2 = before[size_t(heads * dim + h * dim + i + dim / 2)];
        CHECK(x.data<float>()[heads * dim + h * dim + i] == doctest::Approx(x1 * std::cos(angle) - x2 * std::sin(angle)).epsilon(1e-5));
        CHECK(x.data<float>()[heads * dim + h * dim + i + dim / 2] ==
              doctest::Approx(x2 * std::cos(angle) + x1 * std::sin(angle)).epsilon(1e-5));
      }
    }
  }

  SUBCASE("q.k depends only on the position difference") {
    const OwnedTensor q0 = test::random_f32(Shape{1, dim}, 5);
    const OwnedTensor k0 = test::random_f32(Shape{1, dim}, 6);
    const auto score = [&](int qpos, int kpos) {
      OwnedTensor q(DType::F32, Shape{1, dim});
      OwnedTensor k(DType::F32, Shape{1, dim});
      for (int i = 0; i < dim; ++i) {
        q.data<float>()[i] = q0.t.data<float>()[i];
        k.data<float>()[i] = k0.t.data<float>()[i];
      }
      const OwnedTensor qp = test::i32_tensor({qpos});
      const OwnedTensor kp = test::i32_tensor({kpos});
      be.rope_neox(q.t, qp.t, 1, dim, theta);
      be.rope_neox(k.t, kp.t, 1, dim, theta);
      return dot_f32(q.data<float>(), k.data<float>(), dim);
    };
    CHECK(score(5, 3) == doctest::Approx(score(12, 10)).epsilon(1e-4));
  }
}

TEST_CASE("silu_mul, add and gather_rows") {
  CpuBackend be(4);
  const OwnedTensor g = test::random_f32(Shape{3, 16}, 7);
  const OwnedTensor u = test::random_f32(Shape{3, 16}, 8);
  OwnedTensor out(DType::F32, Shape{3, 16});

  be.silu_mul(g.t, u.t, out.t);
  for (int i = 0; i < 48; ++i) {
    const double x = g.t.data<float>()[i];
    CHECK(out.data<float>()[i] == doctest::Approx(x / (1 + std::exp(-x)) * u.t.data<float>()[i]).epsilon(1e-6));
  }

  be.add(g.t, u.t, out.t);
  for (int i = 0; i < 48; ++i) CHECK(out.data<float>()[i] == g.t.data<float>()[i] + u.t.data<float>()[i]);

  const OwnedTensor rows = test::i32_tensor({2, 0});
  OwnedTensor picked(DType::F32, Shape{2, 16});
  be.gather_rows(g.t, rows.t, picked.t);
  for (int i = 0; i < 16; ++i) {
    CHECK(picked.data<float>()[i] == g.t.data<float>()[2 * 16 + i]);
    CHECK(picked.data<float>()[16 + i] == g.t.data<float>()[i]);
  }

  OwnedTensor wrong(DType::F32, Shape{3, 8});
  CHECK_THROWS_AS(be.add(g.t, u.t, wrong.t), InvalidArgument);
}

TEST_CASE("CpuBackend buffers are page aligned and wrap without copying") {
  CpuBackend be(1);
  const auto buf = be.alloc(1000);
  CHECK(reinterpret_cast<uintptr_t>(buf->data()) % kHostAlignment == 0);
  const std::vector<float> host(4, 1.0f);
  const auto wrapped = be.wrap(host.data(), host.size() * sizeof(float));
  CHECK(wrapped->data() == host.data());
}

TEST_CASE("CpuBackend rejects bad dtypes and shapes before running kernels") {
  CpuBackend be(4);
  const OwnedTensor table(DType::I32, Shape{100, 512});
  std::vector<int32_t> many(80);
  for (size_t i = 0; i < many.size(); ++i) many[i] = static_cast<int32_t>(i);
  const OwnedTensor ids = test::i32_tensor(many);
  OwnedTensor out(DType::F32, Shape{80, 512});
  CHECK_THROWS_AS(be.embed(table.t, ids.t, out.t), InvalidArgument);

  const OwnedTensor a = test::random_f32(Shape{3, 16}, 9);
  OwnedTensor half_out(DType::F16, Shape{3, 16});
  CHECK_THROWS_AS(be.add(a.t, a.t, half_out.t), InvalidArgument);

  OwnedTensor x = test::random_f32(Shape{2, 8}, 10);
  const OwnedTensor pos = test::i32_tensor({0, 1, 2});
  CHECK_THROWS_AS(be.rope_neox(x.t, pos.t, 1, 8, 10000.0f), InvalidArgument);
}
