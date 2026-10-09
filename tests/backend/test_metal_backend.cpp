#include <doctest/doctest.h>

#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "backend/metal/metal_backend.h"
#include "core/error.h"
#include "support/tensors.h"

using namespace tie;
using test::OwnedTensor;

TEST_CASE("MetalBackend compiles its shaders and allocates host-visible memory") {
  MetalBackend metal;
  CHECK(metal.name() == "metal");
  const auto buf = metal.alloc(4096);
  static_cast<float*>(buf->data())[3] = 7.0f;  // unified memory: writable from the host
  CHECK(static_cast<float*>(buf->data())[3] == 7.0f);
}

TEST_CASE("Metal add and gather_rows match the CPU exactly") {
  MetalBackend metal;
  CpuBackend cpu(2);
  const OwnedTensor a = test::random_f32(Shape{5, 33}, 1);
  const OwnedTensor b = test::random_f32(Shape{5, 33}, 2);
  OwnedTensor want(DType::F32, Shape{5, 33});
  cpu.add(a.t, b.t, want.t);

  const OwnedTensor ga = test::on(metal, a), gb = test::on(metal, b);
  OwnedTensor got(metal, DType::F32, Shape{5, 33});
  metal.add(ga.t, gb.t, got.t);
  for (int i = 0; i < 5 * 33; ++i) CHECK(got.data<float>()[i] == want.data<float>()[i]);

  const OwnedTensor rows = test::i32_tensor({4, 0, 4});
  const OwnedTensor grows = test::on(metal, rows);
  OwnedTensor want_rows(DType::F32, Shape{3, 33}), got_rows(metal, DType::F32, Shape{3, 33});
  cpu.gather_rows(a.t, rows.t, want_rows.t);
  metal.gather_rows(ga.t, grows.t, got_rows.t);
  for (int i = 0; i < 3 * 33; ++i) CHECK(got_rows.data<float>()[i] == want_rows.data<float>()[i]);
}

TEST_CASE("ops inside one step see each other's results") {
  MetalBackend metal;
  const OwnedTensor a = test::on(metal, test::random_f32(Shape{64}, 3));
  OwnedTensor acc(metal, DType::F32, Shape{64});
  for (int i = 0; i < 64; ++i) acc.data<float>()[i] = 0.0f;
  metal.begin_step();
  for (int k = 0; k < 3; ++k) metal.add(acc.t, a.t, acc.t);  // acc = 3a, in place
  metal.end_step();
  for (int i = 0; i < 64; ++i) CHECK(acc.data<float>()[i] == doctest::Approx(3.0f * a.t.data<float>()[i]));
  CHECK_THROWS_AS(metal.end_step(), InvalidArgument);
  metal.begin_step();
  CHECK_THROWS_AS(metal.begin_step(), InvalidArgument);
  metal.end_step();
}

TEST_CASE("wrap shares caller memory without copying, even at unaligned addresses") {
  MetalBackend metal;
  HostBuffer host(1 << 16);
  float* base = static_cast<float*>(host.data());
  for (int i = 0; i < 1024; ++i) base[i] = float(i);
  // Wrap starting 8 floats in: not page aligned.
  const auto wrapped = metal.wrap(base + 8, 256 * sizeof(float));
  CHECK(wrapped->data() == base + 8);
  const Tensor w(wrapped.get(), 0, DType::F32, Shape{256});
  OwnedTensor out(metal, DType::F32, Shape{256});
  metal.add(w, w, out.t);
  CHECK(out.data<float>()[0] == 16.0f);  // 2 * base[8]
  base[8] = 100.0f;                      // later host writes are visible: no copy was made
  metal.add(w, w, out.t);
  CHECK(out.data<float>()[0] == 200.0f);
}

TEST_CASE("Metal ops reject tensors that are not in Metal memory") {
  MetalBackend metal;
  const OwnedTensor host = test::random_f32(Shape{8}, 4);
  OwnedTensor out(metal, DType::F32, Shape{8});
  CHECK_THROWS_WITH_AS(metal.add(host.t, host.t, out.t), doctest::Contains("Metal"), InvalidArgument);
  const OwnedTensor bad_rows = test::on(metal, test::i32_tensor({9}));
  const OwnedTensor x = test::on(metal, test::random_f32(Shape{2, 4}, 5));
  OwnedTensor picked(metal, DType::F32, Shape{1, 4});
  CHECK_THROWS_AS(metal.gather_rows(x.t, bad_rows.t, picked.t), InvalidArgument);
}

TEST_CASE("Metal add and gather_rows check every operand's dtype") {
  MetalBackend metal;
  const OwnedTensor a(metal, DType::F32, Shape{4});
  const OwnedTensor ints(metal, DType::I32, Shape{4});
  OwnedTensor out(metal, DType::F32, Shape{4});
  OwnedTensor int_out(metal, DType::I32, Shape{4});
  CHECK_THROWS_AS(metal.add(a.t, ints.t, out.t), InvalidArgument);
  CHECK_THROWS_AS(metal.add(a.t, a.t, int_out.t), InvalidArgument);
  const OwnedTensor rows = test::on(metal, test::i32_tensor({0}));
  const OwnedTensor x(metal, DType::F32, Shape{1, 4});
  OwnedTensor bad_out(metal, DType::I32, Shape{1, 4});
  CHECK_THROWS_AS(metal.gather_rows(x.t, rows.t, bad_out.t), InvalidArgument);
}

TEST_CASE("a rejected op leaves the backend usable") {
  MetalBackend metal;
  const OwnedTensor host = test::random_f32(Shape{4}, 6);
  const OwnedTensor one = test::on(metal, test::random_f32(Shape{4}, 7));
  OwnedTensor out(metal, DType::F32, Shape{4});
  CHECK_THROWS_AS(metal.add(host.t, host.t, out.t), InvalidArgument);
  metal.add(one.t, one.t, out.t);
  for (int i = 0; i < 4; ++i) CHECK(out.data<float>()[i] == 2.0f * one.t.data<float>()[i]);
  metal.begin_step();
  metal.end_step();
}

TEST_CASE("a rejected op inside a step discards the step") {
  MetalBackend metal;
  const OwnedTensor host = test::random_f32(Shape{4}, 8);
  const OwnedTensor one = test::on(metal, test::random_f32(Shape{4}, 9));
  OwnedTensor out(metal, DType::F32, Shape{4});
  metal.begin_step();
  CHECK_THROWS_AS(metal.add(host.t, host.t, out.t), InvalidArgument);
  CHECK_THROWS_AS(metal.end_step(), InvalidArgument);
  metal.begin_step();
  metal.add(one.t, one.t, out.t);
  metal.end_step();
  for (int i = 0; i < 4; ++i) CHECK(out.data<float>()[i] == 2.0f * one.t.data<float>()[i]);
}

TEST_CASE("destroying a backend with an open step is safe") {
  {
    MetalBackend metal;
    metal.begin_step();
  }
  CHECK(true);
}

TEST_CASE("a validation failure or stub throw inside a step discards the whole step") {
  MetalBackend metal;
  const OwnedTensor a = test::on(metal, test::random_f32(Shape{4}, 10));
  const OwnedTensor ints = test::on(metal, test::i32_tensor({1, 2, 3, 4}));
  OwnedTensor out(metal, DType::F32, Shape{4});
  for (int i = 0; i < 4; ++i) out.data<float>()[i] = -1.0f;

  metal.begin_step();
  metal.add(a.t, a.t, out.t);
  CHECK_THROWS_AS(metal.add(a.t, ints.t, out.t), InvalidArgument);
  CHECK_THROWS_AS(metal.end_step(), InvalidArgument);
  for (int i = 0; i < 4; ++i) CHECK(out.data<float>()[i] == -1.0f);  // earlier add was dropped

  metal.begin_step();
  metal.add(a.t, a.t, out.t);
  CHECK_THROWS_AS(metal.silu_mul(a.t, a.t, out.t), UnsupportedError);
  CHECK_THROWS_AS(metal.end_step(), InvalidArgument);
  for (int i = 0; i < 4; ++i) CHECK(out.data<float>()[i] == -1.0f);

  metal.begin_step();
  metal.add(a.t, a.t, out.t);
  metal.end_step();
  for (int i = 0; i < 4; ++i) CHECK(out.data<float>()[i] == 2.0f * a.t.data<float>()[i]);
}
