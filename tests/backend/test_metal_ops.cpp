#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "backend/metal/metal_backend.h"
#include "kernels/cpu/q8_0.h"
#include "support/tensors.h"

using namespace tie;
using test::OwnedTensor;

namespace {

void check_close(const OwnedTensor& got, const OwnedTensor& want, float rel, float abs) {
  REQUIRE(got.t.numel() == want.t.numel());
  for (int64_t i = 0; i < got.t.numel(); ++i) {
    const float g = got.t.data<float>()[i], w = want.t.data<float>()[i];
    if (std::abs(g - w) > abs + rel * std::abs(w)) {
      FAIL_CHECK("element " << i << ": metal " << g << " vs cpu " << w);
      return;
    }
  }
}

OwnedTensor quantize(const OwnedTensor& f32) {
  OwnedTensor q(DType::Q8_0, f32.t.shape());
  for (int64_t r = 0; r < f32.t.rows(); ++r) {
    quantize_row_q8_0(f32.t.data<float>() + r * f32.t.cols(),
                      reinterpret_cast<BlockQ8_0*>(static_cast<char*>(q.t.raw()) + r * q.t.row_bytes()), f32.t.cols());
  }
  return q;
}

}  // namespace

TEST_CASE("Metal embed matches the CPU for every table dtype") {
  MetalBackend metal;
  CpuBackend cpu(2);
  const OwnedTensor f32 = test::random_f32(Shape{50, 64}, 1);
  const OwnedTensor ids = test::i32_tensor({49, 0, 7, 7});
  for (DType dtype : {DType::F32, DType::F16, DType::BF16, DType::Q8_0}) {
    CAPTURE(name(dtype));
    const OwnedTensor table = dtype == DType::Q8_0 ? quantize(f32) : test::convert(f32, dtype);
    OwnedTensor want(DType::F32, Shape{4, 64});
    cpu.embed(table.t, ids.t, want.t);
    const OwnedTensor gtable = test::on(metal, table), gids = test::on(metal, ids);
    OwnedTensor got(metal, DType::F32, Shape{4, 64});
    metal.embed(gtable.t, gids.t, got.t);
    check_close(got, want, 0.0f, 0.0f);  // pure conversion: exact
  }
}

TEST_CASE("Metal rms_norm matches the CPU for short and long rows, in place") {
  MetalBackend metal;
  CpuBackend cpu(2);
  for (int64_t dim : {128, 1024, 5000}) {  // 5000 > 4096 takes the looped kernel
    CAPTURE(dim);
    const OwnedTensor x = test::random_f32(Shape{3, dim}, 2, 3.0f);
    const OwnedTensor w = test::random_f32(Shape{dim}, 3);
    OwnedTensor want(DType::F32, Shape{3, dim});
    cpu.rms_norm(x.t, w.t, 1e-6f, want.t);
    OwnedTensor gx = test::on(metal, x);
    const OwnedTensor gw = test::on(metal, w);
    metal.rms_norm(gx.t, gw.t, 1e-6f, gx.t);
    check_close(gx, want, 1e-5f, 1e-6f);
  }
}

TEST_CASE("Metal rope_neox matches the CPU, including large positions") {
  MetalBackend metal;
  CpuBackend cpu(2);
  const int heads = 4, dim = 128;
  const OwnedTensor x = test::random_f32(Shape{4, heads * dim}, 4);
  const OwnedTensor pos = test::i32_tensor({0, 17, 4000, 40000});
  OwnedTensor want = test::on(cpu, x);
  cpu.rope_neox(want.t, pos.t, heads, dim, 1e6f);
  OwnedTensor got = test::on(metal, x);
  const OwnedTensor gpos = test::on(metal, pos);
  metal.rope_neox(got.t, gpos.t, heads, dim, 1e6f);
  check_close(got, want, 1e-4f, 1e-4f);
}

TEST_CASE("Metal silu_mul matches the CPU, in place") {
  MetalBackend metal;
  CpuBackend cpu(2);
  const OwnedTensor g = test::random_f32(Shape{5, 3072}, 5, 4.0f);
  const OwnedTensor u = test::random_f32(Shape{5, 3072}, 6);
  OwnedTensor want(DType::F32, Shape{5, 3072});
  cpu.silu_mul(g.t, u.t, want.t);
  OwnedTensor gg = test::on(metal, g);
  const OwnedTensor gu = test::on(metal, u);
  metal.silu_mul(gg.t, gu.t, gg.t);
  check_close(gg, want, 1e-5f, 1e-6f);
}
