#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "backend/metal/metal_backend.h"
#include "core/error.h"
#include "support/tensors.h"

using namespace tie;
using test::OwnedTensor;

namespace {

constexpr int64_t BS = 16, NB = 12;

OwnedTensor slots_for(const std::vector<int32_t>& table, int64_t begin, int64_t end) {
  std::vector<int32_t> slots;
  for (int64_t p = begin; p < end; ++p) slots.push_back(table[size_t(p / BS)] * int32_t(BS) + int32_t(p % BS));
  return test::i32_tensor(slots);
}

OwnedTensor table_tensor(const std::vector<std::vector<int32_t>>& tables, int64_t max_blocks) {
  OwnedTensor t(DType::I32, Shape{int64_t(tables.size()), max_blocks});
  for (size_t s = 0; s < tables.size(); ++s) {
    for (int64_t b = 0; b < max_blocks; ++b) {
      t.data<int32_t>()[s * size_t(max_blocks) + size_t(b)] = size_t(b) < tables[s].size() ? tables[s][size_t(b)] : 0;
    }
  }
  return t;
}

// Written so that a NaN on either side fails.
void check_close(const float* got, const float* want, int64_t n, float tol) {
  for (int64_t i = 0; i < n; ++i) {
    if (!(std::abs(got[i] - want[i]) <= tol)) {
      FAIL_CHECK("element " << i << ": metal " << got[i] << " vs cpu " << want[i]);
      return;
    }
  }
}

// Bit-for-bit cache comparison (both sides round to F16 the same way).
void check_same_bytes(const OwnedTensor& gpu, const OwnedTensor& cpu, const char* what) {
  CAPTURE(what);
  CHECK(std::memcmp(gpu.t.raw(), cpu.t.raw(), cpu.t.nbytes()) == 0);
}

// Writes two sequences' K/V through both backends, then checks the caches and the attention output.
// Sequence A: 37 tokens, all new (prefill). Sequence B: 70 tokens, 69 cached earlier, 1 new (decode).
void check_against_cpu(int64_t H, int64_t KVH, int64_t D) {
  CAPTURE(D);
  MetalBackend metal;
  CpuBackend cpu(4);
  const float scale = 1.0f / std::sqrt(float(D));
  const std::vector<int32_t> table_a = {9, 2, 5}, table_b = {0, 11, 3, 7, 1};
  const OwnedTensor ka = test::random_f32(Shape{37, KVH * D}, 1), va = test::random_f32(Shape{37, KVH * D}, 2);
  const OwnedTensor kb = test::random_f32(Shape{70, KVH * D}, 3), vb = test::random_f32(Shape{70, KVH * D}, 4);
  const OwnedTensor q = test::random_f32(Shape{38, H * D}, 5);  // rows 0-36: A, row 37: B
  const OwnedTensor qs = test::i32_tensor({0, 37, 38});
  const OwnedTensor lens = test::i32_tensor({37, 70});
  const OwnedTensor tables = table_tensor({table_a, table_b}, 5);

  for (DType cache_dtype : {DType::F32, DType::F16}) {
    CAPTURE(name(cache_dtype));
    OwnedTensor cpu_k(cache_dtype, Shape{NB, KVH, BS, D}), cpu_v(cache_dtype, Shape{NB, KVH, BS, D});
    OwnedTensor gpu_k(metal, cache_dtype, Shape{NB, KVH, BS, D}), gpu_v(metal, cache_dtype, Shape{NB, KVH, BS, D});
    // Blocks no sequence uses are compared too, so start every cache from zeros (host memory is uninitialized).
    for (OwnedTensor* c : {&cpu_k, &cpu_v, &gpu_k, &gpu_v}) std::memset(c->t.raw(), 0, c->t.nbytes());
    const auto write_both = [&](const OwnedTensor& k, const OwnedTensor& v, const OwnedTensor& slots) {
      cpu.kv_write(k.t, v.t, slots.t, cpu_k.t, cpu_v.t);
      const OwnedTensor gk = test::on(metal, k), gv = test::on(metal, v), gs = test::on(metal, slots);
      metal.kv_write(gk.t, gv.t, gs.t, gpu_k.t, gpu_v.t);
    };
    write_both(ka, va, slots_for(table_a, 0, 37));
    write_both(kb, vb, slots_for(table_b, 0, 70));
    check_same_bytes(gpu_k, cpu_k, "K cache");
    check_same_bytes(gpu_v, cpu_v, "V cache");

    OwnedTensor want(DType::F32, Shape{38, H * D});
    cpu.paged_attention(q.t, cpu_k.t, cpu_v.t, AttentionMetadata{2, qs.t, lens.t, tables.t}, int(H), scale, want.t);
    const OwnedTensor gq = test::on(metal, q), gqs = test::on(metal, qs), glens = test::on(metal, lens),
                      gtables = test::on(metal, tables);
    OwnedTensor got(metal, DType::F32, Shape{38, H * D});
    metal.paged_attention(gq.t, gpu_k.t, gpu_v.t, AttentionMetadata{2, gqs.t, glens.t, gtables.t}, int(H), scale,
                          got.t);
    check_close(got.t.data<float>(), want.t.data<float>(), 38 * H * D, 2e-5f);
  }
}

}  // namespace

TEST_CASE("Metal kv_write and paged attention match the CPU across sequences and cache dtypes") {
  check_against_cpu(8, 2, 128);
}

TEST_CASE("Metal paged attention matches the CPU at the smallest and largest head_dim") {
  SUBCASE("head_dim 32: one dimension per lane") { check_against_cpu(4, 2, 32); }
  SUBCASE("head_dim 256: eight dimensions per lane") { check_against_cpu(4, 1, 256); }
}

TEST_CASE("Metal attention rejects what its kernel cannot run") {
  MetalBackend metal;
  OwnedTensor kc(metal, DType::F32, Shape{4, 1, 4, 48}), vc(metal, DType::F32, Shape{4, 1, 4, 48});
  const OwnedTensor q = test::on(metal, test::random_f32(Shape{1, 48}, 6));
  const OwnedTensor qs = test::on(metal, test::i32_tensor({0, 1}));
  const OwnedTensor lens = test::on(metal, test::i32_tensor({1}));
  const OwnedTensor tables = test::on(metal, table_tensor({{0}}, 1));
  OwnedTensor out(metal, DType::F32, Shape{1, 48});
  // head_dim 48 is not a multiple of the 32-lane simdgroup
  CHECK_THROWS_AS(metal.paged_attention(q.t, kc.t, vc.t, {1, qs.t, lens.t, tables.t}, 1, 1.0f, out.t), InvalidArgument);
}

TEST_CASE("Metal attention rejects head_dim beyond its per-lane registers") {
  MetalBackend metal;
  // 288 = 9 * 32 passes the shared checks (<= kMaxHeadDim) but needs 9 dimensions per lane.
  OwnedTensor kc(metal, DType::F32, Shape{4, 1, 4, 288}), vc(metal, DType::F32, Shape{4, 1, 4, 288});
  const OwnedTensor q = test::on(metal, test::random_f32(Shape{1, 288}, 7));
  const OwnedTensor qs = test::on(metal, test::i32_tensor({0, 1}));
  const OwnedTensor lens = test::on(metal, test::i32_tensor({1}));
  const OwnedTensor tables = test::on(metal, table_tensor({{0}}, 1));
  OwnedTensor out(metal, DType::F32, Shape{1, 288});
  CHECK_THROWS_AS(metal.paged_attention(q.t, kc.t, vc.t, {1, qs.t, lens.t, tables.t}, 1, 1.0f, out.t), InvalidArgument);
}

TEST_CASE("KV caches with a zero dimension are rejected by both backends") {
  MetalBackend metal;
  CpuBackend cpu(1);
  // block_size 0 would divide by zero when mapping slots and positions to blocks.
  OwnedTensor kc(metal, DType::F32, Shape{4, 1, 0, 32}), vc(metal, DType::F32, Shape{4, 1, 0, 32});
  const OwnedTensor k = test::on(metal, test::random_f32(Shape{1, 32}, 8));
  const OwnedTensor v = test::on(metal, test::random_f32(Shape{1, 32}, 9));
  const OwnedTensor slots = test::on(metal, test::i32_tensor({0}));
  const OwnedTensor q = test::on(metal, test::random_f32(Shape{1, 32}, 10));
  const OwnedTensor qs = test::on(metal, test::i32_tensor({0, 1}));
  const OwnedTensor lens = test::on(metal, test::i32_tensor({1}));
  const OwnedTensor tables = test::on(metal, table_tensor({{0}}, 1));
  OwnedTensor out(metal, DType::F32, Shape{1, 32});
  for (Backend* backend : {static_cast<Backend*>(&cpu), static_cast<Backend*>(&metal)}) {
    CHECK_THROWS_AS(backend->kv_write(k.t, v.t, slots.t, kc.t, vc.t), InvalidArgument);
    CHECK_THROWS_AS(backend->paged_attention(q.t, kc.t, vc.t, {1, qs.t, lens.t, tables.t}, 1, 1.0f, out.t),
                    InvalidArgument);
  }
}
