#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "backend/cpu/cpu_backend.h"
#include "core/error.h"
#include "kv/paged_kv_cache.h"
#include "support/tensors.h"

using namespace tie;
using test::OwnedTensor;

namespace {

constexpr int64_t H = 4, KVH = 2, D = 16, BS = 4, NB = 16;
const float kScale = 1.0f / std::sqrt(float(D));

// Causal attention for one sequence in double precision. q rows are positions
// first_pos, first_pos + 1, ...; k and v hold every position from 0.
std::vector<double> dense_attention(const float* q, int64_t n_q, int64_t first_pos, const float* k, const float* v) {
  std::vector<double> out(static_cast<size_t>(n_q * H * D), 0.0);
  for (int64_t i = 0; i < n_q; ++i) {
    const int64_t pos = first_pos + i;
    for (int64_t h = 0; h < H; ++h) {
      const int64_t kvh = h / (H / KVH);
      std::vector<double> scores(static_cast<size_t>(pos + 1));
      double max = -1e300;
      for (int64_t j = 0; j <= pos; ++j) {
        double s = 0;
        for (int64_t d = 0; d < D; ++d) s += double(q[i * H * D + h * D + d]) * k[j * KVH * D + kvh * D + d];
        scores[size_t(j)] = s * kScale;
        max = std::max(max, scores[size_t(j)]);
      }
      double sum = 0;
      for (double& s : scores) sum += (s = std::exp(s - max));
      for (int64_t j = 0; j <= pos; ++j) {
        for (int64_t d = 0; d < D; ++d) {
          out[size_t(i * H * D + h * D + d)] += scores[size_t(j)] / sum * v[j * KVH * D + kvh * D + d];
        }
      }
    }
  }
  return out;
}

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

void check_close(const float* got, const std::vector<double>& want, double tol) {
  for (size_t i = 0; i < want.size(); ++i) CHECK(std::abs(got[i] - want[i]) <= tol);
}

}  // namespace

TEST_CASE("paged attention over scattered blocks equals dense causal attention") {
  CpuBackend be(4);
  const std::vector<int32_t> table = {7, 2, 12};  // deliberately non-contiguous and unordered
  const int64_t ctx = 11;
  const OwnedTensor k = test::random_f32(Shape{ctx, KVH * D}, 21);
  const OwnedTensor v = test::random_f32(Shape{ctx, KVH * D}, 22);
  const OwnedTensor q = test::random_f32(Shape{ctx, H * D}, 23);
  const auto want = dense_attention(q.t.data<float>(), ctx, 0, k.t.data<float>(), v.t.data<float>());

  for (DType cache_dtype : {DType::F32, DType::F16}) {
    CAPTURE(name(cache_dtype));
    const double tol = cache_dtype == DType::F32 ? 1e-5 : 3e-3;
    OwnedTensor kc(cache_dtype, Shape{NB, KVH, BS, D});
    OwnedTensor vc(cache_dtype, Shape{NB, KVH, BS, D});
    const OwnedTensor slots = slots_for(table, 0, ctx);
    be.kv_write(k.t, v.t, slots.t, kc.t, vc.t);

    const OwnedTensor tables = table_tensor({table}, 3);

    // Prefill: every position attends causally.
    const OwnedTensor qs_all = test::i32_tensor({0, int32_t(ctx)});
    const OwnedTensor lens = test::i32_tensor({int32_t(ctx)});
    OwnedTensor out_all(DType::F32, Shape{ctx, H * D});
    be.paged_attention(q.t, kc.t, vc.t, AttentionMetadata{1, qs_all.t, lens.t, tables.t}, int(H), kScale, out_all.t);
    check_close(out_all.data<float>(), want, tol);

    // Decode: the last position alone.
    const Tensor last = q.t.slice_rows(ctx - 1, 1);
    const OwnedTensor qs_one = test::i32_tensor({0, 1});
    OwnedTensor out_one(DType::F32, Shape{1, H * D});
    be.paged_attention(last, kc.t, vc.t, AttentionMetadata{1, qs_one.t, lens.t, tables.t}, int(H), kScale, out_one.t);
    check_close(out_one.data<float>(), std::vector<double>(want.end() - H * D, want.end()), tol);
  }
}

TEST_CASE("paged attention handles several sequences of different lengths in one batch") {
  CpuBackend be(4);
  OwnedTensor kc(DType::F32, Shape{NB, KVH, BS, D});
  OwnedTensor vc(DType::F32, Shape{NB, KVH, BS, D});

  // Sequence A: 5 tokens, all new. Sequence B: 7 tokens, the first 5 cached by an earlier step.
  const std::vector<int32_t> table_a = {3, 9}, table_b = {0, 14};
  const OwnedTensor ka = test::random_f32(Shape{5, KVH * D}, 31), va = test::random_f32(Shape{5, KVH * D}, 32);
  const OwnedTensor kb = test::random_f32(Shape{7, KVH * D}, 33), vb = test::random_f32(Shape{7, KVH * D}, 34);
  be.kv_write(ka.t, va.t, slots_for(table_a, 0, 5).t, kc.t, vc.t);
  be.kv_write(kb.t, vb.t, slots_for(table_b, 0, 7).t, kc.t, vc.t);

  const OwnedTensor q = test::random_f32(Shape{7, H * D}, 35);  // rows 0-4: A, rows 5-6: B
  const OwnedTensor qs = test::i32_tensor({0, 5, 7});
  const OwnedTensor lens = test::i32_tensor({5, 7});
  const OwnedTensor tables = table_tensor({table_a, table_b}, 2);
  OwnedTensor out(DType::F32, Shape{7, H * D});
  be.paged_attention(q.t, kc.t, vc.t, AttentionMetadata{2, qs.t, lens.t, tables.t}, int(H), kScale, out.t);

  check_close(out.data<float>(), dense_attention(q.t.data<float>(), 5, 0, ka.t.data<float>(), va.t.data<float>()), 1e-5);
  check_close(out.data<float>() + 5 * H * D,
              dense_attention(q.t.data<float>() + 5 * H * D, 2, 5, kb.t.data<float>(), vb.t.data<float>()), 1e-5);
}

TEST_CASE("kv_write places each head at its slot and rejects bad slots") {
  CpuBackend be(1);
  OwnedTensor kc(DType::F32, Shape{NB, KVH, BS, D});
  OwnedTensor vc(DType::F32, Shape{NB, KVH, BS, D});
  const OwnedTensor k = test::random_f32(Shape{1, KVH * D}, 41);
  const OwnedTensor v = test::random_f32(Shape{1, KVH * D}, 42);
  const OwnedTensor slot = test::i32_tensor({5 * int32_t(BS) + 3});  // block 5, offset 3
  be.kv_write(k.t, v.t, slot.t, kc.t, vc.t);
  const int64_t head1 = ((5 * KVH + 1) * BS + 3) * D;
  CHECK(kc.data<float>()[head1 + 2] == k.t.data<float>()[D + 2]);
  CHECK(vc.data<float>()[head1 + 2] == v.t.data<float>()[D + 2]);

  const OwnedTensor bad = test::i32_tensor({int32_t(NB * BS)});
  CHECK_THROWS_AS(be.kv_write(k.t, v.t, bad.t, kc.t, vc.t), InvalidArgument);
}

TEST_CASE("paged attention rejects inconsistent metadata") {
  CpuBackend be(1);
  OwnedTensor kc(DType::F32, Shape{NB, KVH, BS, D});
  OwnedTensor vc(DType::F32, Shape{NB, KVH, BS, D});
  const OwnedTensor q = test::random_f32(Shape{2, H * D}, 51);
  OwnedTensor out(DType::F32, Shape{2, H * D});
  const OwnedTensor qs = test::i32_tensor({0, 2});

  SUBCASE("block id outside the pool") {
    const OwnedTensor lens = test::i32_tensor({2});
    const OwnedTensor tables = table_tensor({{int32_t(NB)}}, 1);
    CHECK_THROWS_AS(be.paged_attention(q.t, kc.t, vc.t, {1, qs.t, lens.t, tables.t}, int(H), kScale, out.t),
                    InvalidArgument);
  }
  SUBCASE("context shorter than the queries") {
    const OwnedTensor lens = test::i32_tensor({1});
    const OwnedTensor tables = table_tensor({{0}}, 1);
    CHECK_THROWS_AS(be.paged_attention(q.t, kc.t, vc.t, {1, qs.t, lens.t, tables.t}, int(H), kScale, out.t),
                    InvalidArgument);
  }
}
