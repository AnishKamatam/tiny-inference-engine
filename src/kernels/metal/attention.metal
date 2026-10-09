#include "common.h"
#include "tie_args.h"

// One thread per K/V element of this step: scatter into the paged cache at the token's slot.
template <typename T>
kernel void tie_kv_write(constant tie_kv_write_args& args [[buffer(0)]],
                         device const float* k [[buffer(1)]],
                         device const float* v [[buffer(2)]],
                         device const int32_t* slots [[buffer(3)]],
                         device T* k_cache [[buffer(4)]],
                         device T* v_cache [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
  const int row = args.kv_heads * args.head_dim;
  if (int(gid) >= args.num_tokens * row) return;
  const int t = gid / row;
  const int h = (gid % row) / args.head_dim;
  const int d = gid % args.head_dim;
  const int slot = slots[t];
  const int dst = ((slot / args.block_size * args.kv_heads + h) * args.block_size + slot % args.block_size) * args.head_dim + d;
  k_cache[dst] = T(k[gid]);
  v_cache[dst] = T(v[gid]);
}

template [[host_name("tie_kv_write_f32")]] kernel decltype(tie_kv_write<float>) tie_kv_write<float>;
template [[host_name("tie_kv_write_f16")]] kernel decltype(tie_kv_write<half>) tie_kv_write<half>;

constant constexpr int kMaxDimsPerLane = 8;  // head_dim <= 256

// One simdgroup per (query token, query head). Lane i owns dimensions
// i, i + 32, i + 64, ... of the head; scores are reduced with simd_sum and
// folded into an online softmax, so the whole context is one pass over K/V.
template <typename T>
kernel void tie_paged_attention(constant tie_attention_args& args [[buffer(0)]],
                                device const float* q [[buffer(1)]],
                                device const T* k_cache [[buffer(2)]],
                                device const T* v_cache [[buffer(3)]],
                                device const int32_t* query_start [[buffer(4)]],
                                device const int32_t* context_lens [[buffer(5)]],
                                device const int32_t* block_tables [[buffer(6)]],
                                device float* out [[buffer(7)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                ushort lane [[thread_index_in_simdgroup]]) {
  const int h = group.x;
  const int t = group.y;

  int s = 0;  // the sequence owning batch row t (num_seqs is small)
  while (s + 1 < args.num_seqs && query_start[s + 1] <= t) ++s;
  const int n_queries = query_start[s + 1] - query_start[s];
  const int pos = context_lens[s] - n_queries + (t - query_start[s]);
  const int kvh = h / (args.num_heads / args.kv_heads);
  const int per_lane = args.head_dim / 32;
  device const int32_t* table = block_tables + s * args.max_blocks;

  float qv[kMaxDimsPerLane];
  float acc[kMaxDimsPerLane];
  device const float* q_row = q + (t * args.num_heads + h) * args.head_dim;
  for (int i = 0; i < per_lane; ++i) {
    qv[i] = q_row[lane + 32 * i] * args.scale;
    acc[i] = 0.0f;
  }

  // -FLT_MAX rather than -INFINITY: shaders compile with fast math, which assumes no infinities.
  float m = -FLT_MAX;
  float l = 0.0f;
  for (int j = 0; j <= pos; ++j) {
    const int base =
        ((table[j / args.block_size] * args.kv_heads + kvh) * args.block_size + j % args.block_size) * args.head_dim;
    float partial = 0.0f;
    for (int i = 0; i < per_lane; ++i) partial += qv[i] * float(k_cache[base + lane + 32 * i]);
    const float score = simd_sum(partial);
    const float m_new = max(m, score);
    // At j == 0, m - m_new is about -FLT_MAX and fast-math exp (exp2 of x * log2(e)) would
    // overflow to -inf internally; l and acc are still zero, so skip it.
    const float correction = j == 0 ? 0.0f : exp(m - m_new);
    const float p = exp(score - m_new);
    l = l * correction + p;
    for (int i = 0; i < per_lane; ++i) acc[i] = acc[i] * correction + p * float(v_cache[base + lane + 32 * i]);
    m = m_new;
  }

  device float* o = out + (t * args.num_heads + h) * args.head_dim;
  for (int i = 0; i < per_lane; ++i) o[lane + 32 * i] = acc[i] / l;
}

template [[host_name("tie_paged_attention_f32")]] kernel decltype(tie_paged_attention<float>) tie_paged_attention<float>;
template [[host_name("tie_paged_attention_f16")]] kernel decltype(tie_paged_attention<half>) tie_paged_attention<half>;
