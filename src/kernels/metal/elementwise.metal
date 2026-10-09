#include "common.h"
#include "ggml_args.h"

// out[i] = a[i] + b[i]
kernel void tie_add_f32(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant uint& n [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
  if (i < n) out[i] = a[i] + b[i];
}

// out[r][c] = x[rows[r]][c]
kernel void tie_gather_rows_f32(device const float* x [[buffer(0)]],
                                device const int32_t* rows [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant uint& cols [[buffer(3)]],
                                constant uint& n [[buffer(4)]],
                                uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const uint r = i / cols;
  const uint c = i % cols;
  out[i] = x[uint(rows[r]) * cols + c];
}

// ---------------------------------------------------------------------------------------------
// Ported from llama.cpp (https://github.com/ggml-org/llama.cpp), ggml/src/ggml-metal/kernels/
// quantize.metal (get_rows) and unary.metal (swiglu), commit 9c2e0e491a822adae1f0b1c831adb4160057d24f.
// Copyright (c) 2023-2026 The ggml authors. MIT License; see THIRD_PARTY_NOTICES.

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_get_rows_q(
        constant ggml_metal_kargs_get_rows & args,
        device const void * src0,
        device const void * src1,
        device       void * dst,
        uint3               tgpig[[threadgroup_position_in_grid]],
        ushort              tiitg[[thread_index_in_threadgroup]],
        ushort3             ntg  [[threads_per_threadgroup]]) {
    const int32_t iw0 = tgpig.x/args.ne10;
    const int32_t i10 = tgpig.x%args.ne10;
    const int32_t i11 = tgpig.y;
    const int32_t i12 = tgpig.z;

    const int32_t r = ((const device int32_t *) ((const device char *) src1 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10))[0];

    const int32_t i02 = i11;
    const int32_t i03 = i12;

    auto psrc = (device const block_q *) ((const device char *) src0 + i03*args.nb03 + i02*args.nb02 +   r*args.nb01);
    auto pdst = (device      float4x4 *) ((      device char *) dst  + i12*args.nb3  + i11*args.nb2  + i10*args.nb1);

    for (int ind = iw0*ntg.x + tiitg; ind < args.ne00t;) {
        float4x4 temp;
        dequantize_func(psrc + ind/nl, ind%nl, temp);
        pdst[ind] = temp;

        break;
    }
}

template<typename T0, typename T>
kernel void kernel_get_rows_f(
        constant ggml_metal_kargs_get_rows & args,
        device const void * src0,
        device const void * src1,
        device       void * dst,
        uint3               tgpig[[threadgroup_position_in_grid]],
        ushort              tiitg[[thread_index_in_threadgroup]],
        ushort3             ntg [[threads_per_threadgroup]]) {
    const int32_t iw0 = tgpig.x/args.ne10;
    const int32_t i10 = tgpig.x%args.ne10;
    const int32_t i11 = tgpig.y;
    const int32_t i12 = tgpig.z;

    const int32_t r = ((const device int32_t *) ((const device char *) src1 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10))[0];

    const int32_t i02 = i11;
    const int32_t i03 = i12;

    auto psrc = (const device T0 *) ((const device char *) src0 + i03*args.nb03 + i02*args.nb02 +   r*args.nb01);
    auto pdst = (      device T  *) ((      device char *)  dst + i12*args.nb3  + i11*args.nb2  + i10*args.nb1);

    for (int ind = iw0*ntg.x + tiitg; ind < args.ne00t;) {
        pdst[ind] = psrc[ind];

        break;
    }
}

typedef decltype(kernel_get_rows_f<float, float>) get_rows_f_t;

template [[host_name("kernel_get_rows_f32")]]  kernel get_rows_f_t kernel_get_rows_f<float, float>;
template [[host_name("kernel_get_rows_f16")]]  kernel get_rows_f_t kernel_get_rows_f<half,  float>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_get_rows_bf16")]] kernel get_rows_f_t kernel_get_rows_f<bfloat, float>;
#endif

typedef decltype(kernel_get_rows_q<block_q8_0, 2, dequantize_q8_0>) get_rows_q_t;

template [[host_name("kernel_get_rows_q8_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q8_0,    2, dequantize_q8_0>;


template<typename T>
kernel void kernel_swiglu(
        constant ggml_metal_kargs_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const T * src0_row = (device const T *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const T * src1_row = (device const T *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       T * dst_row  = (device       T *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float silu = x0 / (1.0f + exp(-x0));

        dst_row[i0] = (T)(silu*x1);
    }
}

typedef decltype(kernel_swiglu<float>) kernel_swiglu_t;

template [[host_name("kernel_swiglu_f32")]] kernel kernel_swiglu_t kernel_swiglu<float>;
