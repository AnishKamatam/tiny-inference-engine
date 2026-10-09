#include "common.h"

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
