// Saturation for the half shapes nvcc is reluctant to emit: standalone
// HADD2 / HMUL2 (it fuses add+mul into HFMA2 whenever it can) and the
// HMNMX2 / F2FP / F2F conversion forms. These kernels keep the ops
// un-fusable (a lone add, or a lone mul, then store) and hold many lanes
// live at once so nvcc allocates a wide spread of registers.
#include <cuda_fp16.h>

// Pure standalone half adds — no multiply in sight, so they stay HADD2.
extern "C" __global__ void half_add_16(const __half* a, const __half* b,
                                       __half* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half t[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) t[k] = __hadd(a[i*16+k], b[i*16+k]);  // HADD2
    #pragma unroll
    for (int k = 0; k < 16; ++k) c[i*16+k] = t[k];
}

// Pure standalone half multiplies — no add, so they stay HMUL2.
extern "C" __global__ void half_mul_16(int pad, const __half* a,
                                       const __half* b, __half* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n + pad) return;
    __half t[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) t[k] = __hmul(a[i*16+k], b[i*16+k]);  // HMUL2
    #pragma unroll
    for (int k = 0; k < 16; ++k) c[i*16+k] = t[k];
}

// Many independent half min/max -> HMNMX2 register spread.
extern "C" __global__ void half_minmax_16(const __half* a, const __half* b,
                                          __half* lo, __half* hi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half l[16], h[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) { l[k] = __hmin(a[i*16+k], b[i*16+k]);
                                   h[k] = __hmax(a[i*16+k], b[i*16+k]); }
    #pragma unroll
    for (int k = 0; k < 16; ++k) { lo[i*16+k] = l[k]; hi[i*16+k] = h[k]; }
}

// Bulk float->half across many live registers -> F2FP.F16.F32 spread.
extern "C" __global__ void conv_f2h_16(const float* a, __half* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half t[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) t[k] = __float2half(a[i*16+k]);   // F2FP
    #pragma unroll
    for (int k = 0; k < 16; ++k) b[i*16+k] = t[k];
}

// Bulk half->float, different arg layout -> F2F.F32.F16 spread.
extern "C" __global__ void conv_h2f_16(int pad, const __half* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n + pad) return;
    float t[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) t[k] = __half2float(a[i*16+k]);   // F2F
    #pragma unroll
    for (int k = 0; k < 16; ++k) b[i*16+k] = t[k];
}

// Half relu via max-with-zero, many lanes -> more HMNMX2 with an immediate-ish 0.
extern "C" __global__ void half_relu_16(const __half* a, __half* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half z = __float2half(0.f);
    __half t[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) t[k] = __hmax(a[i*16+k], z);
    #pragma unroll
    for (int k = 0; k < 16; ++k) b[i*16+k] = t[k];
}
