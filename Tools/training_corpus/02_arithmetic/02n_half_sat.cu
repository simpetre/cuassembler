// Saturation samples for the packed-half / bf16 ALU (HADD2/HMUL2/HFMA2/
// HMNMX2/F2FP) and the half<->float conversions. 02k_half_scalar.cu gave only
// 1-4 samples per shape; these high-register-pressure, multi-arg-layout kernels
// spread the operands so the learner's basis saturates.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// 8 independent half fma chains -> wide HFMA2 dst/src register spread.
extern "C" __global__ void hfma_8acc(const __half* a, const __half* b,
                                     __half* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half z = __float2half(0.f);
    __half s0=z,s1=z,s2=z,s3=z,s4=z,s5=z,s6=z,s7=z;
    for (int j = 0; j < n; ++j) {
        __half bb = b[j];
        s0=__hfma(a[i*8+0],bb,s0); s1=__hfma(a[i*8+1],bb,s1);
        s2=__hfma(a[i*8+2],bb,s2); s3=__hfma(a[i*8+3],bb,s3);
        s4=__hfma(a[i*8+4],bb,s4); s5=__hfma(a[i*8+5],bb,s5);
        s6=__hfma(a[i*8+6],bb,s6); s7=__hfma(a[i*8+7],bb,s7);
    }
    c[i*8+0]=s0;c[i*8+1]=s1;c[i*8+2]=s2;c[i*8+3]=s3;
    c[i*8+4]=s4;c[i*8+5]=s5;c[i*8+6]=s6;c[i*8+7]=s7;
}

// 12 independent add/mul lanes, extra leading arg to shift registers.
extern "C" __global__ void hadd_hmul_12(int pad, const __half* a,
                                        const __half* b, __half* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n + pad) return;
    __half r[12];
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        __half x = a[i*12+k], y = b[i*12+k];
        r[k] = __hadd(__hmul(x, y), x);     // HMUL2 + HADD2
    }
    #pragma unroll
    for (int k = 0; k < 12; ++k) c[i*12+k] = r[k];
}

// Many independent half min/max -> HMNMX2 operand/predicate variation.
extern "C" __global__ void hmnmx_many(const __half* a, const __half* b,
                                      __half* lo, __half* hi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        __half x = a[i*8+k], y = b[i*8+k];
        lo[i*8+k] = __hmin(x, y);           // HMNMX2 (min)
        hi[i*8+k] = __hmax(x, y);           // HMNMX2 (max)
    }
}

// Bulk float<->half conversions across many registers -> F2FP / F2F samples.
extern "C" __global__ void f2h_bulk(const float* a, __half* b,
                                    const __half* c, float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        b[i*8+k] = __float2half(a[i*8+k]);  // F2FP.F16.F32
        d[i*8+k] = __half2float(c[i*8+k]);  // F2F.F32.F16
    }
}

// bf16 fma chains for HFMA2 sample diversity on the bf16 path.
extern "C" __global__ void bf16_8acc(const __nv_bfloat16* a,
                                     const __nv_bfloat16* b,
                                     __nv_bfloat16* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __nv_bfloat16 z = __float2bfloat16(0.f);
    __nv_bfloat16 s0=z,s1=z,s2=z,s3=z;
    for (int j = 0; j < n; ++j) {
        __nv_bfloat16 bb = b[j];
        s0=__hfma(a[i*4+0],bb,s0); s1=__hfma(a[i*4+1],bb,s1);
        s2=__hfma(a[i*4+2],bb,s2); s3=__hfma(a[i*4+3],bb,s3);
    }
    c[i*4+0]=s0;c[i*4+1]=s1;c[i*4+2]=s2;c[i*4+3]=s3;
}
