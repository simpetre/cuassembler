// Targets: the packed-half datapath — HADD2, HMUL2, HFMA2, HMNMX2 and the
// half<->float conversions (F2FP / F2F.F16). bf16 variants too. Distinct from
// 07_tensor (which exercises HMMA tensor-core ops); this is the elementwise
// half/bf16 ALU used by activation and norm kernels.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void hadd_hmul(const __half* a, const __half* b,
                                     __half* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = __hadd(__hmul(a[i], b[i]), a[i]);   // HMUL2 + HADD2
}

extern "C" __global__ void hfma(const __half* a, const __half* b,
                                const __half* c, __half* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = __hfma(a[i], b[i], c[i]);            // HFMA2
}

extern "C" __global__ void hrelu(const __half* a, __half* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = __hmax(a[i], __float2half(0.f));     // HMNMX2 (half relu)
}

extern "C" __global__ void half_to_float(const __half* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = __half2float(a[i]);                  // F2F.F32.F16
}

extern "C" __global__ void float_to_half(const float* a, __half* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = __float2half(a[i]);                  // F2FP.F16.F32
}

extern "C" __global__ void bf16_fma(const __nv_bfloat16* a,
                                    const __nv_bfloat16* b,
                                    const __nv_bfloat16* c,
                                    __nv_bfloat16* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = __hfma(a[i], b[i], c[i]);            // HFMA2 (bf16)
}
