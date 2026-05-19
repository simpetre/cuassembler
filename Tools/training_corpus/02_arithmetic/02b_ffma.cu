// Targets: FFMA_R_R_R_R (and FFMA with immediate). The fused multiply-add
// is the workhorse of float math; gemm/relu/softmax all rely on it.

extern "C" __global__ void ffma_rrr(const float* a, const float* b, const float* c,
                                    float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = a[i] * b[i] + c[i];
}

extern "C" __global__ void ffma_imm(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] * 2.5f + 1.0f;
}

extern "C" __global__ void ffma_chain(const float* a, const float* b,
                                      const float* c, float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i], y = b[i], z = c[i];
        d[i] = (x * y + z) * x + y;   // chained FFMA
    }
}
