// Targets: LDG_R_ARI samples spanning many destination registers. Each
// loaded value lives in a distinct register so the linear-regression learner
// sees variation across the dst-reg encoding bits, not just one R-value.

extern "C" __global__ void sum8(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + 8 < n) {
        float v0 = a[i + 0], v1 = a[i + 1], v2 = a[i + 2], v3 = a[i + 3];
        float v4 = a[i + 4], v5 = a[i + 5], v6 = a[i + 6], v7 = a[i + 7];
        out[i] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
    }
}

extern "C" __global__ void sum16(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + 16 < n) {
        float s = 0.f;
        // Unroll forces nvcc to schedule loads to multiple regs.
        #pragma unroll
        for (int k = 0; k < 16; ++k) s += a[i + k];
        out[i] = s;
    }
}
