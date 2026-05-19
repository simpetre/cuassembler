// Targets: BRA (with label backedge), plus IADD3/IMAD induction variables
// that loop-using kernels generate.

extern "C" __global__ void sum_range(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = 0.f;
        for (int k = 0; k < 16; ++k) s += a[i * 16 + k];   // small loop -> BRA
        b[i] = s;
    }
}

extern "C" __global__ void prefix_within(const float* a, float* b, int stride, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = 0.f;
        for (int k = 0; k < stride; ++k) s += a[i * stride + k];   // variable trip count
        b[i] = s;
    }
}
