// Targets: BSSY_B_L / BSYNC_B (structured-divergence sync used by larger
// if/else branches the compiler doesn't predicate), plus more BRA samples.

extern "C" __global__ void big_branch(const float* a, float* b, int n, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[i];
    float r;
    if (mode == 0) {
        r = v * v;
        for (int k = 0; k < 4; ++k) r += v * 0.1f;   // big enough to defeat predication
    } else {
        r = v + 1.0f;
        for (int k = 0; k < 4; ++k) r *= 0.9f;
    }
    b[i] = r;
}

extern "C" __global__ void early_exit(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (a[i] < 0.f) return;        // predicated EXIT path
    if (i < n) b[i] = a[i] * 2.f;
}
