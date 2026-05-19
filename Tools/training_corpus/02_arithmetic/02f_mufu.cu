// Targets: MUFU variants — RCP (reciprocal), RSQ (rsqrt), EX2, LG2, SIN,
// COS, TANH. The exp/log family is critical for softmax; rsqrt for
// normalization (layer norm, RMSNorm, attention scale).

extern "C" __global__ void rcp(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = 1.0f / a[i];          // MUFU.RCP
}

extern "C" __global__ void rsqrt_kernel(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = rsqrtf(a[i]);          // MUFU.RSQ
}

extern "C" __global__ void expf_kernel(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = expf(a[i]);            // MUFU.EX2 (with multiplier)
}

extern "C" __global__ void logf_kernel(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = logf(a[i]);            // MUFU.LG2
}

extern "C" __global__ void tanh_kernel(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = tanhf(a[i]);           // MUFU.TANH (sm_75+)
}
