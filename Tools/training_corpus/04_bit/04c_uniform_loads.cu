// Targets: ULDC, ULDC.64, UMOV — kernel scalar args and pointer args get
// loaded into uniform registers via ULDC/ULDC.64 because they are
// warp-invariant. UMOV appears when uniform values get shuffled between
// uniform regs (e.g. through __ldg of a uniform address).

extern "C" __global__ void uldc_scalar(const float* a, float* b,
                                       int n, float scale, float bias) {
    // n, scale, bias are warp-uniform scalars -> ULDC.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] * scale + bias;
}

extern "C" __global__ void uldc_64_ptrs(const float* a, const float* b,
                                        float* c, int n) {
    // 64-bit pointer args go through ULDC.64.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

extern "C" __global__ void uldc_many_scalars(int* out, int a, int b, int c,
                                             int d, int e, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Force many separate ULDC loads with distinct const-cache offsets.
    if (i < n) out[i] = a + b * 2 + c * 3 + d * 4 + e * 5;
}
