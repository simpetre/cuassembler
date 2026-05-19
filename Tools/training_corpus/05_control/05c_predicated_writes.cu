// Targets: SEL (integer select), and additional predicate-driven control
// patterns that get predicated rather than branched.

extern "C" __global__ void sel_int(const int* a, const int* b, const int* c,
                                   int* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (c[i] > 0) ? a[i] : b[i];   // SEL or branchless ISETP+SEL
}

extern "C" __global__ void masked_store(const float* a, const int* mask,
                                        float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i] != 0) b[i] = a[i];   // predicated STG
}

extern "C" __global__ void chain_select(const float* a, const float* b,
                                        const float* c, const float* d,
                                        float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = (a[i] > 0.f) ? b[i] : c[i];
        out[i] = (v < 1.f) ? v : d[i];
    }
}
