// Targets: LDG_R_ARI/.E.64, STG_ARI_R/.E.64 (64-bit loads/stores via double
// and long long).

extern "C" __global__ void copy_f64(const double* a, double* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}

extern "C" __global__ void copy_i64(const long long* a, long long* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}
