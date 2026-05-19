// Targets: SHF_R_R_II_R / SHF_R_R_R_R (shift funnel, used by addressing
// math and bit manipulation).

extern "C" __global__ void shl_int(const int* a, int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] << 4;
}

extern "C" __global__ void shr_int(const int* a, int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] >> 3;
}

extern "C" __global__ void shift_var(const int* a, const int* sh, int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] << (sh[i] & 31);
}
