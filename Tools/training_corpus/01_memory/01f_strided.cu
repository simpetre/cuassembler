// Targets: more LDG_R_dARI / STG_ARI_R operand variation via different
// stride patterns. Each kernel has different access patterns so register
// allocation / address computation differs, expanding the basis.

extern "C" __global__ void stride2(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * i + 1 < n) b[i] = a[2 * i] + a[2 * i + 1];
}

extern "C" __global__ void stride_var(const float* a, float* b, int s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * s < n) b[i] = a[i * s];
}

extern "C" __global__ void gather(const float* a, const int* idx, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[idx[i]];
}

extern "C" __global__ void scatter(const float* a, const int* idx, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[idx[i]] = a[i];
}
