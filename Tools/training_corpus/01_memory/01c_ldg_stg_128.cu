// Targets: LDG_R_ARI/.E.128, STG_ARI_R/.E.128 (vectorized loads/stores).

extern "C" __global__ void copy_f4(const float4* a, float4* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}

extern "C" __global__ void copy_i4(const int4* a, int4* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}
