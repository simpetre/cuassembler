// Targets: LDG_R_ARI, STG_ARI_R (with .E modifier), basic float pointer
// copy/add. Two kernels with different reg pressure to encourage varied
// destination-register encodings in the same shape.

extern "C" __global__ void copy_f32(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}

extern "C" __global__ void add3_f32(const float* a, const float* b, const float* c,
                                    float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = a[i] + b[i] + c[i];
}
