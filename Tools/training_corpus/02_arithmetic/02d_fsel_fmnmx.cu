// Targets: FMNMX_R_R_R_P (float min/max — the relu pattern), FSEL_R_R_R_P
// (predicated select). Both used in softmax and relu agent demos.

extern "C" __global__ void relu(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = fmaxf(a[i], 0.0f);     // FMNMX
}

extern "C" __global__ void clamp01(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = fminf(fmaxf(a[i], 0.0f), 1.0f);   // FMNMX twice
}

extern "C" __global__ void fsel(const float* a, const float* b, const float* c,
                                float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (a[i] > 0.f) ? b[i] : c[i];   // FSEL via ISETP+FSEL or branchless
}
