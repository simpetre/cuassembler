// Targets: FSETP_P_P_R_R_P (float compare). Used by softmax (subtract max,
// compare), and any predicated float pipeline.

extern "C" __global__ void fsetp_gt(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = (a[i] > b[i]) ? a[i] : b[i];   // FSETP.GT + FSEL
}

extern "C" __global__ void fsetp_threshold(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = (a[i] > 0.5f) ? a[i] : 0.f;
}

extern "C" __global__ void fsetp_within(const float* a, int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a[i];
        b[i] = (v > -1.f && v < 1.f) ? 1 : 0;   // chained FSETP
    }
}
