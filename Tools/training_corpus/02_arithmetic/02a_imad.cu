// Targets: IMAD_R_R_R_R (all-register integer multiply-add) plus the
// existing IMAD_R_R_II_R / IMAD_R_R_UR_R shapes with more samples.

extern "C" __global__ void imad_rrr(const int* a, const int* b, const int* c,
                                    int* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = a[i] * b[i] + c[i];   // IMAD R, R, R, R
}

extern "C" __global__ void imad_rrr_wide(const int* a, const int* b,
                                         long long* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (long long)a[i] * b[i];   // IMAD.WIDE variants
}

extern "C" __global__ void imad_imm(const int* a, int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] * 13 + 7;   // IMAD R, R, II, II
}
