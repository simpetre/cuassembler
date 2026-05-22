// Targets: integer min/max. On Blackwell nvcc lowers min()/max() to the
// vector VIMNMX form (VIMNMX_R_P_P_R_{R,UR}_P), not plain IMNMX — so these
// kernels add VIMNMX register/uniform/immediate operand variation plus the
// SEL / ISETP.GT-vs-uniform pattern of a clamp epilogue.

extern "C" __global__ void imnmx(const int* a, const int* b, int* lo, int* hi,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int x = a[i], y = b[i];
        lo[i] = min(x, y);            // IMNMX (min)
        hi[i] = max(x, y);            // IMNMX (max)
    }
}

extern "C" __global__ void clamp_uniform(const int* a, int* b, int lo, int hi,
                                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int v = a[i];
        // clamp against two uniform scalar bounds -> IMNMX with a uniform src.
        b[i] = max(lo, min(v, hi));
    }
}

extern "C" __global__ void greater_uniform(const int* a, int* b, int thr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = (a[i] > thr) ? a[i] : thr;   // ISETP.GT vs uniform + SEL
}
