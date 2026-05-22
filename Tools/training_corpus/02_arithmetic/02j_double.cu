// Targets: DADD, DMUL, DFMA, DSETP, DMNMX and F2F.F64<->F32 conversions —
// the entire fp64 datapath, which no other corpus file exercises. Common in
// reference/accumulator kernels and mixed-precision normalisation.

extern "C" __global__ void dadd_dmul(const double* a, const double* b,
                                     double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i] + a[i];   // DMUL + DADD (or DFMA at -O3)
}

extern "C" __global__ void dfma(const double* a, const double* b,
                                const double* c, double* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = fma(a[i], b[i], c[i]);   // DFMA
}

extern "C" __global__ void dsetp_dmnmx(const double* a, const double* b,
                                       double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = a[i], y = b[i];
        c[i] = (x > y) ? x : y;            // DSETP + selection / DMNMX
    }
}

extern "C" __global__ void d2f_f2d(const double* a, float* b,
                                   const float* c, double* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        b[i] = (float)a[i];               // F2F.F32.F64 (narrow)
        d[i] = (double)c[i];              // F2F.F64.F32 (widen)
    }
}

extern "C" __global__ void daxpy(const double* x, double* y, double alpha,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i] + y[i];   // DFMA with a uniform scalar arg
}
