// Targets: IADD3 (three-input integer add — used in address computation),
// IMUL, IMNMX/IMAX/IMIN (integer min/max). All extremely common.

extern "C" __global__ void iadd3(const int* a, const int* b, const int* c,
                                 int* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = a[i] + b[i] + c[i];   // IADD3
}

extern "C" __global__ void imul(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];          // IMUL or IMAD with RZ
}

extern "C" __global__ void iminmax(const int* a, const int* b, int* lo, int* hi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int x = a[i], y = b[i];
        lo[i] = min(x, y);   // IMNMX
        hi[i] = max(x, y);   // IMNMX
    }
}

extern "C" __global__ void iabs(const int* a, int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = (a[i] < 0) ? -a[i] : a[i];   // IABS or IMNMX
}
