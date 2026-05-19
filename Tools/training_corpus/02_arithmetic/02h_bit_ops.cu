// Targets: LOP3 (logical-op-3-input — combined AND/OR/XOR), POPC, FLO
// (find leading one), BREV, BMSK.

extern "C" __global__ void lop3(const int* a, const int* b, const int* c,
                                int* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (a[i] & b[i]) | c[i];   // LOP3
}

extern "C" __global__ void andor(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = (a[i] & b[i]) | (a[i] ^ b[i]);   // LOP3
}

extern "C" __global__ void popcount(const unsigned* a, int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = __popc(a[i]);     // POPC
}

extern "C" __global__ void clz_kernel(const unsigned* a, int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = __clz(a[i]);      // FLO.U32
}
