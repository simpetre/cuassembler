// Targets: ISETP_P_P_R_R_P (register vs register integer compare, all
// predicates registers), plus more samples for ISETP_P_P_R_UR_P.

extern "C" __global__ void isetp_lt(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = (a[i] < b[i]) ? a[i] : b[i];   // ISETP.LT.AND R, R
}

extern "C" __global__ void isetp_eq_mask(const int* a, int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = (a[i] == 42) ? 1 : 0;   // ISETP.EQ.AND with imm
}

extern "C" __global__ void isetp_chain(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int x = a[i], y = b[i];
        c[i] = (x > 0 && y > 0) ? (x + y) : 0;   // chained ISETP + predicate combiner
    }
}
