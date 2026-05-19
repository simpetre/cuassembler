// Targets: PLOP3 (predicate-LOP3, combines up to 3 predicates with a
// truth table), PSETP (predicate-from-predicate set).

extern "C" __global__ void plop3_and(const int* a, const int* b, const int* c,
                                     int* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Three independent predicates AND'd, then conditional store
        bool p = (a[i] > 0) && (b[i] > 0) && (c[i] > 0);    // PLOP3
        d[i] = p ? 1 : 0;
    }
}

extern "C" __global__ void plop3_xor(const int* a, const int* b, int* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        bool p1 = (a[i] & 1) != 0;
        bool p2 = (b[i] & 1) != 0;
        d[i] = (p1 ^ p2) ? 1 : 0;
    }
}
