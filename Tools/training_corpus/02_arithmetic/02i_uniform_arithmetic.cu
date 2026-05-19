// Targets: UIMAD, UIADD3, UIADD3.X, ULOP3 — arithmetic on warp-uniform
// values lives in uniform registers. Triggered by scalar kernel args
// combined together, or by index math that the compiler proves uniform.

extern "C" __global__ void uniform_imad(int* out, int a, int b, int c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // a*b+c is fully uniform: UIMAD into a uniform reg, then broadcast.
    int k = a * b + c;
    if (i < n) out[i] = k;
}

extern "C" __global__ void uniform_iadd3(int* out, int a, int b, int c,
                                         int d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Three-operand add chain becomes UIADD3 on uniform inputs.
    int k = a + b + c + d;
    if (i < n) out[i] = k;
}

extern "C" __global__ void uniform_lop3(unsigned* out, unsigned a, unsigned b,
                                        unsigned c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Three-way bitwise op compresses to ULOP3.LUT on uniform values.
    unsigned k = (a & b) ^ c;
    if (i < n) out[i] = k;
}

extern "C" __global__ void uniform_wide_addr(const float* base, float* out,
                                             long long off_a, long long off_b,
                                             int n) {
    // 64-bit uniform address arithmetic -> UIADD3 + UIADD3.X carry chain.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = base[off_a + off_b + i];
}
