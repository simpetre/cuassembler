// Targets: bitwise / shift ops on warp-uniform values -> the uniform datapath
// ULOP3 (ULOP3_UR_..._UP), uniform shifts (USHF), UFLO, UPOPC, plus more
// LOP3/SHF samples. Triggered when the operands are scalar kernel args or
// values the compiler proves warp-invariant.

extern "C" __global__ void uniform_lop3(unsigned* out, unsigned a, unsigned b,
                                        unsigned c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // a,b,c are uniform scalar args -> combined bit op in the uniform datapath.
    unsigned m = (a & b) | (~c);          // ULOP3 (3-input truth table)
    if (i < n) out[i] = m;
}

extern "C" __global__ void uniform_shift(unsigned* out, unsigned base,
                                         unsigned sh, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned v = (base << (sh & 31)) ^ (base >> ((32 - sh) & 31));  // USHF / SHF
    if (i < n) out[i] = v;
}

extern "C" __global__ void uniform_mask(unsigned* out, unsigned a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // popcount + find-leading-one on a uniform value.
    unsigned v = __popc(a) + __clz(a);    // UPOPC / UFLO (or POPC/FLO)
    if (i < n) out[i] = v;
}

extern "C" __global__ void per_thread_lop3(const unsigned* a, const unsigned* b,
                                           const unsigned* c, unsigned* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (a[i] & b[i]) ^ c[i];   // LOP3 per-thread (more samples)
}

extern "C" __global__ void bit_reverse_mask(const unsigned* a, unsigned* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = __brev(a[i]) & 0xFF00FF00u;   // BREV + LOP3 with imm
}
