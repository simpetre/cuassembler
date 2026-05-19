// Targets: UISETP.{GE,GT,LT,NE,EQ}.U32.{AND,OR} and ISETP unsigned forms.
// Bounds checks against unsigned counters and uniform tile bounds — the
// dominant comparison shape in linear / softmax outer loops.

extern "C" __global__ void uisetp_ge_u32(unsigned* out, unsigned a, unsigned b,
                                         unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    // Uniform unsigned GE -> UISETP.GE.U32.AND.
    if (a >= b && i < n) out[i] = a - b;
}

extern "C" __global__ void uisetp_gt_u32(unsigned* out, unsigned a, unsigned b,
                                         unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (a > b && i < n) out[i] = a;
}

extern "C" __global__ void uisetp_ne_or(int* out, unsigned a, unsigned b,
                                        unsigned c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // OR-combined predicate -> UISETP.NE.OR.
    if (a != 0 || b != c) {
        if (i < n) out[i] = 1;
    }
}

extern "C" __global__ void isetp_unsigned_bounds(const float* a, float* b,
                                                 unsigned n) {
    // Per-thread unsigned bound check, common in tile-loop epilogues.
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}
