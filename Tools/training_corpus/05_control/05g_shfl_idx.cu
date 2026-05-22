// Targets: SHFL.IDX (warp broadcast) in both immediate-lane and register-lane
// forms, with enough distinct register allocations to give CuAssembler a
// sufficient regression basis. The repo previously had a single SHFL.IDX sample
// per shape, so an agent using other registers (or an immediate source lane,
// e.g. broadcasting a reduction result from lane 0) hit "Insufficient basis".

// (1) Immediate source lanes -> SHFL.IDX with an *immediate* lane operand.
//     Varied lanes + an accumulate chain force distinct dst/src registers.
extern "C" __global__ void bcast_const_lanes(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? a[i] : 0.f;
    float s = 0.f;
    s += __shfl_sync(0xffffffff, v, 0);
    s += __shfl_sync(0xffffffff, v, 1);
    s += __shfl_sync(0xffffffff, v, 3);
    s += __shfl_sync(0xffffffff, v, 7);
    s += __shfl_sync(0xffffffff, v, 15);
    s += __shfl_sync(0xffffffff, v, 31);
    if (i < n) b[i] = s;
}

// (2) Many independent values broadcast from lane 0 -> many distinct register
//     allocations for the same immediate-lane shape.
extern "C" __global__ void bcast_many_vals(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v0 = a[i], v1 = a[i + n], v2 = a[i + 2 * n], v3 = a[i + 3 * n];
    float r0 = __shfl_sync(0xffffffff, v0, 0);
    float r1 = __shfl_sync(0xffffffff, v1, 0);
    float r2 = __shfl_sync(0xffffffff, v2, 0);
    float r3 = __shfl_sync(0xffffffff, v3, 0);
    if (i < n) b[i] = r0 + r1 + r2 + r3;
}

// (3) Register source lanes -> SHFL.IDX with a *register* lane operand, varied.
extern "C" __global__ void bcast_reg_lane(const float* a, const int* lanes,
                                          float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? a[i] : 0.f;
    int l0 = lanes[i & 31];
    int l1 = lanes[(i + 5) & 31];
    int l2 = lanes[(i + 11) & 31];
    float s = __shfl_sync(0xffffffff, v, l0)
            + __shfl_sync(0xffffffff, v, l1)
            + __shfl_sync(0xffffffff, v, l2);
    if (i < n) b[i] = s;
}

// (4) Integer broadcasts too (SHFL.IDX is type-agnostic; covers int reductions
//     that broadcast an index/count from a representative lane).
extern "C" __global__ void bcast_int(const int* a, int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = (i < n) ? a[i] : 0;
    int r = __shfl_sync(0xffffffff, v, 0) + __shfl_sync(0xffffffff, v, 16);
    if (i < n) b[i] = r;
}
