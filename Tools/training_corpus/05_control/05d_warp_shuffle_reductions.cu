// Targets: SHFL (BFLY/DOWN/UP) variants with predicate operands, and the
// FADD/FMAX reduction trees softmax/attention emit. Existing 04b_shfl.cu
// has xor-shuffle FADD; this adds DOWN/UP, conditional reductions, and
// max-tree shapes used by softmax's max-subtraction.

extern "C" __global__ void warp_max_down(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? a[i] : -1e30f;
    // SHFL.DOWN reduction tree (softmax max-pass).
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float w = __shfl_down_sync(0xffffffff, v, o);
        v = fmaxf(v, w);
    }
    if ((threadIdx.x & 31) == 0 && i < n) b[i >> 5] = v;
}

extern "C" __global__ void warp_sum_down(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? a[i] : 0.f;
    // SHFL.DOWN sum tree (softmax denom).
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, o);
    }
    if ((threadIdx.x & 31) == 0 && i < n) b[i >> 5] = v;
}

extern "C" __global__ void warp_predicated_shfl(const float* a, float* b,
                                                int n, int active_mask) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? a[i] : 0.f;
    // Dynamic mask forces predicated SHFL with non-trivial membership reg.
    v += __shfl_xor_sync(active_mask, v, 1);
    v += __shfl_xor_sync(active_mask, v, 2);
    if (i < n) b[i] = v;
}

extern "C" __global__ void warp_broadcast(const float* a, float* b, int n,
                                          int src_lane) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? a[i] : 0.f;
    // SHFL.IDX with dynamic source lane.
    float bcast = __shfl_sync(0xffffffff, v, src_lane);
    if (i < n) b[i] = bcast;
}
