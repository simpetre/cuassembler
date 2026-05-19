// Targets: SHFL_P_R_R_II_II (warp shuffle), used by warp reductions in
// softmax and attention.

extern "C" __global__ void warp_sum(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? a[i] : 0.f;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if ((threadIdx.x & 31) == 0) b[i >> 5] = v;
}

extern "C" __global__ void warp_max(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? a[i] : -1e30f;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float w = __shfl_xor_sync(0xffffffff, v, o);
        v = fmaxf(v, w);
    }
    if ((threadIdx.x & 31) == 0) b[i >> 5] = v;
}
