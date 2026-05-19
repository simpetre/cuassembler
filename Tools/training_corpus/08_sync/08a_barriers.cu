// Targets: BAR (block-wide), WARPSYNC.

extern "C" __global__ void block_sum(const float* a, float* b, int n) {
    __shared__ float s[256];
    int t = threadIdx.x;
    s[t] = (t < n) ? a[t] : 0.f;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (t < o) s[t] += s[t + o];
        __syncthreads();
    }
    if (t == 0) b[blockIdx.x] = s[0];
}

extern "C" __global__ void warp_sync_demo(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] *= 2.f;
        __syncwarp();
    }
}
