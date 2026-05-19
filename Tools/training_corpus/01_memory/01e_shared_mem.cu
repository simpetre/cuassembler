// Targets: LDS_R_ARI, STS_ARI_R, BAR_II (via __syncthreads).

extern "C" __global__ void shared_shift(const float* in, float* out, int n) {
    __shared__ float s[256];
    int t = threadIdx.x;
    if (t < 256) s[t] = in[t];
    __syncthreads();
    if (t < 256) out[t] = s[(t + 1) & 255];
}

extern "C" __global__ void shared_reduce(const float* in, float* out) {
    __shared__ float s[128];
    int t = threadIdx.x;
    s[t] = in[t];
    __syncthreads();
    for (int o = 64; o > 0; o >>= 1) {
        if (t < o) s[t] += s[t + o];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s[0];
}
