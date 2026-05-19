// Targets: ATOMS (shared-memory atomic).

extern "C" __global__ void shared_atomic_add(const int* in, int* out, int n) {
    __shared__ int sum;
    int t = threadIdx.x;
    if (t == 0) sum = 0;
    __syncthreads();
    if (t < n) atomicAdd(&sum, in[t]);   // ATOMS.ADD
    __syncthreads();
    if (t == 0) *out = sum;
}

extern "C" __global__ void shared_atomic_max(const int* in, int* out, int n) {
    __shared__ int maxv;
    int t = threadIdx.x;
    if (t == 0) maxv = -2147483647;
    __syncthreads();
    if (t < n) atomicMax(&maxv, in[t]);  // ATOMS.MAX
    __syncthreads();
    if (t == 0) *out = maxv;
}
