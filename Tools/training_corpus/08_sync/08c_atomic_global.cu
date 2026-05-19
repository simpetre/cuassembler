// Targets: ATOMG / ATOM variants (add, min, max, CAS, exch). Common in
// reductions, histograms.

extern "C" __global__ void atomic_add(const float* a, float* sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(sum, a[i]);   // ATOMG.E.ADD.F32
}

extern "C" __global__ void atomic_max_int(const int* a, int* maxv, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicMax(maxv, a[i]);  // ATOMG.E.MAX
}

extern "C" __global__ void atomic_exch(int* a, int v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicExch(&a[i], v);   // ATOMG.E.EXCH
}

extern "C" __global__ void atomic_cas(int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicCAS(&a[i], 0, 1); // ATOMG.E.CAS
}
