// Targets: MEMBAR (thread/block/system fences).

extern "C" __global__ void publish(float* a, int* flag) {
    a[threadIdx.x] = (float)threadIdx.x;
    __threadfence();      // MEMBAR.GL or similar
    if (threadIdx.x == 0) atomicExch(flag, 1);
}

extern "C" __global__ void block_fence(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = (float)i;
        __threadfence_block();   // MEMBAR.CTA
    }
}
