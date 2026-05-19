// Targets: S2R variants for SR_LANEID, SR_WARPID. Often used inside
// warp-level reductions (softmax, attention).

extern "C" __global__ void lane_and_warp(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = i & 31;
    int warp = (threadIdx.x >> 5);
    if (i < n) out[i] = lane * 100 + warp;
}
