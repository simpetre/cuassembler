// Targets: S2R/S2UR for SR_TID.{X,Y,Z}, SR_CTAID.{X,Y,Z}, SR_NTID.{X,Y,Z},
// SR_NCTAID.{X,Y,Z}. Each is a distinct modifier on the S2R/S2UR shape.

extern "C" __global__ void all_ids_3d(int* out, int n) {
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int dx = blockDim.x, dy = blockDim.y, dz = blockDim.z;
    int gx = gridDim.x, gy = gridDim.y, gz = gridDim.z;
    int i = (bz * gy + by) * gx + bx;
    int j = (tz * dy + ty) * dx + tx;
    int linear = i * (dx * dy * dz) + j;
    if (linear < n) out[linear] = tx + ty + tz + bx + by + bz + dx + dy + dz + gx + gy + gz;
}

extern "C" __global__ void flat_id(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}
