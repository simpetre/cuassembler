// Targets: uniform 64-bit address carry propagation -> UIADD3 + UIADD3.X on
// uniform registers with a carry-in predicate (UP0) and URZ operands. This is
// also the regression test for the URZ-encoding fix: URZ in the uniform
// register field encodes as 255 (8-bit all-ones), not 63 — feeding these
// UIADD3.X samples used to make the linear learner mis-encode the shape when
// URZ shared a field position with real registers.

extern "C" __global__ void strided_accum(const float* a, float* b,
                                          int limit, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    // a[i + k*n] : the k*n stride is warp-uniform, so the 64-bit address
    // increment is computed in the uniform datapath (UIADD3 / UIADD3.X.UP).
    for (int k = 0; k < limit; ++k) acc += a[i + k * n];
    b[i] = acc;
}

extern "C" __global__ void strided_2d(const float* a, float* b,
                                       int rows, int cols, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float acc = 0.f;
    for (int j = 0; j < cols; ++j) acc += a[i * stride + j * rows];
    b[i] = acc;
}
