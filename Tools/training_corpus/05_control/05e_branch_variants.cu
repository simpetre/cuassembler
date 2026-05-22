// Targets: branch-shape variety the splice scheduler must round-trip — plain
// label branches (BRA_L), uniform-predicate branches (BRA_UP / BRA_UP_II) from
// warp-uniform conditions, BREAK/CONT-style nested loops, and switch dispatch
// (a jump ladder of BRAs). Complements 05a_loop / 05b_branches with the forms
// that depend on blockIdx-uniform control flow.

extern "C" __global__ void uniform_branch(const float* a, float* b,
                                          int mode, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // `mode` is a uniform scalar arg -> the branch on it is warp-uniform
    // (BRA_UP / uniform-predicate), not per-lane.
    float v = a[i];
    if (mode == 0)      b[i] = v + 1.0f;
    else if (mode == 1) b[i] = v * 2.0f;
    else                b[i] = -v;
}

extern "C" __global__ void switch_dispatch(const float* a, float* b,
                                           int op, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[i], r;
    switch (op) {                 // jump ladder -> several BRA / BRA_L
        case 0:  r = v + v;   break;
        case 1:  r = v * v;   break;
        case 2:  r = v - 1.f; break;
        case 3:  r = 1.f / v; break;
        default: r = v;       break;
    }
    b[i] = r;
}

extern "C" __global__ void nested_break(const float* a, float* b,
                                        int rows, int cols, float thresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float acc = 0.f;
    for (int j = 0; j < cols; ++j) {        // outer counted loop (backedge BRA)
        float v = a[i * cols + j];
        if (v > thresh) break;              // early-exit BRA out of the loop
        acc += v;
    }
    b[i] = acc;
}

extern "C" __global__ void continue_loop(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int j = 0; j < 32; ++j) {
        float v = a[i * 32 + j];
        if (v < 0.f) continue;              // skip-iteration -> predicated/BRA
        acc += v;
    }
    b[i] = acc;
}

extern "C" __global__ void grid_stride(const float* a, float* b, int n) {
    // Classic grid-stride loop: bound compare against a uniform size, induction
    // var advanced by gridDim*blockDim -> backedge BRA with uniform compare.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridDim.x * blockDim.x) {
        b[i] = a[i] * a[i];
    }
}
