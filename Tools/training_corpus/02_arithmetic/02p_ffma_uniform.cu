// Targets: FFMA_R_R_UR_R -- fused multiply-add whose MULTIPLICAND is a
// warp-uniform float held in a uniform register (ULDC of a scalar arg). The
// repo had a single sample (FFMA R7, R2, UR6, R9), leaving a 5-dim nullspace,
// so any other register/uniform allocation hit "Insufficient basis" (e.g. the
// silu sigmoid idiom FFMA Rd, Ra, URb, RZ). These kernels emit many FFMA-UR
// instances with varied Rd / Ra / URb / Rc to make the basis full rank.

// (1) Wide fan-out: 8 independent (per-thread * distinct-uniform-scalar + add)
//     expressions -> distinct URb (one ULDC per scalar) and distinct Rd/Ra/Rc.
extern "C" __global__ void ffma_ur_fan8(
    const float* a0, const float* a1, const float* a2, const float* a3,
    const float* a4, const float* a5, const float* a6, const float* a7,
    float* o0, float* o1, float* o2, float* o3,
    float* o4, float* o5, float* o6, float* o7,
    int n,
    float s0, float s1, float s2, float s3,
    float s4, float s5, float s6, float s7,
    float t0, float t1, float t2, float t3,
    float t4, float t5, float t6, float t7) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        o0[i] = a0[i] * s0 + t0;
        o1[i] = a1[i] * s1 + t1;
        o2[i] = a2[i] * s2 + t2;
        o3[i] = a3[i] * s3 + t3;
        o4[i] = a4[i] * s4 + t4;
        o5[i] = a5[i] * s5 + t5;
        o6[i] = a6[i] * s6 + t6;
        o7[i] = a7[i] * s7 + t7;
    }
}

// (2) Uniform-scaled chain: reuse one per-thread value with several uniform
//     scalars -> FFMA-UR with the addend being a prior per-thread result (Rc
//     varies as the chain advances).
extern "C" __global__ void ffma_ur_chain(
    const float* a, float* o, int n,
    float s0, float s1, float s2, float s3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        float y = x * s0 + x;
        float z = x * s1 + y;
        float w = y * s2 + z;
        o[i] = z * s3 + w;
    }
}

// (3) Two-input blend with uniform weights -> FFMA-UR feeding distinct regs.
extern "C" __global__ void ffma_ur_blend(
    const float* a, const float* b, float* o, int n,
    float wa, float wb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] * wa + b[i] * wb;
}
