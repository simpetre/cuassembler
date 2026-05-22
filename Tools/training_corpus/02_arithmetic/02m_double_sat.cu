// Saturation samples for the fp64 datapath (DADD/DMUL/DFMA/DSETP). The single
// kernels in 02j_double.cu only gave 1-3 samples per shape, leaving the
// learner's basis under-determined (probe -> InsufficientBasis for arbitrary
// register allocations). These kernels use many independent accumulators /
// high register pressure across several arg layouts so nvcc spreads the
// operands across the full register-field bit range, making each shape's
// operand-value vectors linearly independent.

// 8 independent fma chains -> DFMA dst/src spread across ~16+ registers.
extern "C" __global__ void dfma_8acc(const double* a, const double* b,
                                     double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
    for (int j = 0; j < n; ++j) {
        double bb = b[j];
        s0 = fma(a[i*8+0]*bb, bb, s0);  s1 = fma(a[i*8+1]*bb, bb, s1);
        s2 = fma(a[i*8+2]*bb, bb, s2);  s3 = fma(a[i*8+3]*bb, bb, s3);
        s4 = fma(a[i*8+4]*bb, bb, s4);  s5 = fma(a[i*8+5]*bb, bb, s5);
        s6 = fma(a[i*8+6]*bb, bb, s6);  s7 = fma(a[i*8+7]*bb, bb, s7);
    }
    c[i*8+0]=s0; c[i*8+1]=s1; c[i*8+2]=s2; c[i*8+3]=s3;
    c[i*8+4]=s4; c[i*8+5]=s5; c[i*8+6]=s6; c[i*8+7]=s7;
}

// 12 independent add/mul pairs with an extra leading int arg to shift the
// base register allocation -> different DADD/DMUL operand encodings.
extern "C" __global__ void dadd_dmul_12(int pad, const double* a,
                                        const double* b, double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n + pad) return;
    double r[12];
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        double x = a[i*12+k], y = b[i*12+k];
        r[k] = x*y + (x + y);          // DMUL + DADD per lane
    }
    #pragma unroll
    for (int k = 0; k < 12; ++k) c[i*12+k] = r[k];
}

// Horner polynomial -> a long dependent DFMA chain with a uniform coefficient
// stream (DFMA_R_R_UR_R when coeffs are uniform scalar args).
extern "C" __global__ void dhorner(const double* x, double* y,
                                   double c0, double c1, double c2,
                                   double c3, double c4, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double t = x[i];
    double r = c0;
    r = fma(r, t, c1);  r = fma(r, t, c2);
    r = fma(r, t, c3);  r = fma(r, t, c4);   // DFMA with uniform addends
    y[i] = r;
}

// Many independent double comparisons -> DSETP operand/predicate variation.
extern "C" __global__ void dsetp_many(const double* a, const double* b,
                                      int* mask, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int m = 0;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        m |= (a[i*8+k] > b[i*8+k]) ? (1 << k) : 0;   // DSETP per lane
    }
    mask[i] = m;
}

// Different arg count again, to move the register window.
extern "C" __global__ void daxpy_pair(const double* x, const double* z,
                                      double* y, double alpha, double beta,
                                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fma(alpha, x[i], beta * z[i]);   // DFMA + DMUL, uniform
}
