// Targets: integer arithmetic across widths — 8/16/64-bit. Exercises plain
// IADD (IADD_R_R_R / _R_R_II / _R_R_UR), IMUL/IMAD on 64-bit (IMAD.WIDE,
// IMAD.HI), and I2I sign/zero-extend conversions between widths. The narrow
// types stress the byte/short load-extend paths (LDG.S8/.U8/.S16/.U16).

extern "C" __global__ void i64_add_mul(const long long* a, const long long* b,
                                       long long* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i] + a[i];   // 64-bit IMAD.WIDE/.HI + IADD3.X
}

extern "C" __global__ void i64_scalar_add(const long long* a, long long s,
                                          long long* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + s;             // add a uniform 64-bit scalar arg
}

extern "C" __global__ void i8_add(const signed char* a, const signed char* b,
                                  signed char* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = (signed char)(a[i] + b[i]);   // LDG.S8 + IADD + I2I.S8
}

extern "C" __global__ void u8_add(const unsigned char* a, const unsigned char* b,
                                  unsigned char* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = (unsigned char)(a[i] + b[i]);  // LDG.U8 + IADD
}

extern "C" __global__ void i16_madd(const short* a, const short* b,
                                    short* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = (short)(a[i] * b[i] + a[i]);   // LDG.S16 + IMAD
}

extern "C" __global__ void widen_narrow(const signed char* a, int* b,
                                        const int* c, signed char* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        b[i] = (int)a[i];                 // I2I.S32.S8 (widen)
        d[i] = (signed char)c[i];         // I2I.S8.S32 (narrow)
    }
}

extern "C" __global__ void i64_index(const float* base, const long long* idx,
                                     float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = base[idx[i]];     // 64-bit gather index -> IMAD.WIDE
}
