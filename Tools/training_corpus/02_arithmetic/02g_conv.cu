// Targets: I2F, F2I, F2F (precision change), I2I (sign/zero extend).

extern "C" __global__ void i2f(const int* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = (float)a[i];   // I2F
}

extern "C" __global__ void f2i(const float* a, int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = (int)a[i];     // F2I
}

extern "C" __global__ void f64f32(const double* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = (float)a[i];   // F2F.F32.F64
}

extern "C" __global__ void f32f64(const float* a, double* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = (double)a[i];  // F2F.F64.F32
}
