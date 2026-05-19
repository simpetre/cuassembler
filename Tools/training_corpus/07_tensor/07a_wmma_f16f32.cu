// Targets: HMMA_R_R_R_R (.16816.F32) and the surrounding LDSM/STSM
// fragment loads/stores. Uses the wmma C++ API, which is the only
// nvcc-supported way to emit HMMA without inline PTX.
//
// Compile note: needs `-std=c++14` (or later) which nvcc enables by default.

#include <mma.h>

using namespace nvcuda::wmma;

extern "C" __global__ void wmma_f16f32_16x16x16(const __half* a, const __half* b,
                                                float* c) {
    fragment<matrix_a, 16, 16, 16, __half, row_major> fa;
    fragment<matrix_b, 16, 16, 16, __half, col_major> fb;
    fragment<accumulator, 16, 16, 16, float> fc;
    fill_fragment(fc, 0.0f);
    load_matrix_sync(fa, a, 16);
    load_matrix_sync(fb, b, 16);
    mma_sync(fc, fa, fb, fc);
    store_matrix_sync(c, fc, 16, mem_row_major);
}
