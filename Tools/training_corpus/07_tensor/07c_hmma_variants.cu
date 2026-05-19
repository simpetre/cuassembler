// Targets: more HMMA shape variants. Different fragment shapes/types give
// different HMMA encodings; we hit several so the basis covers them.

#include <mma.h>
using namespace nvcuda::wmma;

// 16x16x16 with col-major A
extern "C" __global__ void wmma_colmajor(const __half* a, const __half* b, float* c) {
    fragment<matrix_a, 16, 16, 16, __half, col_major> fa;
    fragment<matrix_b, 16, 16, 16, __half, row_major> fb;
    fragment<accumulator, 16, 16, 16, float> fc;
    fill_fragment(fc, 0.0f);
    load_matrix_sync(fa, a, 16);
    load_matrix_sync(fb, b, 16);
    mma_sync(fc, fa, fb, fc);
    store_matrix_sync(c, fc, 16, mem_col_major);
}

// 32x8x16 shape
extern "C" __global__ void wmma_32x8x16(const __half* a, const __half* b, float* c) {
    fragment<matrix_a, 32, 8, 16, __half, row_major> fa;
    fragment<matrix_b, 32, 8, 16, __half, col_major> fb;
    fragment<accumulator, 32, 8, 16, float> fc;
    fill_fragment(fc, 0.0f);
    load_matrix_sync(fa, a, 16);
    load_matrix_sync(fb, b, 16);
    mma_sync(fc, fa, fb, fc);
    store_matrix_sync(c, fc, 8, mem_row_major);
}

// 8x32x16 shape
extern "C" __global__ void wmma_8x32x16(const __half* a, const __half* b, float* c) {
    fragment<matrix_a, 8, 32, 16, __half, row_major> fa;
    fragment<matrix_b, 8, 32, 16, __half, col_major> fb;
    fragment<accumulator, 8, 32, 16, float> fc;
    fill_fragment(fc, 0.0f);
    load_matrix_sync(fa, a, 16);
    load_matrix_sync(fb, b, 16);
    mma_sync(fc, fa, fb, fc);
    store_matrix_sync(c, fc, 32, mem_row_major);
}
