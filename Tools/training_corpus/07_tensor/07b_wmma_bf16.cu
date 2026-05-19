// Targets: HMMA_R_R_R_R variants with bf16 inputs (used in modern
// transformers / FlashAttention).

#include <mma.h>
#include <cuda_bf16.h>

using namespace nvcuda::wmma;

extern "C" __global__ void wmma_bf16_16x16x16(const __nv_bfloat16* a,
                                              const __nv_bfloat16* b,
                                              float* c) {
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> fa;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> fb;
    fragment<accumulator, 16, 16, 16, float> fc;
    fill_fragment(fc, 0.0f);
    load_matrix_sync(fa, a, 16);
    load_matrix_sync(fb, b, 16);
    mma_sync(fc, fa, fb, fc);
    store_matrix_sync(c, fc, 16, mem_row_major);
}
