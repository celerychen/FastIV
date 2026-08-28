/*
 * FastIV - Fast image and vision
 * Copyright (C) 2026 Celery Chen
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * See LICENSE file in project root for full license text.
 */

#include "fiv_matrix.h"
#include "fiv_linalg_kernels.h"   /* fiv_transpose_square_inplace_real32 declaration */

#ifndef FIV_ERROR_INFO
#define FIV_ERROR_INFO(msg) ((void)0)
#endif


/* ==================== Copy-style transpose of the last two dims (AVX) ==================== */

#if defined(FIV_USE_AVX)
#include <immintrin.h>

/* 8x8 float transpose inside registers */
static inline void fiv_transpose8x8_ps(__m256* r0, __m256* r1, __m256* r2, __m256* r3,
                                     __m256* r4, __m256* r5, __m256* r6, __m256* r7)
{
    __m256 t0, t1, t2, t3, t4, t5, t6, t7;

    t0 = _mm256_unpacklo_ps(*r0, *r1);
    t1 = _mm256_unpackhi_ps(*r0, *r1);
    t2 = _mm256_unpacklo_ps(*r2, *r3);
    t3 = _mm256_unpackhi_ps(*r2, *r3);
    t4 = _mm256_unpacklo_ps(*r4, *r5);
    t5 = _mm256_unpackhi_ps(*r4, *r5);
    t6 = _mm256_unpacklo_ps(*r6, *r7);
    t7 = _mm256_unpackhi_ps(*r6, *r7);

    *r0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    *r1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    *r2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    *r3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    *r4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    *r5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    *r6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    *r7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

    t0 = _mm256_permute2f128_ps(*r0, *r4, 0x20);
    t1 = _mm256_permute2f128_ps(*r1, *r5, 0x20);
    t2 = _mm256_permute2f128_ps(*r2, *r6, 0x20);
    t3 = _mm256_permute2f128_ps(*r3, *r7, 0x20);
    t4 = _mm256_permute2f128_ps(*r0, *r4, 0x31);
    t5 = _mm256_permute2f128_ps(*r1, *r5, 0x31);
    t6 = _mm256_permute2f128_ps(*r2, *r6, 0x31);
    t7 = _mm256_permute2f128_ps(*r3, *r7, 0x31);

    *r0 = t0; *r1 = t1; *r2 = t2; *r3 = t3;
    *r4 = t4; *r5 = t5; *r6 = t6; *r7 = t7;
}

/* scalar fallback for tile edges */
static void fiv_scalar_transpose_block(float* dst, int ld_dst,
                                   const float* src, int ld_src,
                                   int block_rows, int block_cols)
{
    for (int i = 0; i < block_rows; ++i) {
        const float* src_row = src + i * ld_src;
        for (int j = 0; j < block_cols; ++j) {
            dst[j * ld_dst + i] = src_row[j];
        }
    }
}

#define FIV_TRANSPOSE_CACHE_BLOCK 64
#define FIV_TRANSPOSE_SIMD_BLOCK  8

/* AVX tiled 2D transpose (float32 and int32) */
static void fiv_transpose_avx_tiled(float* dst, int ld_dst,
                                  const float* src, int ld_src,
                                  int rows, int cols)
{
    const int dst_col_step = FIV_TRANSPOSE_SIMD_BLOCK * ld_dst;

    for (int i = 0; i < rows; i += FIV_TRANSPOSE_CACHE_BLOCK) {
        int i_end = (i + FIV_TRANSPOSE_CACHE_BLOCK < rows) ? 
                     i + FIV_TRANSPOSE_CACHE_BLOCK : rows;
        for (int j = 0; j < cols; j += FIV_TRANSPOSE_CACHE_BLOCK) {
            int j_end = (j + FIV_TRANSPOSE_CACHE_BLOCK < cols) ? 
                         j + FIV_TRANSPOSE_CACHE_BLOCK : cols;

            int ii;
            for (ii = i; ii + FIV_TRANSPOSE_SIMD_BLOCK <= i_end; 
                  ii += FIV_TRANSPOSE_SIMD_BLOCK) {
                const float* ps0 = src +  ii      * ld_src + j;
                const float* ps1 = src + (ii + 1) * ld_src + j;
                const float* ps2 = src + (ii + 2) * ld_src + j;
                const float* ps3 = src + (ii + 3) * ld_src + j;
                const float* ps4 = src + (ii + 4) * ld_src + j;
                const float* ps5 = src + (ii + 5) * ld_src + j;
                const float* ps6 = src + (ii + 6) * ld_src + j;
                const float* ps7 = src + (ii + 7) * ld_src + j;

                float* pd0 = dst + j      * ld_dst + ii;
                float* pd1 = dst + (j + 1) * ld_dst + ii;
                float* pd2 = dst + (j + 2) * ld_dst + ii;
                float* pd3 = dst + (j + 3) * ld_dst + ii;
                float* pd4 = dst + (j + 4) * ld_dst + ii;
                float* pd5 = dst + (j + 5) * ld_dst + ii;
                float* pd6 = dst + (j + 6) * ld_dst + ii;
                float* pd7 = dst + (j + 7) * ld_dst + ii;

                int jj = j;
                for (; jj + FIV_TRANSPOSE_SIMD_BLOCK <= j_end; 
                      jj += FIV_TRANSPOSE_SIMD_BLOCK) {
                    __m256 r0 = _mm256_loadu_ps(ps0);
                    __m256 r1 = _mm256_loadu_ps(ps1);
                    __m256 r2 = _mm256_loadu_ps(ps2);
                    __m256 r3 = _mm256_loadu_ps(ps3);
                    __m256 r4 = _mm256_loadu_ps(ps4);
                    __m256 r5 = _mm256_loadu_ps(ps5);
                    __m256 r6 = _mm256_loadu_ps(ps6);
                    __m256 r7 = _mm256_loadu_ps(ps7);

                    fiv_transpose8x8_ps(&r0, &r1, &r2, &r3, 
                                       &r4, &r5, &r6, &r7);

                    _mm256_storeu_ps(pd0, r0);
                    _mm256_storeu_ps(pd1, r1);
                    _mm256_storeu_ps(pd2, r2);
                    _mm256_storeu_ps(pd3, r3);
                    _mm256_storeu_ps(pd4, r4);
                    _mm256_storeu_ps(pd5, r5);
                    _mm256_storeu_ps(pd6, r6);
                    _mm256_storeu_ps(pd7, r7);

                    ps0 += FIV_TRANSPOSE_SIMD_BLOCK;
                    ps1 += FIV_TRANSPOSE_SIMD_BLOCK;
                    ps2 += FIV_TRANSPOSE_SIMD_BLOCK;
                    ps3 += FIV_TRANSPOSE_SIMD_BLOCK;
                    ps4 += FIV_TRANSPOSE_SIMD_BLOCK;
                    ps5 += FIV_TRANSPOSE_SIMD_BLOCK;
                    ps6 += FIV_TRANSPOSE_SIMD_BLOCK;
                    ps7 += FIV_TRANSPOSE_SIMD_BLOCK;

                    pd0 += dst_col_step;
                    pd1 += dst_col_step;
                    pd2 += dst_col_step;
                    pd3 += dst_col_step;
                    pd4 += dst_col_step;
                    pd5 += dst_col_step;
                    pd6 += dst_col_step;
                    pd7 += dst_col_step;
                }

                if (jj < j_end) {
                    fiv_scalar_transpose_block(
                        dst + jj * ld_dst + ii, ld_dst,
                        src + ii * ld_src + jj, ld_src,
                        FIV_TRANSPOSE_SIMD_BLOCK, j_end - jj
                    );
                }
            }

            if (ii < i_end) {
                fiv_scalar_transpose_block(
                    dst + j * ld_dst + ii, ld_dst,
                    src + ii * ld_src + j, ld_src,
                    i_end - ii, j_end - j
                );
            }
        }
    }
}

/* Two-pass out-of-place transpose tuned for memory-bound (write-heavy) blocks.
   Pass A: each 8x8 tile is transposed in registers and its 8 transposed rows
   are stored into a small, L1-resident 64x64 staging buffer. The per-tile
   writes land in the hot buffer, so their scatter costs almost nothing, and in
   that arrangement the buffer ends up holding the fully-transposed block in
   ordinary row-major order.
   Pass B: every whole destination row is then flushed as a sequence of
   contiguous 32-byte stores, which touches many fewer cache lines than the
   direct 8-tile scatter. When the row pointer is 32-byte aligned and the
   transposed width (== source rows) is a multiple of 16, non-temporal
   _mm256_stream_ps stores further exploit the CPU write-combine buffer;
   otherwise plain _mm256_storeu_ps is used. Leftover <8 strips of a block are
   handled by the same scalar fallback as fiv_transpose_avx_tiled. */
static void fiv_transpose_avx_stream_tiled(float* dst, int ld_dst,
                                         const float* src, int ld_src,
                                         int rows, int cols,
                                         int use_stream)
{
    const int T = 8;
    const int B = 64;
    /* 64-byte aligned so every row (256B stride) and every 8-float step (32B)
       inside the buffer is 32-byte aligned: Pass A/B can use the aligned
       _mm256_store_ps / _mm256_load_ps instead of the unaligned variants. */
    FIV_ALIGNED(64) float buf[64][64];

    for (int ib = 0; ib < rows; ib += B) {
        const int bh  = ((ib + B < rows) ? ib + B : rows) - ib; /* block rows  */
        const int bhT = bh - bh % T;                            /* tilable rows */
        for (int jb = 0; jb < cols; jb += B) {
            const int bw  = ((jb + B < cols) ? jb + B : cols) - jb; /* block cols  */
            const int bwT = bw - bw % T;                            /* tilable cols */

            /* ---- Pass A: in-register 8x8 transpose into the staging buffer ---- */
            for (int ty = 0; ty < bhT; ty += T) {
                for (int tx = 0; tx < bwT; tx += T) {
                    const float* ps = src + (size_t)(ib + ty) * ld_src + (jb + tx);
                    __m256 r0 = _mm256_loadu_ps(ps + 0 * ld_src);
                    __m256 r1 = _mm256_loadu_ps(ps + 1 * ld_src);
                    __m256 r2 = _mm256_loadu_ps(ps + 2 * ld_src);
                    __m256 r3 = _mm256_loadu_ps(ps + 3 * ld_src);
                    __m256 r4 = _mm256_loadu_ps(ps + 4 * ld_src);
                    __m256 r5 = _mm256_loadu_ps(ps + 5 * ld_src);
                    __m256 r6 = _mm256_loadu_ps(ps + 6 * ld_src);
                    __m256 r7 = _mm256_loadu_ps(ps + 7 * ld_src);
                    fiv_transpose8x8_ps(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
                    /* tile at rows [ty,ty+8) x cols [tx,tx+8) transposes to
                       buffer rows [tx, tx+8), cols [ty, ty+8) */
                    _mm256_store_ps(&buf[tx + 0][ty], r0);
                    _mm256_store_ps(&buf[tx + 1][ty], r1);
                    _mm256_store_ps(&buf[tx + 2][ty], r2);
                    _mm256_store_ps(&buf[tx + 3][ty], r3);
                    _mm256_store_ps(&buf[tx + 4][ty], r4);
                    _mm256_store_ps(&buf[tx + 5][ty], r5);
                    _mm256_store_ps(&buf[tx + 6][ty], r6);
                    _mm256_store_ps(&buf[tx + 7][ty], r7);
                }
            }

            /* ---- Pass B: flush whole destination rows, stored contiguously ---- */
            for (int R = 0; R < bwT; ++R) {
                float* pd = dst + (size_t)(jb + R) * ld_dst + ib;
                const float* bsrc = buf[R];
                const int aligned = (use_stream && !(((uintptr_t)(pd)) & 31));
                int c = 0;
                /* The aligned flag is loop-invariant here: hoist it out of the
                   per-tile loop by splitting into a stream and a storeu loop. */
                if (aligned) {
                    for (; c + T <= bhT; c += T)
                        _mm256_stream_ps(pd + c, _mm256_load_ps(bsrc + c));
                } else {
                    for (; c + T <= bhT; c += T)
                        _mm256_storeu_ps(pd + c, _mm256_load_ps(bsrc + c));
                }
                for (; c < bhT; ++c) pd[c] = bsrc[c];
            }

            /* ---- Leftover <8 source rows (bottom strip) -> dest columns ---- */
            if (bhT < bh) {
                fiv_scalar_transpose_block(dst + (size_t)jb * ld_dst + (ib + bhT), ld_dst,
                                        src + (size_t)(ib + bhT) * ld_src + jb, ld_src,
                                        bh - bhT, bw);
            }
            /* ---- Leftover <8 source cols (right strip) -> dest rows ---- */
            if (bwT < bw) {
                fiv_scalar_transpose_block(dst + (size_t)(jb + bwT) * ld_dst + ib, ld_dst,
                                        src + (size_t)ib * ld_src + (jb + bwT), ld_src,
                                        bh, bw - bwT);
            }
        }
    }
}

/* In-place square transpose of a contiguous row-major n x n matrix, done by
   swapping the upper/lower 8x8 tiles through the register-transpose primitive
   (fiv_transpose8x8_ps). The diagonal tiles (i == j) are transposed in place; the
   strictly-upper tiles (i > j) are exchanged once with their mirror (j, i).
   A trailing strip of < 8 rows falls back to the scalar triangle swap. The
   whole matrix is touched exactly once (each off-diagonal pair swapped a single
   time), so it moves half the elements of an out-of-place copy. */
static void fiv_transpose_square_inplace_avx(ivf32* a, int n)
{
    const int T = 8;
    int i;
    for (i = 0; i + T <= n; i += T) {
        int j;
        for (j = 0; j <= i; j += T) {
            /* T1 = tile rows [i, i+8) x cols [j, j+8);
               mirror T2 = tile rows [j, j+8) x cols [i, i+8). */
            __m256 r0 = _mm256_loadu_ps(a + (size_t)(i + 0) * n + j);
            __m256 r1 = _mm256_loadu_ps(a + (size_t)(i + 1) * n + j);
            __m256 r2 = _mm256_loadu_ps(a + (size_t)(i + 2) * n + j);
            __m256 r3 = _mm256_loadu_ps(a + (size_t)(i + 3) * n + j);
            __m256 r4 = _mm256_loadu_ps(a + (size_t)(i + 4) * n + j);
            __m256 r5 = _mm256_loadu_ps(a + (size_t)(i + 5) * n + j);
            __m256 r6 = _mm256_loadu_ps(a + (size_t)(i + 6) * n + j);
            __m256 r7 = _mm256_loadu_ps(a + (size_t)(i + 7) * n + j);

            __m256 s0 = _mm256_loadu_ps(a + (size_t)(j + 0) * n + i);
            __m256 s1 = _mm256_loadu_ps(a + (size_t)(j + 1) * n + i);
            __m256 s2 = _mm256_loadu_ps(a + (size_t)(j + 2) * n + i);
            __m256 s3 = _mm256_loadu_ps(a + (size_t)(j + 3) * n + i);
            __m256 s4 = _mm256_loadu_ps(a + (size_t)(j + 4) * n + i);
            __m256 s5 = _mm256_loadu_ps(a + (size_t)(j + 5) * n + i);
            __m256 s6 = _mm256_loadu_ps(a + (size_t)(j + 6) * n + i);
            __m256 s7 = _mm256_loadu_ps(a + (size_t)(j + 7) * n + i);

            fiv_transpose8x8_ps(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
            fiv_transpose8x8_ps(&s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);

            /* T1 transposed -> mirror location (rows [j, j+8) x cols [i, i+8)) */
            _mm256_storeu_ps(a + (size_t)(j + 0) * n + i, r0);
            _mm256_storeu_ps(a + (size_t)(j + 1) * n + i, r1);
            _mm256_storeu_ps(a + (size_t)(j + 2) * n + i, r2);
            _mm256_storeu_ps(a + (size_t)(j + 3) * n + i, r3);
            _mm256_storeu_ps(a + (size_t)(j + 4) * n + i, r4);
            _mm256_storeu_ps(a + (size_t)(j + 5) * n + i, r5);
            _mm256_storeu_ps(a + (size_t)(j + 6) * n + i, r6);
            _mm256_storeu_ps(a + (size_t)(j + 7) * n + i, r7);

            /* T2 transposed -> original location (rows [i, i+8) x cols [j, j+8)) */
            _mm256_storeu_ps(a + (size_t)(i + 0) * n + j, s0);
            _mm256_storeu_ps(a + (size_t)(i + 1) * n + j, s1);
            _mm256_storeu_ps(a + (size_t)(i + 2) * n + j, s2);
            _mm256_storeu_ps(a + (size_t)(i + 3) * n + j, s3);
            _mm256_storeu_ps(a + (size_t)(i + 4) * n + j, s4);
            _mm256_storeu_ps(a + (size_t)(i + 5) * n + j, s5);
            _mm256_storeu_ps(a + (size_t)(i + 6) * n + j, s6);
            _mm256_storeu_ps(a + (size_t)(i + 7) * n + j, s7);
        }
    }
    /* scalar tail: the remaining < 8 rows (i = largest multiple of 8 <= n) */
    for (int ii = i; ii < n; ++ii) {
        ivf32* row_ii = a + (size_t)ii * n;
        ivf32* col_ii = a + ii;
        for (int jj = 0; jj < ii; ++jj) {
            ivf32 tmp = row_ii[jj];
            row_ii[jj] = col_ii[(size_t)jj * n];
            col_ii[(size_t)jj * n] = tmp;
        }
    }
}

#endif /* FIV_USE_AVX */

/* ==================== NEON (aarch64) branch, mirrors the AVX design ====================
   AVX uses 8x8 blocks (256-bit __m256); NEON registers are 128-bit
   (float32x4_t), so a 4x4 block transpose is used; tiling/cache structure
   stays the same. Transpose only moves element positions, so float32 and
   int32 are bit-exact. */
#if defined(__aarch64__)
#include <arm_neon.h>

#ifndef FIV_TRANSPOSE_CACHE_BLOCK
#define FIV_TRANSPOSE_CACHE_BLOCK 64
#endif
#define FIV_TRANSPOSE_NEON_SIMD_BLOCK 4

/* NEON 4x4 float transpose (vtrn -> vuzp -> vtrn) */
static inline void fiv_transpose4x4_ps_neon(float32x4_t* a, float32x4_t* b,
                                          float32x4_t* c, float32x4_t* d)
{
    float32x4x2_t t0 = vtrnq_f32(*a, *b);
    float32x4x2_t t1 = vtrnq_f32(*c, *d);
    float32x4_t P0 = vuzp1q_f32(t0.val[0], t1.val[0]);
    float32x4_t P1 = vuzp2q_f32(t0.val[0], t1.val[0]);
    float32x4_t Q0 = vuzp1q_f32(t0.val[1], t1.val[1]);
    float32x4_t Q1 = vuzp2q_f32(t0.val[1], t1.val[1]);
    *a = vtrn1q_f32(P0, P1);  /* col0 = [a0,b0,c0,d0] */
    *b = vtrn1q_f32(Q0, Q1);  /* col1 = [a1,b1,c1,d1] */
    *c = vtrn2q_f32(P0, P1);  /* col2 = [a2,b2,c2,d2] */
    *d = vtrn2q_f32(Q0, Q1);  /* col3 = [a3,b3,c3,d3] */
}

/* scalar edge fallback, NEON-specific name to avoid clashing with AVX */
static void fiv_neon_transpose_block(float* dst, int ld_dst,
                                          const float* src, int ld_src,
                                          int block_rows, int block_cols)
{
    for (int i = 0; i < block_rows; ++i) {
        const float* src_row = src + i * ld_src;
        for (int j = 0; j < block_cols; ++j) {
            dst[j * ld_dst + i] = src_row[j];
        }
    }
}

/* NEON tiled 2D transpose (float32 and int32) */
static void fiv_transpose_neon_tiled(float* dst, int ld_dst,
                                  const float* src, int ld_src,
                                  int rows, int cols)
{
    const int dst_col_step = FIV_TRANSPOSE_NEON_SIMD_BLOCK * ld_dst;

    for (int i = 0; i < rows; i += FIV_TRANSPOSE_CACHE_BLOCK) {
        int i_end = (i + FIV_TRANSPOSE_CACHE_BLOCK < rows) ?
                     i + FIV_TRANSPOSE_CACHE_BLOCK : rows;
        for (int j = 0; j < cols; j += FIV_TRANSPOSE_CACHE_BLOCK) {
            int j_end = (j + FIV_TRANSPOSE_CACHE_BLOCK < cols) ?
                         j + FIV_TRANSPOSE_CACHE_BLOCK : cols;

            int ii;
            for (ii = i; ii + FIV_TRANSPOSE_NEON_SIMD_BLOCK <= i_end;
                  ii += FIV_TRANSPOSE_NEON_SIMD_BLOCK) {
                const float* ps0 = src +  ii      * ld_src + j;
                const float* ps1 = src + (ii + 1) * ld_src + j;
                const float* ps2 = src + (ii + 2) * ld_src + j;
                const float* ps3 = src + (ii + 3) * ld_src + j;

                float* pd0 = dst + j      * ld_dst + ii;
                float* pd1 = dst + (j + 1) * ld_dst + ii;
                float* pd2 = dst + (j + 2) * ld_dst + ii;
                float* pd3 = dst + (j + 3) * ld_dst + ii;

                int jj = j;
                for (; jj + FIV_TRANSPOSE_NEON_SIMD_BLOCK <= j_end;
                      jj += FIV_TRANSPOSE_NEON_SIMD_BLOCK) {
                    float32x4_t r0 = vld1q_f32(ps0);
                    float32x4_t r1 = vld1q_f32(ps1);
                    float32x4_t r2 = vld1q_f32(ps2);
                    float32x4_t r3 = vld1q_f32(ps3);

                    fiv_transpose4x4_ps_neon(&r0, &r1, &r2, &r3);

                    vst1q_f32(pd0, r0);
                    vst1q_f32(pd1, r1);
                    vst1q_f32(pd2, r2);
                    vst1q_f32(pd3, r3);

                    ps0 += FIV_TRANSPOSE_NEON_SIMD_BLOCK;
                    ps1 += FIV_TRANSPOSE_NEON_SIMD_BLOCK;
                    ps2 += FIV_TRANSPOSE_NEON_SIMD_BLOCK;
                    ps3 += FIV_TRANSPOSE_NEON_SIMD_BLOCK;

                    pd0 += dst_col_step;
                    pd1 += dst_col_step;
                    pd2 += dst_col_step;
                    pd3 += dst_col_step;
                }

                if (jj < j_end) {
                    fiv_neon_transpose_block(
                        dst + jj * ld_dst + ii, ld_dst,
                        src + ii * ld_src + jj, ld_src,
                        FIV_TRANSPOSE_NEON_SIMD_BLOCK, j_end - jj
                    );
                }
            }

            if (ii < i_end) {
                fiv_neon_transpose_block(
                    dst + j * ld_dst + ii, ld_dst,
                    src + ii * ld_src + j, ld_src,
                    i_end - ii, j_end - j
                );
            }
        }
    }
}

#endif /* __aarch64__ */


/* ==================== Generic scalar fallback ====================
   Transpose is type-agnostic (only byte positions move), so a plain
   float* copy is bit-exact for any 4-byte element type. */
#if !defined(FIV_USE_AVX) && !defined(__aarch64__)
static void fiv_transpose_scalar(float* dst, int ld_dst,
                              const float* src, int ld_src,
                              int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        const float* s = src + (size_t)i * ld_src;
        for (int j = 0; j < cols; ++j) {
            dst[(size_t)j * ld_dst + i] = s[j];
        }
    }
}
#endif


/* In-place transpose of a square n x n contiguous row-major matrix via
   triangle swaps (O(n^2) element moves, half of an out-of-place copy).
   Used by the public fiv_matrix_transpose for the dst==src square case and by
   the Cholesky driver. AVX2 builds take the vectorized tile path above; other
   builds fall back to the scalar triangle swap. */
void fiv_transpose_square_inplace_real32(ivf32* a, int n)
{
#if defined(FIV_USE_AVX)
    fiv_transpose_square_inplace_avx(a, n);
#else
    for (int i = 1; i < n; ++i) {
        ivf32* row_i = a + (size_t)i * n;
        ivf32* col_i = a + i;
        for (int j = 0; j < i; ++j) {
            ivf32 tmp = row_i[j];
            row_i[j] = col_i[(size_t)j * n];
            col_i[(size_t)j * n] = tmp;
        }
    }
#endif
}


/* supported element types: 4-byte (32U / 32S / 32F families) */
static int fiv_is_4byte_dtype(fiv_data_type t)
{
    int m = (int)t % 16;
    return m == 4 || m == 5 || m == 8;
}


/* ==================== Public API: transpose src (m x n) into dst (n x m) ====================
   - dst must already hold a buffer of >= m*n*eb bytes.
   - Both src and dst must be contiguous (data_continue == 1) and share
     the same 4-byte dtype; other dtypes return FIV_RET_ERR_NOT_SUPPORT.
   - In-place (dst aliases src) IS supported for SQUARE matrices (m == n):
     the lower/upper triangles are swapped in place and the call returns
     FIV_RET_OK with metadata unchanged. In-place on a NON-square matrix has
     no second buffer to hold the result and returns FIV_RET_ERR_PARA.
   - On return, dst's metadata is rewritten to describe the transposed
     matrix (rows/cols swapped, strides/total_bytes updated). */
fiv_ret fiv_matrix_transpose(fiv_mat* dst, const fiv_mat* src)
{
    /* ---- validation (consistent with fiv_tensor_add) ---- */
    if (dst == NULL || src == NULL) return FIV_RET_ERR_PARA;
    if (dst->data.ptr == NULL || src->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (dst->data_continue == 0 || src->data_continue == 0) return FIV_RET_ERR_PARA;
    if (src->dtype != dst->dtype) return FIV_RET_ERR_PARA;
    if (!fiv_is_4byte_dtype(src->dtype)) return FIV_RET_ERR_NOT_SUPPORT;

    const size_t eb = (size_t)src->element_bytes;
    const size_t m  = src->shapes[0];   /* src rows */
    const size_t n  = src->shapes[1];   /* src cols */

    if (dst->total_bytes < eb * m * n) return FIV_RET_ERR_PARA;

    float* src_f = (float*)src->data.ptr;
    float* dst_f = (float*)dst->data.ptr;

    if (dst_f == src_f) {
        /* In-place: only a square matrix can be transposed without a second
           buffer (lower/upper triangles are swapped in place). Metadata is
           unchanged for an n x n (rows/cols and strides are symmetric). */
        if (m != n) return FIV_RET_ERR_PARA;
        fiv_transpose_square_inplace_real32((ivf32*)dst_f, (int)n);
        return FIV_RET_OK;
    }

    /* out-of-place copy */
    const int  ld_src = (int)n;          /* src stride (elements per row) */
    const int  ld_dst = (int)m;          /* dst stride (transposed cols)   */

#if defined(FIV_USE_AVX)
    /* Three-tier dispatch (benchmarked on AVX2, see build/bench_mat_transpose.c):
       - large  (m,n >= 2048, m%16==0): two-pass staging + non-temporal stream
         stores -> ~+5..23% (write-combine on long aligned dest rows);
       - medium (256 <= m,n <= 1024):   two-pass staging, plain storeu -> ~+9..27%
         (matrix is cache resident, contiguous row flushes beat the direct
          scattered 8-tile stores);
       - small / non-cache-friendly (large with m%16!=0, etc.): direct
         cache-blocked tiled path (the staging extra pass is pure overhead). */
    if ((int)m >= 2048 && (int)n >= 2048 && ((int)m % 16) == 0)
        fiv_transpose_avx_stream_tiled(dst_f, ld_dst, src_f, ld_src, (int)m, (int)n, 1);
    else if ((int)m >= 256 && (int)n >= 256 && (int)m <= 1024 && (int)n <= 1024)
        fiv_transpose_avx_stream_tiled(dst_f, ld_dst, src_f, ld_src, (int)m, (int)n, 0);
    else
        fiv_transpose_avx_tiled(dst_f, ld_dst, src_f, ld_src, (int)m, (int)n);
#elif defined(__aarch64__)
    fiv_transpose_neon_tiled(dst_f, ld_dst, src_f, ld_src, (int)m, (int)n);
#else
    fiv_transpose_scalar(dst_f, ld_dst, src_f, ld_src, (int)m, (int)n);
#endif

    /* rewrite dst metadata to describe the transposed matrix */
    dst->id            = src->id;
    dst->dtype         = src->dtype;
    dst->shapes[0]     = n;
    dst->shapes[1]     = m;
    dst->strides[0]    = eb * m;   /* transposed matrix is n x m: row stride = m */
    dst->strides[1]    = eb;
    dst->total_bytes   = eb * m * n;
    dst->data_continue = 1;
    dst->element_bytes = (iv8u)eb;

    return FIV_RET_OK;
}
