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

/* 64-bit (ivf64 / double) matrix multiplication. Algorithmic structure is
   ported from fiv_mat_mul.c (the float32 path): the same four transpose
   variants, the small (non-blocked) vs blocked dispatch, the cache-blocked
   panel packing, the 8x8 micro-kernel, and the public-API contract. The
   float32 path uses width-specific SIMD; here the SIMD is the double-precision
   equivalent: NEON float64x2_t (2 lanes) and AVX2 __m256d (4 lanes). The
   small kernels block 4 output rows at a time and unroll k by 2, mirroring
   the float32 kernels; the micro-kernel broadcasts each a[] scalar across the
   b rows exactly like the float32 8x8 kernel. */

#include "fiv_mat_mul_db.h"
#include "fiv_common.h"

#include <string.h>   /* memset */
#include <stdio.h>    /* printf (kept verbatim from the verified kernels) */

#if defined(FIV_USE_AVX) || defined(FIV_USE_AVX2)
#include <immintrin.h>
#endif
#if defined(FIV_USE_ARM_NEON)
#include <arm_neon.h>
#endif

#ifndef FIV_INLINE
#define FIV_INLINE static inline
#endif

#ifndef FIV_PRINT_LOG
#define FIV_PRINT_LOG(msg) ((void)0)
#endif

#ifndef FIV_DALIGNED
#define FIV_DALIGNED
#endif

/* ============================================================================
   Small-matrix kernels. Copied structurally from the verified
   fiv_small_matrix_mul_matrix_*_real32 (double only). The only edits are:
   SIMD feature macros mapped to FIV_USE_AVX2 / FIV_USE_ARM_NEON, ivf32 ->
   ivf64, float32x4_t -> float64x2_t / __m256d, and the vector width scaled
   (NEON 2 lanes vs 4, AVX2 4 lanes vs 8).
   ========================================================================== */

#if defined(FIV_USE_AVX)
FIV_INLINE ivf64 fiv_mm_hsum_pd_sse3(__m128d v)
{
    __m128d shuf = _mm_movehdup_pd(v);
    __m128d sums = _mm_add_pd(v, shuf);
    return _mm_cvtsd_f64(sums);
}
#endif

#if defined(FIV_USE_AVX2)
FIV_INLINE ivf64 fiv_mm256_hsum_pd(__m256d v)
{
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    return fiv_mm_hsum_pd_sse3(vlow);
}
#endif

/* C = A * B, A: rows_a x cols_a, B: cols_a x cols_b */
static void fiv_small_matrix_mul_matrix_real64(
    ivf64* data_a, int rows_a, int cols_a, int stride_a,
    ivf64* data_b, int cols_b, int stride_b,
    ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta)
{
    int i = 0;
    for (; i <= rows_a - 4; i += 4){
        ivf64* ptr_a[4], *ptr_c[4];
        ptr_a[0] = &data_a[i * stride_a];
        ptr_a[1] = ptr_a[0] + stride_a;
        ptr_a[2] = ptr_a[1] + stride_a;
        ptr_a[3] = ptr_a[2] + stride_a;

        ptr_c[0] = &data_c[i * stride_c];
        ptr_c[1] = ptr_c[0] + stride_c;
        ptr_c[2] = ptr_c[1] + stride_c;
        ptr_c[3] = ptr_c[2] + stride_c;

        if (beta == 0.0) {
            memset(ptr_c[0], 0, sizeof(ivf64) * cols_b);
            memset(ptr_c[1], 0, sizeof(ivf64) * cols_b);
            memset(ptr_c[2], 0, sizeof(ivf64) * cols_b);
            memset(ptr_c[3], 0, sizeof(ivf64) * cols_b);
        }  else if (beta != 1.0) {
            for (int l = 0; l < cols_b; l++){
                ptr_c[0][l] *= beta;
                ptr_c[1][l] *= beta;
                ptr_c[2][l] *= beta;
                ptr_c[3][l] *= beta;
            }
        }
        int k = 0;
        for (; k <= cols_a - 2; k += 2)	{
            ivf64 t_a[8];
            if (alpha != 1.0) {
                t_a[0] = ptr_a[0][k + 0] * alpha;
                t_a[1] = ptr_a[1][k + 0] * alpha;
                t_a[2] = ptr_a[2][k + 0] * alpha;
                t_a[3] = ptr_a[3][k + 0] * alpha;
                t_a[4] = ptr_a[0][k + 1] * alpha;
                t_a[5] = ptr_a[1][k + 1] * alpha;
                t_a[6] = ptr_a[2][k + 1] * alpha;
                t_a[7] = ptr_a[3][k + 1] * alpha;
            }	else {
                t_a[0] = ptr_a[0][k + 0] ;
                t_a[1] = ptr_a[1][k + 0] ;
                t_a[2] = ptr_a[2][k + 0] ;
                t_a[3] = ptr_a[3][k + 0] ;
                t_a[4] = ptr_a[0][k + 1] ;
                t_a[5] = ptr_a[1][k + 1] ;
                t_a[6] = ptr_a[2][k + 1] ;
                t_a[7] = ptr_a[3][k + 1] ;

            }
            ivf64* ptr_b[2] = {&data_b[k * stride_b], &data_b[(k + 1)* stride_b]};
            int j = 0;
#if defined(FIV_USE_AVX2)
            __m256d m_t_a1 = _mm256_broadcast_sd(&t_a[0]);
            __m256d m_t_a2 = _mm256_broadcast_sd(&t_a[1]);
            __m256d m_t_a3 = _mm256_broadcast_sd(&t_a[2]);
            __m256d m_t_a4 = _mm256_broadcast_sd(&t_a[3]);
            __m256d m_t_a5 = _mm256_broadcast_sd(&t_a[4]);
            __m256d m_t_a6 = _mm256_broadcast_sd(&t_a[5]);
            __m256d m_t_a7 = _mm256_broadcast_sd(&t_a[6]);
            __m256d m_t_a8 = _mm256_broadcast_sd(&t_a[7]);

            for (; j <= cols_b - 4; j += 4)	{
                __m256d m_t_c0 = _mm256_loadu_pd(&ptr_c[0][j]);
                __m256d m_t_c1 = _mm256_loadu_pd(&ptr_c[1][j]);
                __m256d m_t_c2 = _mm256_loadu_pd(&ptr_c[2][j]);
                __m256d m_t_c3 = _mm256_loadu_pd(&ptr_c[3][j]);
                __m256d m_t_b0 = _mm256_loadu_pd(&ptr_b[0][j]);
                __m256d m_t_b1 = _mm256_loadu_pd(&ptr_b[1][j]);

                m_t_c0 = _mm256_fmadd_pd(m_t_a1, m_t_b0, m_t_c0);
                m_t_c1 = _mm256_fmadd_pd(m_t_a2, m_t_b0, m_t_c1);
                m_t_c2 = _mm256_fmadd_pd(m_t_a3, m_t_b0, m_t_c2);
                m_t_c3 = _mm256_fmadd_pd(m_t_a4, m_t_b0, m_t_c3);

                m_t_c0 = _mm256_fmadd_pd(m_t_a5, m_t_b1, m_t_c0);
                m_t_c1 = _mm256_fmadd_pd(m_t_a6, m_t_b1, m_t_c1);
                m_t_c2 = _mm256_fmadd_pd(m_t_a7, m_t_b1, m_t_c2);
                m_t_c3 = _mm256_fmadd_pd(m_t_a8, m_t_b1, m_t_c3);

                _mm256_storeu_pd(&ptr_c[0][j], m_t_c0);
                _mm256_storeu_pd(&ptr_c[1][j], m_t_c1);
                _mm256_storeu_pd(&ptr_c[2][j], m_t_c2);
                _mm256_storeu_pd(&ptr_c[3][j], m_t_c3);

            }
#elif defined(FIV_USE_ARM_NEON)
            for (; j <= cols_b - 4; j += 4) {
                float64x2x2_t b0 = { { vld1q_f64(&ptr_b[0][j]),     vld1q_f64(&ptr_b[0][j + 2]) } };
                float64x2x2_t b1 = { { vld1q_f64(&ptr_b[1][j]),     vld1q_f64(&ptr_b[1][j + 2]) } };
                float64x2x2_t c0 = { { vld1q_f64(&ptr_c[0][j]),     vld1q_f64(&ptr_c[0][j + 2]) } };
                float64x2x2_t c1 = { { vld1q_f64(&ptr_c[1][j]),     vld1q_f64(&ptr_c[1][j + 2]) } };
                float64x2x2_t c2 = { { vld1q_f64(&ptr_c[2][j]),     vld1q_f64(&ptr_c[2][j + 2]) } };
                float64x2x2_t c3 = { { vld1q_f64(&ptr_c[3][j]),     vld1q_f64(&ptr_c[3][j + 2]) } };

                c0.val[0] = vfmaq_n_f64(c0.val[0], b0.val[0], t_a[0]);
                c0.val[1] = vfmaq_n_f64(c0.val[1], b0.val[1], t_a[0]);
                c1.val[0] = vfmaq_n_f64(c1.val[0], b0.val[0], t_a[1]);
                c1.val[1] = vfmaq_n_f64(c1.val[1], b0.val[1], t_a[1]);
                c2.val[0] = vfmaq_n_f64(c2.val[0], b0.val[0], t_a[2]);
                c2.val[1] = vfmaq_n_f64(c2.val[1], b0.val[1], t_a[2]);
                c3.val[0] = vfmaq_n_f64(c3.val[0], b0.val[0], t_a[3]);
                c3.val[1] = vfmaq_n_f64(c3.val[1], b0.val[1], t_a[3]);

                c0.val[0] = vfmaq_n_f64(c0.val[0], b1.val[0], t_a[4]);
                c0.val[1] = vfmaq_n_f64(c0.val[1], b1.val[1], t_a[4]);
                c1.val[0] = vfmaq_n_f64(c1.val[0], b1.val[0], t_a[5]);
                c1.val[1] = vfmaq_n_f64(c1.val[1], b1.val[1], t_a[5]);
                c2.val[0] = vfmaq_n_f64(c2.val[0], b1.val[0], t_a[6]);
                c2.val[1] = vfmaq_n_f64(c2.val[1], b1.val[1], t_a[6]);
                c3.val[0] = vfmaq_n_f64(c3.val[0], b1.val[0], t_a[7]);
                c3.val[1] = vfmaq_n_f64(c3.val[1], b1.val[1], t_a[7]);

                vst1q_f64(&ptr_c[0][j],     c0.val[0]);
                vst1q_f64(&ptr_c[0][j + 2], c0.val[1]);
                vst1q_f64(&ptr_c[1][j],     c1.val[0]);
                vst1q_f64(&ptr_c[1][j + 2], c1.val[1]);
                vst1q_f64(&ptr_c[2][j],     c2.val[0]);
                vst1q_f64(&ptr_c[2][j + 2], c2.val[1]);
                vst1q_f64(&ptr_c[3][j],     c3.val[0]);
                vst1q_f64(&ptr_c[3][j + 2], c3.val[1]);
            }
#endif
            for (; j < cols_b; j++){
                ptr_c[0][j] += t_a[0] * ptr_b[0][j];
                ptr_c[1][j] += t_a[1] * ptr_b[0][j];
                ptr_c[2][j] += t_a[2] * ptr_b[0][j];
                ptr_c[3][j] += t_a[3] * ptr_b[0][j];

                ptr_c[0][j] += t_a[4] * ptr_b[1][j];
                ptr_c[1][j] += t_a[5] * ptr_b[1][j];
                ptr_c[2][j] += t_a[6] * ptr_b[1][j];
                ptr_c[3][j] += t_a[7] * ptr_b[1][j];
            }
        }

        for (; k < cols_a; k++)	{
            ivf64 t_a[4] = {
                ptr_a[0][k] * alpha,ptr_a[1][k] * alpha,
                ptr_a[2][k] * alpha,ptr_a[3][k] * alpha
            };
            ivf64* ptr_b = &data_b[k * stride_b];

            int j = 0;
#if defined(FIV_USE_AVX2)
            __m256d m_t_a1 = _mm256_broadcast_sd(&t_a[0]);
            __m256d m_t_a2 = _mm256_broadcast_sd(&t_a[1]);
            __m256d m_t_a3 = _mm256_broadcast_sd(&t_a[2]);
            __m256d m_t_a4 = _mm256_broadcast_sd(&t_a[3]);

            for (; j <= cols_b - 4; j += 4){
                __m256d m_t_c0 = _mm256_loadu_pd(&ptr_c[0][j]);
                __m256d m_t_c1 = _mm256_loadu_pd(&ptr_c[1][j]);
                __m256d m_t_c2 = _mm256_loadu_pd(&ptr_c[2][j]);
                __m256d m_t_c3 = _mm256_loadu_pd(&ptr_c[3][j]);
                __m256d m_t_b1 = _mm256_loadu_pd(&ptr_b[j]);

                m_t_c0 = _mm256_fmadd_pd(m_t_a1, m_t_b1, m_t_c0);
                m_t_c1 = _mm256_fmadd_pd(m_t_a2, m_t_b1, m_t_c1);
                m_t_c2 = _mm256_fmadd_pd(m_t_a3, m_t_b1, m_t_c2);
                m_t_c3 = _mm256_fmadd_pd(m_t_a4, m_t_b1, m_t_c3);

                _mm256_storeu_pd(&ptr_c[0][j], m_t_c0);
                _mm256_storeu_pd(&ptr_c[1][j], m_t_c1);
                _mm256_storeu_pd(&ptr_c[2][j], m_t_c2);
                _mm256_storeu_pd(&ptr_c[3][j], m_t_c3);
            }
#elif defined(FIV_USE_ARM_NEON)
            for (; j <= cols_b - 4; j += 4) {
                float64x2x2_t b  = { { vld1q_f64(&ptr_b[j]),         vld1q_f64(&ptr_b[j + 2]) } };
                float64x2x2_t c0 = { { vld1q_f64(&ptr_c[0][j]),     vld1q_f64(&ptr_c[0][j + 2]) } };
                float64x2x2_t c1 = { { vld1q_f64(&ptr_c[1][j]),     vld1q_f64(&ptr_c[1][j + 2]) } };
                float64x2x2_t c2 = { { vld1q_f64(&ptr_c[2][j]),     vld1q_f64(&ptr_c[2][j + 2]) } };
                float64x2x2_t c3 = { { vld1q_f64(&ptr_c[3][j]),     vld1q_f64(&ptr_c[3][j + 2]) } };

                c0.val[0] = vfmaq_n_f64(c0.val[0], b.val[0], t_a[0]);
                c0.val[1] = vfmaq_n_f64(c0.val[1], b.val[1], t_a[0]);
                c1.val[0] = vfmaq_n_f64(c1.val[0], b.val[0], t_a[1]);
                c1.val[1] = vfmaq_n_f64(c1.val[1], b.val[1], t_a[1]);
                c2.val[0] = vfmaq_n_f64(c2.val[0], b.val[0], t_a[2]);
                c2.val[1] = vfmaq_n_f64(c2.val[1], b.val[1], t_a[2]);
                c3.val[0] = vfmaq_n_f64(c3.val[0], b.val[0], t_a[3]);
                c3.val[1] = vfmaq_n_f64(c3.val[1], b.val[1], t_a[3]);

                vst1q_f64(&ptr_c[0][j],     c0.val[0]);
                vst1q_f64(&ptr_c[0][j + 2], c0.val[1]);
                vst1q_f64(&ptr_c[1][j],     c1.val[0]);
                vst1q_f64(&ptr_c[1][j + 2], c1.val[1]);
                vst1q_f64(&ptr_c[2][j],     c2.val[0]);
                vst1q_f64(&ptr_c[2][j + 2], c2.val[1]);
                vst1q_f64(&ptr_c[3][j],     c3.val[0]);
                vst1q_f64(&ptr_c[3][j + 2], c3.val[1]);
            }
#endif
            for (; j < cols_b; j++){
                ptr_c[0][j] += t_a[0] * ptr_b[j];
                ptr_c[1][j] += t_a[1] * ptr_b[j];
                ptr_c[2][j] += t_a[2] * ptr_b[j];
                ptr_c[3][j] += t_a[3] * ptr_b[j];
            }
        }
    }

    for (; i < rows_a; i++){
        ivf64* ptr_a = &data_a[i * stride_a];
        ivf64* ptr_c = &data_c[i * stride_c];
        if (beta == 0.0){
            memset(ptr_c, 0, sizeof(ivf64) * cols_b);
        }	else if (beta != 1.0) {
            for (int l = 0; l < cols_b; l++){
                ptr_c[l] *= beta;
            }
        }
        int k = 0;
        for (; k < cols_a; k++) {
            ivf64 t_a_d = ptr_a[k] * alpha;
            ivf64* ptr_b = &data_b[k * stride_b];
            for (int j = 0; j < cols_b; j++){
                ptr_c[j] += t_a_d * ptr_b[j];
            }
        }
    }
}

/* C = A * B^T, A: rows_a x cols_a, B: rows_b x cols_a, result rows_a x rows_b */
static void fiv_small_matrix_mul_matrix_t_real64(
    ivf64* data_a, int rows_a, int cols_a, int stride_a,
    ivf64* data_b, int rows_b, int stride_b,
    ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta)
{
    int i = 0;
    for (; i <= rows_a - 4; i += 4) {
        ivf64* ptr_lines_a[4], *ptr_lines_c[4];
        ptr_lines_a[0] = &data_a[i * stride_a];
        ptr_lines_a[1] = ptr_lines_a[0] + stride_a;
        ptr_lines_a[2] = ptr_lines_a[1] + stride_a;
        ptr_lines_a[3] = ptr_lines_a[2] + stride_a;

        ptr_lines_c[0] = &data_c[i * stride_c];
        ptr_lines_c[1] = ptr_lines_c[0] + stride_c;
        ptr_lines_c[2] = ptr_lines_c[1] + stride_c;
        ptr_lines_c[3] = ptr_lines_c[2] + stride_c;

        int j = 0;

        for (; j <= rows_b - 2; j += 2) {
            ivf64* ptr_lines_b[2];
            ptr_lines_b[0] = &data_b[j * stride_b];
            ptr_lines_b[1] = ptr_lines_b[0] + stride_b;

            ivf64 FIV_DALIGNED sum[8] = { 0 };
            int k = 0;
#if defined(FIV_USE_AVX2)
            __m256d m_s[8] = { 0 };
            for (; k <= cols_a - 4; k += 4) {
                __m256d t_b1 = _mm256_loadu_pd(&ptr_lines_b[0][k]);
                __m256d t_b2 = _mm256_loadu_pd(&ptr_lines_b[1][k]);

                m_s[0] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[0][k]), t_b1, m_s[0]);
                m_s[1] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[1][k]), t_b1, m_s[1]);
                m_s[2] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[2][k]), t_b1, m_s[2]);
                m_s[3] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[3][k]), t_b1, m_s[3]);

                m_s[4] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[0][k]), t_b2, m_s[4]);
                m_s[5] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[1][k]), t_b2, m_s[5]);
                m_s[6] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[2][k]), t_b2, m_s[6]);
                m_s[7] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[3][k]), t_b2, m_s[7]);
            }
            sum[0] = fiv_mm256_hsum_pd(m_s[0]);
            sum[1] = fiv_mm256_hsum_pd(m_s[1]);
            sum[2] = fiv_mm256_hsum_pd(m_s[2]);
            sum[3] = fiv_mm256_hsum_pd(m_s[3]);
            sum[4] = fiv_mm256_hsum_pd(m_s[4]);
            sum[5] = fiv_mm256_hsum_pd(m_s[5]);
            sum[6] = fiv_mm256_hsum_pd(m_s[6]);
            sum[7] = fiv_mm256_hsum_pd(m_s[7]);
#elif defined(FIV_USE_ARM_NEON)
            float64x2x2_t m_s[8];
            for (int r = 0; r < 8; r++) {
                m_s[r].val[0] = vdupq_n_f64(0);
                m_s[r].val[1] = vdupq_n_f64(0);
            }
            for (; k <= cols_a - 4; k += 4) {
                float64x2x2_t t_b1 = { { vld1q_f64(&ptr_lines_b[0][k]),
                                         vld1q_f64(&ptr_lines_b[0][k + 2]) } };
                float64x2x2_t t_b2 = { { vld1q_f64(&ptr_lines_b[1][k]),
                                         vld1q_f64(&ptr_lines_b[1][k + 2]) } };
                float64x2x2_t a0 = { { vld1q_f64(&ptr_lines_a[0][k]),
                                       vld1q_f64(&ptr_lines_a[0][k + 2]) } };
                float64x2x2_t a1 = { { vld1q_f64(&ptr_lines_a[1][k]),
                                       vld1q_f64(&ptr_lines_a[1][k + 2]) } };
                float64x2x2_t a2 = { { vld1q_f64(&ptr_lines_a[2][k]),
                                       vld1q_f64(&ptr_lines_a[2][k + 2]) } };
                float64x2x2_t a3 = { { vld1q_f64(&ptr_lines_a[3][k]),
                                       vld1q_f64(&ptr_lines_a[3][k + 2]) } };

                m_s[0].val[0] = vfmaq_f64(m_s[0].val[0], a0.val[0], t_b1.val[0]);
                m_s[0].val[1] = vfmaq_f64(m_s[0].val[1], a0.val[1], t_b1.val[1]);
                m_s[1].val[0] = vfmaq_f64(m_s[1].val[0], a1.val[0], t_b1.val[0]);
                m_s[1].val[1] = vfmaq_f64(m_s[1].val[1], a1.val[1], t_b1.val[1]);
                m_s[2].val[0] = vfmaq_f64(m_s[2].val[0], a2.val[0], t_b1.val[0]);
                m_s[2].val[1] = vfmaq_f64(m_s[2].val[1], a2.val[1], t_b1.val[1]);
                m_s[3].val[0] = vfmaq_f64(m_s[3].val[0], a3.val[0], t_b1.val[0]);
                m_s[3].val[1] = vfmaq_f64(m_s[3].val[1], a3.val[1], t_b1.val[1]);

                m_s[4].val[0] = vfmaq_f64(m_s[4].val[0], a0.val[0], t_b2.val[0]);
                m_s[4].val[1] = vfmaq_f64(m_s[4].val[1], a0.val[1], t_b2.val[1]);
                m_s[5].val[0] = vfmaq_f64(m_s[5].val[0], a1.val[0], t_b2.val[0]);
                m_s[5].val[1] = vfmaq_f64(m_s[5].val[1], a1.val[1], t_b2.val[1]);
                m_s[6].val[0] = vfmaq_f64(m_s[6].val[0], a2.val[0], t_b2.val[0]);
                m_s[6].val[1] = vfmaq_f64(m_s[6].val[1], a2.val[1], t_b2.val[1]);
                m_s[7].val[0] = vfmaq_f64(m_s[7].val[0], a3.val[0], t_b2.val[0]);
                m_s[7].val[1] = vfmaq_f64(m_s[7].val[1], a3.val[1], t_b2.val[1]);
            }
            vst1q_f64(sum,
                vpaddq_f64(vaddq_f64(m_s[0].val[0], m_s[0].val[1]),
                           vaddq_f64(m_s[1].val[0], m_s[1].val[1])));
            vst1q_f64(sum + 2,
                vpaddq_f64(vaddq_f64(m_s[2].val[0], m_s[2].val[1]),
                           vaddq_f64(m_s[3].val[0], m_s[3].val[1])));
            vst1q_f64(sum + 4,
                vpaddq_f64(vaddq_f64(m_s[4].val[0], m_s[4].val[1]),
                           vaddq_f64(m_s[5].val[0], m_s[5].val[1])));
            vst1q_f64(sum + 6,
                vpaddq_f64(vaddq_f64(m_s[6].val[0], m_s[6].val[1]),
                           vaddq_f64(m_s[7].val[0], m_s[7].val[1])));
#endif
            for (; k < cols_a; k++){
                sum[0] += ptr_lines_a[0][k] * ptr_lines_b[0][k];
                sum[1] += ptr_lines_a[1][k] * ptr_lines_b[0][k];
                sum[2] += ptr_lines_a[2][k] * ptr_lines_b[0][k];
                sum[3] += ptr_lines_a[3][k] * ptr_lines_b[0][k];
                sum[4] += ptr_lines_a[0][k] * ptr_lines_b[1][k];
                sum[5] += ptr_lines_a[1][k] * ptr_lines_b[1][k];
                sum[6] += ptr_lines_a[2][k] * ptr_lines_b[1][k];
                sum[7] += ptr_lines_a[3][k] * ptr_lines_b[1][k];
            }

            if (alpha != 1.0){
                sum[0] *= alpha;
                sum[1] *= alpha;
                sum[2] *= alpha;
                sum[3] *= alpha;
                sum[4] *= alpha;
                sum[5] *= alpha;
                sum[6] *= alpha;
                sum[7] *= alpha;
            }
            if (beta == 0.0){
                ptr_lines_c[0][j] = sum[0];
                ptr_lines_c[1][j] = sum[1];
                ptr_lines_c[2][j] = sum[2];
                ptr_lines_c[3][j] = sum[3];
                ptr_lines_c[0][j + 1] = sum[4];
                ptr_lines_c[1][j + 1] = sum[5];
                ptr_lines_c[2][j + 1] = sum[6];
                ptr_lines_c[3][j + 1] = sum[7];
            }  else {
                ptr_lines_c[0][j] = beta * ptr_lines_c[0][j] + sum[0];
                ptr_lines_c[1][j] = beta * ptr_lines_c[1][j] + sum[1];
                ptr_lines_c[2][j] = beta * ptr_lines_c[2][j] + sum[2];
                ptr_lines_c[3][j] = beta * ptr_lines_c[3][j] + sum[3];
                ptr_lines_c[0][j + 1] = beta * ptr_lines_c[0][j + 1] + sum[4];
                ptr_lines_c[1][j + 1] = beta * ptr_lines_c[1][j + 1] + sum[5];
                ptr_lines_c[2][j + 1] = beta * ptr_lines_c[2][j + 1] + sum[6];
                ptr_lines_c[3][j + 1] = beta * ptr_lines_c[3][j + 1] + sum[7];
            }
        }

        for (; j < rows_b; j++){
            ivf64* ptr_line_b = &data_b[j * stride_b];
            ivf64 sum[4] = { 0 };
            int k = 0;
#if defined(FIV_USE_AVX2)
            __m256d m_s[4] = { 0 };
            for (; k <= cols_a - 4; k += 4){
                __m256d t_b = _mm256_loadu_pd(&ptr_line_b[k]);
                m_s[0] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[0][k]), t_b, m_s[0]);
                m_s[1] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[1][k]), t_b, m_s[1]);
                m_s[2] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[2][k]), t_b, m_s[2]);
                m_s[3] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[3][k]), t_b, m_s[3]);
            }
            sum[0] = fiv_mm256_hsum_pd(m_s[0]);
            sum[1] = fiv_mm256_hsum_pd(m_s[1]);
            sum[2] = fiv_mm256_hsum_pd(m_s[2]);
            sum[3] = fiv_mm256_hsum_pd(m_s[3]);
#elif defined(FIV_USE_ARM_NEON)
            float64x2x2_t m_s[4];
            for (int r = 0; r < 4; r++) {
                m_s[r].val[0] = vdupq_n_f64(0);
                m_s[r].val[1] = vdupq_n_f64(0);
            }
            for (; k <= cols_a - 4; k += 4){
                float64x2x2_t t_b = { { vld1q_f64(&ptr_line_b[k]),
                                        vld1q_f64(&ptr_line_b[k + 2]) } };
                float64x2x2_t a0 = { { vld1q_f64(&ptr_lines_a[0][k]),
                                       vld1q_f64(&ptr_lines_a[0][k + 2]) } };
                float64x2x2_t a1 = { { vld1q_f64(&ptr_lines_a[1][k]),
                                       vld1q_f64(&ptr_lines_a[1][k + 2]) } };
                float64x2x2_t a2 = { { vld1q_f64(&ptr_lines_a[2][k]),
                                       vld1q_f64(&ptr_lines_a[2][k + 2]) } };
                float64x2x2_t a3 = { { vld1q_f64(&ptr_lines_a[3][k]),
                                       vld1q_f64(&ptr_lines_a[3][k + 2]) } };

                m_s[0].val[0] = vfmaq_f64(m_s[0].val[0], a0.val[0], t_b.val[0]);
                m_s[0].val[1] = vfmaq_f64(m_s[0].val[1], a0.val[1], t_b.val[1]);
                m_s[1].val[0] = vfmaq_f64(m_s[1].val[0], a1.val[0], t_b.val[0]);
                m_s[1].val[1] = vfmaq_f64(m_s[1].val[1], a1.val[1], t_b.val[1]);
                m_s[2].val[0] = vfmaq_f64(m_s[2].val[0], a2.val[0], t_b.val[0]);
                m_s[2].val[1] = vfmaq_f64(m_s[2].val[1], a2.val[1], t_b.val[1]);
                m_s[3].val[0] = vfmaq_f64(m_s[3].val[0], a3.val[0], t_b.val[0]);
                m_s[3].val[1] = vfmaq_f64(m_s[3].val[1], a3.val[1], t_b.val[1]);
            }
            vst1q_f64(sum,
                vpaddq_f64(vaddq_f64(m_s[0].val[0], m_s[0].val[1]),
                           vaddq_f64(m_s[1].val[0], m_s[1].val[1])));
            vst1q_f64(sum + 2,
                vpaddq_f64(vaddq_f64(m_s[2].val[0], m_s[2].val[1]),
                           vaddq_f64(m_s[3].val[0], m_s[3].val[1])));
#endif
            for (; k < cols_a; k++){
                sum[0] += ptr_lines_a[0][k] * ptr_line_b[k];
                sum[1] += ptr_lines_a[1][k] * ptr_line_b[k];
                sum[2] += ptr_lines_a[2][k] * ptr_line_b[k];
                sum[3] += ptr_lines_a[3][k] * ptr_line_b[k];
            }

            if (beta == 0.0){
                ptr_lines_c[0][j] = alpha * sum[0];
                ptr_lines_c[1][j] = alpha * sum[1];
                ptr_lines_c[2][j] = alpha * sum[2];
                ptr_lines_c[3][j] = alpha * sum[3];
            }  else {
                ptr_lines_c[0][j] = beta * ptr_lines_c[0][j] + alpha * sum[0];
                ptr_lines_c[1][j] = beta * ptr_lines_c[1][j] + alpha * sum[1];
                ptr_lines_c[2][j] = beta * ptr_lines_c[2][j] + alpha * sum[2];
                ptr_lines_c[3][j] = beta * ptr_lines_c[3][j] + alpha * sum[3];
            }
        }
    }
    for (; i < rows_a; i++) {
        ivf64* ptr_line_a = &data_a[i * stride_a];
        ivf64* ptr_line_c = &data_c[i * stride_c];
        for (int j = 0; j < rows_b; j++){
            ivf64* ptr_line_b = &data_b[j * stride_b];
            ivf64 sum = 0;
            for (int k = 0; k < cols_a; k++) {
                sum += ptr_line_a[k] * ptr_line_b[k];
            }
            ptr_line_c[j] = beta * ptr_line_c[j] + alpha * sum;
        }
    }
}

/* C = A^T * B, A: rows_a x cols_a, B: rows_a x cols_b, result cols_a x cols_b */
static void fiv_small_matrix_t_mul_matrix_real64(
    ivf64* data_a, int rows_a, int cols_a, int stride_a,
    ivf64* data_b, int cols_b, int stride_b,
    ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta)
{
    int i = 0;
    for (; i <= cols_a - 4; i += 4){
        ivf64* ptr_c_lines[4];
        ptr_c_lines[0] = &data_c[i * stride_c];
        ptr_c_lines[1] = ptr_c_lines[0] + stride_c;
        ptr_c_lines[2] = ptr_c_lines[1] + stride_c;
        ptr_c_lines[3] = ptr_c_lines[2] + stride_c;

        if (beta == 0.0){
            memset(ptr_c_lines[0], 0, sizeof(ivf64) * cols_b);
            memset(ptr_c_lines[1], 0, sizeof(ivf64) * cols_b);
            memset(ptr_c_lines[2], 0, sizeof(ivf64) * cols_b);
            memset(ptr_c_lines[3], 0, sizeof(ivf64) * cols_b);
        }	else if (beta != 1.0) {
            for (int l = 0; l < cols_b; l++){
                ptr_c_lines[0][l] *= beta;
                ptr_c_lines[1][l] *= beta;
                ptr_c_lines[2][l] *= beta;
                ptr_c_lines[3][l] *= beta;
            }
        }
        int k = 0;
        for (; k <= rows_a - 2; k += 2){
            ivf64* ptr_data_a_tmp = &data_a[k * stride_a + i];
            ivf64 t_a[8] = {ptr_data_a_tmp[0],ptr_data_a_tmp[1], ptr_data_a_tmp[2], ptr_data_a_tmp[3],
            ptr_data_a_tmp[0 + stride_a], ptr_data_a_tmp[1 + stride_a],
            ptr_data_a_tmp[2 + stride_a], ptr_data_a_tmp[3 + stride_a]};
            if (alpha != 1.0) {
                t_a[0] *= alpha; t_a[1] *= alpha;
                t_a[2] *= alpha; t_a[3] *= alpha;
                t_a[4] *= alpha; t_a[5] *= alpha;
                t_a[6] *= alpha; t_a[7] *= alpha;
            }
            ivf64* ptr_b_line1 = &data_b[k *stride_b];
            ivf64* ptr_b_line2 = &data_b[k *stride_b + stride_b];
            int j = 0;
#if defined(FIV_USE_AVX2)
            __m256d m_t_a1 = _mm256_broadcast_sd(&t_a[0]);
            __m256d m_t_a2 = _mm256_broadcast_sd(&t_a[1]);
            __m256d m_t_a3 = _mm256_broadcast_sd(&t_a[2]);
            __m256d m_t_a4 = _mm256_broadcast_sd(&t_a[3]);
            __m256d m_t_a5 = _mm256_broadcast_sd(&t_a[4]);
            __m256d m_t_a6 = _mm256_broadcast_sd(&t_a[5]);
            __m256d m_t_a7 = _mm256_broadcast_sd(&t_a[6]);
            __m256d m_t_a8 = _mm256_broadcast_sd(&t_a[7]);

            for (; j <= cols_b - 4; j += 4){
                __m256d m_t_c0 = _mm256_loadu_pd(&ptr_c_lines[0][j]);
                __m256d m_t_c1 = _mm256_loadu_pd(&ptr_c_lines[1][j]);
                __m256d m_t_c2 = _mm256_loadu_pd(&ptr_c_lines[2][j]);
                __m256d m_t_c3 = _mm256_loadu_pd(&ptr_c_lines[3][j]);

                __m256d m_t_b1 = _mm256_loadu_pd(&ptr_b_line1[j]);
                __m256d m_t_b2 = _mm256_loadu_pd(&ptr_b_line2[j]);

                m_t_c0 = _mm256_fmadd_pd(m_t_a1, m_t_b1, m_t_c0);
                m_t_c1 = _mm256_fmadd_pd(m_t_a2, m_t_b1, m_t_c1);
                m_t_c2 = _mm256_fmadd_pd(m_t_a3, m_t_b1, m_t_c2);
                m_t_c3 = _mm256_fmadd_pd(m_t_a4, m_t_b1, m_t_c3);

                m_t_c0 = _mm256_fmadd_pd(m_t_a5, m_t_b2, m_t_c0);
                m_t_c1 = _mm256_fmadd_pd(m_t_a6, m_t_b2, m_t_c1);
                m_t_c2 = _mm256_fmadd_pd(m_t_a7, m_t_b2, m_t_c2);
                m_t_c3 = _mm256_fmadd_pd(m_t_a8, m_t_b2, m_t_c3);

                _mm256_storeu_pd(&ptr_c_lines[0][j], m_t_c0);
                _mm256_storeu_pd(&ptr_c_lines[1][j], m_t_c1);
                _mm256_storeu_pd(&ptr_c_lines[2][j], m_t_c2);
                _mm256_storeu_pd(&ptr_c_lines[3][j], m_t_c3);

            }
#elif defined(FIV_USE_ARM_NEON)
            for (; j <= cols_b - 4; j += 4){
                float64x2x2_t b1 = { { vld1q_f64(&ptr_b_line1[j]),     vld1q_f64(&ptr_b_line1[j + 2]) } };
                float64x2x2_t b2 = { { vld1q_f64(&ptr_b_line2[j]),     vld1q_f64(&ptr_b_line2[j + 2]) } };
                float64x2x2_t c0 = { { vld1q_f64(&ptr_c_lines[0][j]), vld1q_f64(&ptr_c_lines[0][j + 2]) } };
                float64x2x2_t c1 = { { vld1q_f64(&ptr_c_lines[1][j]), vld1q_f64(&ptr_c_lines[1][j + 2]) } };
                float64x2x2_t c2 = { { vld1q_f64(&ptr_c_lines[2][j]), vld1q_f64(&ptr_c_lines[2][j + 2]) } };
                float64x2x2_t c3 = { { vld1q_f64(&ptr_c_lines[3][j]), vld1q_f64(&ptr_c_lines[3][j + 2]) } };

                c0.val[0] = vfmaq_n_f64(c0.val[0], b1.val[0], t_a[0]);
                c0.val[1] = vfmaq_n_f64(c0.val[1], b1.val[1], t_a[0]);
                c1.val[0] = vfmaq_n_f64(c1.val[0], b1.val[0], t_a[1]);
                c1.val[1] = vfmaq_n_f64(c1.val[1], b1.val[1], t_a[1]);
                c2.val[0] = vfmaq_n_f64(c2.val[0], b1.val[0], t_a[2]);
                c2.val[1] = vfmaq_n_f64(c2.val[1], b1.val[1], t_a[2]);
                c3.val[0] = vfmaq_n_f64(c3.val[0], b1.val[0], t_a[3]);
                c3.val[1] = vfmaq_n_f64(c3.val[1], b1.val[1], t_a[3]);

                c0.val[0] = vfmaq_n_f64(c0.val[0], b2.val[0], t_a[4]);
                c0.val[1] = vfmaq_n_f64(c0.val[1], b2.val[1], t_a[4]);
                c1.val[0] = vfmaq_n_f64(c1.val[0], b2.val[0], t_a[5]);
                c1.val[1] = vfmaq_n_f64(c1.val[1], b2.val[1], t_a[5]);
                c2.val[0] = vfmaq_n_f64(c2.val[0], b2.val[0], t_a[6]);
                c2.val[1] = vfmaq_n_f64(c2.val[1], b2.val[1], t_a[6]);
                c3.val[0] = vfmaq_n_f64(c3.val[0], b2.val[0], t_a[7]);
                c3.val[1] = vfmaq_n_f64(c3.val[1], b2.val[1], t_a[7]);

                vst1q_f64(&ptr_c_lines[0][j],     c0.val[0]);
                vst1q_f64(&ptr_c_lines[0][j + 2], c0.val[1]);
                vst1q_f64(&ptr_c_lines[1][j],     c1.val[0]);
                vst1q_f64(&ptr_c_lines[1][j + 2], c1.val[1]);
                vst1q_f64(&ptr_c_lines[2][j],     c2.val[0]);
                vst1q_f64(&ptr_c_lines[2][j + 2], c2.val[1]);
                vst1q_f64(&ptr_c_lines[3][j],     c3.val[0]);
                vst1q_f64(&ptr_c_lines[3][j + 2], c3.val[1]);
            }
#endif
            for (; j < cols_b; j++){
                ptr_c_lines[0][j] += t_a[0] * ptr_b_line1[j];
                ptr_c_lines[0][j] += t_a[4] * ptr_b_line2[j];

                ptr_c_lines[1][j] += t_a[1] * ptr_b_line1[j];
                ptr_c_lines[1][j] += t_a[5] * ptr_b_line2[j];

                ptr_c_lines[2][j] += t_a[2] * ptr_b_line1[j];
                ptr_c_lines[2][j] += t_a[6] * ptr_b_line2[j];

                ptr_c_lines[3][j] += t_a[3] * ptr_b_line1[j];
                ptr_c_lines[3][j] += t_a[7] * ptr_b_line2[j];
            }
        }

        for (; k < rows_a; k++){
            ivf64 t_a1 = data_a[k * stride_a + i + 0];
            ivf64 t_a2 = data_a[k * stride_a + i + 1];
            ivf64 t_a3 = data_a[k * stride_a + i + 2];
            ivf64 t_a4 = data_a[k * stride_a + i + 3];
            if (alpha != 1.0){
                t_a1 *= alpha; t_a2 *= alpha;
                t_a3 *= alpha; t_a4 *= alpha;
            }
            ivf64* ptr_b_line = &data_b[k * stride_b];
            int j = 0;
#if defined(FIV_USE_AVX2)
            __m256d m_t_a1 = _mm256_broadcast_sd(&t_a1);
            __m256d m_t_a2 = _mm256_broadcast_sd(&t_a2);
            __m256d m_t_a3 = _mm256_broadcast_sd(&t_a3);
            __m256d m_t_a4 = _mm256_broadcast_sd(&t_a4);

            for (; j <= cols_b - 4; j += 4){
                __m256d m_t_c0 = _mm256_loadu_pd(&ptr_c_lines[0][j]);
                __m256d m_t_c1 = _mm256_loadu_pd(&ptr_c_lines[1][j]);
                __m256d m_t_c2 = _mm256_loadu_pd(&ptr_c_lines[2][j]);
                __m256d m_t_c3 = _mm256_loadu_pd(&ptr_c_lines[3][j]);
                __m256d m_t_b = _mm256_loadu_pd(&ptr_b_line[j]);

                m_t_c0 = _mm256_fmadd_pd(m_t_a1, m_t_b, m_t_c0);
                m_t_c1 = _mm256_fmadd_pd(m_t_a2, m_t_b, m_t_c1);
                m_t_c2 = _mm256_fmadd_pd(m_t_a3, m_t_b, m_t_c2);
                m_t_c3 = _mm256_fmadd_pd(m_t_a4, m_t_b, m_t_c3);

                _mm256_storeu_pd(&ptr_c_lines[0][j], m_t_c0);
                _mm256_storeu_pd(&ptr_c_lines[1][j], m_t_c1);
                _mm256_storeu_pd(&ptr_c_lines[2][j], m_t_c2);
                _mm256_storeu_pd(&ptr_c_lines[3][j], m_t_c3);

            }
#elif defined(FIV_USE_ARM_NEON)
            for (; j <= cols_b - 4; j += 4){
                float64x2x2_t b  = { { vld1q_f64(&ptr_b_line[j]),     vld1q_f64(&ptr_b_line[j + 2]) } };
                float64x2x2_t c0 = { { vld1q_f64(&ptr_c_lines[0][j]), vld1q_f64(&ptr_c_lines[0][j + 2]) } };
                float64x2x2_t c1 = { { vld1q_f64(&ptr_c_lines[1][j]), vld1q_f64(&ptr_c_lines[1][j + 2]) } };
                float64x2x2_t c2 = { { vld1q_f64(&ptr_c_lines[2][j]), vld1q_f64(&ptr_c_lines[2][j + 2]) } };
                float64x2x2_t c3 = { { vld1q_f64(&ptr_c_lines[3][j]), vld1q_f64(&ptr_c_lines[3][j + 2]) } };

                c0.val[0] = vfmaq_n_f64(c0.val[0], b.val[0], t_a1);
                c0.val[1] = vfmaq_n_f64(c0.val[1], b.val[1], t_a1);
                c1.val[0] = vfmaq_n_f64(c1.val[0], b.val[0], t_a2);
                c1.val[1] = vfmaq_n_f64(c1.val[1], b.val[1], t_a2);
                c2.val[0] = vfmaq_n_f64(c2.val[0], b.val[0], t_a3);
                c2.val[1] = vfmaq_n_f64(c2.val[1], b.val[1], t_a3);
                c3.val[0] = vfmaq_n_f64(c3.val[0], b.val[0], t_a4);
                c3.val[1] = vfmaq_n_f64(c3.val[1], b.val[1], t_a4);

                vst1q_f64(&ptr_c_lines[0][j],     c0.val[0]);
                vst1q_f64(&ptr_c_lines[0][j + 2], c0.val[1]);
                vst1q_f64(&ptr_c_lines[1][j],     c1.val[0]);
                vst1q_f64(&ptr_c_lines[1][j + 2], c1.val[1]);
                vst1q_f64(&ptr_c_lines[2][j],     c2.val[0]);
                vst1q_f64(&ptr_c_lines[2][j + 2], c2.val[1]);
                vst1q_f64(&ptr_c_lines[3][j],     c3.val[0]);
                vst1q_f64(&ptr_c_lines[3][j + 2], c3.val[1]);
            }
#endif
            for (; j < cols_b; j++){
                ptr_c_lines[0][j] += t_a1 * ptr_b_line[j];
                ptr_c_lines[1][j] += t_a2 * ptr_b_line[j];
                ptr_c_lines[2][j] += t_a3 * ptr_b_line[j];
                ptr_c_lines[3][j] += t_a4 * ptr_b_line[j];
            }
        }
    }

    for (; i < cols_a; i++){
        ivf64* ptr_c_line = &data_c[i * stride_c];
        if (beta == 0.0){
            memset(ptr_c_line, 0, sizeof(ivf64) * cols_b);
        }	else if(beta != 1.0){
            for (int k = 0; k < cols_b; k++){
                ptr_c_line[k] *= beta;
            }
        }
        for (int k = 0; k < rows_a; k++){
            ivf64 t_a = data_a[k * stride_a + i] * alpha;
            ivf64* ptr_b_line = &data_b[k * stride_b];
            for (int j = 0; j < cols_b; j++) {
                ptr_c_line[j] += t_a * ptr_b_line[j];
            }
        }
    }
}

/* C = A^T * B^T, A: rows_a x cols_a, B: rows_b x cols_a, result cols_a x rows_b */
static void fiv_small_matrix_t_mul_matrix_t_real64(
    ivf64* data_a, int rows_a, int cols_a, int stride_a,
    ivf64* data_b, int rows_b, int stride_b,
    ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta)
{
    int i = 0;
    ivf64* ptr_mem_a = (ivf64*)fiv_malloc(sizeof(ivf64) * 4 * rows_a);
    if (ptr_mem_a == NULL) {
        printf("MEM ERROR\n");
        return;
    }
    for (; i <= cols_a - 4; i += 4){
        ivf64* ptr_c_lines[4];
        ptr_c_lines[0] = &data_c[i * stride_c];
        ptr_c_lines[1] = ptr_c_lines[0] + stride_c;
        ptr_c_lines[2] = ptr_c_lines[1] + stride_c;
        ptr_c_lines[3] = ptr_c_lines[2] + stride_c;
        ivf64* ptr_a_i_col = &data_a[i];
        for (int k = 0; k < rows_a; k++) {
            ptr_mem_a[4 * k + 0] = ptr_a_i_col[k * stride_a + 0];
            ptr_mem_a[4 * k + 1] = ptr_a_i_col[k * stride_a + 1];
            ptr_mem_a[4 * k + 2] = ptr_a_i_col[k * stride_a + 2];
            ptr_mem_a[4 * k + 3] = ptr_a_i_col[k * stride_a + 3];
        }

        int j = 0;
        for (; j <= rows_b - 2; j += 2) {
            ivf64* ptr_a = ptr_mem_a;
            ivf64* ptr_b[2] = {&data_b[j * stride_b], &data_b[(j + 1) * stride_b]};
            ivf64 FIV_DALIGNED sum[8] = { 0 };
            int k = 0;
#if defined(FIV_USE_AVX2)
            __m256d m_sum[8] = { 0 };
            for (; k <= rows_a - 4; k += 4) {
                m_sum[0] = _mm256_fmadd_pd(_mm256_loadu_pd(ptr_a), _mm256_broadcast_sd(&ptr_b[0][k]), m_sum[0]);
                m_sum[4] = _mm256_fmadd_pd(_mm256_loadu_pd(ptr_a), _mm256_broadcast_sd(&ptr_b[1][k]), m_sum[4]);

                m_sum[1] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_a[4]), _mm256_broadcast_sd(&ptr_b[0][k + 1]), m_sum[1]);
                m_sum[5] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_a[4]), _mm256_broadcast_sd(&ptr_b[1][k + 1]), m_sum[5]);

                m_sum[2] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_a[8]), _mm256_broadcast_sd(&ptr_b[0][k + 2]), m_sum[2]);
                m_sum[6] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_a[8]), _mm256_broadcast_sd(&ptr_b[1][k + 2]), m_sum[6]);

                m_sum[3] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_a[12]), _mm256_broadcast_sd(&ptr_b[0][k + 3]), m_sum[3]);
                m_sum[7] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_a[12]), _mm256_broadcast_sd(&ptr_b[1][k + 3]), m_sum[7]);
                ptr_a += 16;
            }
            m_sum[0] = _mm256_add_pd(m_sum[0], m_sum[1]);
            m_sum[2] = _mm256_add_pd(m_sum[2], m_sum[3]);
            m_sum[0] = _mm256_add_pd(m_sum[0], m_sum[2]);

            m_sum[4] = _mm256_add_pd(m_sum[4], m_sum[5]);
            m_sum[6] = _mm256_add_pd(m_sum[6], m_sum[7]);
            m_sum[4] = _mm256_add_pd(m_sum[4], m_sum[6]);

            for (; k < rows_a; k++){
                m_sum[0] = _mm256_fmadd_pd(_mm256_loadu_pd(ptr_a), _mm256_broadcast_sd(&ptr_b[0][k]), m_sum[0]);
                m_sum[4] = _mm256_fmadd_pd(_mm256_loadu_pd(ptr_a), _mm256_broadcast_sd(&ptr_b[1][k]), m_sum[4]);
                ptr_a += 4;
            }
            _mm256_storeu_pd(sum,    m_sum[0]);
            _mm256_storeu_pd(sum + 4, m_sum[4]);
#elif defined(FIV_USE_ARM_NEON)
            float64x2_t s0a = vdupq_n_f64(0), s0b = vdupq_n_f64(0);
            float64x2_t s1a = vdupq_n_f64(0), s1b = vdupq_n_f64(0);
            for (; k <= rows_a - 4; k += 4) {
                float64x2x2_t a0 = { { vld1q_f64(&ptr_mem_a[4 * k]),       vld1q_f64(&ptr_mem_a[4 * k + 2]) } };
                float64x2x2_t a1 = { { vld1q_f64(&ptr_mem_a[4 * (k + 1)]), vld1q_f64(&ptr_mem_a[4 * (k + 1) + 2]) } };
                float64x2x2_t a2 = { { vld1q_f64(&ptr_mem_a[4 * (k + 2)]), vld1q_f64(&ptr_mem_a[4 * (k + 2) + 2]) } };
                float64x2x2_t a3 = { { vld1q_f64(&ptr_mem_a[4 * (k + 3)]), vld1q_f64(&ptr_mem_a[4 * (k + 3) + 2]) } };

                s0a = vfmaq_n_f64(s0a, a0.val[0], ptr_b[0][k]);
                s0b = vfmaq_n_f64(s0b, a0.val[1], ptr_b[0][k]);
                s1a = vfmaq_n_f64(s1a, a0.val[0], ptr_b[1][k]);
                s1b = vfmaq_n_f64(s1b, a0.val[1], ptr_b[1][k]);

                s0a = vfmaq_n_f64(s0a, a1.val[0], ptr_b[0][k + 1]);
                s0b = vfmaq_n_f64(s0b, a1.val[1], ptr_b[0][k + 1]);
                s1a = vfmaq_n_f64(s1a, a1.val[0], ptr_b[1][k + 1]);
                s1b = vfmaq_n_f64(s1b, a1.val[1], ptr_b[1][k + 1]);

                s0a = vfmaq_n_f64(s0a, a2.val[0], ptr_b[0][k + 2]);
                s0b = vfmaq_n_f64(s0b, a2.val[1], ptr_b[0][k + 2]);
                s1a = vfmaq_n_f64(s1a, a2.val[0], ptr_b[1][k + 2]);
                s1b = vfmaq_n_f64(s1b, a2.val[1], ptr_b[1][k + 2]);

                s0a = vfmaq_n_f64(s0a, a3.val[0], ptr_b[0][k + 3]);
                s0b = vfmaq_n_f64(s0b, a3.val[1], ptr_b[0][k + 3]);
                s1a = vfmaq_n_f64(s1a, a3.val[0], ptr_b[1][k + 3]);
                s1b = vfmaq_n_f64(s1b, a3.val[1], ptr_b[1][k + 3]);
            }
            for (; k < rows_a; k++){
                float64x2x2_t a0 = { { vld1q_f64(&ptr_mem_a[4 * k]),       vld1q_f64(&ptr_mem_a[4 * k + 2]) } };
                s0a = vfmaq_n_f64(s0a, a0.val[0], ptr_b[0][k]);
                s0b = vfmaq_n_f64(s0b, a0.val[1], ptr_b[0][k]);
                s1a = vfmaq_n_f64(s1a, a0.val[0], ptr_b[1][k]);
                s1b = vfmaq_n_f64(s1b, a0.val[1], ptr_b[1][k]);
            }
            vst1q_f64(sum, s0a);
            vst1q_f64(sum + 2, s0b);
            vst1q_f64(sum + 4, s1a);
            vst1q_f64(sum + 6, s1b);
#endif
            for (; k < rows_a; k++){
                sum[0] += ptr_a[0] * ptr_b[0][k];
                sum[1] += ptr_a[1] * ptr_b[0][k];
                sum[2] += ptr_a[2] * ptr_b[0][k];
                sum[3] += ptr_a[3] * ptr_b[0][k];

                sum[4]  += ptr_a[0] * ptr_b[1][k];
                sum[5]  += ptr_a[1] * ptr_b[1][k];
                sum[6]  += ptr_a[2] * ptr_b[1][k];
                sum[7]  += ptr_a[3] * ptr_b[1][k];
                ptr_a += 4;
            }

            sum[0] *= alpha; sum[1] *= alpha;
            sum[2] *= alpha; sum[3] *= alpha;
            sum[4] *= alpha; sum[5] *= alpha;
            sum[6] *= alpha; sum[7] *= alpha;

            if (beta == 0.0){
                ptr_c_lines[0][j + 0] = sum[0];
                ptr_c_lines[0][j + 1] = sum[4];
                ptr_c_lines[1][j + 0] = sum[1];
                ptr_c_lines[1][j + 1] = sum[5];
                ptr_c_lines[2][j + 0] = sum[2];
                ptr_c_lines[2][j + 1] = sum[6];
                ptr_c_lines[3][j + 0] = sum[3];
                ptr_c_lines[3][j + 1] = sum[7];
            }	else {
                ptr_c_lines[0][j + 0] = beta * ptr_c_lines[0][j + 0] + sum[0];
                ptr_c_lines[0][j + 1] = beta * ptr_c_lines[0][j + 1] + sum[4];
                ptr_c_lines[1][j + 0] = beta * ptr_c_lines[1][j + 0] + sum[1];
                ptr_c_lines[1][j + 1] = beta * ptr_c_lines[1][j + 1] + sum[5];
                ptr_c_lines[2][j + 0] = beta * ptr_c_lines[2][j + 0] + sum[2];
                ptr_c_lines[2][j + 1] = beta * ptr_c_lines[2][j + 1] + sum[6];
                ptr_c_lines[3][j + 0] = beta * ptr_c_lines[3][j + 0] + sum[3];
                ptr_c_lines[3][j + 1] = beta * ptr_c_lines[3][j + 1] + sum[7];
            }
        }

        for (; j < rows_b; j++){
            ivf64* ptr_a = ptr_mem_a;
            ivf64* ptr_b = &data_b[j * stride_b];
            ivf64 FIV_DALIGNED sum[4] = { 0 };
            int k = 0;
#if defined(FIV_USE_AVX2)
            __m256d m_sum[4] = { 0 };
            for (; k <= rows_a - 4; k += 4) {
                m_sum[0] = _mm256_fmadd_pd(_mm256_loadu_pd(ptr_a), _mm256_broadcast_sd(&ptr_b[k]), m_sum[0]);
                m_sum[1] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_a[4]), _mm256_broadcast_sd(&ptr_b[k + 1]), m_sum[1]);
                m_sum[2] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_a[8]), _mm256_broadcast_sd(&ptr_b[k + 2]), m_sum[2]);
                m_sum[3] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_a[12]), _mm256_broadcast_sd(&ptr_b[k + 3]), m_sum[3]);
                ptr_a += 16;
            }
            m_sum[0] = _mm256_add_pd(m_sum[0], m_sum[1]);
            m_sum[2] = _mm256_add_pd(m_sum[2], m_sum[3]);
            m_sum[0] = _mm256_add_pd(m_sum[0], m_sum[2]);

            for (; k < rows_a; k++){
                m_sum[0] = _mm256_fmadd_pd(_mm256_loadu_pd(ptr_a), _mm256_broadcast_sd(&ptr_b[k]), m_sum[0]);
                ptr_a += 4;
            }
            _mm256_storeu_pd(sum, m_sum[0]);
#elif defined(FIV_USE_ARM_NEON)
            float64x2_t s0a = vdupq_n_f64(0), s0b = vdupq_n_f64(0);
            for (; k <= rows_a - 4; k += 4) {
                float64x2x2_t a0 = { { vld1q_f64(&ptr_mem_a[4 * k]),       vld1q_f64(&ptr_mem_a[4 * k + 2]) } };
                float64x2x2_t a1 = { { vld1q_f64(&ptr_mem_a[4 * (k + 1)]), vld1q_f64(&ptr_mem_a[4 * (k + 1) + 2]) } };
                float64x2x2_t a2 = { { vld1q_f64(&ptr_mem_a[4 * (k + 2)]), vld1q_f64(&ptr_mem_a[4 * (k + 2) + 2]) } };
                float64x2x2_t a3 = { { vld1q_f64(&ptr_mem_a[4 * (k + 3)]), vld1q_f64(&ptr_mem_a[4 * (k + 3) + 2]) } };

                s0a = vfmaq_n_f64(s0a, a0.val[0], ptr_b[k]);
                s0b = vfmaq_n_f64(s0b, a0.val[1], ptr_b[k]);
                s0a = vfmaq_n_f64(s0a, a1.val[0], ptr_b[k + 1]);
                s0b = vfmaq_n_f64(s0b, a1.val[1], ptr_b[k + 1]);
                s0a = vfmaq_n_f64(s0a, a2.val[0], ptr_b[k + 2]);
                s0b = vfmaq_n_f64(s0b, a2.val[1], ptr_b[k + 2]);
                s0a = vfmaq_n_f64(s0a, a3.val[0], ptr_b[k + 3]);
                s0b = vfmaq_n_f64(s0b, a3.val[1], ptr_b[k + 3]);
            }
            for (; k < rows_a; k++){
                float64x2x2_t a0 = { { vld1q_f64(&ptr_mem_a[4 * k]),       vld1q_f64(&ptr_mem_a[4 * k + 2]) } };
                s0a = vfmaq_n_f64(s0a, a0.val[0], ptr_b[k]);
                s0b = vfmaq_n_f64(s0b, a0.val[1], ptr_b[k]);
            }
            vst1q_f64(sum, s0a);
            vst1q_f64(sum + 2, s0b);
#endif
            for (; k < rows_a; k++){
                sum[0] += ptr_a[0] * ptr_b[k];
                sum[1] += ptr_a[1] * ptr_b[k];
                sum[2] += ptr_a[2] * ptr_b[k];
                sum[3] += ptr_a[3] * ptr_b[k];
                ptr_a += 4;
            }

            sum[0] *= alpha; sum[1] *= alpha;
            sum[2] *= alpha; sum[3] *= alpha;

            if (beta == 0.0){
                ptr_c_lines[0][j] = sum[0];
                ptr_c_lines[1][j] = sum[1];
                ptr_c_lines[2][j] = sum[2];
                ptr_c_lines[3][j] = sum[3];
            }	else {
                ptr_c_lines[0][j] = beta * ptr_c_lines[0][j] + sum[0];
                ptr_c_lines[1][j] = beta * ptr_c_lines[1][j] + sum[1];
                ptr_c_lines[2][j] = beta * ptr_c_lines[2][j] + sum[2];
                ptr_c_lines[3][j] = beta * ptr_c_lines[3][j] + sum[3];
            }
        }
    }
    for (; i < cols_a; i++){
        ivf64* ptr_c_line = &data_c[i * stride_c];
        ivf64* ptr_a_i_col = &data_a[i];
        for (int k = 0; k < rows_a; k++) {
            ptr_mem_a[k] = ptr_a_i_col[k * stride_a];
        }
        for (int j = 0; j < rows_b; j++) {
            ivf64* ptr_a = ptr_mem_a;
            ivf64* ptr_b = &data_b[j * stride_b];
            ivf64 sum = 0;
            for (int k = 0; k < rows_a; k++) {
                sum += ptr_a[k] * ptr_b[k];
            }
            ptr_c_line[j] = beta * ptr_c_line[j] + alpha * sum;
        }
    }
    fiv_free(ptr_mem_a);
}

/* ============================================================================
   Blocked matrix multiplication (ivf64). Panel packing and the 8x8 micro-kernel
   mirror the float32 blocked path; the micro-kernel uses float64x2 NEON /
   __m256d AVX2 instead of the float32 SIMD.
   ========================================================================== */

typedef void(*ptr_func_mat_mul_mxkxn_kernel_db)
    (int kc, ivf64 alpha, ivf64* a, ivf64* b, ivf64 beta, ivf64* c, int inc_row_c, int inc_col_c);

#ifndef FIV_L3_CACHE_BYTES
#define FIV_L3_CACHE_BYTES (8 * 1024 * 1024)
#endif
#if _DEBUG
#define FIV_M_BLOCK_DB 32
#define FIV_K_BLOCK_DB 32
#define FIV_N_BLOCK_DB 32
#else
#define FIV_M_BLOCK_DB 512
#define FIV_N_BLOCK_DB 480
#define FIV_BLOCK_K_CALC_DB() \
    ((((FIV_L3_CACHE_BYTES / 2) / (8 * (FIV_M_BLOCK_DB + FIV_N_BLOCK_DB))) / 128) * 128)
#define FIV_K_BLOCK_DB (FIV_BLOCK_K_CALC_DB() < 64 ? 64 : FIV_BLOCK_K_CALC_DB())
#endif

static void copy_mrxk_blocked_db(
    int k, int kernel_m, ivf64* a, int inc_row_a, int inc_col_a, ivf64* buffer)
{
    int i, j;
    for (j = 0; j < k; j++) {
        for (i = 0; i < kernel_m; i++) buffer[i] = a[i * inc_row_a];
        buffer += kernel_m;
        a += inc_col_a;
    }
}

static void copy_a_blocked_db(int mc, int kc, int kernel_m, ivf64* a, int inc_row_a, int inc_col_a, ivf64* buffer)
{
    int mp = mc / kernel_m;
    int mr = mc % kernel_m;
    int i, j;
    for (i = 0; i < mp; i++) {
        copy_mrxk_blocked_db(kc, kernel_m, a, inc_row_a, inc_col_a, buffer);
        buffer += kc * kernel_m;
        a += kernel_m * inc_row_a;
    }
    if (mr > 0) {
        for (j = 0; j < kc; j++) {
            for (i = 0; i < mr; i++) buffer[i] = a[i * inc_row_a];
            for (i = mr; i < kernel_m; i++) buffer[i] = 0.0;
            buffer += kernel_m;
            a += inc_col_a;
        }
    }
}

static void copy_nrxk_blocked_db(int k, int kernel_n, ivf64* b, int inc_row_b, int inc_col_b, ivf64* buffer)
{
    int i, j;
    for (i = 0; i < k; i++) {
        for (j = 0; j < kernel_n; j++) buffer[j] = b[j * inc_col_b];
        buffer += kernel_n;
        b += inc_row_b;
    }
}

static void copy_b_blocked_db(int kc, int nc, int kernel_n, ivf64* b, int inc_row_b, int inc_col_b, ivf64* buffer)
{
    int np = nc / kernel_n;
    int nr = nc % kernel_n;
    int i, j;
    for (j = 0; j < np; j++) {
        copy_nrxk_blocked_db(kc, kernel_n, b, inc_row_b, inc_col_b, buffer);
        buffer += kc * kernel_n;
        b += kernel_n * inc_col_b;
    }
    if (nr > 0) {
        for (i = 0; i < kc; i++) {
            for (j = 0; j < nr; j++) buffer[j] = b[j * inc_col_b];
            for (j = nr; j < kernel_n; j++) buffer[j] = 0.0;
            buffer += kernel_n;
            b += inc_row_b;
        }
    }
}

#define FIV_KERNEL_M_8_DB  (8)
#define FIV_KERNEL_N_8_DB  (8)

/* 8x8 micro-kernel: C_tile += alpha * A_panel(8xkc) * B_panel(kc x 8) + beta * C_tile.
   c is addressed with (row=j, col=i) in the tile via c[j*inc_row_c + i*inc_col_c],
   matching the float32 micro-kernel so the packing/layout stays identical. */
static void mat_mul_8xkx8_kernel_db(
    int kc, ivf64 alpha, ivf64* a, ivf64* b, ivf64 beta,
    ivf64* c, int inc_row_c, int inc_col_c)
{
    ivf64 ab[FIV_KERNEL_M_8_DB * FIV_KERNEL_N_8_DB];
    int i, j, l;
#if defined(FIV_USE_AVX2)
    __m256d ab0, ab1, ab2, ab3;
    __m256d ab4, ab5, ab6, ab7;
    ab0 = _mm256_setzero_pd(); ab1 = _mm256_setzero_pd();
    ab2 = _mm256_setzero_pd(); ab3 = _mm256_setzero_pd();
    ab4 = _mm256_setzero_pd(); ab5 = _mm256_setzero_pd();
    ab6 = _mm256_setzero_pd(); ab7 = _mm256_setzero_pd();

    __m256d b0 = _mm256_load_pd(b);
    b += 8;
    for (l = 0; l < kc; l++){
        __m256d a0 = _mm256_broadcast_sd(a);
        __m256d a1 = _mm256_broadcast_sd(a + 1);
        __m256d a2 = _mm256_broadcast_sd(a + 2);
        __m256d a3 = _mm256_broadcast_sd(a + 3);

        ab0 = _mm256_fmadd_pd(b0, a0, ab0);
        ab1 = _mm256_fmadd_pd(b0, a1, ab1);
        ab2 = _mm256_fmadd_pd(b0, a2, ab2);
        ab3 = _mm256_fmadd_pd(b0, a3, ab3);

        __m256d a4 = _mm256_broadcast_sd(a + 4);
        __m256d a5 = _mm256_broadcast_sd(a + 5);

         a0 = _mm256_broadcast_sd(a + 6);
         a1 = _mm256_broadcast_sd(a + 7);
        a += 8;

        ab4 = _mm256_fmadd_pd(b0, a4, ab4);
        ab5 = _mm256_fmadd_pd(b0, a5, ab5);
        ab6 = _mm256_fmadd_pd(b0, a0, ab6);
        ab7 = _mm256_fmadd_pd(b0, a1, ab7);

        b0 = _mm256_load_pd(b);
        b += 8;
    }

    _mm256_store_pd(ab + 0,  ab0);
    _mm256_store_pd(ab + 8,  ab1);
    _mm256_store_pd(ab + 16, ab2);
    _mm256_store_pd(ab + 24, ab3);
    _mm256_store_pd(ab + 32, ab4);
    _mm256_store_pd(ab + 40, ab5);
    _mm256_store_pd(ab + 48, ab6);
    _mm256_store_pd(ab + 56, ab7);

#elif defined(FIV_USE_ARM_NEON)
    float64x2_t ab0_0 = vdupq_n_f64(0), ab0_1 = vdupq_n_f64(0), ab0_2 = vdupq_n_f64(0), ab0_3 = vdupq_n_f64(0);
    float64x2_t ab1_0 = vdupq_n_f64(0), ab1_1 = vdupq_n_f64(0), ab1_2 = vdupq_n_f64(0), ab1_3 = vdupq_n_f64(0);
    float64x2_t ab2_0 = vdupq_n_f64(0), ab2_1 = vdupq_n_f64(0), ab2_2 = vdupq_n_f64(0), ab2_3 = vdupq_n_f64(0);
    float64x2_t ab3_0 = vdupq_n_f64(0), ab3_1 = vdupq_n_f64(0), ab3_2 = vdupq_n_f64(0), ab3_3 = vdupq_n_f64(0);
    float64x2_t ab4_0 = vdupq_n_f64(0), ab4_1 = vdupq_n_f64(0), ab4_2 = vdupq_n_f64(0), ab4_3 = vdupq_n_f64(0);
    float64x2_t ab5_0 = vdupq_n_f64(0), ab5_1 = vdupq_n_f64(0), ab5_2 = vdupq_n_f64(0), ab5_3 = vdupq_n_f64(0);
    float64x2_t ab6_0 = vdupq_n_f64(0), ab6_1 = vdupq_n_f64(0), ab6_2 = vdupq_n_f64(0), ab6_3 = vdupq_n_f64(0);
    float64x2_t ab7_0 = vdupq_n_f64(0), ab7_1 = vdupq_n_f64(0), ab7_2 = vdupq_n_f64(0), ab7_3 = vdupq_n_f64(0);

    for (l = 0; l < kc; l++){
        float64x2_t b0 = vld1q_f64(b);
        float64x2_t b1 = vld1q_f64(b + 2);
        float64x2_t b2 = vld1q_f64(b + 4);
        float64x2_t b3 = vld1q_f64(b + 6);

        ab0_0 = vfmaq_n_f64(ab0_0, b0, a[0]); ab0_1 = vfmaq_n_f64(ab0_1, b1, a[0]);
        ab0_2 = vfmaq_n_f64(ab0_2, b2, a[0]); ab0_3 = vfmaq_n_f64(ab0_3, b3, a[0]);
        ab1_0 = vfmaq_n_f64(ab1_0, b0, a[1]); ab1_1 = vfmaq_n_f64(ab1_1, b1, a[1]);
        ab1_2 = vfmaq_n_f64(ab1_2, b2, a[1]); ab1_3 = vfmaq_n_f64(ab1_3, b3, a[1]);
        ab2_0 = vfmaq_n_f64(ab2_0, b0, a[2]); ab2_1 = vfmaq_n_f64(ab2_1, b1, a[2]);
        ab2_2 = vfmaq_n_f64(ab2_2, b2, a[2]); ab2_3 = vfmaq_n_f64(ab2_3, b3, a[2]);
        ab3_0 = vfmaq_n_f64(ab3_0, b0, a[3]); ab3_1 = vfmaq_n_f64(ab3_1, b1, a[3]);
        ab3_2 = vfmaq_n_f64(ab3_2, b2, a[3]); ab3_3 = vfmaq_n_f64(ab3_3, b3, a[3]);
        ab4_0 = vfmaq_n_f64(ab4_0, b0, a[4]); ab4_1 = vfmaq_n_f64(ab4_1, b1, a[4]);
        ab4_2 = vfmaq_n_f64(ab4_2, b2, a[4]); ab4_3 = vfmaq_n_f64(ab4_3, b3, a[4]);
        ab5_0 = vfmaq_n_f64(ab5_0, b0, a[5]); ab5_1 = vfmaq_n_f64(ab5_1, b1, a[5]);
        ab5_2 = vfmaq_n_f64(ab5_2, b2, a[5]); ab5_3 = vfmaq_n_f64(ab5_3, b3, a[5]);
        ab6_0 = vfmaq_n_f64(ab6_0, b0, a[6]); ab6_1 = vfmaq_n_f64(ab6_1, b1, a[6]);
        ab6_2 = vfmaq_n_f64(ab6_2, b2, a[6]); ab6_3 = vfmaq_n_f64(ab6_3, b3, a[6]);
        ab7_0 = vfmaq_n_f64(ab7_0, b0, a[7]); ab7_1 = vfmaq_n_f64(ab7_1, b1, a[7]);
        ab7_2 = vfmaq_n_f64(ab7_2, b2, a[7]); ab7_3 = vfmaq_n_f64(ab7_3, b3, a[7]);

        a += 8;
        b += 8;
    }
    vst1q_f64(ab + 0,  ab0_0); vst1q_f64(ab + 2,  ab0_1);
    vst1q_f64(ab + 4,  ab0_2); vst1q_f64(ab + 6,  ab0_3);
    vst1q_f64(ab + 8,  ab1_0); vst1q_f64(ab + 10, ab1_1);
    vst1q_f64(ab + 12, ab1_2); vst1q_f64(ab + 14, ab1_3);
    vst1q_f64(ab + 16, ab2_0); vst1q_f64(ab + 18, ab2_1);
    vst1q_f64(ab + 20, ab2_2); vst1q_f64(ab + 22, ab2_3);
    vst1q_f64(ab + 24, ab3_0); vst1q_f64(ab + 26, ab3_1);
    vst1q_f64(ab + 28, ab3_2); vst1q_f64(ab + 30, ab3_3);
    vst1q_f64(ab + 32, ab4_0); vst1q_f64(ab + 34, ab4_1);
    vst1q_f64(ab + 36, ab4_2); vst1q_f64(ab + 38, ab4_3);
    vst1q_f64(ab + 40, ab5_0); vst1q_f64(ab + 42, ab5_1);
    vst1q_f64(ab + 44, ab5_2); vst1q_f64(ab + 46, ab5_3);
    vst1q_f64(ab + 48, ab6_0); vst1q_f64(ab + 50, ab6_1);
    vst1q_f64(ab + 52, ab6_2); vst1q_f64(ab + 54, ab6_3);
    vst1q_f64(ab + 56, ab7_0); vst1q_f64(ab + 58, ab7_1);
    vst1q_f64(ab + 60, ab7_2); vst1q_f64(ab + 62, ab7_3);
#else
    memset(ab, 0, sizeof(ivf64) * FIV_KERNEL_M_8_DB * FIV_KERNEL_N_8_DB);
    for (l = 0; l < kc; l++) {
        for (j = 0; j < FIV_KERNEL_N_8_DB; j++) {
            for (i = 0; i < FIV_KERNEL_M_8_DB; i++) {
                ab[i * FIV_KERNEL_M_8_DB + j] += a[i] * b[j];
            }
        }
        a += FIV_KERNEL_M_8_DB;
        b += FIV_KERNEL_N_8_DB;
    }
#endif
    if (beta == 0.0) {
        if (inc_row_c > inc_col_c) {
            for (i = 0; i < FIV_KERNEL_M_8_DB; i++) {
                for (j = 0; j < FIV_KERNEL_N_8_DB; j++) {
                    c[i * inc_row_c + j * inc_col_c] = 0.0;
                }
            }
        }    else {
            for (j = 0; j < FIV_KERNEL_N_8_DB; j++) {
                for (i = 0; i < FIV_KERNEL_M_8_DB; i++) {
                    c[i * inc_row_c + j * inc_col_c] = 0.0;
                }
            }
        }
    }    else if(beta != 1.0){
        for (j = 0; j < FIV_KERNEL_N_8_DB; j++) {
            for (i = 0; i < FIV_KERNEL_M_8_DB; i++) {
                c[i * inc_row_c + j * inc_col_c] *= beta;
            }
        }
    }

    if (alpha == 1.0){
        for (j = 0; j < FIV_KERNEL_N_8_DB; j++) {
            for (i = 0; i < FIV_KERNEL_M_8_DB; i++) {
                c[j * inc_row_c + i * inc_col_c] += ab[i + j * FIV_KERNEL_M_8_DB];
            }
        }
    }  else {
        for (j = 0; j < FIV_KERNEL_N_8_DB; j++) {
            for (i = 0; i < FIV_KERNEL_M_8_DB; i++) {
                c[j * inc_row_c + i * inc_col_c] += alpha * ab[i + j * FIV_KERNEL_M_8_DB];
            }
        }
    }
}

#undef FIV_KERNEL_M_8_DB
#undef FIV_KERNEL_N_8_DB

#define FIV_MAX_KERNEL_SIZE_DB 96   /* >= 8*8, holds the remainder tile */

static void dgeaxpy_row_major_db(
    int m, int n, ivf64 alpha,
    ivf64* x, int inc_row_x, int inc_col_x,
    ivf64* y, int inc_row_y, int inc_col_y)
{
    int i, j;
    if (alpha != 1.0) {
        for (i = 0; i < m; i++)
            for (j = 0; j < n; j++)
                y[i * inc_row_y + j * inc_col_y] += alpha * x[i * inc_row_x + j * inc_col_x];
    } else {
        for (i = 0; i < m; i++)
            for (j = 0; j < n; j++)
                y[i * inc_row_y + j * inc_col_y] += x[i * inc_row_x + j * inc_col_x];
    }
}

static void dgescal_row_major_db(
    int m, int n, ivf64 alpha,
    ivf64* x, int inc_row_x, int inc_col_x)
{
    int i, j;
    if (alpha != 1.0) {
        for (i = 0; i < m; i++)
            for (j = 0; j < n; j++)
                x[i * inc_row_x + j * inc_col_x] *= alpha;
    }
}

static void mat_mul_kernel_row_major_db(
    int mc, int nc, int kc,
    ivf64 alpha, ivf64 beta,
    ivf64* c, int inc_row_c, int inc_col_c,
    ivf64* blocked_a, ivf64* blocked_b,
    int kernel_m_size, int kernel_n_size,
    ptr_func_mat_mul_mxkxn_kernel_db kernel)
{
    int mp = (mc + kernel_m_size - 1) / kernel_m_size;
    int np = (nc + kernel_n_size - 1) / kernel_n_size;
    int _mr = mc % kernel_m_size;
    int _nr = nc % kernel_n_size;
    int i;
    for (i = 0; i < mp; i++) {
        int j, mr, nr;
        ivf64* ptr_a = &blocked_a[i * kc * kernel_m_size];
        ivf64* ptr_c = &c[i * kernel_m_size * inc_row_c];
        mr = (i != mp - 1 || _mr == 0) ? kernel_m_size : _mr;
        for (j = 0; j < np; j++) {
            ivf64* ptr_b_j = &blocked_b[j * kc * kernel_n_size];
            ivf64* ptr_c_j = &ptr_c[j * kernel_n_size * inc_col_c];
            nr = (j != np - 1 || _nr == 0) ? kernel_n_size : _nr;
            if (mr == kernel_m_size && nr == kernel_n_size) {
                kernel(kc, alpha, ptr_a, ptr_b_j, beta, ptr_c_j, inc_row_c, inc_col_c);
            } else {
                ivf64 blocked_c[FIV_MAX_KERNEL_SIZE_DB];
                kernel(kc, alpha, ptr_a, ptr_b_j, 0.0, blocked_c, 1, kernel_m_size);
                dgescal_row_major_db(mr, nr, beta, ptr_c_j, inc_row_c, inc_col_c);
                dgeaxpy_row_major_db(mr, nr, 1.0, blocked_c, 1, kernel_m_size, ptr_c_j, inc_row_c, inc_col_c);
            }
        }
    }
}

static void blocked_mat_mul_row_major_real64(
    int m, int n, int k,
    ivf64 alpha,
    ivf64* a, int inc_row_a, int inc_col_a,
    ivf64* b, int inc_row_b, int inc_col_b,
    ivf64 beta,
    ivf64* c, int inc_row_c, int inc_col_c)
{
    int mb = (m + FIV_M_BLOCK_DB - 1) / FIV_M_BLOCK_DB;
    int nb = (n + FIV_N_BLOCK_DB - 1) / FIV_N_BLOCK_DB;
    int kb = (k + FIV_K_BLOCK_DB - 1) / FIV_K_BLOCK_DB;
    int _mc = m % FIV_M_BLOCK_DB;
    int _nc = n % FIV_N_BLOCK_DB;
    int _kc = k % FIV_K_BLOCK_DB;

    if (alpha == 0.0 || k == 0) {
        dgescal_row_major_db(m, n, beta, c, inc_row_c, inc_col_c);
        return;
    }

    int mc, nc, kc, i, j, l;
    int kernel_m_size = 8, kernel_n_size = 8;
    ptr_func_mat_mul_mxkxn_kernel_db kernel_func = mat_mul_8xkx8_kernel_db;

    int a_zero_rest = FIV_M_BLOCK_DB % kernel_m_size == 0 ? 0 : kernel_m_size - FIV_M_BLOCK_DB % kernel_m_size;
    int b_zero_rest = FIV_N_BLOCK_DB % kernel_n_size == 0 ? 0 : kernel_n_size - FIV_N_BLOCK_DB % kernel_n_size;

    ivf64* blocked_a = (ivf64*)fiv_malloc(sizeof(ivf64) * (FIV_M_BLOCK_DB + a_zero_rest) * FIV_K_BLOCK_DB);
    ivf64* blocked_b = (ivf64*)fiv_malloc(sizeof(ivf64) * (FIV_N_BLOCK_DB + b_zero_rest) * FIV_K_BLOCK_DB);
    if (blocked_a && blocked_b) {
        for (i = 0; i < mb; i++) {
            ivf64* ptr_a = &a[i * FIV_M_BLOCK_DB * inc_row_a];
            ivf64* ptr_c = &c[i * FIV_M_BLOCK_DB * inc_row_c];
            mc = (i != mb - 1 || _mc == 0) ? FIV_M_BLOCK_DB : _mc;
            for (l = 0; l < kb; l++) {
                ivf64* ptr_a_l = &ptr_a[l * FIV_K_BLOCK_DB * inc_col_a];
                ivf64* ptr_b   = &b[l * FIV_K_BLOCK_DB * inc_row_b];
                kc = (l != kb - 1 || _kc == 0) ? FIV_K_BLOCK_DB : _kc;
                ivf64 _beta = (l == 0) ? beta : 1.0;
                copy_a_blocked_db(mc, kc, kernel_m_size, ptr_a_l, inc_row_a, inc_col_a, blocked_a);
                for (j = 0; j < nb; j++) {
                    ivf64* ptr_c_j = &ptr_c[j * FIV_N_BLOCK_DB * inc_col_c];
                    ivf64* ptr_b_j = &ptr_b[j * FIV_N_BLOCK_DB * inc_col_b];
                    nc = (j != nb - 1 || _nc == 0) ? FIV_N_BLOCK_DB : _nc;
                    copy_b_blocked_db(kc, nc, kernel_n_size, ptr_b_j, inc_row_b, inc_col_b, blocked_b);
                    mat_mul_kernel_row_major_db(
                        mc, nc, kc, alpha, _beta,
                        ptr_c_j, inc_row_c, inc_col_c,
                        blocked_a, blocked_b,
                        kernel_m_size, kernel_n_size,
                        kernel_func);
                }
            }
        }
    } else {
        FIV_PRINT_LOG("memory alloc error!");
    }

    fiv_free(blocked_b);
    fiv_free(blocked_a);
}

/* Blocked GEMM entry (ivf64). a_t/b_t: 1 means the operand is used transposed.
   m/n/k are the effective dims of op(A)/op(B)/op(C). */
static void fiv_matrix_mul_blocked_real64(
    int a_t, int b_t, int m, int n, int k,
    ivf64 alpha,
    ivf64* a, int lda,
    ivf64* b, int ldb,
    ivf64 beta,
    ivf64* c, int ldc)
{
    int i, j;
    if (m <= 0 || n <= 0 || ((alpha == 0.0 || k <= 0) && (beta == 1.0))) {
        return;
    }
    if (alpha == 0.0) {
        if (beta == 0.0) {
            for (j = 0; j < n; j++)
                for (i = 0; i < m; i++)
                    c[j * ldc + i] = 0.0;
        } else {
            for (j = 0; j < n; j++)
                for (i = 0; i < m; i++)
                    c[j * ldc + i] *= beta;
        }
        return;
    }
    if (a_t == 0 && b_t == 0) {
        blocked_mat_mul_row_major_real64(m, n, k, alpha, a, lda, 1, b, ldb, 1, beta, c, ldc, 1);
    } else if (a_t && b_t == 0) {
        blocked_mat_mul_row_major_real64(m, n, k, alpha, a, 1, lda, b, ldb, 1, beta, c, ldc, 1);
    } else if (a_t == 0 && b_t) {
        blocked_mat_mul_row_major_real64(m, n, k, alpha, a, lda, 1, b, 1, ldb, beta, c, ldc, 1);
    } else {
        blocked_mat_mul_row_major_real64(m, n, k, alpha, a, 1, lda, b, 1, ldb, beta, c, ldc, 1);
    }
}

/* ============================================================================
   64-bit (ivf64 / double) full API: dst = alpha * op(A) * op(B) + beta * dst.
   This is the dtype-specific backend invoked by the generic fiv_matrix_mul
   (api/fiv_matrix.h) when the operands are FIV_64F1; it is not a standalone
   public interface. Dispatched to the small (non-blocked) path when A+B+C fit
   in the L3-cache budget, otherwise to the blocked path.
   ========================================================================== */

fiv_ret fiv_matrix_mul_real64(fiv_mat* dst, const fiv_mat* A, const fiv_mat* B,
                              int a_transpose, int b_transpose, fiv_scalar alpha, fiv_scalar beta)
{
    if (dst == NULL || A == NULL || B == NULL) return FIV_RET_ERR_PARA;
    if (dst->data.ptr == NULL || A->data.ptr == NULL || B->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (dst->data_continue == 0 || A->data_continue == 0 || B->data_continue == 0) return FIV_RET_ERR_PARA;
    if (A->dtype != FIV_64F1 || B->dtype != FIV_64F1 || dst->dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (alpha.id != FIV_ID_SCALAR || alpha.dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (beta.id != FIV_ID_SCALAR || beta.dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;
    ivf64 alpha_f = alpha.data.value_fp64;
    ivf64 beta_f  = beta.data.value_fp64;
    if (dst->data.ptr == A->data.ptr || dst->data.ptr == B->data.ptr) return FIV_RET_ERR_PARA;

    const int ra = (int)A->shapes[0];
    const int ca = (int)A->shapes[1];
    const int rb = (int)B->shapes[0];
    const int cb = (int)B->shapes[1];
    if (ra <= 0 || ca <= 0 || rb <= 0 || cb <= 0) return FIV_RET_ERR_PARA;

    const int M  = a_transpose ? ca : ra;
    const int N  = b_transpose ? rb : cb;
    const int K  = a_transpose ? ra : ca;    /* cols of op(A) */
    const int Kb = b_transpose ? cb : rb;    /* rows of op(B) */
    if (K != Kb) return FIV_RET_ERR_PARA;

    if ((int)dst->shapes[0] != M || (int)dst->shapes[1] != N) return FIV_RET_ERR_PARA;
    if (dst->total_bytes < (size_t)M * (size_t)N * (size_t)dst->element_bytes) return FIV_RET_ERR_PARA;

    ivf64* a = (ivf64*)A->data.ptr;
    ivf64* b = (ivf64*)B->data.ptr;
    ivf64* c = (ivf64*)dst->data.ptr;

    const size_t ws_bytes = ((size_t)ra * (size_t)ca + (size_t)rb * (size_t)cb +
                             (size_t)M * (size_t)N) * (size_t)A->element_bytes;

    if (ws_bytes <= FIV_MAT_MUL_DB_L3_LIMIT_BYTES) {
        if (!a_transpose && !b_transpose)
            fiv_small_matrix_mul_matrix_real64(a, ra, ca, ca, b, cb, cb, c, N, alpha_f, beta_f);
        else if (!a_transpose && b_transpose)
            fiv_small_matrix_mul_matrix_t_real64(a, ra, ca, ca, b, rb, cb, c, N, alpha_f, beta_f);
        else if (a_transpose && !b_transpose)
            fiv_small_matrix_t_mul_matrix_real64(a, ra, ca, ca, b, cb, cb, c, N, alpha_f, beta_f);
        else
            fiv_small_matrix_t_mul_matrix_t_real64(a, ra, ca, ca, b, rb, cb, c, N, alpha_f, beta_f);
    } else {
        fiv_matrix_mul_blocked_real64(a_transpose, b_transpose, M, N, K, alpha_f, a, ca, b, cb, beta_f, c, N);
    }

    dst->shapes[0]   = (size_t)M;
    dst->shapes[1]   = (size_t)N;
    dst->strides[0]  = (size_t)N * (size_t)dst->element_bytes;
    dst->strides[1]  = (size_t)dst->element_bytes;
    dst->total_bytes = (size_t)M * (size_t)N * (size_t)dst->element_bytes;
    dst->data_continue = 1;
    return FIV_RET_OK;
}
