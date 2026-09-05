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

#include "fiv_math_softmax.h"
#include <math.h>     /* expf / exp / fmaxf */
#include <string.h>   /* memcpy (out-of-place AVX path) */


/* ==================== Scalar backends ==================== */

/* One row: stable scalar softmax, max subtracted before the exponentials.
   Works in-place (dst aliases src): the exp write-back consumes src[j] before
   overwriting it, then the normalization pass scales in place. FIV_64F1. */
static void fiv_math_softmax_row_real64(ivf64* dst, const ivf64* src, size_t cols)
{
    ivf64 max_value = src[0];
    for (size_t j = 1; j < cols; j++)
        if (src[j] > max_value) max_value = src[j];

    ivf64 sum = 0.0;
    for (size_t j = 0; j < cols; j++) {
        ivf64 e = exp(src[j] - max_value);
        dst[j] = e;
        sum += e;
    }

    const ivf64 inv_sum = 1.0 / sum;
    for (size_t j = 0; j < cols; j++)
        dst[j] *= inv_sum;
}

void fiv_math_softmax_real64(ivf64* dst, const ivf64* src, size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows; i++)
        fiv_math_softmax_row_real64(dst + i * cols, src + i * cols, cols);
}

/* Scalar FIV_32F1 fallback, only compiled when the AVX2 kernel is not
   available. Same stable row algorithm as the FIV_64F1 variant. */
#if !defined(FIV_USE_AVX2)
static void fiv_math_softmax_row_real32(ivf32* dst, const ivf32* src, size_t cols)
{
    ivf32 max_value = src[0];
    for (size_t j = 1; j < cols; j++)
        if (src[j] > max_value) max_value = src[j];

    ivf32 sum = 0.0f;
    for (size_t j = 0; j < cols; j++) {
        ivf32 e = expf(src[j] - max_value);
        dst[j] = e;
        sum += e;
    }

    const ivf32 inv_sum = 1.0f / sum;
    for (size_t j = 0; j < cols; j++)
        dst[j] *= inv_sum;
}

void fiv_math_softmax_real32(ivf32* dst, const ivf32* src, size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows; i++)
        fiv_math_softmax_row_real32(dst + i * cols, src + i * cols, cols);
}
#endif  /* !FIV_USE_AVX2 */


/* ==================== Softmax (AVX2+FMA row kernel, FIV_32F1) ====================
   Math is the same stable 3-pass algorithm as the scalar row; only pass 2 uses
   the dual-vector packed-coefficient exp256_ps4 instead of per-chain broadcast
   constants. Handles any n (n<8 falls back to scalar inside the kernel).
   Ported from the reference softmax_avx_v1_4.c; the vector exp and reductions
   come from fiv_math_kernels.h. */
#if defined(FIV_USE_AVX2)

#include "fiv_math_kernels.h"

/* Normalize ONE row in place, three passes over x[0..n-1]:
     1) max, 4-way unrolled (32 floats/iter);
     2) exp(x - max) write-back + sum, using the dual-vector interleaved,
        packed-coefficient exp256_ps4;
     3) in-place division by sum, 4-way unrolled. */
static void fiv_math_softmax_row_avx2_ps(ivf32* x, size_t n)
{
    if (n == 0) return;

    /* Pass 1: max, 4-way unrolled (32 floats/iter) */
    float max_val;
    if (n >= 8) {
        __m256 vmax0 = _mm256_loadu_ps(x);
        __m256 vmax1 = vmax0;
        __m256 vmax2 = vmax0;
        __m256 vmax3 = vmax0;
        size_t i = 8;
        for (; i + 32 <= n; i += 32) {
            vmax0 = _mm256_max_ps(vmax0, _mm256_loadu_ps(x + i));
            vmax1 = _mm256_max_ps(vmax1, _mm256_loadu_ps(x + i + 8));
            vmax2 = _mm256_max_ps(vmax2, _mm256_loadu_ps(x + i + 16));
            vmax3 = _mm256_max_ps(vmax3, _mm256_loadu_ps(x + i + 24));
        }
        __m256 vmax = _mm256_max_ps(_mm256_max_ps(vmax0, vmax1),
                                    _mm256_max_ps(vmax2, vmax3));
        for (; i + 8 <= n; i += 8)
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + i));
        max_val = fiv_math_hmax256_ps(vmax);
        for (; i < n; i++)
            max_val = fmaxf(max_val, x[i]);
    } else {
        max_val = x[0];
        for (size_t i = 1; i < n; i++)
            max_val = fmaxf(max_val, x[i]);
    }

    /* Pass 2: exp(x-m) write-back + accumulate sum. exp256_ps4 processes
       2 blocks (16 floats) at once with two interleaved exp chains sharing
       the packed constants. */
    const __m256 vmax_bc = _mm256_set1_ps(max_val);
    __m256 vs0 = _mm256_setzero_ps();
    __m256 vs1 = _mm256_setzero_ps();
    __m256 vs2 = _mm256_setzero_ps();
    __m256 vs3 = _mm256_setzero_ps();
    size_t j = 0;
    for (; j + 32 <= n; j += 32) {
        __m256 b0 = _mm256_sub_ps(_mm256_loadu_ps(x + j),      vmax_bc);
        __m256 b1 = _mm256_sub_ps(_mm256_loadu_ps(x + j + 8),  vmax_bc);
        __m256 b2 = _mm256_sub_ps(_mm256_loadu_ps(x + j + 16), vmax_bc);
        __m256 b3 = _mm256_sub_ps(_mm256_loadu_ps(x + j + 24), vmax_bc);
        fiv_m256x2 e01 = fiv_math_exp256_ps4(b0, b1);
        fiv_m256x2 e23 = fiv_math_exp256_ps4(b2, b3);
        _mm256_storeu_ps(x + j,      e01.a);
        _mm256_storeu_ps(x + j + 8,  e01.b);
        _mm256_storeu_ps(x + j + 16, e23.a);
        _mm256_storeu_ps(x + j + 24, e23.b);
        vs0 = _mm256_add_ps(vs0, e01.a);
        vs1 = _mm256_add_ps(vs1, e01.b);
        vs2 = _mm256_add_ps(vs2, e23.a);
        vs3 = _mm256_add_ps(vs3, e23.b);
    }
    __m256 vsum = _mm256_add_ps(_mm256_add_ps(vs0, vs1), _mm256_add_ps(vs2, vs3));
    for (; j + 8 <= n; j += 8) {
        __m256 e = fiv_math_exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x + j), vmax_bc));
        _mm256_storeu_ps(x + j, e);
        vsum = _mm256_add_ps(vsum, e);
    }
    float sum = fiv_math_hsum256_ps(vsum);
    for (; j < n; j++) {
        float e = expf(x[j] - max_val);
        x[j] = e;
        sum += e;
    }

    /* Pass 3: in-place division by sum (exact division, 4-way unrolled) */
    const float inv = 1.0f / sum;
    const __m256 vinv = _mm256_set1_ps(inv);
    size_t k = 0;
    for (; k + 32 <= n; k += 32) {
        _mm256_storeu_ps(x + k,      _mm256_mul_ps(_mm256_loadu_ps(x + k),      vinv));
        _mm256_storeu_ps(x + k + 8,  _mm256_mul_ps(_mm256_loadu_ps(x + k + 8),  vinv));
        _mm256_storeu_ps(x + k + 16, _mm256_mul_ps(_mm256_loadu_ps(x + k + 16), vinv));
        _mm256_storeu_ps(x + k + 24, _mm256_mul_ps(_mm256_loadu_ps(x + k + 24), vinv));
    }
    for (; k + 8 <= n; k += 8)
        _mm256_storeu_ps(x + k, _mm256_mul_ps(_mm256_loadu_ps(x + k), vinv));
    for (; k < n; k++)
        x[k] *= inv;
}

/* Full-matrix AVX2 backend: the row kernel normalizes one row in place, so for
   an out-of-place call the src buffer is copied into dst first, then each row
   is normalized. */
void fiv_math_softmax_avx2_ps(ivf32* dst, const ivf32* src, size_t rows, size_t cols)
{
    if (dst != src)
        memcpy(dst, src, (size_t)rows * cols * sizeof(ivf32));
    for (size_t i = 0; i < rows; i++)
        fiv_math_softmax_row_avx2_ps((ivf32*)dst + i * cols, cols);
}

#endif  /* FIV_USE_AVX2 */
