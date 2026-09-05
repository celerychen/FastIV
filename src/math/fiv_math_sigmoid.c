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

#include "fiv_math_sigmoid.h"
#include <math.h>  /* expf (scalar fallback / scalar tail) */


/* ==================== Scalar backends ==================== */

/* dst[i] = 1 / (1 + exp(-src[i])), FIV_32F1 — scalar fallback, only compiled
   when the AVX2 kernel is not available. */
#if !defined(FIV_USE_AVX2)
void fiv_math_sigmoid_real32(ivf32* dst, const ivf32* src, size_t element_count)
{
    for (size_t i = 0; i < element_count; i++)
        dst[i] = 1.0f / (1.0f + expf(-src[i]));
}
#endif  /* !FIV_USE_AVX2 */

/* dst[i] = 1 / (1 + exp(-src[i])), FIV_64F1 */
void fiv_math_sigmoid_real64(ivf64* dst, const ivf64* src, size_t element_count)
{
    for (size_t i = 0; i < element_count; i++)
        dst[i] = 1.0 / (1.0 + exp(-src[i]));
}


/* ==================== Sigmoid (AVX2+FMA, element-wise) ====================
   Numerically stable form based on |x| so only ONE exp is needed per element
   and its argument is always <= 0:
     e = exp(-|x|)  (e in (0,1]; never overflows, even for extreme x)
     t = 1 / (1 + e)   (t in [0.5, 1); denominator is in (1,2], so the division
                        is always well-conditioned)
     sigmoid(x) = x >= 0 ? t : 1 - t, using sigmoid(-a) = 1 - sigmoid(a)
   The reciprocal is _mm256_rcp_ps + one Newton refinement (~full float
   precision), keeping the kernel free of any division. Reuses the vector exp
   helpers from fiv_math_kernels.h (single-chain exp256_ps for tails,
   packed-coefficient dual-vector exp256_ps4 for the main loop). */
#if defined(FIV_USE_AVX2)

#include "fiv_math_kernels.h"

/* one vector: sigmoid(x) for 8 elements (single-chain exp) */
static inline __m256 fiv_math_sigmoid256_ps(__m256 x)
{
    const __m256 c_one     = _mm256_set1_ps(1.0f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    __m256 ax = _mm256_andnot_ps(sign_mask, x);                    /* |x|          */
    __m256 e  = fiv_math_exp256_ps(_mm256_xor_ps(ax, sign_mask));  /* exp(-|x|)    */
    __m256 t  = fiv_math_rcp256_ps(_mm256_add_ps(c_one, e));       /* 1/(1+e)      */
    return _mm256_blendv_ps(t, _mm256_sub_ps(c_one, t), x);        /* x<0 ? 1-t : t */
}

/* two vectors: sigmoid for 16 elements, sharing the packed-coefficient
   dual-vector exp256_ps4 */
static inline fiv_m256x2 fiv_math_sigmoid256_ps2(__m256 x, __m256 y)
{
    const __m256 c_one     = _mm256_set1_ps(1.0f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    __m256 ax = _mm256_andnot_ps(sign_mask, x);
    __m256 ay = _mm256_andnot_ps(sign_mask, y);
    fiv_m256x2 e = fiv_math_exp256_ps4(_mm256_xor_ps(ax, sign_mask),
                                       _mm256_xor_ps(ay, sign_mask));

    fiv_m256x2 r;
    __m256 ta = fiv_math_rcp256_ps(_mm256_add_ps(c_one, e.a));
    __m256 tb = fiv_math_rcp256_ps(_mm256_add_ps(c_one, e.b));
    r.a = _mm256_blendv_ps(ta, _mm256_sub_ps(c_one, ta), x);
    r.b = _mm256_blendv_ps(tb, _mm256_sub_ps(c_one, tb), y);
    return r;
}

/* Element-wise sigmoid over the whole buffer, in-place safe (dst may alias
   src). Main loop 4-way unrolled (32 floats/iter) via the dual-vector exp;
   the tail blocks use the single-vector exp, then a scalar tail. */
void fiv_math_sigmoid_avx2_ps(ivf32* dst, const ivf32* src, size_t element_count)
{
    size_t i = 0;
    for (; i + 32 <= element_count; i += 32) {
        fiv_m256x2 e01 = fiv_math_sigmoid256_ps2(_mm256_loadu_ps(src + i),
                                                 _mm256_loadu_ps(src + i + 8));
        fiv_m256x2 e23 = fiv_math_sigmoid256_ps2(_mm256_loadu_ps(src + i + 16),
                                                 _mm256_loadu_ps(src + i + 24));
        _mm256_storeu_ps(dst + i,      e01.a);
        _mm256_storeu_ps(dst + i + 8,  e01.b);
        _mm256_storeu_ps(dst + i + 16, e23.a);
        _mm256_storeu_ps(dst + i + 24, e23.b);
    }
    for (; i + 8 <= element_count; i += 8)
        _mm256_storeu_ps(dst + i, fiv_math_sigmoid256_ps(_mm256_loadu_ps(src + i)));
    for (; i < element_count; i++)
        dst[i] = 1.0f / (1.0f + expf(-src[i]));
}

#endif  /* FIV_USE_AVX2 */
