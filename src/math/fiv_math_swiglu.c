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

#include "fiv_math_swiglu.h"
#include <math.h>  /* expf / exp (scalar fallback / scalar tail) */


/* ==================== Scalar backends ==================== */

/* dst[i] = gate[i] / (1 + exp(-gate[i])) * up[i], FIV_32F1 — scalar fallback,
   only compiled when the AVX2 kernel is not available. */
#if !defined(FIV_USE_AVX2)
void fiv_math_swiglu_real32(ivf32* dst, const ivf32* gate, const ivf32* up, size_t element_count)
{
    for (size_t i = 0; i < element_count; i++)
        dst[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
}
#endif  /* !FIV_USE_AVX2 */

/* dst[i] = gate[i] / (1 + exp(-gate[i])) * up[i], FIV_64F1 */
void fiv_math_swiglu_real64(ivf64* dst, const ivf64* gate, const ivf64* up, size_t element_count)
{
    for (size_t i = 0; i < element_count; i++)
        dst[i] = gate[i] / (1.0 + exp(-gate[i])) * up[i];
}


/* ==================== SwiGLU (AVX2+FMA, element-wise) ====================
   dst = silu(gate) * up, where silu(g) = g * sigmoid(g). The sigmoid factor
   reuses the numerically stable |g| form from the sigmoid kernel so only ONE
   exp is needed per element and its argument is always <= 0:
     e   = exp(-|g|)                  (e in (0,1]; never overflows)
     t   = 1 / (1 + e)                (t in [0.5, 1), via rcp + Newton)
     sig = g >= 0 ? t : 1 - t
     dst = g * sig * up
   Reuses the vector exp helpers from fiv_math_kernels.h (single-chain
   exp256_ps for tails, packed-coefficient dual-vector exp256_ps4 for the main
   loop). */
#if defined(FIV_USE_AVX2)

#include "fiv_math_kernels.h"

/* one vector: silu(g) = g * sigmoid(g) for 8 elements (single-chain exp) */
static inline __m256 fiv_math_silu256_ps(__m256 g)
{
    const __m256 c_one     = _mm256_set1_ps(1.0f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    __m256 ag = _mm256_andnot_ps(sign_mask, g);                    /* |g|          */
    __m256 e  = fiv_math_exp256_ps(_mm256_xor_ps(ag, sign_mask));  /* exp(-|g|)    */
    __m256 t  = fiv_math_rcp256_ps(_mm256_add_ps(c_one, e));       /* 1/(1+e)      */
    __m256 sig = _mm256_blendv_ps(t, _mm256_sub_ps(c_one, t), g);  /* g<0 ? 1-t : t */
    return _mm256_mul_ps(g, sig);                                  /* silu(g)      */
}

/* two vectors: silu for 16 elements, sharing the packed-coefficient
   dual-vector exp256_ps4 */
static inline fiv_m256x2 fiv_math_silu256_ps2(__m256 g0, __m256 g1)
{
    const __m256 c_one     = _mm256_set1_ps(1.0f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    __m256 ag0 = _mm256_andnot_ps(sign_mask, g0);
    __m256 ag1 = _mm256_andnot_ps(sign_mask, g1);
    fiv_m256x2 e = fiv_math_exp256_ps4(_mm256_xor_ps(ag0, sign_mask),
                                       _mm256_xor_ps(ag1, sign_mask));

    fiv_m256x2 r;
    __m256 t0 = fiv_math_rcp256_ps(_mm256_add_ps(c_one, e.a));
    __m256 t1 = fiv_math_rcp256_ps(_mm256_add_ps(c_one, e.b));
    __m256 sig0 = _mm256_blendv_ps(t0, _mm256_sub_ps(c_one, t0), g0);
    __m256 sig1 = _mm256_blendv_ps(t1, _mm256_sub_ps(c_one, t1), g1);
    r.a = _mm256_mul_ps(g0, sig0);
    r.b = _mm256_mul_ps(g1, sig1);
    return r;
}

/* Element-wise SwiGLU over the whole buffer, in-place safe (dst may alias
   gate or up). Main loop 4-way unrolled (32 floats/iter) via the dual-vector
   exp; the tail blocks use the single-vector exp, then a scalar tail. */
void fiv_math_swiglu_avx2_ps(ivf32* dst, const ivf32* gate, const ivf32* up, size_t element_count)
{
    size_t i = 0;
    for (; i + 32 <= element_count; i += 32) {
        fiv_m256x2 s01 = fiv_math_silu256_ps2(_mm256_loadu_ps(gate + i),
                                              _mm256_loadu_ps(gate + i + 8));
        fiv_m256x2 s23 = fiv_math_silu256_ps2(_mm256_loadu_ps(gate + i + 16),
                                              _mm256_loadu_ps(gate + i + 24));
        _mm256_storeu_ps(dst + i,      _mm256_mul_ps(s01.a, _mm256_loadu_ps(up + i)));
        _mm256_storeu_ps(dst + i + 8,  _mm256_mul_ps(s01.b, _mm256_loadu_ps(up + i + 8)));
        _mm256_storeu_ps(dst + i + 16, _mm256_mul_ps(s23.a, _mm256_loadu_ps(up + i + 16)));
        _mm256_storeu_ps(dst + i + 24, _mm256_mul_ps(s23.b, _mm256_loadu_ps(up + i + 24)));
    }
    for (; i + 8 <= element_count; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(fiv_math_silu256_ps(_mm256_loadu_ps(gate + i)),
                                                _mm256_loadu_ps(up + i)));
    for (; i < element_count; i++)
        dst[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
}

#endif  /* FIV_USE_AVX2 */
