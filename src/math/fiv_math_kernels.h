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

#ifndef _FIV_MATH_KERNELS_H_
#define _FIV_MATH_KERNELS_H_

#include "fiv_data_typedefs.h"

/* =========================================================================
   Shared AVX2+FMA vector primitives for the fiv_math kernels (sigmoid /
   softmax). Only the generic building blocks live here -- the vector exp
   chains, the horizontal reductions and the approximate reciprocal -- so both
   kernel files reuse a single implementation. The op-specific compositions
   (sigmoid's |x| form, the softmax three-pass row loop) stay in their own
   files.

   The vector exp is a 6th-order Taylor on r = x - k*ln2 (k = round(x/ln2))
   with the input clamped to [-88, 88], so it never over/underflows and needs
   no division and no libcall inside the vector path.
   ======================================================================== */

#if defined(FIV_USE_AVX2)
#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dual-vector result packing */
typedef struct { __m256 a; __m256 b; } fiv_m256x2;

/* Single-vector exp (inline constants), reused by the <8 tail and the scalar
   remainder of the kernels. */
static inline __m256 fiv_math_exp256_ps(__m256 x)
{
    const __m256 c_invln2 = _mm256_set1_ps(1.4426950408889634f); /* 1/ln2  */
    const __m256 c_ln2    = _mm256_set1_ps(0.6931471805599453f); /* ln2    */
    const __m256 c_half   = _mm256_set1_ps(0.5f);

    /* clamp the input to avoid exp overflow/underflow */
    x = _mm256_min_ps(x, _mm256_set1_ps( 88.0f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));

    /* k = round(x / ln2), r = x - k*ln2 */
    __m256 k = _mm256_floor_ps(_mm256_fmadd_ps(x, c_invln2, c_half));
    __m256 r = _mm256_fnmadd_ps(k, c_ln2, x);

    /* 6th-order Taylor of exp(r), Horner all-FMA (ends on the constant term 1) */
    __m256 y = _mm256_set1_ps(0.001388888888888889f);                  /* 1/6! */
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(0.008333333333333333f));  /* +1/5! */
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(0.041666666666666664f));  /* +1/4! */
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(0.16666666666666666f));   /* +1/3! */
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(0.5f));                   /* +1/2! */
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f));                   /* +1/1! */
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f));                   /* +1/0! (constant) */

    /* 2^k = ldexp(1, k): move (k+127) into the exponent field (no multiply) */
    __m256i ki  = _mm256_cvtps_epi32(k);
    __m256i ex = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(ex));
}

/* Dual-vector interleaved + packed-coefficient exp: the x and y Horner chains
   are interleaved beat by beat and share the packed constants K0/K1 (taken via
   vpermilps), returning {exp(x), exp(y)}. Packing convention: vpermilps does
   not cross the 128-bit boundary, so both halves hold the same coefficient
   order, making an in-half broadcast equivalent to a full 8-lane broadcast.
     K0 = [1/6!, 1/5!, 1/4!, 1/3! | same]
     K1 = [1/2!,   1,    1,    1   | same] */
static inline fiv_m256x2 fiv_math_exp256_ps4(__m256 x, __m256 y)
{
    /* only the 3 constants used across steps are kept (k/r are dead after) */
    const __m256 c_invln2 = _mm256_set1_ps(1.4426950408889634f);
    const __m256 c_ln2    = _mm256_set1_ps(0.6931471805599453f);
    const __m256 c_half   = _mm256_set1_ps(0.5f);

    /* clamp (inline constants, do not hold registers for the whole pass) */
    x = _mm256_min_ps(x, _mm256_set1_ps( 88.0f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    y = _mm256_min_ps(y, _mm256_set1_ps( 88.0f));
    y = _mm256_max_ps(y, _mm256_set1_ps(-88.0f));

    /* k = round(x/ln2); r = x - k*ln2, interleaved for x and y */
    __m256 kx = _mm256_floor_ps(_mm256_fmadd_ps(x, c_invln2, c_half));
    __m256 ky = _mm256_floor_ps(_mm256_fmadd_ps(y, c_invln2, c_half));
    __m256 rx = _mm256_fnmadd_ps(kx, c_ln2, x);
    __m256 ry = _mm256_fnmadd_ps(ky, c_ln2, y);

    /* packed coefficients: only 2 constant registers live through the Horner
       pass, shared by both chains */
    const __m256 K0 = _mm256_set_ps(
        0.16666666666666666f, 0.041666666666666664f, 0.008333333333333333f, 0.001388888888888889f,
        0.16666666666666666f, 0.041666666666666664f, 0.008333333333333333f, 0.001388888888888889f);
    const __m256 K1 = _mm256_set_ps(
        1.0f, 1.0f, 1.0f, 0.5f,
        1.0f, 1.0f, 1.0f, 0.5f);

    /* 6th-order Horner, x/y chains interleaved beat by beat; every step takes
       the coefficient via immediate vpermilps from K0/K1 (no extra registers,
       no memory broadcast loads). Sharing one packed constant set drops the
       12 broadcast loads of v1_2 down to 2 loads + 12 shuffles (port 5 only,
       no memory port pressure). */
    __m256 yx = _mm256_permute_ps(K0, _MM_SHUFFLE(0, 0, 0, 0));                   /* 1/6! */
    __m256 yy = _mm256_permute_ps(K0, _MM_SHUFFLE(0, 0, 0, 0));
    yx = _mm256_fmadd_ps(yx, rx, _mm256_permute_ps(K0, _MM_SHUFFLE(1, 1, 1, 1))); /* 1/5! */
    yy = _mm256_fmadd_ps(yy, ry, _mm256_permute_ps(K0, _MM_SHUFFLE(1, 1, 1, 1)));
    yx = _mm256_fmadd_ps(yx, rx, _mm256_permute_ps(K0, _MM_SHUFFLE(2, 2, 2, 2))); /* 1/4! */
    yy = _mm256_fmadd_ps(yy, ry, _mm256_permute_ps(K0, _MM_SHUFFLE(2, 2, 2, 2)));
    yx = _mm256_fmadd_ps(yx, rx, _mm256_permute_ps(K0, _MM_SHUFFLE(3, 3, 3, 3))); /* 1/3! */
    yy = _mm256_fmadd_ps(yy, ry, _mm256_permute_ps(K0, _MM_SHUFFLE(3, 3, 3, 3)));
    yx = _mm256_fmadd_ps(yx, rx, _mm256_permute_ps(K1, _MM_SHUFFLE(0, 0, 0, 0))); /* 1/2! */
    yy = _mm256_fmadd_ps(yy, ry, _mm256_permute_ps(K1, _MM_SHUFFLE(0, 0, 0, 0)));
    yx = _mm256_fmadd_ps(yx, rx, _mm256_permute_ps(K1, _MM_SHUFFLE(1, 1, 1, 1))); /* 1/1! */
    yy = _mm256_fmadd_ps(yy, ry, _mm256_permute_ps(K1, _MM_SHUFFLE(1, 1, 1, 1)));
    yx = _mm256_fmadd_ps(yx, rx, _mm256_permute_ps(K1, _MM_SHUFFLE(1, 1, 1, 1))); /* 1/0! */
    yy = _mm256_fmadd_ps(yy, ry, _mm256_permute_ps(K1, _MM_SHUFFLE(1, 1, 1, 1)));

    /* 2^k (exponent-field shift, no multiply) */
    __m256i ex = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(kx),
                                                    _mm256_set1_epi32(127)), 23);
    __m256i ey = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(ky),
                                                    _mm256_set1_epi32(127)), 23);

    fiv_m256x2 r;
    r.a = _mm256_mul_ps(yx, _mm256_castsi256_ps(ex));
    r.b = _mm256_mul_ps(yy, _mm256_castsi256_ps(ey));
    return r;
}

/* 256-bit vector horizontal max */
static inline float fiv_math_hmax256_ps(__m256 v)
{
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m = _mm_max_ps(hi, lo);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(m);
}

/* 256-bit vector horizontal sum */
static inline float fiv_math_hsum256_ps(__m256 v)
{
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(s);
}

/* approximate reciprocal 1/a: rcp + one Newton step (rel. err ~2^-24) */
static inline __m256 fiv_math_rcp256_ps(__m256 a)
{
    __m256 r0 = _mm256_rcp_ps(a);
    return _mm256_mul_ps(r0, _mm256_fnmadd_ps(a, r0, _mm256_set1_ps(2.0f)));
}

#ifdef __cplusplus
}
#endif

#endif  /* FIV_USE_AVX2 */

#endif  /* _FIV_MATH_KERNELS_H_ */
