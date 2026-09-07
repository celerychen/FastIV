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
   Shared SIMD vector primitives for the fiv_math kernels (sigmoid /
   softmax). Only the generic building blocks live here -- the vector exp
   chains, the horizontal reductions and the approximate reciprocal -- so the
   kernel files reuse a single implementation. The op-specific compositions
   (sigmoid's |x| form, the softmax three-pass row loop) stay in their own
   files.

   Two backends are provided, one per active SIMD macro:
     - FIV_USE_AVX2 : 256-bit / 8-lane (fiv_math_exp256_ps & exp256_ps4)
     - FIV_USE_ARM_NEON : 128-bit / 4-lane (fiv_math_exp128_ps & exp128_ps2)
   NEON has its own instruction set and register width, so the 4-lane helpers
   below are native NEON implementations (not transliterations of the AVX
   intrinsics); each maps 1:1 onto the same-named-role AVX2 helper so the
   kernel call sites stay symmetric per architecture.

   The vector exp is a 6th-order Taylor on r = x - k*ln2 (k = round(x/ln2))
   with the input clamped to [-88, 88], so it needs no libcall inside the
   vector path and cannot overflow; 2^k is built with an exponent-field shift.
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

#elif defined(FIV_USE_ARM_NEON)

#include <arm_neon.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 4-lane packed result of the dual-vector exp (NEON counterpart of fiv_m256x2;
   lanes of a and b are processed together so one dual call covers 8 floats). */
typedef struct { float32x4_t a; float32x4_t b; } fiv_f32x4x2;

/* Single-vector exp (4 lanes), NEON counterpart of fiv_math_exp256_ps.
   Same algorithm as the AVX2 path: clamp to [-88, 88], k = floor(x/ln2 + 0.5),
   r = x - k*ln2 (single-rounding FMS), 6th-order Taylor in Horner form with
   FMA, then scale by 2^k built from the exponent field. */
static inline float32x4_t fiv_math_exp128_ps(float32x4_t x)
{
    const float32x4_t invln2 = vdupq_n_f32(1.4426950408889634f); /* 1/ln2  */
    const float32x4_t ln2    = vdupq_n_f32(0.6931471805599453f); /* ln2    */
    const float32x4_t half   = vdupq_n_f32(0.5f);

    /* clamp the input to avoid exp overflow/underflow */
    x = vminq_f32(x, vdupq_n_f32( 88.0f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));

    /* k = floor(x / ln2 + 0.5), r = x - k*ln2 */
    float32x4_t kf = vrndmq_f32(vfmaq_f32(half, x, invln2));
    float32x4_t r  = vfmsq_f32(x, kf, ln2);

    /* 6th-order Taylor of exp(r), Horner all-FMA (ends on the constant 1).
       vfmaq_f32(a, b, c) = a + b*c; the Horner step is y = y*r + c, so the
       coefficient vector is the accumulate operand and (y, r) the fma pair. */
    float32x4_t y = vdupq_n_f32(0.001388888888888889f);                  /* 1/6! */
    y = vfmaq_f32(vdupq_n_f32(0.008333333333333333f), y, r);             /* +1/5! */
    y = vfmaq_f32(vdupq_n_f32(0.041666666666666664f), y, r);             /* +1/4! */
    y = vfmaq_f32(vdupq_n_f32(0.16666666666666666f), y, r);              /* +1/3! */
    y = vfmaq_f32(vdupq_n_f32(0.5f), y, r);                              /* +1/2! */
    y = vfmaq_f32(vdupq_n_f32(1.0f), y, r);                              /* +1/1! */
    y = vfmaq_f32(vdupq_n_f32(1.0f), y, r);                              /* +1/0! (constant) */

    /* 2^k: move (k+127) into the exponent field (no multiply) */
    int32x4_t ki = vcvtq_s32_f32(kf);
    int32x4_t ex = vshlq_n_s32(vaddq_s32(ki, vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(ex));
}

/* Dual-vector exp: two 4-lane chains processed together (8 floats per call),
   NEON counterpart of fiv_math_exp256_ps4. Both Horner chains run beat by
   beat and pull their coefficients with vdupq_lane_f32 from two packed
   constant vectors K0/K1 (4 lanes each; K1 lanes 2/3 are the repeated 1s of
   the last two steps):
     K0 = [1/6!, 1/5!, 1/4!, 1/3!]
     K1 = [1/2!,   1,    1,    1   ] */
static inline fiv_f32x4x2 fiv_math_exp128_ps2(float32x4_t x, float32x4_t y)
{
    const float32x4_t invln2 = vdupq_n_f32(1.4426950408889634f);
    const float32x4_t ln2    = vdupq_n_f32(0.6931471805599453f);
    const float32x4_t half   = vdupq_n_f32(0.5f);

    x = vminq_f32(x, vdupq_n_f32( 88.0f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
    y = vminq_f32(y, vdupq_n_f32( 88.0f));
    y = vmaxq_f32(y, vdupq_n_f32(-88.0f));

    float32x4_t kx = vrndmq_f32(vfmaq_f32(half, x, invln2));
    float32x4_t ky = vrndmq_f32(vfmaq_f32(half, y, invln2));
    float32x4_t rx = vfmsq_f32(x, kx, ln2);
    float32x4_t ry = vfmsq_f32(y, ky, ln2);

    /* Packed coefficients, one 128-bit vector each. vdupq_lane_f32 only
       accepts a 64-bit source, so each 4-lane vector is broadcast through its
       low/high half (on A64 both still compile to a single lane-immediate
       DUP against the same physical register - no GP round trip). */
    const float32x4_t K0 = { 0.001388888888888889f, 0.008333333333333333f,
                             0.041666666666666664f, 0.16666666666666666f };
    const float32x4_t K1 = { 0.5f, 1.0f, 1.0f, 1.0f };
    const float32x2_t K0_lo = vget_low_f32(K0);
    const float32x2_t K0_hi = vget_high_f32(K0);
    const float32x2_t K1_lo = vget_low_f32(K1);

    float32x4_t yx = vdupq_lane_f32(K0_lo, 0);       /* 1/6! */
    float32x4_t yy = yx;
    yx = vfmaq_f32(vdupq_lane_f32(K0_lo, 1), yx, rx); /* +1/5! */
    yy = vfmaq_f32(vdupq_lane_f32(K0_lo, 1), yy, ry);
    yx = vfmaq_f32(vdupq_lane_f32(K0_hi, 0), yx, rx); /* +1/4! */
    yy = vfmaq_f32(vdupq_lane_f32(K0_hi, 0), yy, ry);
    yx = vfmaq_f32(vdupq_lane_f32(K0_hi, 1), yx, rx); /* +1/3! */
    yy = vfmaq_f32(vdupq_lane_f32(K0_hi, 1), yy, ry);
    yx = vfmaq_f32(vdupq_lane_f32(K1_lo, 0), yx, rx); /* +1/2! */
    yy = vfmaq_f32(vdupq_lane_f32(K1_lo, 0), yy, ry);
    yx = vfmaq_f32(vdupq_lane_f32(K1_lo, 1), yx, rx); /* +1/1! */
    yy = vfmaq_f32(vdupq_lane_f32(K1_lo, 1), yy, ry);
    yx = vfmaq_f32(vdupq_lane_f32(K1_lo, 1), yx, rx); /* +1/0! */
    yy = vfmaq_f32(vdupq_lane_f32(K1_lo, 1), yy, ry);

    int32x4_t ex = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(kx), vdupq_n_s32(127)), 23);
    int32x4_t ey = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(ky), vdupq_n_s32(127)), 23);

    fiv_f32x4x2 out;
    out.a = vmulq_f32(yx, vreinterpretq_f32_s32(ex));
    out.b = vmulq_f32(yy, vreinterpretq_f32_s32(ey));
    return out;
}

/* 4-lane horizontal max (NEON counterpart of fiv_math_hmax256_ps):
   pairwise max across the low/high halves, then across the two lanes. */
static inline float fiv_math_hmax128_ps(float32x4_t v)
{
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t m  = vpmax_f32(lo, hi);
    m = vpmax_f32(m, m);
    return vget_lane_f32(m, 0);
}

/* 4-lane horizontal sum (NEON counterpart of fiv_math_hsum256_ps). */
static inline float fiv_math_hsum128_ps(float32x4_t v)
{
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t s  = vpadd_f32(lo, hi);
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
}

/* Approximate reciprocal 1/a (NEON counterpart of fiv_math_rcp256_ps).
   vrecpeq_f32 gives ~2^-8 accuracy; two Newton refinements (vrecpsq_f32 =
   2 - a*r) push it to ~full float precision, so no division is needed. */
static inline float32x4_t fiv_math_rcp128_ps(float32x4_t a)
{
    float32x4_t r = vrecpeq_f32(a);
    r = vmulq_f32(r, vrecpsq_f32(a, r));   /* Newton 1 -> ~2^-16 */
    r = vmulq_f32(r, vrecpsq_f32(a, r));   /* Newton 2 -> ~2^-32 */
    return r;
}

#ifdef __cplusplus
}
#endif

#endif  /* FIV_USE_AVX2 / FIV_USE_ARM_NEON */

#endif  /* _FIV_MATH_KERNELS_H_ */
