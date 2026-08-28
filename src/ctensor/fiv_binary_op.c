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

#include "fiv_binary_op.h"
#include <string.h>

/* int32 scalar fallback */
#if !defined(FIV_USE_AVX2) && !defined(FIV_USE_ARM_NEON)
void fiv_add_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n)
{
    for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
}

void fiv_sub_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n)
{
    for (size_t i = 0; i < n; i++) c[i] = a[i] - b[i];
}

void fiv_mul_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n)
{
    for (size_t i = 0; i < n; i++) c[i] = a[i] * b[i];
}

void fiv_div_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n)
{
    for (size_t i = 0; i < n; i++) c[i] = a[i] / b[i];
}
#endif /* int32 scalar fallback */

/* int32 AVX2 (16 int32/step, unrolled x2) */
#if defined(FIV_USE_AVX2)
#include <immintrin.h>

void fiv_add_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i va0 = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb0 = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i va1 = _mm256_loadu_si256((const __m256i*)(a + i + 8));
        __m256i vb1 = _mm256_loadu_si256((const __m256i*)(b + i + 8));
        __m256i vc0 = _mm256_add_epi32(va0, vb0);
        __m256i vc1 = _mm256_add_epi32(va1, vb1);
        _mm256_storeu_si256((__m256i*)(c + i), vc0);
        _mm256_storeu_si256((__m256i*)(c + i + 8), vc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        _mm256_storeu_si256((__m256i*)(c + i), _mm256_add_epi32(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
}

void fiv_sub_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i va0 = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb0 = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i va1 = _mm256_loadu_si256((const __m256i*)(a + i + 8));
        __m256i vb1 = _mm256_loadu_si256((const __m256i*)(b + i + 8));
        __m256i vc0 = _mm256_sub_epi32(va0, vb0);
        __m256i vc1 = _mm256_sub_epi32(va1, vb1);
        _mm256_storeu_si256((__m256i*)(c + i), vc0);
        _mm256_storeu_si256((__m256i*)(c + i + 8), vc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        _mm256_storeu_si256((__m256i*)(c + i), _mm256_sub_epi32(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] - b[i];
}

void fiv_mul_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i va0 = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb0 = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i va1 = _mm256_loadu_si256((const __m256i*)(a + i + 8));
        __m256i vb1 = _mm256_loadu_si256((const __m256i*)(b + i + 8));
        __m256i vc0 = _mm256_mullo_epi32(va0, vb0);
        __m256i vc1 = _mm256_mullo_epi32(va1, vb1);
        _mm256_storeu_si256((__m256i*)(c + i), vc0);
        _mm256_storeu_si256((__m256i*)(c + i + 8), vc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        _mm256_storeu_si256((__m256i*)(c + i), _mm256_mullo_epi32(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] * b[i];
}

void fiv_div_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n)
{
    for (size_t i = 0; i < n; i++) c[i] = a[i] / b[i];
}

/* int32 NEON (8 int32/step, unrolled x2) */
#elif defined(FIV_USE_ARM_NEON)
#include <arm_neon.h>

void fiv_add_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int32x4_t va0 = vld1q_s32(a + i);
        int32x4_t vb0 = vld1q_s32(b + i);
        int32x4_t va1 = vld1q_s32(a + i + 4);
        int32x4_t vb1 = vld1q_s32(b + i + 4);
        int32x4_t vc0 = vaddq_s32(va0, vb0);
        int32x4_t vc1 = vaddq_s32(va1, vb1);
        vst1q_s32(c + i, vc0);
        vst1q_s32(c + i + 4, vc1);
    }
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(a + i);
        int32x4_t vb = vld1q_s32(b + i);
        vst1q_s32(c + i, vaddq_s32(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
}

void fiv_sub_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int32x4_t va0 = vld1q_s32(a + i);
        int32x4_t vb0 = vld1q_s32(b + i);
        int32x4_t va1 = vld1q_s32(a + i + 4);
        int32x4_t vb1 = vld1q_s32(b + i + 4);
        int32x4_t vc0 = vsubq_s32(va0, vb0);
        int32x4_t vc1 = vsubq_s32(va1, vb1);
        vst1q_s32(c + i, vc0);
        vst1q_s32(c + i + 4, vc1);
    }
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(a + i);
        int32x4_t vb = vld1q_s32(b + i);
        vst1q_s32(c + i, vsubq_s32(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] - b[i];
}

void fiv_mul_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int32x4_t va0 = vld1q_s32(a + i);
        int32x4_t vb0 = vld1q_s32(b + i);
        int32x4_t va1 = vld1q_s32(a + i + 4);
        int32x4_t vb1 = vld1q_s32(b + i + 4);
        int32x4_t vc0 = vmulq_s32(va0, vb0);
        int32x4_t vc1 = vmulq_s32(va1, vb1);
        vst1q_s32(c + i, vc0);
        vst1q_s32(c + i + 4, vc1);
    }
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(a + i);
        int32x4_t vb = vld1q_s32(b + i);
        vst1q_s32(c + i, vmulq_s32(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] * b[i];
}

void fiv_div_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n)
{
    for (size_t i = 0; i < n; i++) c[i] = a[i] / b[i];
}
#endif /* int32 NEON */

/* float32 scalar fallback */
#if !defined(FIV_USE_AVX) && !defined(FIV_USE_ARM_NEON)
void fiv_add_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n)
{
    for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
}

void fiv_sub_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n)
{
    for (size_t i = 0; i < n; i++) c[i] = a[i] - b[i];
}

void fiv_mul_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n)
{
    for (size_t i = 0; i < n; i++) c[i] = a[i] * b[i];
}

void fiv_div_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n)
{
    for (size_t i = 0; i < n; i++) c[i] = a[i] / b[i];
}
#endif /* float32 scalar fallback */

/* float32 AVX (16 float32/step, unrolled x2) */
#if defined(FIV_USE_AVX)
#include <immintrin.h>

void fiv_add_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 vc0 = _mm256_add_ps(va0, vb0);
        __m256 vc1 = _mm256_add_ps(va1, vb1);
        _mm256_storeu_ps(c + i, vc0);
        _mm256_storeu_ps(c + i + 8, vc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
}

void fiv_sub_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 vc0 = _mm256_sub_ps(va0, vb0);
        __m256 vc1 = _mm256_sub_ps(va1, vb1);
        _mm256_storeu_ps(c + i, vc0);
        _mm256_storeu_ps(c + i + 8, vc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_sub_ps(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] - b[i];
}

void fiv_mul_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 vc0 = _mm256_mul_ps(va0, vb0);
        __m256 vc1 = _mm256_mul_ps(va1, vb1);
        _mm256_storeu_ps(c + i, vc0);
        _mm256_storeu_ps(c + i + 8, vc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_mul_ps(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] * b[i];
}

void fiv_div_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 vc0 = _mm256_div_ps(va0, vb0);
        __m256 vc1 = _mm256_div_ps(va1, vb1);
        _mm256_storeu_ps(c + i, vc0);
        _mm256_storeu_ps(c + i + 8, vc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_div_ps(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] / b[i];
}

/* float32 NEON (8 float32/step, unrolled x2) */
#elif defined(FIV_USE_ARM_NEON)
#include <arm_neon.h>

void fiv_add_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t vc0 = vaddq_f32(va0, vb0);
        float32x4_t vc1 = vaddq_f32(va1, vb1);
        vst1q_f32(c + i, vc0);
        vst1q_f32(c + i + 4, vc1);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(c + i, vaddq_f32(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
}

void fiv_sub_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t vc0 = vsubq_f32(va0, vb0);
        float32x4_t vc1 = vsubq_f32(va1, vb1);
        vst1q_f32(c + i, vc0);
        vst1q_f32(c + i + 4, vc1);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(c + i, vsubq_f32(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] - b[i];
}

void fiv_mul_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t vc0 = vmulq_f32(va0, vb0);
        float32x4_t vc1 = vmulq_f32(va1, vb1);
        vst1q_f32(c + i, vc0);
        vst1q_f32(c + i + 4, vc1);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(c + i, vmulq_f32(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] * b[i];
}

#if defined(__aarch64__)
/* AArch64 has a native float-divide instruction, use it directly. */
void fiv_div_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t vc0 = vdivq_f32(va0, vb0);
        float32x4_t vc1 = vdivq_f32(va1, vb1);
        vst1q_f32(c + i, vc0);
        vst1q_f32(c + i + 4, vc1);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(c + i, vdivq_f32(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] / b[i];
}
#else
/* Plain NEON (armv7 / aarch32) has no float-divide instruction;
 * use vrecpeq_f32 + 2x Newton-Raphson refinement (~24-bit accuracy). */
void fiv_div_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t recip0 = vrecpeq_f32(vb0);                 /* initial reciprocal ~8 bits */
        recip0 = vmulq_f32(recip0, vrecpsq_f32(vb0, recip0));  /* 1st refine ~16 bits */
        recip0 = vmulq_f32(recip0, vrecpsq_f32(vb0, recip0));  /* 2nd refine ~24 bits */
        float32x4_t vc0 = vmulq_f32(va0, recip0);

        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t recip1 = vrecpeq_f32(vb1);
        recip1 = vmulq_f32(recip1, vrecpsq_f32(vb1, recip1));
        recip1 = vmulq_f32(recip1, vrecpsq_f32(vb1, recip1));
        float32x4_t vc1 = vmulq_f32(va1, recip1);

        vst1q_f32(c + i, vc0);
        vst1q_f32(c + i + 4, vc1);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t recip = vrecpeq_f32(vb);
        recip = vmulq_f32(recip, vrecpsq_f32(vb, recip));
        recip = vmulq_f32(recip, vrecpsq_f32(vb, recip));
        vst1q_f32(c + i, vmulq_f32(va, recip));
    }
    for (; i < n; i++) c[i] = a[i] / b[i];
}
#endif /* __aarch64__ */
#endif /* float32 NEON */

/* float64 scalar binary ops: portable scalar loops (mirrors the float32 scalar
   fallback; float64 SIMD was not provided upstream, so kept scalar). */
void fiv_add_ivf64(ivf64* restrict c, const ivf64* restrict a, const ivf64* restrict b, size_t n)
{
    for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
}

void fiv_sub_ivf64(ivf64* restrict c, const ivf64* restrict a, const ivf64* restrict b, size_t n)
{
    for (size_t i = 0; i < n; i++) c[i] = a[i] - b[i];
}

void fiv_mul_ivf64(ivf64* restrict c, const ivf64* restrict a, const ivf64* restrict b, size_t n)
{
    for (size_t i = 0; i < n; i++) c[i] = a[i] * b[i];
}

void fiv_div_ivf64(ivf64* restrict c, const ivf64* restrict a, const ivf64* restrict b, size_t n)
{
    for (size_t i = 0; i < n; i++) c[i] = a[i] / b[i];
}
