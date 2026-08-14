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

/*
 * matvet_v0.c - 矩阵-向量乘法实现
 *
 * 三套独立实现, 函数名后缀区分指令集:
 *   (无后缀)  纯 C (标量 + 4 路循环展开)
 *   _avx      AVX2 + FMA (_mm256_fmadd_ps 乘累加)
 *   _neon     ARM NEON (vmlaq_f32 乘累加)
 *
 * 关键修复 (相对初版): 行间步长统一使用 mat_stride, 不再误用 cols,
 * 因此 mat_stride > cols (带 padding) 的矩阵也能正确计算。
 */

#include "matvet.h"
#include <string.h>   /* memset: 转置乘向量需要先清零 dst */

#ifndef IVF32_DEFINED
/* 自包含回退定义; 若工程已定义 ivf32, 在包含 matvet.h 前 #define IVF32_DEFINED */
typedef float ivf32;
#endif


/* ============================ 纯 C 实现 ============================ */

/* dst = mat · vec  (每行一个点积, 4 行一组复用向量加载) */
void mat_mul_vet_real32(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec)
{
    int i, j;
    for (i = 0; i <= rows - 4; i += 4)
    {
        const ivf32* r0 = mat + (size_t)(i + 0) * mat_stride;
        const ivf32* r1 = mat + (size_t)(i + 1) * mat_stride;
        const ivf32* r2 = mat + (size_t)(i + 2) * mat_stride;
        const ivf32* r3 = mat + (size_t)(i + 3) * mat_stride;
        const ivf32* v  = vec;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

        for (j = 0; j <= cols - 4; j += 4)
        {
            float t0 = v[0], t1 = v[1], t2 = v[2], t3 = v[3]; v += 4;
            s0 += r0[0]*t0 + r0[1]*t1 + r0[2]*t2 + r0[3]*t3;
            s1 += r1[0]*t0 + r1[1]*t1 + r1[2]*t2 + r1[3]*t3;
            s2 += r2[0]*t0 + r2[1]*t1 + r2[2]*t2 + r2[3]*t3;
            s3 += r3[0]*t0 + r3[1]*t1 + r3[2]*t2 + r3[3]*t3;
            r0 += 4; r1 += 4; r2 += 4; r3 += 4;
        }
        for (; j < cols; j++)
        {
            float t = *v++;
            s0 += r0[0]*t; r0++;
            s1 += r1[0]*t; r1++;
            s2 += r2[0]*t; r2++;
            s3 += r3[0]*t; r3++;
        }
        dst[i + 0] = (ivf32)s0;
        dst[i + 1] = (ivf32)s1;
        dst[i + 2] = (ivf32)s2;
        dst[i + 3] = (ivf32)s3;
    }
    for (; i < rows; i++)
    {
        const ivf32* r = mat + (size_t)i * mat_stride;
        const ivf32* v = vec;
        float s = 0.0f; int j;
        for (j = 0; j <= cols - 4; j += 4)
        {
            s += r[0]*v[0] + r[1]*v[1] + r[2]*v[2] + r[3]*v[3];
            r += 4; v += 4;
        }
        for (; j < cols; j++) { s += r[0]*v[0]; r++; v++; }
        dst[i] = (ivf32)s;
    }
}

/* dst = mat^T · vec  (先清零, 再按行流式累加到 dst[0..cols)) */
void mat_t_mul_vet_real32(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec)
{
    int i, j;
    memset(dst, 0, (size_t)cols * sizeof(ivf32));

    for (i = 0; i <= rows - 4; i += 4)
    {
        const ivf32* r0 = mat + (size_t)(i + 0) * mat_stride;
        const ivf32* r1 = mat + (size_t)(i + 1) * mat_stride;
        const ivf32* r2 = mat + (size_t)(i + 2) * mat_stride;
        const ivf32* r3 = mat + (size_t)(i + 3) * mat_stride;
        float v0 = (float)vec[i + 0], v1 = (float)vec[i + 1], v2 = (float)vec[i + 2], v3 = (float)vec[i + 3];

        for (j = 0; j <= cols - 4; j += 4)
        {
            dst[j + 0] += (ivf32)(r0[0]*v0 + r1[0]*v1 + r2[0]*v2 + r3[0]*v3);
            dst[j + 1] += (ivf32)(r0[1]*v0 + r1[1]*v1 + r2[1]*v2 + r3[1]*v3);
            dst[j + 2] += (ivf32)(r0[2]*v0 + r1[2]*v1 + r2[2]*v2 + r3[2]*v3);
            dst[j + 3] += (ivf32)(r0[3]*v0 + r1[3]*v1 + r2[3]*v2 + r3[3]*v3);
            r0 += 4; r1 += 4; r2 += 4; r3 += 4;
        }
        for (; j < cols; j++)
        {
            dst[j] += (ivf32)(r0[0]*v0 + r1[0]*v1 + r2[0]*v2 + r3[0]*v3);
            r0++; r1++; r2++; r3++;
        }
    }
    for (; i < rows; i++)
    {
        const ivf32* r = mat + (size_t)i * mat_stride;
        float vi = (float)vec[i];
        int j;
        for (j = 0; j <= cols - 4; j += 4)
        {
            dst[j + 0] += (ivf32)(r[0]*vi);
            dst[j + 1] += (ivf32)(r[1]*vi);
            dst[j + 2] += (ivf32)(r[2]*vi);
            dst[j + 3] += (ivf32)(r[3]*vi);
            r += 4;
        }
        for (; j < cols; j++) { dst[j] += (ivf32)(r[0]*vi); r++; }
    }
}


/* ====================== AVX2 + FMA 实现 ====================== */
#if defined(__AVX2__)

#include <immintrin.h>

/* 256 位向量水平求和, 返回标量 */
static inline float matvet_hsum256(__m256 v)
{
    __m128 v128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    v128 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
    v128 = _mm_add_ss(v128, _mm_shuffle_ps(v128, v128, 0x55));
    return _mm_cvtss_f32(v128);
}

/* dst = mat · vec  (8 行一组, 每组 8 路 FMA 累加) */
void mat_mul_vet_real32_avx(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec)
{
    int i, j;
    for (i = 0; i <= rows - 8; i += 8)
    {
        const ivf32* r0 = mat + (size_t)(i + 0) * mat_stride;
        const ivf32* r1 = mat + (size_t)(i + 1) * mat_stride;
        const ivf32* r2 = mat + (size_t)(i + 2) * mat_stride;
        const ivf32* r3 = mat + (size_t)(i + 3) * mat_stride;
        const ivf32* r4 = mat + (size_t)(i + 4) * mat_stride;
        const ivf32* r5 = mat + (size_t)(i + 5) * mat_stride;
        const ivf32* r6 = mat + (size_t)(i + 6) * mat_stride;
        const ivf32* r7 = mat + (size_t)(i + 7) * mat_stride;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps(), a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
        const ivf32* v = vec;

        for (j = 0; j <= cols - 8; j += 8)
        {
            __m256 t = _mm256_loadu_ps(v); v += 8;
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0), t, a0); r0 += 8;
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1), t, a1); r1 += 8;
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2), t, a2); r2 += 8;
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r3), t, a3); r3 += 8;
            a4 = _mm256_fmadd_ps(_mm256_loadu_ps(r4), t, a4); r4 += 8;
            a5 = _mm256_fmadd_ps(_mm256_loadu_ps(r5), t, a5); r5 += 8;
            a6 = _mm256_fmadd_ps(_mm256_loadu_ps(r6), t, a6); r6 += 8;
            a7 = _mm256_fmadd_ps(_mm256_loadu_ps(r7), t, a7); r7 += 8;
        }
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
        for (; j < cols; j++)
        {
            float t = *v++;
            s0 += r0[0]*t; r0++;
            s1 += r1[0]*t; r1++;
            s2 += r2[0]*t; r2++;
            s3 += r3[0]*t; r3++;
            s4 += r4[0]*t; r4++;
            s5 += r5[0]*t; r5++;
            s6 += r6[0]*t; r6++;
            s7 += r7[0]*t; r7++;
        }
        dst[i + 0] = (ivf32)(matvet_hsum256(a0) + s0);
        dst[i + 1] = (ivf32)(matvet_hsum256(a1) + s1);
        dst[i + 2] = (ivf32)(matvet_hsum256(a2) + s2);
        dst[i + 3] = (ivf32)(matvet_hsum256(a3) + s3);
        dst[i + 4] = (ivf32)(matvet_hsum256(a4) + s4);
        dst[i + 5] = (ivf32)(matvet_hsum256(a5) + s5);
        dst[i + 6] = (ivf32)(matvet_hsum256(a6) + s6);
        dst[i + 7] = (ivf32)(matvet_hsum256(a7) + s7);
    }
    for (; i < rows; i++)
    {
        const ivf32* r = mat + (size_t)i * mat_stride;
        __m256 acc = _mm256_setzero_ps();
        const ivf32* v = vec; int j;
        for (j = 0; j <= cols - 8; j += 8)
        {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(r), _mm256_loadu_ps(v), acc);
            r += 8; v += 8;
        }
        float s = matvet_hsum256(acc);
        for (; j < cols; j++) { s += r[0]*v[0]; r++; v++; }
        dst[i] = (ivf32)s;
    }
}

/* dst = mat^T · vec  (先清零, 8 行一组, 每组把向量分量广播后 FMA 累加到 dst) */
void mat_t_mul_vet_real32_avx(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec)
{
    int i, j;
    memset(dst, 0, (size_t)cols * sizeof(ivf32));

    for (i = 0; i <= rows - 8; i += 8)
    {
        const ivf32* r0 = mat + (size_t)(i + 0) * mat_stride;
        const ivf32* r1 = mat + (size_t)(i + 1) * mat_stride;
        const ivf32* r2 = mat + (size_t)(i + 2) * mat_stride;
        const ivf32* r3 = mat + (size_t)(i + 3) * mat_stride;
        const ivf32* r4 = mat + (size_t)(i + 4) * mat_stride;
        const ivf32* r5 = mat + (size_t)(i + 5) * mat_stride;
        const ivf32* r6 = mat + (size_t)(i + 6) * mat_stride;
        const ivf32* r7 = mat + (size_t)(i + 7) * mat_stride;
        __m256 v0 = _mm256_set1_ps((float)vec[i + 0]);
        __m256 v1 = _mm256_set1_ps((float)vec[i + 1]);
        __m256 v2 = _mm256_set1_ps((float)vec[i + 2]);
        __m256 v3 = _mm256_set1_ps((float)vec[i + 3]);
        __m256 v4 = _mm256_set1_ps((float)vec[i + 4]);
        __m256 v5 = _mm256_set1_ps((float)vec[i + 5]);
        __m256 v6 = _mm256_set1_ps((float)vec[i + 6]);
        __m256 v7 = _mm256_set1_ps((float)vec[i + 7]);

        for (j = 0; j <= cols - 8; j += 8)
        {
            __m256 d = _mm256_loadu_ps(&dst[j]);
            d = _mm256_fmadd_ps(_mm256_loadu_ps(r0), v0, d); r0 += 8;
            d = _mm256_fmadd_ps(_mm256_loadu_ps(r1), v1, d); r1 += 8;
            d = _mm256_fmadd_ps(_mm256_loadu_ps(r2), v2, d); r2 += 8;
            d = _mm256_fmadd_ps(_mm256_loadu_ps(r3), v3, d); r3 += 8;
            d = _mm256_fmadd_ps(_mm256_loadu_ps(r4), v4, d); r4 += 8;
            d = _mm256_fmadd_ps(_mm256_loadu_ps(r5), v5, d); r5 += 8;
            d = _mm256_fmadd_ps(_mm256_loadu_ps(r6), v6, d); r6 += 8;
            d = _mm256_fmadd_ps(_mm256_loadu_ps(r7), v7, d); r7 += 8;
            _mm256_storeu_ps(&dst[j], d);
        }
        for (; j < cols; j++)
        {
            float d = (float)dst[j];
            d += r0[0] * (float)vec[i + 0]; r0++;
            d += r1[0] * (float)vec[i + 1]; r1++;
            d += r2[0] * (float)vec[i + 2]; r2++;
            d += r3[0] * (float)vec[i + 3]; r3++;
            d += r4[0] * (float)vec[i + 4]; r4++;
            d += r5[0] * (float)vec[i + 5]; r5++;
            d += r6[0] * (float)vec[i + 6]; r6++;
            d += r7[0] * (float)vec[i + 7]; r7++;
            dst[j] = (ivf32)d;
        }
    }
    for (; i < rows; i++)
    {
        const ivf32* r = mat + (size_t)i * mat_stride;
        __m256 v = _mm256_set1_ps((float)vec[i]);
        int j;
        for (j = 0; j <= cols - 8; j += 8)
        {
            __m256 d = _mm256_loadu_ps(&dst[j]);
            d = _mm256_fmadd_ps(_mm256_loadu_ps(r), v, d);
            _mm256_storeu_ps(&dst[j], d);
            r += 8;
        }
        for (; j < cols; j++) { dst[j] += (ivf32)(r[0] * (float)vec[i]); r++; }
    }
}

#endif /* __AVX2__ */


/* ============================ ARM NEON 实现 ============================ */
#if defined(__ARM_NEON) || defined(__aarch64__)

#include <arm_neon.h>

/* 128 位向量 (4×float) 水平求和 */
static inline float matvet_neon_hsum(float32x4_t v)
{
    float32x2_t s = vadd_f32(vget_high_f32(v), vget_low_f32(v));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
}

/* dst = mat · vec  (4 行一组, 每组 4 路乘累加 vmlaq_f32) */
void mat_mul_vet_real32_neon(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec)
{
    int i, j;
    for (i = 0; i <= rows - 4; i += 4)
    {
        const ivf32* r0 = mat + (size_t)(i + 0) * mat_stride;
        const ivf32* r1 = mat + (size_t)(i + 1) * mat_stride;
        const ivf32* r2 = mat + (size_t)(i + 2) * mat_stride;
        const ivf32* r3 = mat + (size_t)(i + 3) * mat_stride;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f), a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        const ivf32* v = vec;

        for (j = 0; j <= cols - 4; j += 4)
        {
            float32x4_t t = vld1q_f32(v); v += 4;
            a0 = vmlaq_f32(a0, vld1q_f32(r0), t); r0 += 4;
            a1 = vmlaq_f32(a1, vld1q_f32(r1), t); r1 += 4;
            a2 = vmlaq_f32(a2, vld1q_f32(r2), t); r2 += 4;
            a3 = vmlaq_f32(a3, vld1q_f32(r3), t); r3 += 4;
        }
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (; j < cols; j++)
        {
            float t = *v++;
            s0 += r0[0]*t; r0++;
            s1 += r1[0]*t; r1++;
            s2 += r2[0]*t; r2++;
            s3 += r3[0]*t; r3++;
        }
        dst[i + 0] = (ivf32)(matvet_neon_hsum(a0) + s0);
        dst[i + 1] = (ivf32)(matvet_neon_hsum(a1) + s1);
        dst[i + 2] = (ivf32)(matvet_neon_hsum(a2) + s2);
        dst[i + 3] = (ivf32)(matvet_neon_hsum(a3) + s3);
    }
    for (; i < rows; i++)
    {
        const ivf32* r = mat + (size_t)i * mat_stride;
        float32x4_t a = vdupq_n_f32(0.0f);
        const ivf32* v = vec; int j;
        for (j = 0; j <= cols - 4; j += 4)
        {
            a = vmlaq_f32(a, vld1q_f32(r), vld1q_f32(v));
            r += 4; v += 4;
        }
        float s = matvet_neon_hsum(a);
        for (; j < cols; j++) { s += r[0]*v[0]; r++; v++; }
        dst[i] = (ivf32)s;
    }
}

/* dst = mat^T · vec  (先清零, 4 行一组, 向量分量广播后乘累加) */
void mat_t_mul_vet_real32_neon(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec)
{
    int i, j;
    memset(dst, 0, (size_t)cols * sizeof(ivf32));

    for (i = 0; i <= rows - 4; i += 4)
    {
        const ivf32* r0 = mat + (size_t)(i + 0) * mat_stride;
        const ivf32* r1 = mat + (size_t)(i + 1) * mat_stride;
        const ivf32* r2 = mat + (size_t)(i + 2) * mat_stride;
        const ivf32* r3 = mat + (size_t)(i + 3) * mat_stride;
        float32x4_t v0 = vdupq_n_f32((float)vec[i + 0]);
        float32x4_t v1 = vdupq_n_f32((float)vec[i + 1]);
        float32x4_t v2 = vdupq_n_f32((float)vec[i + 2]);
        float32x4_t v3 = vdupq_n_f32((float)vec[i + 3]);

        for (j = 0; j <= cols - 4; j += 4)
        {
            float32x4_t d = vld1q_f32(&dst[j]);
            d = vmlaq_f32(d, vld1q_f32(r0), v0); r0 += 4;
            d = vmlaq_f32(d, vld1q_f32(r1), v1); r1 += 4;
            d = vmlaq_f32(d, vld1q_f32(r2), v2); r2 += 4;
            d = vmlaq_f32(d, vld1q_f32(r3), v3); r3 += 4;
            vst1q_f32(&dst[j], d);
        }
        for (; j < cols; j++)
        {
            float d = (float)dst[j];
            d += r0[0] * (float)vec[i + 0]; r0++;
            d += r1[0] * (float)vec[i + 1]; r1++;
            d += r2[0] * (float)vec[i + 2]; r2++;
            d += r3[0] * (float)vec[i + 3]; r3++;
            dst[j] = (ivf32)d;
        }
    }
    for (; i < rows; i++)
    {
        const ivf32* r = mat + (size_t)i * mat_stride;
        float32x4_t v = vdupq_n_f32((float)vec[i]);
        int j;
        for (j = 0; j <= cols - 4; j += 4)
        {
            float32x4_t d = vld1q_f32(&dst[j]);
            d = vmlaq_f32(d, vld1q_f32(r), v);
            vst1q_f32(&dst[j], d);
            r += 4;
        }
        for (; j < cols; j++) { dst[j] += (ivf32)(r[0] * (float)vec[i]); r++; }
    }
}

#endif /* __ARM_NEON || __aarch64__ */
