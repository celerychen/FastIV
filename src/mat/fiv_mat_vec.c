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

#include "fiv_mat_vec.h"
#include "fiv_mat_vec_db.h"
#include <string.h>   /* memset for the transposed mat*vec path */
#include <math.h>     /* sqrt / sqrtf for L2 norm */
#include <stdint.h>   /* uint32_t / uint64_t for bit-twiddle abs */

/* ==================== Scalar implementation (4-way unrolled) ==================== */
#if !defined(FIV_USE_AVX2) && !defined(FIV_USE_ARM_NEON)

/* dst = mat * vec, one dot product per row */
static void _fiv_mat_mul_vec_real32(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec)
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

/* dst = mat^T * vec: zero dst first, then stream rows into dst[0..cols) */
static void _fiv_mat_t_mul_vec_real32(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec)
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

#endif /* !FIV_USE_AVX2 && !FIV_USE_ARM_NEON */


/* ==================== AVX2 + FMA implementation ==================== */
#if defined(FIV_USE_AVX2)

#include <immintrin.h>

/* horizontal sum of a 256-bit vector into a scalar */
static inline float _fiv_matvec_hsum256(__m256 v)
{
    __m128 v128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    v128 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
    v128 = _mm_add_ss(v128, _mm_shuffle_ps(v128, v128, 0x55));
    return _mm_cvtss_f32(v128);
}

/* dst = mat * vec: 8 rows per group, 8-way FMA accumulation */
static void _fiv_mat_mul_vec_real32_avx(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec)
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
        dst[i + 0] = (ivf32)(_fiv_matvec_hsum256(a0) + s0);
        dst[i + 1] = (ivf32)(_fiv_matvec_hsum256(a1) + s1);
        dst[i + 2] = (ivf32)(_fiv_matvec_hsum256(a2) + s2);
        dst[i + 3] = (ivf32)(_fiv_matvec_hsum256(a3) + s3);
        dst[i + 4] = (ivf32)(_fiv_matvec_hsum256(a4) + s4);
        dst[i + 5] = (ivf32)(_fiv_matvec_hsum256(a5) + s5);
        dst[i + 6] = (ivf32)(_fiv_matvec_hsum256(a6) + s6);
        dst[i + 7] = (ivf32)(_fiv_matvec_hsum256(a7) + s7);
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
        float s = _fiv_matvec_hsum256(acc);
        for (; j < cols; j++) { s += r[0]*v[0]; r++; v++; }
        dst[i] = (ivf32)s;
    }
}

/* dst = mat^T * vec: zero dst, 8 rows per group, broadcast vector components and FMA into dst */
static void _fiv_mat_t_mul_vec_real32_avx(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec)
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

#endif /* FIV_USE_AVX2 */


/* ==================== ARM NEON implementation ==================== */
#if defined(FIV_USE_ARM_NEON)

#include <arm_neon.h>

/* horizontal sum of a 128-bit (4x float) vector */
static inline float _fiv_matvec_neon_hsum(float32x4_t v)
{
    float32x2_t s = vadd_f32(vget_high_f32(v), vget_low_f32(v));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
}

/* dst = mat * vec: 4 rows per group, 4-way vmlaq_f32 accumulation */
static void _fiv_mat_mul_vec_real32_neon(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec)
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
        dst[i + 0] = (ivf32)(_fiv_matvec_neon_hsum(a0) + s0);
        dst[i + 1] = (ivf32)(_fiv_matvec_neon_hsum(a1) + s1);
        dst[i + 2] = (ivf32)(_fiv_matvec_neon_hsum(a2) + s2);
        dst[i + 3] = (ivf32)(_fiv_matvec_neon_hsum(a3) + s3);
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
        float s = _fiv_matvec_neon_hsum(a);
        for (; j < cols; j++) { s += r[0]*v[0]; r++; v++; }
        dst[i] = (ivf32)s;
    }
}

/* dst = mat^T * vec: zero dst, 4 rows per group, broadcast and accumulate */
static void _fiv_mat_t_mul_vec_real32_neon(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec)
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

#endif /* FIV_USE_ARM_NEON */


/* ==================== Public API ==================== */
fiv_ret fiv_matrix_mul_vec(fiv_vec* dst, const fiv_mat* mat, const fiv_vec* vec, int transpose)
{
    if (dst == NULL || mat == NULL || vec == NULL) return FIV_RET_ERR_PARA;
    if (dst->data.ptr == NULL || mat->data.ptr == NULL || vec->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (dst->data.ptr == vec->data.ptr) return FIV_RET_ERR_PARA;   /* in-place not supported */
    if (dst->data_continue == 0 || mat->data_continue == 0 || vec->data_continue == 0) return FIV_RET_ERR_PARA;
    if (mat->dtype == FIV_64F1)
        return fiv_matrix_mul_vec_real64(dst, mat, vec, transpose);
    if (mat->dtype != FIV_32F1 || vec->dtype != FIV_32F1 || dst->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;

    const size_t rows = mat->shapes[0];   /* mat rows */
    const size_t cols = mat->shapes[1];   /* mat cols */
    const int    mat_stride = (int)(mat->strides[0] / (size_t)mat->element_bytes);   /* elements per row */
    ivf32*       dst_f = (ivf32*)dst->data.ptr;
    const ivf32* vec_f = (const ivf32*)vec->data.ptr;
    const ivf32* mat_f = (const ivf32*)mat->data.ptr;

    if (transpose == 0) {
        /* dst = mat * vec: vec holds cols entries, result holds rows entries */
        if (vec->shapes[0] < cols) return FIV_RET_ERR_PARA;
        if (dst->shapes[0] < rows) return FIV_RET_ERR_PARA;
#if defined(FIV_USE_AVX2)
        _fiv_mat_mul_vec_real32_avx(dst_f, mat_f, (int)rows, (int)cols, mat_stride, vec_f);
#elif defined(FIV_USE_ARM_NEON)
        _fiv_mat_mul_vec_real32_neon(dst_f, mat_f, (int)rows, (int)cols, mat_stride, vec_f);
#else
        _fiv_mat_mul_vec_real32(dst_f, mat_f, (int)rows, (int)cols, mat_stride, vec_f);
#endif
    } else {
        /* dst = mat^T * vec: vec holds rows entries, result holds cols entries */
        if (vec->shapes[0] < rows) return FIV_RET_ERR_PARA;
        if (dst->shapes[0] < cols) return FIV_RET_ERR_PARA;
#if defined(FIV_USE_AVX2)
        _fiv_mat_t_mul_vec_real32_avx(dst_f, mat_f, (int)rows, (int)cols, mat_stride, vec_f);
#elif defined(FIV_USE_ARM_NEON)
        _fiv_mat_t_mul_vec_real32_neon(dst_f, mat_f, (int)rows, (int)cols, mat_stride, vec_f);
#else
        _fiv_mat_t_mul_vec_real32(dst_f, mat_f, (int)rows, (int)cols, mat_stride, vec_f);
#endif
    }

    /* rewrite dst metadata to describe the result vector */
    dst->dtype         = FIV_32F1;
    dst->shapes[0]     = (transpose == 0) ? rows : cols;
    dst->element_bytes = sizeof(ivf32);
    dst->strides[0]    = sizeof(ivf32);
    dst->data_continue = 1;
    dst->total_bytes   = (size_t)dst->shapes[0] * sizeof(ivf32);

    return FIV_RET_OK;
}


/* ==================== Matrix + vector (broadcast add) ==================== */
fiv_ret fiv_matrix_add_vec(fiv_mat* dst, const fiv_mat* src, const fiv_vec* vec, int dim)
{
    if (dst == NULL || src == NULL || vec == NULL) return FIV_RET_ERR_PARA;
    if (dst->data.ptr == NULL || src->data.ptr == NULL || vec->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (src->dtype == FIV_64F1)
        return fiv_matrix_add_vec_real64(dst, src, vec, dim);
    if (dst->dtype != FIV_32F1 || src->dtype != FIV_32F1 || vec->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (dst->data_continue == 0 || src->data_continue == 0 || vec->data_continue == 0) return FIV_RET_ERR_PARA;
    if (dst->rows != src->rows || dst->cols != src->cols) return FIV_RET_ERR_PARA;

    const size_t rows = src->rows;
    const size_t cols = src->cols;
    const ivf32* s = (const ivf32*)src->data.ptr;
          ivf32* d = (      ivf32*)dst->data.ptr;
    const ivf32* v = (const ivf32*)vec->data.ptr;

    if (dim == 0) {
        /* vector added to each row: vec length must equal cols */
        if (vec->length != cols) return FIV_RET_ERR_PARA;
        for (size_t i = 0; i < rows; i++) {
            const ivf32* sv = s + i * cols;
                  ivf32* dv = d + i * cols;
            for (size_t j = 0; j < cols; j++) dv[j] = sv[j] + v[j];
        }
    } else if (dim == 1) {
        /* vector added to each column: vec length must equal rows */
        if (vec->length != rows) return FIV_RET_ERR_PARA;
        for (size_t i = 0; i < rows; i++) {
            const ivf32* sv = s + i * cols;
                  ivf32* dv = d + i * cols;
            ivf32 vi = v[i];
            for (size_t j = 0; j < cols; j++) dv[j] = sv[j] + vi;
        }
    } else {
        return FIV_RET_ERR_PARA;
    }
    return FIV_RET_OK;
}


/* ==================== Vector norm (L1 / L2) ==================== */

/* |x| via sign-bit clear: no branch, valid for normal/subnormal/0/NaN */
static inline ivf32 _fiv_abs_f32(ivf32 x)
{
    union { ivf32 f; uint32_t u; } p;
    p.f = x;
    p.u &= 0x7fffffffU;
    return p.f;
}

static inline ivf64 _fiv_abs_f64(ivf64 x)
{
    union { ivf64 f; uint64_t u; } p;
    p.f = x;
    p.u &= 0x7fffffffffffffffULL;
    return p.f;
}

/* L1 norm (sum of absolute values) for ivf32 */
static ivf32 _fiv_vec_norm_l1_real32(const ivf32* data, size_t element_count)
{
    ivf32 acc = 0.0f;
    for (size_t idx = 0; idx < element_count; idx++)
    {
        acc += _fiv_abs_f32(data[idx]);
    }
    return acc;
}

/* L2 norm (two-pass, max_abs scaling for overflow safety) for ivf32 */
static ivf32 _fiv_vec_norm_l2_real32(const ivf32* data, size_t element_count)
{
    ivf32 max_abs = 0.0f;
    for (size_t idx = 0; idx < element_count; idx++)
    {
        ivf32 a = _fiv_abs_f32(data[idx]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0f) return 0.0f;

    ivf32 sum_sq = 0.0f;
    for (size_t idx = 0; idx < element_count; idx++)
    {
        ivf32 scaled = data[idx] / max_abs;
        sum_sq += scaled * scaled;
    }
    return max_abs * sqrtf(sum_sq);
}

/* L1 norm (sum of absolute values) for ivf64 */
static ivf64 _fiv_vec_norm_l1_real64(const ivf64* data, size_t element_count)
{
    ivf64 acc = 0.0;
    for (size_t idx = 0; idx < element_count; idx++)
    {
        acc += _fiv_abs_f64(data[idx]);
    }
    return acc;
}

/* L2 norm (two-pass, max_abs scaling for overflow safety) for ivf64 */
static ivf64 _fiv_vec_norm_l2_real64(const ivf64* data, size_t element_count)
{
    ivf64 max_abs = 0.0;
    for (size_t idx = 0; idx < element_count; idx++)
    {
        ivf64 a = _fiv_abs_f64(data[idx]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0) return 0.0;

    ivf64 sum_sq = 0.0;
    for (size_t idx = 0; idx < element_count; idx++)
    {
        ivf64 scaled = data[idx] / max_abs;
        sum_sq += scaled * scaled;
    }
    return max_abs * sqrt(sum_sq);
}

/* Infinity norm (max absolute value) for ivf32 */
static ivf32 _fiv_vec_norm_inf_real32(const ivf32* data, size_t element_count)
{
    ivf32 max_abs = 0.0f;
    for (size_t idx = 0; idx < element_count; idx++)
    {
        ivf32 a = _fiv_abs_f32(data[idx]);
        if (a > max_abs) max_abs = a;
    }
    return max_abs;
}

/* Infinity norm (max absolute value) for ivf64 */
static ivf64 _fiv_vec_norm_inf_real64(const ivf64* data, size_t element_count)
{
    ivf64 max_abs = 0.0;
    for (size_t idx = 0; idx < element_count; idx++)
    {
        ivf64 a = _fiv_abs_f64(data[idx]);
        if (a > max_abs) max_abs = a;
    }
    return max_abs;
}

fiv_ret fiv_vec_norm(fiv_scalar* norm_value, fiv_vec* vec, fiv_norm_type norm_type)
{
    if (norm_value == NULL || vec == NULL) return FIV_RET_ERR_PARA;
    if (vec->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (vec->data_continue == 0) return FIV_RET_ERR_PARA;
    if (norm_type != FIV_L1_NORM && norm_type != FIV_L2_NORM && norm_type != FIV_INF_NORM)
        return FIV_RET_ERR_NOT_SUPPORT;

    size_t element_count = vec->shapes[0];

    if (vec->dtype == FIV_64F1)
    {
        const ivf64* data = (const ivf64*)vec->data.db;
        ivf64 result = (norm_type == FIV_L1_NORM) ? _fiv_vec_norm_l1_real64(data, element_count)
                     : (norm_type == FIV_L2_NORM) ? _fiv_vec_norm_l2_real64(data, element_count)
                                                  : _fiv_vec_norm_inf_real64(data, element_count);
        norm_value->id            = FIV_ID_SCALAR;
        norm_value->dtype         = FIV_64F1;
        norm_value->data.value_fp64 = result;
        return FIV_RET_OK;
    }
    if (vec->dtype == FIV_32F1)
    {
        const ivf32* data = (const ivf32*)vec->data.fl;
        ivf32 result = (norm_type == FIV_L1_NORM) ? _fiv_vec_norm_l1_real32(data, element_count)
                     : (norm_type == FIV_L2_NORM) ? _fiv_vec_norm_l2_real32(data, element_count)
                                                  : _fiv_vec_norm_inf_real32(data, element_count);
        norm_value->id            = FIV_ID_SCALAR;
        norm_value->dtype         = FIV_32F1;
        norm_value->data.value_fp32 = result;
        return FIV_RET_OK;
    }
    return FIV_RET_ERR_NOT_SUPPORT;
}


/* ==================== Vector axpy: y = a * x + y ==================== */

/* ivf32 backend: y += a*x, in-place (y aliases x) allowed. */
static void _fiv_vec_axpy_real32(ivf32* y, ivf32 a, const ivf32* x, size_t element_count)
{
    for (size_t idx = 0; idx < element_count; idx++)
    {
        y[idx] = a * x[idx] + y[idx];
    }
}

/* ivf64 backend: y += a*x, in-place (y aliases x) allowed. */
static void _fiv_vec_axpy_real64(ivf64* y, ivf64 a, const ivf64* x, size_t element_count)
{
    for (size_t idx = 0; idx < element_count; idx++)
    {
        y[idx] = a * x[idx] + y[idx];
    }
}

fiv_ret fiv_vec_axpy(fiv_vec* y, fiv_scalar a, fiv_vec* x)
{
    if (y == NULL || x == NULL) return FIV_RET_ERR_PARA;
    if (y->data.ptr == NULL || x->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (y->data_continue == 0 || x->data_continue == 0) return FIV_RET_ERR_PARA;
    if (y->dtype != x->dtype) return FIV_RET_ERR_PARA;
    if (y->dtype != FIV_32F1 && y->dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (y->shapes[0] != x->shapes[0]) return FIV_RET_ERR_PARA;

    size_t element_count = y->shapes[0];

    if (y->dtype == FIV_64F1)
    {
        _fiv_vec_axpy_real64((ivf64*)y->data.db, a.data.value_fp64, (const ivf64*)x->data.db, element_count);
        return FIV_RET_OK;
    }
    _fiv_vec_axpy_real32((ivf32*)y->data.fl, a.data.value_fp32, (const ivf32*)x->data.fl, element_count);
    return FIV_RET_OK;
}


/* ==================== Vector scale: y = x * scale ==================== */

/* ivf32 backend: y = x * scale, in-place (y aliases x) allowed. */
static void _fiv_vec_scale_real32(ivf32* y, ivf32 scale_value, const ivf32* x, size_t element_count)
{
    for (size_t idx = 0; idx < element_count; idx++)
    {
        y[idx] = scale_value * x[idx];
    }
}

/* ivf64 backend: y = x * scale, in-place (y aliases x) allowed. */
static void _fiv_vec_scale_real64(ivf64* y, ivf64 scale_value, const ivf64* x, size_t element_count)
{
    for (size_t idx = 0; idx < element_count; idx++)
    {
        y[idx] = scale_value * x[idx];
    }
}

fiv_ret fiv_vec_scale(fiv_vec* y, fiv_vec* x, fiv_scalar scale)
{
    if (y == NULL || x == NULL) return FIV_RET_ERR_PARA;
    if (y->data.ptr == NULL || x->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (y->data_continue == 0 || x->data_continue == 0) return FIV_RET_ERR_PARA;
    if (y->dtype != x->dtype) return FIV_RET_ERR_PARA;
    if (y->dtype != FIV_32F1 && y->dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (y->shapes[0] != x->shapes[0]) return FIV_RET_ERR_PARA;

    const size_t element_count = y->shapes[0];

    if (y->dtype == FIV_64F1)
    {
        const ivf64 scale_value = (scale.dtype == FIV_64F1) ? scale.data.value_fp64
                                                        : (ivf64)scale.data.value_fp32;
        _fiv_vec_scale_real64((ivf64*)y->data.db, scale_value,
                              (const ivf64*)x->data.db, element_count);
        return FIV_RET_OK;
    }
    const ivf32 scale_value = (scale.dtype == FIV_64F1) ? (ivf32)scale.data.value_fp64
                                                      : scale.data.value_fp32;
    _fiv_vec_scale_real32((ivf32*)y->data.fl, scale_value,
                          (const ivf32*)x->data.fl, element_count);
    return FIV_RET_OK;
}


/* ==================== Vector dot product: sum_i a[i] * b[i] ==================== */

#if !defined(FIV_USE_ARM_NEON)
/* ivf32 backend (scalar): C baseline for non-NEON builds. */
static ivf32 _fiv_vec_dot_real32_scalar(const ivf32* a, const ivf32* b, size_t element_count)
{
    ivf32 acc = 0.0f;
    for (size_t idx = 0; idx < element_count; idx++)
    {
        acc += a[idx] * b[idx];
    }
    return acc;
}
#endif

#if defined(FIV_USE_ARM_NEON)
/* ivf32 backend (NEON): 8x float32x4_t accumulators (32 floats / iteration)
   with load-ahead software pipelining, mirroring dot_avx_sp's scheme. */
static ivf32 _fiv_vec_dot_real32_neon(const ivf32* a, const ivf32* b, size_t n)
{
    if (n == 0) return 0.0f;

    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f),
                a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f),
                a4 = vdupq_n_f32(0.0f), a5 = vdupq_n_f32(0.0f),
                a6 = vdupq_n_f32(0.0f), a7 = vdupq_n_f32(0.0f);

    size_t i = 0;
    size_t nb = n / 32;

    if (nb >= 1)
    {
        float32x4_t x0 = vld1q_f32(a +  0), y0 = vld1q_f32(b +  0);
        float32x4_t x1 = vld1q_f32(a +  4), y1 = vld1q_f32(b +  4);
        float32x4_t x2 = vld1q_f32(a +  8), y2 = vld1q_f32(b +  8);
        float32x4_t x3 = vld1q_f32(a + 12), y3 = vld1q_f32(b + 12);
        float32x4_t x4 = vld1q_f32(a + 16), y4 = vld1q_f32(b + 16);
        float32x4_t x5 = vld1q_f32(a + 20), y5 = vld1q_f32(b + 20);
        float32x4_t x6 = vld1q_f32(a + 24), y6 = vld1q_f32(b + 24);
        float32x4_t x7 = vld1q_f32(a + 28), y7 = vld1q_f32(b + 28);

        for (size_t k = 1; k < nb; k++)
        {
            const float* pa = a + k * 32;
            const float* pb = b + k * 32;
            a0 = vfmaq_f32(a0, x0, y0);
            a1 = vfmaq_f32(a1, x1, y1);
            a2 = vfmaq_f32(a2, x2, y2);
            a3 = vfmaq_f32(a3, x3, y3);
            a4 = vfmaq_f32(a4, x4, y4);
            a5 = vfmaq_f32(a5, x5, y5);
            a6 = vfmaq_f32(a6, x6, y6);
            a7 = vfmaq_f32(a7, x7, y7);
            x0 = vld1q_f32(pa +  0); y0 = vld1q_f32(pb +  0);
            x1 = vld1q_f32(pa +  4); y1 = vld1q_f32(pb +  4);
            x2 = vld1q_f32(pa +  8); y2 = vld1q_f32(pb +  8);
            x3 = vld1q_f32(pa + 12); y3 = vld1q_f32(pb + 12);
            x4 = vld1q_f32(pa + 16); y4 = vld1q_f32(pb + 16);
            x5 = vld1q_f32(pa + 20); y5 = vld1q_f32(pb + 20);
            x6 = vld1q_f32(pa + 24); y6 = vld1q_f32(pb + 24);
            x7 = vld1q_f32(pa + 28); y7 = vld1q_f32(pb + 28);
        }

        a0 = vfmaq_f32(a0, x0, y0);
        a1 = vfmaq_f32(a1, x1, y1);
        a2 = vfmaq_f32(a2, x2, y2);
        a3 = vfmaq_f32(a3, x3, y3);
        a4 = vfmaq_f32(a4, x4, y4);
        a5 = vfmaq_f32(a5, x5, y5);
        a6 = vfmaq_f32(a6, x6, y6);
        a7 = vfmaq_f32(a7, x7, y7);
        i = nb * 32;
    }

    float32x4_t sum4 = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    float32x4_t sum4b = vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7));
    float32x4_t acc = vaddq_f32(sum4, sum4b);

    for (; i + 4 <= n; i += 4)
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));

    float s = vaddvq_f32(acc);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
#endif /* FIV_USE_ARM_NEON */

/* ivf64 backend: scalar accumulation. */
static ivf64 _fiv_vec_dot_real64(const ivf64* a, const ivf64* b, size_t element_count)
{
    ivf64 acc = 0.0;
    for (size_t idx = 0; idx < element_count; idx++)
    {
        acc += a[idx] * b[idx];
    }
    return acc;
}

fiv_ret fiv_vec_dot(fiv_scalar* dot_value, const fiv_vec* a, const fiv_vec* b)
{
    if (dot_value == NULL || a == NULL || b == NULL) return FIV_RET_ERR_PARA;
    if (a->data.ptr == NULL || b->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (a->data_continue == 0 || b->data_continue == 0) return FIV_RET_ERR_PARA;
    if (a->dtype != b->dtype) return FIV_RET_ERR_PARA;
    if (a->dtype != FIV_32F1 && a->dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (a->shapes[0] != b->shapes[0]) return FIV_RET_ERR_PARA;

    size_t element_count = a->shapes[0];

    if (a->dtype == FIV_64F1)
    {
        dot_value->id    = FIV_ID_SCALAR;
        dot_value->dtype = FIV_64F1;
        dot_value->data.value_fp64 = _fiv_vec_dot_real64((const ivf64*)a->data.db, (const ivf64*)b->data.db, element_count);
        return FIV_RET_OK;
    }
    dot_value->id    = FIV_ID_SCALAR;
    dot_value->dtype = FIV_32F1;
#if defined(FIV_USE_ARM_NEON)
    dot_value->data.value_fp32 = _fiv_vec_dot_real32_neon((const ivf32*)a->data.fl, (const ivf32*)b->data.fl, element_count);
#else
    dot_value->data.value_fp32 = _fiv_vec_dot_real32_scalar((const ivf32*)a->data.fl, (const ivf32*)b->data.fl, element_count);
#endif
    return FIV_RET_OK;
}
