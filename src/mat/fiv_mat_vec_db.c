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

/* 64-bit (ivf64 / double) matrix-vector multiply. The float32 path in
 * fiv_mat_vec.c is bit-width-specific (SSE/AVX/NEON micro-kernels); float64
 * double-precision SIMD micro-kernels are not provided there, so this file
 * keeps the full algorithm as scalar loops (4-way unrolled dot products),
 * identical in structure to the float32 path. */

#include "fiv_mat_vec_db.h"
#include <string.h>   /* memset for the transposed mat*vec path */

/* dst = mat * vec, one dot product per row */
static void _fiv_mat_mul_vec_real64(ivf64* dst, const ivf64* mat, int rows, int cols, int mat_stride, const ivf64* vec)
{
    int i, j;
    for (i = 0; i <= rows - 4; i += 4)
    {
        const ivf64* r0 = mat + (size_t)(i + 0) * mat_stride;
        const ivf64* r1 = mat + (size_t)(i + 1) * mat_stride;
        const ivf64* r2 = mat + (size_t)(i + 2) * mat_stride;
        const ivf64* r3 = mat + (size_t)(i + 3) * mat_stride;
        const ivf64* v  = vec;
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;

        for (j = 0; j <= cols - 4; j += 4)
        {
            double t0 = v[0], t1 = v[1], t2 = v[2], t3 = v[3]; v += 4;
            s0 += r0[0]*t0 + r0[1]*t1 + r0[2]*t2 + r0[3]*t3;
            s1 += r1[0]*t0 + r1[1]*t1 + r1[2]*t2 + r1[3]*t3;
            s2 += r2[0]*t0 + r2[1]*t1 + r2[2]*t2 + r2[3]*t3;
            s3 += r3[0]*t0 + r3[1]*t1 + r3[2]*t2 + r3[3]*t3;
            r0 += 4; r1 += 4; r2 += 4; r3 += 4;
        }
        for (; j < cols; j++)
        {
            double t = *v++;
            s0 += r0[0]*t; r0++;
            s1 += r1[0]*t; r1++;
            s2 += r2[0]*t; r2++;
            s3 += r3[0]*t; r3++;
        }
        dst[i + 0] = (ivf64)s0;
        dst[i + 1] = (ivf64)s1;
        dst[i + 2] = (ivf64)s2;
        dst[i + 3] = (ivf64)s3;
    }
    for (; i < rows; i++)
    {
        const ivf64* r = mat + (size_t)i * mat_stride;
        const ivf64* v = vec;
        double s = 0.0; int j;
        for (j = 0; j <= cols - 4; j += 4)
        {
            s += r[0]*v[0] + r[1]*v[1] + r[2]*v[2] + r[3]*v[3];
            r += 4; v += 4;
        }
        for (; j < cols; j++) { s += r[0]*v[0]; r++; v++; }
        dst[i] = (ivf64)s;
    }
}

/* dst = mat^T * vec: zero dst first, then stream rows into dst[0..cols) */
static void _fiv_mat_t_mul_vec_real64(ivf64* dst, const ivf64* mat, int rows, int cols, int mat_stride, const ivf64* vec)
{
    int i, j;
    memset(dst, 0, (size_t)cols * sizeof(ivf64));

    for (i = 0; i <= rows - 4; i += 4)
    {
        const ivf64* r0 = mat + (size_t)(i + 0) * mat_stride;
        const ivf64* r1 = mat + (size_t)(i + 1) * mat_stride;
        const ivf64* r2 = mat + (size_t)(i + 2) * mat_stride;
        const ivf64* r3 = mat + (size_t)(i + 3) * mat_stride;
        double v0 = (double)vec[i + 0], v1 = (double)vec[i + 1], v2 = (double)vec[i + 2], v3 = (double)vec[i + 3];

        for (j = 0; j <= cols - 4; j += 4)
        {
            dst[j + 0] += (ivf64)(r0[0]*v0 + r1[0]*v1 + r2[0]*v2 + r3[0]*v3);
            dst[j + 1] += (ivf64)(r0[1]*v0 + r1[1]*v1 + r2[1]*v2 + r3[1]*v3);
            dst[j + 2] += (ivf64)(r0[2]*v0 + r1[2]*v1 + r2[2]*v2 + r3[2]*v3);
            dst[j + 3] += (ivf64)(r0[3]*v0 + r1[3]*v1 + r2[3]*v2 + r3[3]*v3);
            r0 += 4; r1 += 4; r2 += 4; r3 += 4;
        }
        for (; j < cols; j++)
        {
            dst[j] += (ivf64)(r0[0]*v0 + r1[0]*v1 + r2[0]*v2 + r3[0]*v3);
            r0++; r1++; r2++; r3++;
        }
    }
    for (; i < rows; i++)
    {
        const ivf64* r = mat + (size_t)i * mat_stride;
        double vi = (double)vec[i];
        int j;
        for (j = 0; j <= cols - 4; j += 4)
        {
            dst[j + 0] += (ivf64)(r[0]*vi);
            dst[j + 1] += (ivf64)(r[1]*vi);
            dst[j + 2] += (ivf64)(r[2]*vi);
            dst[j + 3] += (ivf64)(r[3]*vi);
            r += 4;
        }
        for (; j < cols; j++) { dst[j] += (ivf64)(r[0]*vi); r++; }
    }
}

/* 64-bit backend for the generic fiv_matrix_mul_vec: validates the FIV_64F1
   operands and dispatches to the scalar mat*vec / mat^T*vec kernels. */
fiv_ret fiv_matrix_mul_vec_real64(fiv_vec* dst, const fiv_mat* mat, const fiv_vec* vec, int transpose)
{
    if (dst == NULL || mat == NULL || vec == NULL) return FIV_RET_ERR_PARA;
    if (dst->data.ptr == NULL || mat->data.ptr == NULL || vec->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (dst->data.ptr == vec->data.ptr) return FIV_RET_ERR_PARA;
    if (dst->data_continue == 0 || mat->data_continue == 0 || vec->data_continue == 0) return FIV_RET_ERR_PARA;
    if (mat->dtype != FIV_64F1 || vec->dtype != FIV_64F1 || dst->dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;

    const size_t rows = mat->shapes[0];
    const size_t cols = mat->shapes[1];
    const int    mat_stride = (int)(mat->strides[0] / (size_t)mat->element_bytes);
    ivf64*       dst_f = (ivf64*)dst->data.ptr;
    const ivf64* vec_f = (const ivf64*)vec->data.ptr;
    const ivf64* mat_f = (const ivf64*)mat->data.ptr;

    if (transpose == 0) {
        if (vec->shapes[0] < cols) return FIV_RET_ERR_PARA;
        if (dst->shapes[0] < rows) return FIV_RET_ERR_PARA;
        _fiv_mat_mul_vec_real64(dst_f, mat_f, (int)rows, (int)cols, mat_stride, vec_f);
    } else {
        if (vec->shapes[0] < rows) return FIV_RET_ERR_PARA;
        if (dst->shapes[0] < cols) return FIV_RET_ERR_PARA;
        _fiv_mat_t_mul_vec_real64(dst_f, mat_f, (int)rows, (int)cols, mat_stride, vec_f);
    }

    dst->dtype         = FIV_64F1;
    dst->shapes[0]     = (transpose == 0) ? rows : cols;
    dst->element_bytes = sizeof(ivf64);
    dst->strides[0]    = sizeof(ivf64);
    dst->data_continue = 1;
    dst->total_bytes   = (size_t)dst->shapes[0] * sizeof(ivf64);

    return FIV_RET_OK;
}

/* 64-bit backend for the generic fiv_matrix_add_vec: broadcasts an FIV_64F1
   vector over the rows (dim == 0) or columns (dim == 1) of an FIV_64F1 matrix.
   Mirrors the float32 fiv_matrix_add_vec, scalar loops, no SIMD. */
fiv_ret fiv_matrix_add_vec_real64(fiv_mat* dst, const fiv_mat* src, const fiv_vec* vec, int dim)
{
    if (dst == NULL || src == NULL || vec == NULL) return FIV_RET_ERR_PARA;
    if (dst->data.ptr == NULL || src->data.ptr == NULL || vec->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (dst->dtype != FIV_64F1 || src->dtype != FIV_64F1 || vec->dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (dst->data_continue == 0 || src->data_continue == 0 || vec->data_continue == 0) return FIV_RET_ERR_PARA;
    if (dst->rows != src->rows || dst->cols != src->cols) return FIV_RET_ERR_PARA;

    const size_t rows = src->rows;
    const size_t cols = src->cols;
    const ivf64* s = (const ivf64*)src->data.ptr;
          ivf64* d = (      ivf64*)dst->data.ptr;
    const ivf64* v = (const ivf64*)vec->data.ptr;

    if (dim == 0) {
        if (vec->length != cols) return FIV_RET_ERR_PARA;
        for (size_t i = 0; i < rows; i++) {
            const ivf64* sv = s + i * cols;
                  ivf64* dv = d + i * cols;
            for (size_t j = 0; j < cols; j++) dv[j] = sv[j] + v[j];
        }
    } else if (dim == 1) {
        if (vec->length != rows) return FIV_RET_ERR_PARA;
        for (size_t i = 0; i < rows; i++) {
            const ivf64* sv = s + i * cols;
                  ivf64* dv = d + i * cols;
            ivf64 vi = v[i];
            for (size_t j = 0; j < cols; j++) dv[j] = sv[j] + vi;
        }
    } else {
        return FIV_RET_ERR_PARA;
    }
    return FIV_RET_OK;
}
