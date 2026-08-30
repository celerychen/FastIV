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

/* Unified thin SVD entry: fiv_matrix_svd (declared in api/fiv_matrix.h).
 *
 * Thin wrapper that routes the caller's fiv_svd_type selection to one of
 * the two backends and adapts the V output layout to the requested
 * v_transpose form. Neither backend source is modified here:
 *
 *   FIV_JACOBI_SVD -> fiv_matrix_svd_jacobi: one-sided Jacobi sweeps
 *   (OpenCV SVD::compute lineage); native output is V^T in rows (dim x
 *   cols), so v_transpose != 0 passes straight through and v_transpose == 0
 *   transposes the internal V^T into the caller's V-in-columns buffer.
 *
 *   FIV_BCD_SVD    -> fiv_matrix_svd_bcd: blocked Householder
 *   bidiagonalization + Golub-Kahan QR (LAPACK dgesdd lineage); native
 *   output is V in columns (cols x dim), so v_transpose == 0 passes
 *   straight through and v_transpose != 0 builds an internal V first and
 *   transposes it into the caller's V^T-shaped buffer.
 *
 * Both backends preserve the input A and return descending singular values
 * with U columns / V (or V^T) rows or columns orthonormal.
 */

#include "fiv_matrix.h"
#include "fiv_mat_svd_backends.h"
#include "fiv_common.h"

#include <stdlib.h>

fiv_ret fiv_matrix_svd(fiv_mat* mat_a, ivf32* sing_vals,
                       fiv_mat* mat_u, fiv_mat* mat_v,
                       int v_transpose, fiv_svd_type type)
{
    if (mat_a == NULL || mat_a->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (sing_vals == NULL)                        return FIV_RET_ERR_PARA;
    if (mat_a->data_continue == 0)                return FIV_RET_ERR_PARA;
    if (mat_a->dtype != FIV_32F1)                 return FIV_RET_ERR_NOT_SUPPORT;
    if (type != FIV_JACOBI_SVD && type != FIV_BCD_SVD) return FIV_RET_ERR_NOT_SUPPORT;

    const size_t rows_s = mat_a->shapes[0];
    const size_t cols_s = mat_a->shapes[1];
    if (rows_s == 0 || cols_s == 0)               return FIV_RET_ERR_PARA;
    const int rows = (int)rows_s;
    const int cols = (int)cols_s;
    const int dim = rows < cols ? rows : cols;
    if ((size_t)rows * cols > (size_t)(SIZE_MAX / sizeof(ivf32))) return FIV_RET_ERR_PARA;
    if (mat_a->total_bytes < rows_s * cols_s * (size_t)mat_a->element_bytes) {
        return FIV_RET_ERR_PARA;
    }

    /* mat_u: rows x dim in both layouts; mat_v: (cols x dim) when the
       caller wants V in columns, (dim x cols) when v_transpose != 0 */
    const size_t want_v_rows = v_transpose ? (size_t)dim : cols_s;
    const size_t want_v_cols = v_transpose ? cols_s : (size_t)dim;

    fiv_mat* out_mats[2];
    out_mats[0] = mat_u;
    out_mats[1] = mat_v;
    const size_t want_shapes[2][2] = { { rows_s, (size_t)dim },
                                       { want_v_rows, want_v_cols } };
    for (int idx = 0; idx < 2; idx++) {
        const fiv_mat* out_mat = out_mats[idx];
        if (out_mat == NULL) continue;
        if (out_mat->data.ptr == NULL || out_mat->data_continue == 0) return FIV_RET_ERR_PARA;
        if (out_mat->dtype != FIV_32F1)               return FIV_RET_ERR_NOT_SUPPORT;
        if (out_mat->shapes[0] != want_shapes[idx][0] ||
            out_mat->shapes[1] != want_shapes[idx][1]) {
            return FIV_RET_ERR_PARA;
        }
        if (out_mat->total_bytes <
                want_shapes[idx][0] * want_shapes[idx][1] *
                (size_t)out_mat->element_bytes) {
            return FIV_RET_ERR_PARA;
        }
        if (out_mat->data.ptr == mat_a->data.ptr)     return FIV_RET_ERR_PARA;
    }
    if (mat_u != NULL && mat_v != NULL && mat_u->data.ptr == mat_v->data.ptr) {
        return FIV_RET_ERR_PARA;
    }

    fiv_ret ret = FIV_RET_ERR_UNKNOWN;

    if (type == FIV_BCD_SVD) {
        if (mat_v == NULL || !v_transpose) {
            /* backend natively produces V in columns */
            ret = fiv_matrix_svd_bcd(mat_a, sing_vals, mat_u, mat_v);
        } else {
            /* internal V (cols x dim), then transpose into mat_v (dim x cols) */
            fiv_mat vbuf = *mat_a;
            vbuf.data.ptr    = fiv_malloc(sizeof(ivf32) * (size_t)cols * dim);
            if (vbuf.data.ptr == NULL) return FIV_RET_ERR_MEM;
            vbuf.shapes[0]   = (size_t)cols;
            vbuf.shapes[1]   = (size_t)dim;
            vbuf.strides[0]  = (size_t)dim * mat_a->element_bytes;
            vbuf.strides[1]  = mat_a->element_bytes;
            vbuf.total_bytes = (size_t)cols * dim * mat_a->element_bytes;
            ret = fiv_matrix_svd_bcd(mat_a, sing_vals, mat_u, &vbuf);
            if (ret == FIV_RET_OK &&
                fiv_matrix_transpose(mat_v, &vbuf) != FIV_RET_OK) {
                ret = FIV_RET_ERR_UNKNOWN;
            }
            fiv_free(vbuf.data.ptr);
        }
    } else { /* FIV_JACOBI_SVD */
        if (mat_v == NULL || v_transpose) {
            /* backend natively produces V^T in rows (dim x cols) */
            ret = fiv_matrix_svd_jacobi(mat_a, sing_vals, mat_u, mat_v);
        } else {
            /* internal V^T (dim x cols), then transpose into mat_v (cols x dim) */
            fiv_mat vtbuf = *mat_a;
            vtbuf.data.ptr    = fiv_malloc(sizeof(ivf32) * (size_t)dim * cols);
            if (vtbuf.data.ptr == NULL) return FIV_RET_ERR_MEM;
            vtbuf.shapes[0]   = (size_t)dim;
            vtbuf.shapes[1]   = (size_t)cols;
            vtbuf.strides[0]  = (size_t)cols * mat_a->element_bytes;
            vtbuf.strides[1]  = mat_a->element_bytes;
            vtbuf.total_bytes = (size_t)dim * cols * mat_a->element_bytes;
            ret = fiv_matrix_svd_jacobi(mat_a, sing_vals, mat_u, &vtbuf);
            if (ret == FIV_RET_OK &&
                fiv_matrix_transpose(mat_v, &vtbuf) != FIV_RET_OK) {
                ret = FIV_RET_ERR_UNKNOWN;
            }
            fiv_free(vtbuf.data.ptr);
        }
    }
    return ret;
}
