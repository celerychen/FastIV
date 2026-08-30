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

#include "fiv_lp_mat.h"
#include "fiv_sp_matrix.h"   /* fiv_create_sp_matrix_from_coo / fiv_sparse_* / fiv_release_sp_matrix */
#include "fiv_matrix.h"      /* fiv_matrix_mul_vec (dense backend) */
#include "fiv_common.h"      /* fiv_malloc / fiv_free */

#include <math.h>            /* fabs, pow */

/* Read one element of a dense matrix (row-major, honoring its BYTE strides) as
 * f64. The matrix may be FIV_64F1 or FIV_32F1 (the latter is widened). */
static ivf64 fiv_lp_mat_dense_get(const fiv_mat *dense_matrix, size_t row_index, size_t col_index)
{
    const size_t element_bytes = dense_matrix->element_bytes;
    const size_t flat_index = (row_index * dense_matrix->strides[0] +
                               col_index * dense_matrix->strides[1]) / element_bytes;
    if (dense_matrix->dtype == FIV_64F1) return dense_matrix->data.db[flat_index];
    return (ivf64)dense_matrix->data.fl[flat_index];   /* FIV_32F1 widened */
}

/* Dense per-row / per-column max of |entry|. */
static fiv_ret fiv_lp_mat_dense_reduce_abs_max(fiv_vec *dst, const fiv_mat *dense_matrix, int dim)
{
    if (dim == 0) {
        if (dst->length != dense_matrix->rows) return FIV_RET_ERR_PARA;
        for (size_t row = 0; row < dense_matrix->rows; row++) {
            ivf64 row_max = 0.0;
            for (size_t col = 0; col < dense_matrix->cols; col++) {
                ivf64 abs_value = fabs(fiv_lp_mat_dense_get(dense_matrix, row, col));
                if (abs_value > row_max) row_max = abs_value;
            }
            dst->data.db[row] = row_max;
        }
    } else if (dim == 1) {
        if (dst->length != dense_matrix->cols) return FIV_RET_ERR_PARA;
        for (size_t col = 0; col < dense_matrix->cols; col++) {
            ivf64 col_max = 0.0;
            for (size_t row = 0; row < dense_matrix->rows; row++) {
                ivf64 abs_value = fabs(fiv_lp_mat_dense_get(dense_matrix, row, col));
                if (abs_value > col_max) col_max = abs_value;
            }
            dst->data.db[col] = col_max;
        }
    } else {
        return FIV_RET_ERR_PARA;
    }
    return FIV_RET_OK;
}

/* Dense per-row / per-column sum of |entry|^exponent. */
static fiv_ret fiv_lp_mat_dense_reduce_abs_pow(fiv_vec *dst, const fiv_mat *dense_matrix,
                                              int dim, ivf64 exponent)
{
    if (dim == 0) {
        if (dst->length != dense_matrix->rows) return FIV_RET_ERR_PARA;
        for (size_t row = 0; row < dense_matrix->rows; row++) {
            ivf64 row_sum = 0.0;
            for (size_t col = 0; col < dense_matrix->cols; col++)
                row_sum += pow(fabs(fiv_lp_mat_dense_get(dense_matrix, row, col)), exponent);
            dst->data.db[row] = row_sum;
        }
    } else if (dim == 1) {
        if (dst->length != dense_matrix->cols) return FIV_RET_ERR_PARA;
        for (size_t col = 0; col < dense_matrix->cols; col++) {
            ivf64 col_sum = 0.0;
            for (size_t row = 0; row < dense_matrix->rows; row++)
                col_sum += pow(fabs(fiv_lp_mat_dense_get(dense_matrix, row, col)), exponent);
            dst->data.db[col] = col_sum;
        }
    } else {
        return FIV_RET_ERR_PARA;
    }
    return FIV_RET_OK;
}


/* ============================ Build / release ============================ */

fiv_lp_mat *fiv_lp_mat_wrap_dense(fiv_mat *dense_matrix)
{
    if (dense_matrix == NULL) return NULL;
    fiv_lp_mat *wrapper = (fiv_lp_mat *)fiv_malloc(sizeof(fiv_lp_mat));
    if (wrapper == NULL) return NULL;
    wrapper->kind = FIV_LP_MAT_DENSE;
    wrapper->rows = dense_matrix->rows;
    wrapper->cols = dense_matrix->cols;
    wrapper->owns_data = 0;
    wrapper->as.dense = dense_matrix;
    wrapper->transpose_view = NULL;
    return wrapper;
}

fiv_lp_mat *fiv_lp_mat_wrap_sparse(fiv_sparse_mat *sparse_matrix)
{
    if (sparse_matrix == NULL) return NULL;
    fiv_lp_mat *wrapper = (fiv_lp_mat *)fiv_malloc(sizeof(fiv_lp_mat));
    if (wrapper == NULL) return NULL;
    wrapper->kind = FIV_LP_MAT_SPARSE;
    wrapper->rows = sparse_matrix->rows;
    wrapper->cols = sparse_matrix->cols;
    wrapper->owns_data = 0;
    wrapper->as.sparse = sparse_matrix;
    wrapper->transpose_view = NULL;
    return wrapper;
}

fiv_lp_mat *fiv_create_lp_mat_from_coo(const int *row_indices, const int *col_indices,
                                       const void *values, fiv_data_type value_dtype,
                                       size_t num_nonzeros, size_t num_rows, size_t num_cols)
{
    fiv_sparse_mat *sparse_matrix = fiv_create_sp_matrix_from_coo(row_indices, col_indices,
                                                                  values, value_dtype,
                                                                  num_nonzeros, num_rows, num_cols);
    if (sparse_matrix == NULL) return NULL;

    fiv_lp_mat *wrapper = (fiv_lp_mat *)fiv_malloc(sizeof(fiv_lp_mat));
    if (wrapper == NULL) { fiv_release_sp_matrix(&sparse_matrix); return NULL; }
    wrapper->kind = FIV_LP_MAT_SPARSE;
    wrapper->rows = sparse_matrix->rows;
    wrapper->cols = sparse_matrix->cols;
    wrapper->owns_data = 1;
    wrapper->as.sparse = sparse_matrix;
    wrapper->transpose_view = NULL;
    return wrapper;
}

fiv_ret fiv_release_lp_mat(fiv_lp_mat **lp_matrix)
{
    if (lp_matrix == NULL) return FIV_RET_ERR_PARA;
    if (*lp_matrix == NULL) return FIV_RET_OK;

    fiv_lp_mat *wrapper = *lp_matrix;
    /* transpose_view is always owned by the wrapper (materialized by
     * fiv_lp_mat_build_transpose), so free it regardless of owns_data. */
    if (wrapper->transpose_view != NULL)
        fiv_release_sp_matrix(&wrapper->transpose_view);
    if (wrapper->owns_data && wrapper->kind == FIV_LP_MAT_SPARSE && wrapper->as.sparse != NULL)
        fiv_release_sp_matrix(&wrapper->as.sparse);
    fiv_free(wrapper);
    *lp_matrix = NULL;
    return FIV_RET_OK;
}


fiv_ret fiv_lp_mat_build_transpose(fiv_lp_mat *lp_matrix)
{
    if (lp_matrix == NULL) return FIV_RET_ERR_PARA;
    if (lp_matrix->kind == FIV_LP_MAT_DENSE)
        return FIV_RET_OK;   /* FastIV handles transpose natively; nothing to build */

    /* sparse: already built? */
    if (lp_matrix->transpose_view != NULL)
        return FIV_RET_OK;
    if (lp_matrix->as.sparse == NULL)
        return FIV_RET_ERR_PARA;
    if (fiv_sparse_get_fmt(lp_matrix->as.sparse) != FIV_SPARSE_CSR)
        return FIV_RET_ERR_PARA;   /* canonical format must be CSR */

    fiv_sparse_mat *transpose_csc = NULL;
    fiv_ret status = fiv_sparse_transpose(&transpose_csc, lp_matrix->as.sparse);
    if (status != FIV_RET_OK) return status;
    lp_matrix->transpose_view = transpose_csc;
    return FIV_RET_OK;
}


/* ============================ Operations ============================ */

fiv_ret fiv_lp_mat_matvec(fiv_vec *dst, const fiv_lp_mat *lp_matrix,
                          const fiv_vec *x, int transpose)
{
    if (dst == NULL || lp_matrix == NULL || x == NULL) return FIV_RET_ERR_PARA;
    if (lp_matrix->kind == FIV_LP_MAT_DENSE)
        return fiv_matrix_mul_vec(dst, lp_matrix->as.dense, x, transpose);

    /* sparse: A * x uses the canonical CSR (as.sparse); Aᵀ * x uses the CSC
     * transpose_view when available, otherwise falls back to as.sparse (which
     * the caller may have set to a CSC, e.g. the M_sparseT wrap convention). */
    if (transpose == 0)
        return fiv_sparse_matmul_vec(dst, lp_matrix->as.sparse, x, 0);
    const fiv_sparse_mat *Kt = (lp_matrix->transpose_view != NULL)
                                 ? lp_matrix->transpose_view : lp_matrix->as.sparse;
    return fiv_sparse_matmul_vec(dst, Kt, x, 1);
}

fiv_ret fiv_lp_mat_reduce_abs_max(fiv_vec *dst, const fiv_lp_mat *lp_matrix, int dim)
{
    if (dst == NULL || lp_matrix == NULL) return FIV_RET_ERR_PARA;
    if (lp_matrix->kind == FIV_LP_MAT_DENSE)
        return fiv_lp_mat_dense_reduce_abs_max(dst, lp_matrix->as.dense, dim);
    return fiv_sparse_reduce_abs_max(lp_matrix->as.sparse, dim, dst);
}

fiv_ret fiv_lp_mat_reduce_abs_pow(fiv_vec *dst, const fiv_lp_mat *lp_matrix,
                                  int dim, ivf64 exponent)
{
    if (dst == NULL || lp_matrix == NULL) return FIV_RET_ERR_PARA;
    if (lp_matrix->kind == FIV_LP_MAT_DENSE)
        return fiv_lp_mat_dense_reduce_abs_pow(dst, lp_matrix->as.dense, dim, exponent);
    return fiv_sparse_reduce_pow_abs_sum(lp_matrix->as.sparse, dim, exponent, dst);
}
