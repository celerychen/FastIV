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

#include "fiv_lp_rescale.h"
#include "fiv_lp_mat.h"      /* fiv_lp_mat, fiv_lp_mat_reduce_abs_pow */
#include "fiv_sp_matrix.h"   /* fiv_sparse_transpose / fiv_sparse_reduce_abs_pow */
#include "fiv_matrix.h"      /* fiv_vec_norm / fiv_vec_dot (future KKT use) */
#include "fiv_ctensor.h"     /* fiv_create_tensor1d / fiv_release_tensor1d /
                                fiv_tensor_div / fiv_tensor_mul */
#include "fiv_common.h"      /* fiv_malloc / fiv_free */

#include <math.h>


/* ------------------------------------------------------------------ */
/* Object lifecycle                                                    */
/* ------------------------------------------------------------------ */

fiv_lp_rescaling *fiv_create_lp_rescaling(size_t num_variables, size_t num_constraints)
{
    fiv_lp_rescaling *rescaling = (fiv_lp_rescaling *)fiv_malloc(sizeof(fiv_lp_rescaling));
    if (rescaling == NULL) return NULL;
    rescaling->variable_rescaling = fiv_create_tensor1d(num_variables, FIV_64F1);
    rescaling->constraint_rescaling = fiv_create_tensor1d(num_constraints, FIV_64F1);
    if (rescaling->variable_rescaling == NULL || rescaling->constraint_rescaling == NULL) {
        fiv_release_tensor1d(&rescaling->variable_rescaling);
        fiv_release_tensor1d(&rescaling->constraint_rescaling);
        fiv_free(rescaling);
        return NULL;
    }
    for (size_t index = 0; index < num_variables; index++)
        rescaling->variable_rescaling->data.db[index] = 1.0;
    for (size_t index = 0; index < num_constraints; index++)
        rescaling->constraint_rescaling->data.db[index] = 1.0;
    return rescaling;
}

fiv_ret fiv_release_lp_rescaling(fiv_lp_rescaling **rescaling)
{
    if (rescaling == NULL || *rescaling == NULL) return FIV_RET_OK;
    fiv_release_tensor1d(&(*rescaling)->variable_rescaling);
    fiv_release_tensor1d(&(*rescaling)->constraint_rescaling);
    fiv_free(*rescaling);
    *rescaling = NULL;
    return FIV_RET_OK;
}


/* ------------------------------------------------------------------ */
/* Internal helpers                                                    */
/* ------------------------------------------------------------------ */

/* Reduce sum |value|^exponent along `dim` of the CURRENT K (Pock-Chambolle).
 *   dim == 0 -> per row  (length m)  : uses CSR directly
 *   dim == 1 -> per col  (length n)  : for SPARSE a transient CSC is built
 *                                       from the live K values, then freed. */
static fiv_ret fiv_rescale_reduce_pow_dim(const fiv_lp_mat *lp_matrix, int dim,
                                          ivf64 exponent, fiv_vec *dst)
{
    if (lp_matrix->kind == FIV_LP_MAT_DENSE)
        return fiv_lp_mat_reduce_abs_pow(dst, lp_matrix, dim, exponent);

    /* sparse */
    if (dim == 0)
        return fiv_lp_mat_reduce_abs_pow(dst, lp_matrix, 0, exponent);  /* CSR */

    /* dim == 1 needs column-major access -> materialize CSC from current K */
    fiv_sparse_mat *csc_matrix = NULL;
    fiv_ret ret_code = fiv_sparse_transpose(&csc_matrix, lp_matrix->as.sparse);
    if (ret_code != FIV_RET_OK) return ret_code;
    ret_code = fiv_sparse_reduce_pow_abs_sum(csc_matrix, 1, exponent, dst);
    fiv_release_sp_matrix(&csc_matrix);
    return ret_code;
}

/* Reduce max |value| along `dim` of the CURRENT K (Ruiz). Same dim/spark rule
 * as the pow variant, but uses the abs-max reductions. */
static fiv_ret fiv_rescale_reduce_max_dim(const fiv_lp_mat *lp_matrix, int dim, fiv_vec *dst)
{
    if (lp_matrix->kind == FIV_LP_MAT_DENSE)
        return fiv_lp_mat_reduce_abs_max(dst, lp_matrix, dim);

    /* sparse */
    if (dim == 0)
        return fiv_sparse_reduce_abs_max(lp_matrix->as.sparse, 0, dst);  /* CSR */

    /* dim == 1 needs CSC */
    fiv_sparse_mat *csc_matrix = NULL;
    fiv_ret ret_code = fiv_sparse_transpose(&csc_matrix, lp_matrix->as.sparse);
    if (ret_code != FIV_RET_OK) return ret_code;
    ret_code = fiv_sparse_reduce_abs_max(csc_matrix, 1, dst);
    fiv_release_sp_matrix(&csc_matrix);
    return ret_code;
}

/* Multiply every entry a_ij of K by (1 / row_factor[i]) * (1 / col_factor[j]).
 * Mutates K in place; row_factor length m, col_factor length n. */
static fiv_ret fiv_rescale_apply_matrix(fiv_lp_mat *lp_matrix,
                                        const fiv_vec *row_factor, const fiv_vec *col_factor)
{
    const ivf64 *row_scale = row_factor->data.db;
    const ivf64 *col_scale = col_factor->data.db;

    if (lp_matrix->kind == FIV_LP_MAT_DENSE) {
        fiv_mat *dense_matrix = lp_matrix->as.dense;
        const size_t num_rows = dense_matrix->rows;
        const size_t num_cols = dense_matrix->cols;
        const size_t element_bytes = dense_matrix->element_bytes;
        ivf64 *data_ptr = dense_matrix->data.db;
        const size_t stride_row = dense_matrix->strides[0] / element_bytes;
        const size_t stride_col = dense_matrix->strides[1] / element_bytes;
        for (size_t row = 0; row < num_rows; row++) {
            const ivf64 reciprocal_row = 1.0 / row_scale[row];
            for (size_t col = 0; col < num_cols; col++) {
                const size_t flat_index = row * stride_row + col * stride_col;
                data_ptr[flat_index] *= reciprocal_row * (1.0 / col_scale[col]);
            }
        }
        return FIV_RET_OK;
    }

    fiv_sparse_mat *sparse_matrix = lp_matrix->as.sparse;
    ivf64 *values = (ivf64 *)sparse_matrix->hdr.data.ptr;
    for (size_t row = 0; row < sparse_matrix->rows; row++) {
        const ivf64 reciprocal_row = 1.0 / row_scale[row];
        for (int entry = sparse_matrix->indptr[row]; entry < sparse_matrix->indptr[row + 1]; entry++) {
            const ivf64 reciprocal_col = 1.0 / col_scale[sparse_matrix->indices[entry]];
            values[entry] *= reciprocal_row * reciprocal_col;
        }
    }
    return FIV_RET_OK;
}

/* Apply the per-vector rescalings (mirrors pdlp.py lines 172-177 / 193-198):
 *   c /= col_factor ; l *= col_factor ; u *= col_factor
 *   q /= row_factor
 *   var_rescale *= col_factor ; con_rescale *= row_factor */
static fiv_ret fiv_rescale_apply_vectors(fiv_lp_rescaling *rescaling,
                                        fiv_vec *c, fiv_vec *l, fiv_vec *u, fiv_vec *q,
                                        const fiv_vec *row_factor, const fiv_vec *col_factor)
{
    fiv_ret ret_code = fiv_tensor_div(c, c, col_factor);
    if (ret_code != FIV_RET_OK) return ret_code;
    fiv_tensor_mul(l, l, col_factor);
    fiv_tensor_mul(u, u, col_factor);
    fiv_tensor_div(q, q, row_factor);
    fiv_tensor_mul(rescaling->variable_rescaling, rescaling->variable_rescaling, col_factor);
    fiv_tensor_mul(rescaling->constraint_rescaling, rescaling->constraint_rescaling, row_factor);
    return FIV_RET_OK;
}

/* One equilibration pass (used by both Ruiz and Pock-Chambolle).
 *   use_max          : 1 -> abs-MAX reduction (Ruiz); 0 -> abs-POW-SUM (Pock-Chambolle)
 *   exponent_for_row : power for the row reduction when use_max==0 (PC=alpha)
 *   exponent_for_col : power for the col reduction when use_max==0 (PC=2-alpha)
 *   combine_with_c   : Ruiz folds |c| into the column max; PC does not. */
static fiv_ret fiv_lp_rescale_once(fiv_lp_rescaling *rescaling, fiv_lp_mat *lp_matrix,
                                   fiv_vec *c, fiv_vec *l, fiv_vec *u, fiv_vec *q,
                                   int use_max, ivf64 exponent_for_row, ivf64 exponent_for_col,
                                   int combine_with_c, ivf64 eps_zero)
{
    const size_t num_rows = lp_matrix->rows;
    const size_t num_cols = lp_matrix->cols;

    fiv_vec *row_reduction = fiv_create_tensor1d(num_rows, FIV_64F1);
    fiv_vec *col_reduction = fiv_create_tensor1d(num_cols, FIV_64F1);
    fiv_vec *row_rescale_vec = fiv_create_tensor1d(num_rows, FIV_64F1);
    fiv_vec *col_rescale_vec = fiv_create_tensor1d(num_cols, FIV_64F1);
    if (row_reduction == NULL || col_reduction == NULL ||
        row_rescale_vec == NULL || col_rescale_vec == NULL) {
        fiv_release_tensor1d(&row_reduction);
        fiv_release_tensor1d(&col_reduction);
        fiv_release_tensor1d(&row_rescale_vec);
        fiv_release_tensor1d(&col_rescale_vec);
        return FIV_RET_ERR_MEM;
    }

    fiv_ret ret_code;
    if (use_max) {
        ret_code = fiv_rescale_reduce_max_dim(lp_matrix, 0, row_reduction);
        if (ret_code != FIV_RET_OK) goto cleanup;
        ret_code = fiv_rescale_reduce_max_dim(lp_matrix, 1, col_reduction);
        if (ret_code != FIV_RET_OK) goto cleanup;
    } else {
        ret_code = fiv_rescale_reduce_pow_dim(lp_matrix, 0, exponent_for_row, row_reduction);
        if (ret_code != FIV_RET_OK) goto cleanup;
        ret_code = fiv_rescale_reduce_pow_dim(lp_matrix, 1, exponent_for_col, col_reduction);
        if (ret_code != FIV_RET_OK) goto cleanup;
    }

    /* row rescale = sqrt(max(row_reduction, eps_zero)) */
    for (size_t row = 0; row < num_rows; row++) {
        const ivf64 value = row_reduction->data.db[row];
        row_rescale_vec->data.db[row] = (value > eps_zero) ? sqrt(value) : 1.0;
    }
    /* col rescale = sqrt(max(col_reduction [or max(|c|)], eps_zero)) */
    const ivf64 *cdata = c->data.db;
    for (size_t col = 0; col < num_cols; col++) {
        ivf64 value = col_reduction->data.db[col];
        if (combine_with_c) {
            const ivf64 abs_c = fabs(cdata[col]);
            if (abs_c > value) value = abs_c;
        }
        col_rescale_vec->data.db[col] = (value > eps_zero) ? sqrt(value) : 1.0;
    }

    ret_code = fiv_rescale_apply_vectors(rescaling, c, l, u, q, row_rescale_vec, col_rescale_vec);
    if (ret_code != FIV_RET_OK) goto cleanup;
    ret_code = fiv_rescale_apply_matrix(lp_matrix, row_rescale_vec, col_rescale_vec);

cleanup:
    fiv_release_tensor1d(&row_reduction);
    fiv_release_tensor1d(&col_reduction);
    fiv_release_tensor1d(&row_rescale_vec);
    fiv_release_tensor1d(&col_rescale_vec);
    return ret_code;
}


/* ------------------------------------------------------------------ */
/* Public rescale entry points                                        */
/* ------------------------------------------------------------------ */

fiv_ret fiv_lp_rescale_ruiz(fiv_lp_rescaling *rescaling, fiv_lp_mat *lp_matrix,
                            fiv_vec *c, fiv_vec *l, fiv_vec *u, fiv_vec *q,
                            int ruiz_iterations, ivf64 eps_zero)
{
    if (rescaling == NULL || lp_matrix == NULL || c == NULL || l == NULL ||
        u == NULL || q == NULL)
        return FIV_RET_ERR_PARA;
    if (ruiz_iterations <= 0) return FIV_RET_OK;
    for (int iteration = 0; iteration < ruiz_iterations; iteration++) {
        fiv_ret ret_code = fiv_lp_rescale_once(rescaling, lp_matrix, c, l, u, q,
                                               1, 1.0, 1.0, 1, eps_zero);
        if (ret_code != FIV_RET_OK) return ret_code;
    }
    return FIV_RET_OK;
}

fiv_ret fiv_lp_rescale_pock_chambolle(fiv_lp_rescaling *rescaling, fiv_lp_mat *lp_matrix,
                                      fiv_vec *c, fiv_vec *l, fiv_vec *u, fiv_vec *q,
                                      ivf64 alpha, ivf64 eps_zero)
{
    if (rescaling == NULL || lp_matrix == NULL || c == NULL || l == NULL ||
        u == NULL || q == NULL)
        return FIV_RET_ERR_PARA;
    if (alpha <= 0.0) return FIV_RET_OK;   /* disabled */
    return fiv_lp_rescale_once(rescaling, lp_matrix, c, l, u, q,
                               0, alpha, 2.0 - alpha, 0, eps_zero);
}

fiv_ret fiv_lp_rescale_solve(fiv_lp_rescaling *rescaling, fiv_lp_mat *lp_matrix,
                             fiv_vec *c, fiv_vec *l, fiv_vec *u, fiv_vec *q,
                             int ruiz_iterations, ivf64 pock_chambolle_alpha, ivf64 eps_zero)
{
    if (rescaling == NULL || lp_matrix == NULL || c == NULL || l == NULL ||
        u == NULL || q == NULL)
        return FIV_RET_ERR_PARA;
    fiv_ret ret_code = fiv_lp_rescale_ruiz(rescaling, lp_matrix, c, l, u, q,
                                           ruiz_iterations, eps_zero);
    if (ret_code != FIV_RET_OK) return ret_code;
    return fiv_lp_rescale_pock_chambolle(rescaling, lp_matrix, c, l, u, q,
                                         pock_chambolle_alpha, eps_zero);
}


/* ------------------------------------------------------------------ */
/* Unscaling                                                           */
/* ------------------------------------------------------------------ */

fiv_ret fiv_lp_unscale_primal(fiv_vec *x_unscaled, const fiv_lp_rescaling *rescaling,
                              const fiv_vec *x)
{
    if (x_unscaled == NULL || rescaling == NULL || x == NULL)
        return FIV_RET_ERR_PARA;
    if (x_unscaled->length != x->length ||
        x_unscaled->length != rescaling->variable_rescaling->length)
        return FIV_RET_ERR_PARA;
    return fiv_tensor_div(x_unscaled, x, rescaling->variable_rescaling);
}

fiv_ret fiv_lp_unscale_dual(fiv_vec *y_unscaled, const fiv_lp_rescaling *rescaling,
                            const fiv_vec *y)
{
    if (y_unscaled == NULL || rescaling == NULL || y == NULL)
        return FIV_RET_ERR_PARA;
    if (y_unscaled->length != y->length ||
        y_unscaled->length != rescaling->constraint_rescaling->length)
        return FIV_RET_ERR_PARA;
    return fiv_tensor_div(y_unscaled, y, rescaling->constraint_rescaling);
}
