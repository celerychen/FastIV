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

#ifndef _FIV_LP_MAT_H_
#define _FIV_LP_MAT_H_

#include "fiv_ctensor.h"   /* fiv_vec, fiv_ret, fiv_data_type */
#include "fiv_matrix.h"    /* fiv_mat (dense backend) */
#include "fiv_sp_matrix.h" /* fiv_sparse_mat (sparse backend) */

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Unified matrix abstraction for PDLP.
 *
 * The PDHG core never touches the concrete storage; it only calls
 * fiv_lp_mat_matvec / fiv_lp_mat_reduce_abs_* on an fiv_lp_mat. The backend is
 * either a DENSE fiv_mat or a SPARSE fiv_sparse_mat (CSR/CSC/COO), dispatched
 * inside these functions.
 *
 * Ownership model:
 *   - fiv_lp_mat_wrap_dense / fiv_lp_mat_wrap_sparse build a NON-owning view
 *     over a matrix the caller already owns and must release separately.
 *   - fiv_create_lp_mat_from_coo is the only OWNING entry: it builds the CSR
 *     matrix internally and fiv_release_lp_mat frees both the wrapper and the
 *     underlying sparse matrix in that case.
 * ========================================================================= */

typedef FIV_ENUM(iv8u) {
    FIV_LP_MAT_DENSE = 0,
    FIV_LP_MAT_SPARSE,
} fiv_lp_mat_kind;

typedef struct {
    fiv_lp_mat_kind kind;
    size_t rows;
    size_t cols;
    int    owns_data;             /* 1 only for fiv_create_lp_mat_from_coo */
    union {
        fiv_mat        *dense;
        fiv_sparse_mat *sparse;   /* canonical CSR for A * x (transpose == 0) */
    } as;
    /* Optional transposed view (CSC) for Aᵀ * x (transpose != 0).
       For DENSE it stays NULL (FastIV fiv_matrix_mul_vec handles transpose
       natively). For SPARSE it is materialized by fiv_lp_mat_build_transpose
       (owns the buffer, freed by fiv_release_lp_mat). When NULL, the sparse
       transpose call falls back to as.sparse (which the caller may have set to
       a CSC, e.g. the M_sparseT wrap convention in the tests). */
    fiv_sparse_mat *transpose_view;
} fiv_lp_mat;


/* Non-owning view over an existing dense matrix. Returns the wrapper (NULL on
 * allocation failure); release with fiv_release_lp_mat (does NOT free dense). */
fiv_lp_mat *fiv_lp_mat_wrap_dense(fiv_mat *dense_matrix);

/* Non-owning view over an existing sparse matrix. Returns the wrapper. */
fiv_lp_mat *fiv_lp_mat_wrap_sparse(fiv_sparse_mat *sparse_matrix);

/* Owning entry: build a CSR matrix from COO triplets then wrap it. The result
 * is released with fiv_release_lp_mat, which also frees the underlying sparse
 * matrix. NULL on error. */
fiv_lp_mat *fiv_create_lp_mat_from_coo(const int *row_indices, const int *col_indices,
                                       const void *values, fiv_data_type value_dtype,
                                       size_t num_nonzeros, size_t num_rows, size_t num_cols);

/* Free the wrapper (and the owned sparse matrix when owns_data). *lp_matrix is
 * set to NULL. Safe when *lp_matrix == NULL. Also frees transpose_view when set
 * (always owned by the wrapper). */
fiv_ret fiv_release_lp_mat(fiv_lp_mat **lp_matrix);

/* Materialize the transposed view (CSC for sparse, no-op for dense) so that
 * fiv_lp_mat_matvec(..., transpose != 0) can compute Aᵀ * x. Idempotent: a
 * second call is a no-op returning FIV_RET_OK. For sparse, as.sparse must be a
 * CSR (the canonical format built by fiv_create_lp_mat_from_coo). */
fiv_ret fiv_lp_mat_build_transpose(fiv_lp_mat *lp_matrix);


/* dst = M * x (transpose==0) or Mᵀ * x (transpose!=0).
 * Lengths are validated by the backend (rows/cols of M vs x / dst length). */
fiv_ret fiv_lp_mat_matvec(fiv_vec *dst, const fiv_lp_mat *lp_matrix,
                          const fiv_vec *x, int transpose);

/* dst = max over entries of |value| along dim.
 *   dim == 0 -> per row  (dst length == rows)
 *   dim == 1 -> per column (dst length == cols) */
fiv_ret fiv_lp_mat_reduce_abs_max(fiv_vec *dst, const fiv_lp_mat *lp_matrix, int dim);

/* dst = sum over entries of |value|^exponent along dim (same dim rule).
 * Used by Ruiz rescaling (exponent == 1) and Pock-Chambolle rescaling
 * (exponent == 2 - alpha for columns, exponent == alpha for rows). */
fiv_ret fiv_lp_mat_reduce_abs_pow(fiv_vec *dst, const fiv_lp_mat *lp_matrix,
                                  int dim, ivf64 exponent);


#ifdef __cplusplus
}
#endif

#endif  /* _FIV_LP_MAT_H_ */
