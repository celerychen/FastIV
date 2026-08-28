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

#ifndef _FIV_MATRIX_H_
#define _FIV_MATRIX_H_

#include "fiv_ctensor.h"

#ifdef __cplusplus
extern "C" {
#endif


/* Transpose src (m x n) into dst (n x m). Both must be contiguous, share the
   same dtype and hold data; in-place is not supported. */
fiv_ret fiv_matrix_transpose(fiv_mat* dst, const fiv_mat* src);

/* dst = mat * vec (transpose==0) or mat^T * vec (transpose!=0). All operands
   share a float dtype (FIV_32F1 or FIV_64F1) and must be contiguous; in-place
   (dst aliasing vec) is not supported. */
fiv_ret fiv_matrix_mul_vec(fiv_vec* dst, const fiv_mat* mat, const fiv_vec* vec, int transpose);

/* dst = alpha * op(A) * op(B) + beta * dst. Inner dims of op(A)/op(B) must
   match; A/B/dst share a float dtype (FIV_32F1 or FIV_64F1); in-place (dst
   aliasing A or B) is not supported. */
fiv_ret fiv_matrix_mul(fiv_mat* mat_c, const fiv_mat* mat_a, const fiv_mat* mat_b,
                       int a_transpose, int b_transpose, fiv_scalar alpha, fiv_scalar beta);

/* Broadcast-add vec to each row (dim==0) or column (dim==1) of src.
   src/vec/dst share a float dtype (FIV_32F1 or FIV_64F1); dst may alias src. */
fiv_ret fiv_matrix_add_vec(fiv_mat* dst, const fiv_mat* src, const fiv_vec* vec, int dim);

/* dst = beta * dst + sum(src along dim), dim in {-1,0,1} (-1 sums all).
   src/dst/beta share a float dtype (FIV_32F1 or FIV_64F1). */
fiv_ret fiv_matrix_reduce_sum(void* dst, fiv_mat* src, int dim, fiv_scalar beta);

/* In-place blocked Cholesky of a square, contiguous float32 SPD matrix A_io.
   lower!=0 -> lower triangle holds L with A = L*L^T; lower==0 -> holds L^T
   with A = U^T*U. Returns NOT_POS_DEF on a non-positive diagonal; the
   unreferenced triangle is overwritten as scratch. */
fiv_ret fiv_matrix_cholesky(fiv_mat* mat_a, int lower);

/* In-place blocked LU with partial pivoting of a contiguous float32 matrix
   A_io (rectangular ok). piv[] needs min(rows,cols) ints recording the row
   interchanges so P*A = L*U; A_io packs L (unit diagonal implicit) + U. A zero
   pivot does not abort: the column is skipped and SINGULAR is returned after
   the pass (factors/pivots still written). */
fiv_ret fiv_matrix_lu(fiv_mat* mat_a, int* piv);

typedef enum : iv32u {
   FIV_L1_NORM,
   FIV_L2_NORM,
   FIV_INF_NORM,
}fiv_norm_type;


/* norm = L1(sum|x|) / L2(sqrt(sum x^2)) / L∞(max|x|). vec dtype must be FIV_32F1 or FIV_64F1 and contiguous;
   result scalar inherits the dtype. */
fiv_ret fiv_vec_norm(fiv_scalar* norm_value, fiv_vec* vec,  fiv_norm_type  norm_type);

/* y = a * x + y (BLAS axpy). x and y share a float dtype (FIV_32F1 or FIV_64F1),
   must have equal length and be contiguous; y may alias x (in-place). */
fiv_ret fiv_vec_axpy(fiv_vec* y, fiv_scalar a, fiv_vec* x);

/* dot = sum_i a[i] * b[i]. a and b share a float dtype (FIV_32F1 or FIV_64F1),
   must have equal length and be contiguous. Result scalar inherits the dtype. */
fiv_ret fiv_vec_dot(fiv_scalar* dot_value, const fiv_vec* a, const fiv_vec* b);


#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MATRIX_H_ */
