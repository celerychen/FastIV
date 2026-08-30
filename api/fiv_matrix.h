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


/* Transpose src (m x n) into dst (n x m). Float dtype, contiguous. In-place
   (dst aliasing src) only for square matrices. */
fiv_ret fiv_matrix_transpose(fiv_mat* dst, const fiv_mat* src);

/* dst = mat * vec, or mat^T * vec if transpose!=0. Float dtype, contiguous;
   in-place not supported. */
fiv_ret fiv_matrix_mul_vec(fiv_vec* dst, const fiv_mat* mat, const fiv_vec* vec, int transpose);

/* dst = alpha * op(A) * op(B) + beta * dst. Float dtype, contiguous;
   in-place not supported. */
fiv_ret fiv_matrix_mul(fiv_mat* mat_c, const fiv_mat* mat_a, const fiv_mat* mat_b,
                       int a_transpose, int b_transpose, fiv_scalar alpha, fiv_scalar beta);

/* Broadcast-add vec to each row (dim==0) or column (dim==1) of src;
   dst may alias src. */
fiv_ret fiv_matrix_add_vec(fiv_mat* dst, const fiv_mat* src, const fiv_vec* vec, int dim);

/* dst = beta * dst + sum(src along dim), dim in {-1,0,1} (-1 = all). */
fiv_ret fiv_matrix_reduce_sum(void* dst, fiv_mat* src, int dim, fiv_scalar beta);

/* In-place blocked Cholesky of a square, contiguous float32 SPD matrix A_io.
   lower!=0: A = L*L^T, L in the lower triangle; lower==0: A = U^T*U, U = L^T
   in the upper. Only the referenced triangle is read; the other is scratch.
   NOT_POS_DEF if not positive definite. */
fiv_ret fiv_matrix_cholesky(fiv_mat* mat_a, int lower);

/* In-place blocked LU with partial pivoting of a contiguous float32 matrix
   A_io. piv[min(rows,cols)] records row swaps, P*A = L*U; A_io packs L (unit
   diagonal implicit) + U. SINGULAR on a zero pivot (factors still written). */
fiv_ret fiv_matrix_lu(fiv_mat* mat_a, int* piv);

/* In-place symmetric eigen decomposition (Householder tridiagonalization +
   implicit-shift QL) of a square, contiguous float32 symmetric matrix A_io
   (destroyed). evals[dim] gets the ascending eigenvalues; optional mat_evec
   (dim x dim, must not alias A_io) gets orthonormal eigenvectors in its
   columns. Only the triangle selected by upper is read; the other is
   scratch. UNKNOWN if the QL iteration fails to converge. */
fiv_ret fiv_matrix_eig_sym(fiv_mat* mat_a, ivf32* evals, fiv_mat* mat_evec, int upper);

/* Thin SVD of a contiguous float32 matrix A (input is preserved):
   A = U * diag(sing_vals) * V^T with sing_vals[k] descending, k =
   min(rows,cols); sing_vals may be NULL together with both factor outputs
   for a values-only run.
   mat_u (rows x k, optional): orthonormal LEFT singular vectors in its
   COLUMNS, column j matching sing_vals[j].
   mat_v (optional): the RIGHT singular vectors in one of two layouts,
   selected by v_transpose:
     v_transpose == 0 -> mat_v is (cols x k), vectors in its COLUMNS;
     v_transpose != 0 -> mat_v is (k x cols), vectors in its ROWS (V^T, the
                          OpenCV SVD::compute "vt" convention).
   type selects the backend: FIV_JACOBI_SVD runs one-sided Jacobi sweeps
   (OpenCV lineage; highest accuracy; prefer for small and ill-conditioned
   inputs), FIV_BCD_SVD runs the blocked Householder bidiagonalization +
   Golub-Kahan QR path (LAPACK dgesdd lineage; fastest for large inputs).
   Both share this exact contract and preserve A. UNKNOWN if the selected
   backend fails to converge; NOT_SUPPORT for an unknown type. */

typedef FIV_ENUM(iv32u){
     FIV_JACOBI_SVD,
     FIV_BCD_SVD,
}fiv_svd_type;


fiv_ret fiv_matrix_svd(fiv_mat* mat_a, ivf32* sing_vals, fiv_mat* mat_u, fiv_mat* mat_v, int v_transpose, fiv_svd_type type);

typedef FIV_ENUM(iv32u) {
   FIV_L1_NORM,
   FIV_L2_NORM,
   FIV_INF_NORM,
}fiv_norm_type;


/* norm = L1 / L2 / L-inf of vec (FIV_32F1 or FIV_64F1, contiguous);
   result scalar inherits the dtype. */
fiv_ret fiv_vec_norm(fiv_scalar* norm_value, fiv_vec* vec,  fiv_norm_type  norm_type);

/* y = a * x + y (axpy). x/y equal length, float dtype, contiguous;
   y may alias x. */
fiv_ret fiv_vec_axpy(fiv_vec* y, fiv_scalar a, fiv_vec* x);


/*
  y = x * scale 
*/

fiv_ret fiv_vec_scale(fiv_vec* y, fiv_vec* x, fiv_scalar scale);

/* dot = sum_i a[i] * b[i]. a/b equal length, float dtype, contiguous;
   result scalar inherits the dtype. */
fiv_ret fiv_vec_dot(fiv_scalar* dot_value, const fiv_vec* a, const fiv_vec* b);


#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MATRIX_H_ */
