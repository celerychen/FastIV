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
fiv_ret fiv_matrix_mul(fiv_mat* dst, const fiv_mat* A, const fiv_mat* B,
                       int a_transpose, int b_transpose, fiv_scalar alpha, fiv_scalar beta);

/* Broadcast-add vec to each row (dim==0) or column (dim==1) of src.
   src/vec/dst share a float dtype (FIV_32F1 or FIV_64F1); dst may alias src. */
fiv_ret fiv_matrix_add_vec(fiv_mat* dst, const fiv_mat* src, const fiv_vec* vec, int dim);

/* dst = beta * dst + sum(src along dim), dim in {-1,0,1} (-1 sums all).
   src/dst/beta share a float dtype (FIV_32F1 or FIV_64F1). */
fiv_ret fiv_matrix_reduce_sum(void* dst, fiv_mat* src, int dim, fiv_scalar beta);


#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MATRIX_H_ */
