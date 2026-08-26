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

#ifndef _FIV_MAT_MUL_DB_H_
#define _FIV_MAT_MUL_DB_H_

#include "fiv_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Working set (A + B + C in bytes) below which the non-blocked small-matrix
   path is used; above it the blocked path takes over. Mirrors
   FIV_MAT_MUL_L3_LIMIT_BYTES but kept separate so the float32 and float64
   dispatchers can be tuned independently. Override with
   -DFIV_MAT_MUL_DB_L3_LIMIT_BYTES=<n>. */
#ifndef FIV_MAT_MUL_DB_L3_LIMIT_BYTES
#define FIV_MAT_MUL_DB_L3_LIMIT_BYTES (8u * 1024u * 1024u)
#endif

/* 64-bit (ivf64 / double) matrix multiply, full API. This is the dtype-specific
   backend invoked by the generic fiv_matrix_mul (api/fiv_matrix.h) when the
   operands are FIV_64F1; it is NOT a standalone public interface.
   dst = alpha * op(A) * op(B) + beta * dst, with op(X) selected by
   a_transpose / b_transpose. A, B and dst must be contiguous FIV_64F1 tensors;
   alpha and beta are fiv_scalar of dtype FIV_64F1. In-place aliasing of dst
   with A or B is not supported. */
fiv_ret fiv_matrix_mul_real64(fiv_mat* dst, const fiv_mat* A, const fiv_mat* B,
                              int a_transpose, int b_transpose, fiv_scalar alpha, fiv_scalar beta);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MAT_MUL_DB_H_ */
