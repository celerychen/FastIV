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


/* ============================== Matrix ops ============================== */

/* Transpose src (rows x cols) into dst (cols x rows). dst must already hold a
   buffer of >= rows*cols*element_bytes. Both tensors must be contiguous
   (data_continue == 1), share the same 4-byte dtype (32U/32S/32F families) and
   hold data; other dtypes return FIV_RET_ERR_NOT_SUPPORT, any mismatch returns
   FIV_RET_ERR_PARA. In-place (dst aliasing src) is NOT supported and returns
   FIV_RET_ERR_PARA. On success dst's metadata is rewritten to describe the
   transposed matrix. */
fiv_ret fiv_matrix_transpose(fiv_mat* dst, const fiv_mat* src);

/* Compute dst = mat * vec (transpose == 0) or dst = mat^T * vec (transpose != 0).
   mat is a rows x cols matrix. Non-transposed: vec holds cols entries, result
   dst holds rows entries. Transposed: vec holds rows entries, result dst holds
   cols entries. All tensors must be contiguous (data_continue == 1), hold data
   and share float32 dtype (32F); other dtypes return FIV_RET_ERR_NOT_SUPPORT,
   any mismatch returns FIV_RET_ERR_PARA. dst must be a 1D tensor (fiv_vec) with
   a buffer large enough for the result. In-place (dst aliasing vec) is NOT
   supported and returns FIV_RET_ERR_PARA. */
fiv_ret fiv_matrix_mul_vec(fiv_vec* dst, const fiv_mat* mat, const fiv_vec* vec, int transpose);

/* General matrix multiply: dst = alpha * op(A) * op(B) + beta * dst, where
   op(X) is X itself when the transpose flag is 0 and X^T otherwise. A is stored
   as rows_a x cols_a, B as rows_b x cols_b (both contiguous float32, holding
   data). The inner dims must match: a_transpose ? rows_a : cols_a must equal
   b_transpose ? cols_b : rows_b. Result dims: M = a_transpose ? cols_a : rows_a,
   N = b_transpose ? rows_b : cols_b. dst must be a contiguous float32 matrix
   of shape M x N whose buffer holds >= M*N*4 bytes; in-place (dst aliasing A or
   B) is NOT supported and returns FIV_RET_ERR_PARA. Non-32F dtypes return
   FIV_RET_ERR_NOT_SUPPORT, any shape/dtype mismatch returns FIV_RET_ERR_PARA.
   On success dst's metadata is rewritten to describe the M x N result. */
fiv_ret fiv_matrix_mul(fiv_mat* dst, const fiv_mat* A, const fiv_mat* B,
                       int a_transpose, int b_transpose, ivf32 alpha, ivf32 beta);


#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MATRIX_H_ */
