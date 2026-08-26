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

#ifndef _FIV_MAT_VEC_DB_H_
#define _FIV_MAT_VEC_DB_H_

#include "fiv_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 64-bit (ivf64 / double) matrix-vector multiply, dtype-specific backend.
   This is invoked by the generic fiv_matrix_mul_vec (api/fiv_matrix.h) when the
   operands are FIV_64F1; it is NOT a standalone public interface.
   dst = mat * vec (transpose == 0) or dst = mat^T * vec (transpose != 0), with
   the same shape/dtype/in-place rules as the float32 path. mat, vec and dst
   must be contiguous FIV_64F1 tensors; in-place aliasing of dst with vec is not
   supported. On success dst's metadata is rewritten to describe the result. */
fiv_ret fiv_matrix_mul_vec_real64(fiv_vec* dst, const fiv_mat* mat, const fiv_vec* vec, int transpose);

/* 64-bit (ivf64 / double) matrix + vector broadcast-add, dtype-specific backend.
   Invoked by the generic fiv_matrix_add_vec (api/fiv_matrix.h) when the operands
   are FIV_64F1; NOT a standalone public interface. dim == 0 broadcasts vec over
   each row (vec->length == cols, out[i,j] = src[i,j] + vec[j]); dim == 1
   broadcasts vec over each column (vec->length == rows, out[i,j] = src[i,j] +
   vec[i]); any other dim returns FIV_RET_ERR_PARA. tensors must be contiguous
   FIV_64F1 holding data; dst may alias src (in-place). On success dst's buffer
   holds src + broadcast(vec); its dtype/shape metadata is unchanged. */
fiv_ret fiv_matrix_add_vec_real64(fiv_mat* dst, const fiv_mat* src, const fiv_vec* vec, int dim);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MAT_VEC_DB_H_ */
