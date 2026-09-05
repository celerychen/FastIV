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

/* Matrix reduce-sum (float32, native row-major) driving the public
   fiv_matrix_reduce_sum (api/fiv_matrix.h). The 64-bit dtype-specific backend
   lives in fiv_mat_reduce_db.h. */
#ifndef _FIV_MAT_REDUCE_H_
#define _FIV_MAT_REDUCE_H_

#include "fiv_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* dst = beta * dst + sum(src along dim), dim in {-1,0,1} (-1 = all):
   dim == 0 sums over rows (dst: FIV_32F1 fiv_vec, length == cols);
   dim == 1 sums over cols (dst: FIV_32F1 fiv_vec, length == rows);
   dim == -1 sums all elements (dst: FIV_32F1 fiv_scalar).
   beta must be an FIV_32F1 scalar matching the dst type, otherwise
   FIV_RET_ERR_NOT_SUPPORT is returned; 64-bit inputs route to
   fiv_matrix_reduce_sum_real64. */
fiv_ret fiv_matrix_reduce_sum(void* dst, fiv_mat* src, int dim, fiv_scalar beta);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MAT_REDUCE_H_ */