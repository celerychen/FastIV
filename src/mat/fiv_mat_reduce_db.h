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

#ifndef _FIV_MAT_REDUCE_DB_H_
#define _FIV_MAT_REDUCE_DB_H_

#include "fiv_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 64-bit (ivf64 / double) matrix reduce-sum, dtype-specific backend.
   Invoked by the generic fiv_matrix_reduce_sum (api/fiv_matrix.h) when src is
   FIV_64F1; NOT a standalone public interface. dst = beta*dst + sum(src along
   dim): dim 0/1 writes a FIV_64F1 fiv_vec (length == cols/rows), dim -1 writes a
   FIV_64F1 fiv_scalar; beta must be an FIV_64F1 scalar, otherwise
   FIV_RET_ERR_NOT_SUPPORT is returned. */
fiv_ret fiv_matrix_reduce_sum_real64(void* dst, const fiv_mat* src, int dim, fiv_scalar beta);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MAT_REDUCE_DB_H_ */
