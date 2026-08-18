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

#include <string.h>
#include "fiv_matrix.h"
#include "fiv_common.h"

/* Reduce a matrix to a vector or a scalar by summing along one axis.
   dim == 0: sum over rows -> dst[j] = sum_i src[i, j]  (dst: vector, length == cols)
   dim == 1: sum over cols -> dst[i] = sum_j src[i, j]  (dst: vector, length == rows)
   dim == -1: total sum -> dst[0] = sum_{i,j} src[i, j] (dst: scalar) */
fiv_ret fiv_matrix_reduce_sum(void* dst, fiv_mat* src, int dim)
{
    if (!dst || !src) return FIV_RET_ERR_PARA;
    if (src->id != FIV_ID_TENSOR2D || src->dtype != FIV_32F1 || src->data_continue == 0)
        return FIV_RET_ERR_PARA;

    size_t rows = src->rows;
    size_t cols = src->cols;
    const ivf32* s = src->data.fl;

    if (dim == 0 || dim == 1) {
        fiv_vec* out = (fiv_vec*)dst;
        if (out->id != FIV_ID_TENSOR1D || out->dtype != FIV_32F1 || out->data_continue == 0)
            return FIV_RET_ERR_PARA;
        ivf32* d = out->data.fl;
        if (dim == 0) {
            if (out->length != cols) return FIV_RET_ERR_PARA;
            for (size_t j = 0; j < cols; j++) {
                float acc = 0.0f;
                for (size_t i = 0; i < rows; i++) acc += s[i * cols + j];
                d[j] = acc;
            }
        } else {
            if (out->length != rows) return FIV_RET_ERR_PARA;
            for (size_t i = 0; i < rows; i++) {
                float acc = 0.0f;
                for (size_t j = 0; j < cols; j++) acc += s[i * cols + j];
                d[i] = acc;
            }
        }
    } else if (dim == -1) {
        fiv_scalar* out = (fiv_scalar*)dst;
        if (out->id != FIV_ID_SCALAR || out->dtype != FIV_32F1)
            return FIV_RET_ERR_PARA;
        float acc = 0.0f;
        size_t n = rows * cols;
        for (size_t k = 0; k < n; k++) acc += s[k];
        out->data.value_fp32 = acc;
    } else {
        return FIV_RET_ERR_PARA;
    }
    return FIV_RET_OK;
}
