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

#ifndef FIV_BINARY_OP_H
#define FIV_BINARY_OP_H

#include "fiv_data_typedefs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* int32 scalar binary ops */
void fiv_add_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n);
void fiv_sub_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n);
void fiv_mul_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n);
void fiv_div_iv32s(iv32s* restrict c, const iv32s* restrict a, const iv32s* restrict b, size_t n);

/* float32 scalar binary ops */
void fiv_add_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n);
void fiv_sub_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n);
void fiv_mul_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n);
void fiv_div_ivf32(ivf32* restrict c, const ivf32* restrict a, const ivf32* restrict b, size_t n);

/* float64 scalar binary ops */
void fiv_add_ivf64(ivf64* restrict c, const ivf64* restrict a, const ivf64* restrict b, size_t n);
void fiv_sub_ivf64(ivf64* restrict c, const ivf64* restrict a, const ivf64* restrict b, size_t n);
void fiv_mul_ivf64(ivf64* restrict c, const ivf64* restrict a, const ivf64* restrict b, size_t n);
void fiv_div_ivf64(ivf64* restrict c, const ivf64* restrict a, const ivf64* restrict b, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* FIV_BINARY_OP_H */
