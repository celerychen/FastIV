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

#ifndef _FIV_MATH_SOFTMAX_H_
#define _FIV_MATH_SOFTMAX_H_

#include "fiv_data_typedefs.h"
#include <stddef.h>   /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Internal softmax backends invoked by the public fiv_math_softmax
   (api/fiv_math.h). Row-wise normalization of a whole matrix: every row is
   processed independently, dst[i,j] = exp(src[i,j] - max_j) / sum_j exp(...)
   with the per-row max subtraction for stability. dst may alias src
   (in-place). NOT standalone public interfaces. */

/* FIV_64F1 scalar backend. */
void fiv_math_softmax_real64(ivf64* dst, const ivf64* src, size_t rows, size_t cols);

/* FIV_32F1 scalar fallback, only compiled when the AVX2 kernel is absent. */
#if !defined(FIV_USE_AVX2)
void fiv_math_softmax_real32(ivf32* dst, const ivf32* src, size_t rows, size_t cols);
#endif

/* FIV_32F1 AVX2+FMA backend, only compiled when FIV_USE_AVX2 is defined.
   Handles the out-of-place copy internally, then normalizes each row in place
   in dst. */
#if defined(FIV_USE_AVX2)
void fiv_math_softmax_avx2_ps(ivf32* dst, const ivf32* src, size_t rows, size_t cols);
#endif

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MATH_SOFTMAX_H_ */
