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

#ifndef _FIV_MATH_SIGMOID_H_
#define _FIV_MATH_SIGMOID_H_

#include "fiv_data_typedefs.h"
#include <stddef.h>   /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Internal sigmoid backends invoked by the public fiv_math_sigmoid
   (api/fiv_math.h). Element-wise dst[i] = 1 / (1 + exp(-src[i])) over a whole
   contiguous buffer; dst may alias src (in-place). NOT standalone public
   interfaces. */

/* FIV_64F1 scalar backend. */
void fiv_math_sigmoid_real64(ivf64* dst, const ivf64* src, size_t element_count);

/* FIV_32F1 scalar fallback, only compiled when the AVX2 kernel is absent. */
#if !defined(FIV_USE_AVX2)
void fiv_math_sigmoid_real32(ivf32* dst, const ivf32* src, size_t element_count);
#endif

/* FIV_32F1 AVX2+FMA backend, only compiled when FIV_USE_AVX2 is defined. */
#if defined(FIV_USE_AVX2)
void fiv_math_sigmoid_avx2_ps(ivf32* dst, const ivf32* src, size_t element_count);
#endif

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MATH_SIGMOID_H_ */
