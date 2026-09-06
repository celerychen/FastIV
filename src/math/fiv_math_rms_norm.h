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

#ifndef _FIV_MATH_RMS_NORM_H_
#define _FIV_MATH_RMS_NORM_H_

#include "fiv_data_typedefs.h"
#include <stddef.h>   /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Internal RMSNorm backends invoked by the public fiv_math_rms_norm
   (api/fiv_math.h). Row-wise normalization of a whole matrix: for each row i
     rms_i = sqrt( mean_j( src[i,j]^2 ) + eps )
     dst[i,j] = src[i,j] / rms_i * gain[j]
   gain is a per-column scale vector of length cols, broadcast over rows. dst
   may alias src (in-place). NOT standalone public interfaces.

   dst_stride / src_stride are the row strides in elements (distance between
   consecutive rows in the flat buffer), matching the lda/ldb/ldc convention of
   the low-level matrix-multiply interface. A contiguous row-major matrix has
   stride == cols. */

/* FIV_64F1 scalar backend. */
void fiv_math_rms_norm_real64(ivf64* dst, int dst_stride,
                              const ivf64* src, int src_stride,
                              const ivf64* gain,
                              size_t rows, size_t cols, ivf64 eps);

/* FIV_32F1 scalar backend. */
void fiv_math_rms_norm_real32(ivf32* dst, int dst_stride,
                              const ivf32* src, int src_stride,
                              const ivf32* gain,
                              size_t rows, size_t cols, ivf32 eps);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MATH_RMS_NORM_H_ */
