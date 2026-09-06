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

#include "fiv_math_rms_norm.h"
#include <math.h>     /* sqrt / sqrtf */


/* ==================== Scalar backends ==================== */

/* One row: rms = sqrt(mean(x^2) + eps); dst[j] = x[j] / rms * gain[j].
   Works in-place (dst aliases src): the sum-of-squares pass consumes all of
   src[j] before the write-back pass scales and stores dst[j]. gain is a
   per-column vector shared by every row. FIV_64F1. */
static void fiv_math_rms_norm_row_real64(ivf64* dst, const ivf64* src,
                                         const ivf64* gain, size_t cols, ivf64 eps)
{
    ivf64 ss = 0.0;
    for (size_t j = 0; j < cols; j++) {
        ivf64 v = src[j];
        ss += v * v;
    }
    const ivf64 inv_rms = 1.0 / sqrt(ss / (ivf64)cols + eps);
    for (size_t j = 0; j < cols; j++)
        dst[j] = src[j] * inv_rms * gain[j];
}

void fiv_math_rms_norm_real64(ivf64* dst, int dst_stride,
                              const ivf64* src, int src_stride,
                              const ivf64* gain,
                              size_t rows, size_t cols, ivf64 eps)
{
    for (size_t i = 0; i < rows; i++)
        fiv_math_rms_norm_row_real64(dst + (size_t)dst_stride * i,
                                     src + (size_t)src_stride * i,
                                     gain, cols, eps);
}

/* Scalar FIV_32F1 backend, same per-row algorithm as the FIV_64F1 variant. */
static void fiv_math_rms_norm_row_real32(ivf32* dst, const ivf32* src,
                                         const ivf32* gain, size_t cols, ivf32 eps)
{
    ivf32 ss = 0.0f;
    for (size_t j = 0; j < cols; j++) {
        ivf32 v = src[j];
        ss += v * v;
    }
    const ivf32 inv_rms = 1.0f / sqrtf(ss / (ivf32)cols + eps);
    for (size_t j = 0; j < cols; j++)
        dst[j] = src[j] * inv_rms * gain[j];
}

void fiv_math_rms_norm_real32(ivf32* dst, int dst_stride,
                              const ivf32* src, int src_stride,
                              const ivf32* gain,
                              size_t rows, size_t cols, ivf32 eps)
{
    for (size_t i = 0; i < rows; i++)
        fiv_math_rms_norm_row_real32(dst + (size_t)dst_stride * i,
                                     src + (size_t)src_stride * i,
                                     gain, cols, eps);
}
