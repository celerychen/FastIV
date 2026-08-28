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

#ifndef _FIV_IMAGE_GAUSSION_BLUR_H_
#define _FIV_IMAGE_GAUSSION_BLUR_H_

#include "fiv_ctensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Separable Gaussian blur (horizontal 1-D convolution followed by a vertical
 * 1-D convolution with the same 1-D Gaussian kernel).
 *
 * Supports iv8u tensors (8U1 / 8U3 / 8U4) and ivf32 tensors (32F1 / 32F3 /
 * 32F4), i.e. both gray and interleaved color images. Boundary handling is
 * BORDER_REPLICATE (edge replication), handled separately for the first/last
 * `a` columns and rows -- matching the reference conv2d_3x1_1x3_v1 approach.
 *
 * Parameters:
 *   dst    output tensor, must be pre-allocated with the same dtype and the
 *          same height/width/stride as src (in-place is NOT supported here;
 *          pass a separate dst)
 *   src    input tensor
 *   sigma  Gaussian standard deviation (sigma > 0.1); the kernel size is
 *          derived from sigma unless `size` is given explicitly
 *   size   Gaussian kernel size:
 *            size <= 0 : kernel size auto-derived from sigma using OpenCV's
 *                        rule  ksize = round(sigma * (float?4:3) * 2 + 1) | 1
 *            size  > 0 : used as-is when odd; when even, ksize = 2*size + 1
 *
 * Returns FIV_RET_OK on success, or an error code otherwise. */
fiv_ret fiv_image_gaussian_blur(fiv_mat* dst, fiv_mat* src, ivf32 sigma, int size);

/* Precision-aligned variant of fiv_image_gaussian_blur for 8-bit inputs.
 *
 * The default 8U path rounds twice (row->u8->col->u8), diverging from
 * OpenCV's single-round 8U separable GaussianBlur by up to 3 on full-noise
 * inputs.  This variant keeps the row-filter result in a 16-bit intermediate
 * and rounds only once at the end of the column filter, matching OpenCV's
 * single-round 8U path (max|Δ| vs OpenCV drops to <=1) at the cost of a
 * 2x-larger intermediate ring buffer and slightly different SIMD scheduling.
 *
 * 32F inputs are already single-round and are routed to the standard path.
 * Same signatures / semantics as fiv_image_gaussian_blur otherwise. */
fiv_ret fiv_image_gaussian_blur_precise(fiv_mat* dst, fiv_mat* src, ivf32 sigma, int size);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_IMAGE_GAUSSION_BLUR_H_ */
