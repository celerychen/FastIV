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

#ifndef _FIV_IMAGE_COLOR_SPACE_H_
#define _FIV_IMAGE_COLOR_SPACE_H_

#include "fiv_image.h"

/* Scalar grayscale baseline (exact integer core). Exposed so the perf test can
   benchmark it against the SIMD kernels on equal footing. red/green/blue_index
   select the source channel order (0,1,2 = RGB; 2,1,0 = BGR). */
fiv_ret fiv_cs_to_gray_scalar(fiv_mat* image_dst, const fiv_mat* image_src,
                              int red_index, int green_index, int blue_index);

#endif  /* _FIV_IMAGE_COLOR_SPACE_H_ */
