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

#ifndef _FIV_IMAGE_RESIZER_H_
#define _FIV_IMAGE_RESIZER_H_

#include "fiv_image.h"

/*
 * Image resizer module. The public entry fiv_image_resize() is declared in
 * fiv_image.h. Both nearest-neighbor and bilinear support single-channel
 * (FIV_8U1) and three-channel interleaved (FIV_8U3) 8-bit images.
 */

/* Nearest-neighbor resize, exposed for benchmarking and the public entry. */
fiv_ret fiv_image_resize_nn(fiv_mat* dst, fiv_mat* src);

/* Bilinear resize (Q11 fixed point, clamped edge sampling). */
fiv_ret fiv_image_resize_bilinear(fiv_mat* dst, fiv_mat* src);

#endif  /* _FIV_IMAGE_RESIZER_H_ */