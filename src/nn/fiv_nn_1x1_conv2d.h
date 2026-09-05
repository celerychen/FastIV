/*
 * FastIV - Fast image and vision
 * Copyright (C) 2026 Celery Chen
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * See LICENSE file in project root for full license text.
 *
 * 1x1 kernel plane convolution (STD / POINTWISE fast path, stride 1).
 */

#ifndef _FIV_NN_1X1_CONV2D_H_
#define _FIV_NN_1X1_CONV2D_H_

#include <stddef.h>
#include "fiv_nn_conv2d.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 1x1 (pointwise) conv: all input channels at one spatial pixel -> one output
   channel per pixel, same hw = height*width layout. Called once per batch via
   the conv dispatch. No padding (spatial neighborhood is 1), so the pad params
   are ignored here. */
void fiv_conv2d_pw(ivf32* d, const ivf32* s, const ivf32* w, int c_in, int c_out, size_t hw);

#ifdef __cplusplus
}
#endif

#endif /* _FIV_NN_1X1_CONV2D_H_ */