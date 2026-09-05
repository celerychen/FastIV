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
 * 2x2 kernel plane convolution (STD fast path, stride 2 only).
 */

#ifndef _FIV_NN_2X2_CONV2D_H_
#define _FIV_NN_2X2_CONV2D_H_

#include <stddef.h>
#include "fiv_nn_conv2d.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Dense (STD) 2x2 stride-2 conv over all (oc, ic): d[oc] = sum_ic conv(s[ic],
   w[oc][ic]). Owns the oc / ic loops so the module can be tuned in place.
   zero_pad: 0 = replicate edge, 1 = zero padding. pt / pl: explicit start pads
   (0 for the landmark even-dim downsample layers). */
void fiv_conv2d_std_2x2_s2(ivf32* d, const ivf32* s, const ivf32* w,
                           int c_in, int c_out, int width, int height,
                           int oh, int ow, int zero_pad, int pt, int pl);

/* Single src plane x single 2x2 coef -> one half-sized dst plane (stride 2). */
void fiv_conv2d_plane_2x2_s2(ivf32* dst, int ow, int oh, int stride_dst,
                             const ivf32* src, int width, int height, int stride_src,
                             const ivf32 coef[4], int accumulate, int zero_pad,
                             int pt, int pl);

#ifdef __cplusplus
}
#endif

#endif /* _FIV_NN_2X2_CONV2D_H_ */