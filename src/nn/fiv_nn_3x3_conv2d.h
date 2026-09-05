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
 * 3x3 kernel plane convolutions (STD fast paths).
 */

#ifndef _FIV_NN_3X3_CONV2D_H_
#define _FIV_NN_3X3_CONV2D_H_

#include <stddef.h>
#include "fiv_nn_conv2d.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Dense (STD) 3x3 conv over all (oc, ic): d[oc] = sum_ic conv(s[ic], w[oc][ic]).
   These own the oc / ic loops so each module can be tuned in place. stride-1
   leaves the plane size unchanged; stride-2 halves it. zero_pad: 0 = replicate
   edge (clamp), 1 = zero padding (skip out-of-range). stride-2 uses SAME pad
   (pt = pl = 1). */
void fiv_conv2d_std_3x3_s1(ivf32* d, const ivf32* s, const ivf32* w,
                           int c_in, int c_out, int width, int height,
                           int zero_pad);

/* Winograd F(2,3) dense 3x3 stride-1 layer (inference only). Same NCHW layout
   and weight order as fiv_conv2d_std_3x3_s1. NOT bit-identical to the direct
   kernel (transformed arithmetic); validation uses a small tolerance. */
void fiv_conv2d_std_3x3_s1_wino(ivf32* d, const ivf32* s, const ivf32* w,
                                int c_in, int c_out, int width, int height,
                                int zero_pad);
void fiv_conv2d_std_3x3_s2(ivf32* d, const ivf32* s, const ivf32* w,
                           int c_in, int c_out, int width, int height,
                           int oh, int ow, int zero_pad);

/* Single src plane x single 3x3 coef -> one dst plane. stride-1 keeps the
   size; stride-2 halves it. accumulate: (ic > 0) adds into dst (except the
   depthwise path, which calls it with accumulate=0 once per output channel).
   zero_pad: 0 = replicate edge, 1 = zero padding. stride-2 uses SAME pad. */
void fiv_conv2d_plane_3x3_s1(ivf32* dst, int width_dst, int height_dst, int stride_dst,
                             const ivf32* src, int width_src, int height_src, int stride_src,
                             const ivf32 coef[9], int accumulate, int zero_pad);
void fiv_conv2d_plane_3x3_s2(ivf32* dst, int ow, int oh, int stride_dst,
                             const ivf32* src, int width, int height, int stride_src,
                             const ivf32 coef[9], int accumulate, int zero_pad);

/* DEPTHWISE 3x3 over all output channels: one coef per oc, single input
   channel ic = oc / mult. Owns the oc loop so the module can be tuned. */
void fiv_conv2d_dw_3x3_s1(ivf32* d, const ivf32* s, const ivf32* w,
                          int c_out, int mult, int width, int height, int zero_pad);
void fiv_conv2d_dw_3x3_s2(ivf32* d, const ivf32* s, const ivf32* w,
                          int c_out, int mult, int width, int height,
                          int oh, int ow, int zero_pad);

#ifdef __cplusplus
}
#endif

#endif /* _FIV_NN_3X3_CONV2D_H_ */