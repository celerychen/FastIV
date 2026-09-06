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

/* Proto26 mask-prototype network weights + forward (YOLO26 instance seg).
 * All convs carry BN-folded weight/bias; every conv applies SiLU. */

#ifndef FIV_YOLO26_PROTO_H
#define FIV_YOLO26_PROTO_H

#include "fiv_data_typedefs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const ivf32* refine0_w;   /* [64, 128, 1, 1] */
    const ivf32* refine0_b;   /* [64] */
    const ivf32* refine1_w;   /* [64, 256, 1, 1] */
    const ivf32* refine1_b;   /* [64] */
    const ivf32* fuse_w;      /* [64, 64, 3, 3] */
    const ivf32* fuse_b;      /* [64] */
    const ivf32* cv1_w;       /* [64, 64, 3, 3] */
    const ivf32* cv1_b;       /* [64] */
    const ivf32* upsample_w;  /* [64, 64, 2, 2] (ConvTranspose2d, in,out) */
    const ivf32* upsample_b;  /* [64] */
    const ivf32* cv2_w;       /* [64, 64, 3, 3] */
    const ivf32* cv2_b;       /* [64] */
    const ivf32* cv3_w;       /* [32, 64, 1, 1] */
    const ivf32* cv3_b;       /* [32] */
} fiv_yolo26_proto_weights;

/* Run Proto26: P3 (c=64,h,w), P4 (c=128,h/2,w/2), P5 (c=256,h/4,w/4) flat
 * channel-major arrays. proto gets [32, 2h, 2w]; all sizes filled back.
 * Returns 0 on success. */
int fiv_yolo26_proto_run(const ivf32* p3, int p3_channels,
                         const ivf32* p4, int p4_channels,
                         const ivf32* p5, int p5_channels,
                         int height, int width,
                         const fiv_yolo26_proto_weights* weights,
                         ivf32* proto, int* out_channels, int* out_height,
                         int* out_width);

#ifdef __cplusplus
}
#endif

#endif /* FIV_YOLO26_PROTO_H */
