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

/* YOLO26 NMS-free decode (reg_max=1, end2end). Turns the six Detect one2one
 * raw heads (box0..2 then cls0..2, each a contiguous NCHW tensor) into top-k
 * detections mirroring the torch reference (ultralytics head.py + tal.py):
 *   anchors   make_anchors: per level y-major flatten of (x+0.5, y+0.5),
 *             levels concatenated in stride order 8/16/32.
 *   decode    dist2bbox, xyxy: box channels ordered l,t,r,b;
 *             x1 = ax - l, y1 = ay - t, x2 = ax + r, y2 = ay + b
 *             in grid units, then scaled by the level stride (pixels).
 *   scores    sigmoid of the raw class head.
 *   top-k     two exact passes: per-anchor class max keeps k = min(max_k, A)
 *             anchors; the k*nc candidate scores are reduced to the final k.
 * Output rows are [x1, y1, x2, y2, score, class], score-descending.
 */

#ifndef FIV_YOLO26_POST_H
#define FIV_YOLO26_POST_H

#include "fiv_ctensor.h"
#include "fiv_data_typedefs.h"

/* heads[0..2] box heads (4 channels), heads[3..5] class heads (nc channels);
 * level 0..2 in stride order (8/16/32 for yolo26n). strides[3] are the level
 * strides. out must hold max_k * 6 floats; returns the kept count or -1. */
int fiv_yolo26_postprocess(const fiv_tensor4d* heads[6], const int strides[3],
                           ivf32* out, int max_k);

/* Segment26 (yolo26-seg) variant: heads[0..5] are the same box/cls heads and
 * heads[6..8] the one2one_cv4 mask-coefficient heads (nm channels each level,
 * nm = heads[6]->channels, 32 in yolo26n-seg). The two-stage top-k is driven
 * by the class scores only; the winning anchor carries its full nm-coefficient
 * row. Output rows are [x1, y1, x2, y2, score, class, coef(0..nm-1)], so the
 * row width is 6 + nm and `out` must hold max_k * (6 + nm) floats. Returns the
 * kept count or -1. */
int fiv_yolo26_postprocess_seg(const fiv_tensor4d* heads[9], const int strides[3],
                               ivf32* out, int max_k);

/* Pose26 (yolo26-pose): heads[0..5] box/cls as detection, heads[6..8] the
 * one2one_cv4_kpts heads (3*nk channels = 51 for COCO-17, channel order
 * [x0,y0,v0, x1,y1,v1, ...]). The kpt rows are decoded per winning anchor:
 *   x_k = (raw_x + anchor_x) * stride, y_k = (raw_y + anchor_y) * stride,
 *   v_k = sigmoid(raw_v).
 * Output rows are [x1,y1,x2,y2,score,class, kpt(51)] (width 6+3*nk). */
int fiv_yolo26_postprocess_pose(const fiv_tensor4d* heads[9], const int strides[3],
                                ivf32* out, int max_k);

#endif /* FIV_YOLO26_POST_H */
