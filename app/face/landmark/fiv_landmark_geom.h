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
 * FaceMesh 478-landmark geometry (port of the reference lm_geom.c). All math
 * mirrors MediaPipe's proto/C++ float32 arithmetic (operand order preserved)
 * so the ROI / warp-matrix chain matches the reference to float32 precision:
 *   - fiv_lm_expanded_face_rect   detection -> normalized rect (with rotation)
 *   - fiv_lm_roi_to_tensor        warp the rotated ROI to the 256x256 tensor
 *   - fiv_lm_sub_rect_to_rect_matrix / fiv_lm_project_xy / fiv_lm_calc_z_scale
 *                                 project normalized landmarks back to the image
 *   - fiv_lm_letterbox_removal    undo the (identity here) letterbox padding
 */

#ifndef _FIV_LANDMARK_GEOM_H_
#define _FIV_LANDMARK_GEOM_H_

#include "fiv_data_typedefs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* NormalizedRect: center/size in [0,1], rotation in radians. */
typedef struct {
    ivf32 xc, yc, w, h, rot;
} fiv_lm_rect;

/* NormalizeRadians (detections_to_rects_calculator.h). */
ivf32 fiv_lm_normalize_radians(ivf32 angle);

/* DetectionsToRectsCalculator::DetectionToNormalizedRect + ComputeRotation.
   bbox = (xmin, ymin, width, height) normalized; kp0/kp1 normalized [0,1]. */
fiv_lm_rect fiv_lm_detection_to_norm_rect(ivf32 xmin, ivf32 ymin, ivf32 w, ivf32 h,
                                          ivf32 k0x, ivf32 k0y, ivf32 k1x, ivf32 k1y,
                                          int img_w, int img_h, ivf32 target_angle);

/* RectTransformationCalculator (scale_x/scale_y, square_long). */
fiv_lm_rect fiv_lm_rect_transform(const fiv_lm_rect* r, int img_w, int img_h,
                                  ivf32 scale_x, ivf32 scale_y, int square_long);

/* Expanded ROI from a detection (Tasks FaceDetectorGraph):
   bbox = (x_center, y_center, width, height) normalized; kps[6][2] normalized. */
fiv_lm_rect fiv_lm_expanded_face_rect(ivf32 bxc, ivf32 byc, ivf32 bw, ivf32 bh,
                                      const ivf32 kps[6][2], int img_w, int img_h);

/* GetRoi: normalized rect -> pixel-space [cx, cy, w, h, rotation]. */
void fiv_lm_get_roi(const fiv_lm_rect* r, int img_w, int img_h, ivf32 roi[5]);

/* roi.rotation * 180 / M_PI (float multiply, double divide). */
ivf32 fiv_lm_rect_deg(ivf32 rot_rad);

/* OpenCV RotatedRect::points() (float32, exact 3.4/4.x formula).
   pts = 4 (x,y) pairs; angle_deg is the rotated-rect angle in degrees. */
void fiv_lm_box_points(ivf32 cx, ivf32 cy, ivf32 w, ivf32 h, ivf32 angle_deg,
                       ivf32 pts[8]);

/* Warp the rotated ROI of an h x w x cn uint8 image into the ts x ts tensor
   buffer (ts = network input size: 256 or 192). `warped` must hold >= ts*ts*cn
   bytes of scratch; `tensor` receives ts*ts*cn float32 pixels normalized to
   [0,1] (OpenCV convertTo 1/255). */
void fiv_lm_roi_to_tensor(const iv8u* img, int h, int w, int cn,
                          const fiv_lm_rect* rect, int ts,
                          iv8u* warped, ivf32* tensor);

/* GetRotatedSubRectToRectTransformMatrix (row-major 4x4 float32). */
void fiv_lm_sub_rect_to_rect_matrix(const ivf32 roi[5], int rect_w, int rect_h,
                                    int flip_horizontally, ivf32 m[16]);

/* LandmarkProjectionNodeImpl::ProjectXY (float32). */
void fiv_lm_project_xy(ivf32 x, ivf32 y, ivf32 z, const ivf32 m[16],
                       ivf32* nx, ivf32* ny);

/* CalculateZScale. */
ivf32 fiv_lm_calc_z_scale(const ivf32 m[16]);

/* LandmarkLetterboxRemovalCalculator (float32); lms is the 478x3 landmark array. */
void fiv_lm_letterbox_removal(ivf32 lms[][3], ivf32 h_pad, ivf32 v_pad);

#ifdef __cplusplus
}
#endif

#endif /* _FIV_LANDMARK_GEOM_H_ */
