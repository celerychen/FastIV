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
 * FaceMesh 478-landmark geometry (see fiv_landmark_geom.h). Port of the
 * reference lm_geom.c; every value is kept in float32 (operand order
 * preserved) so the ROI and projection chain match MediaPipe to float32
 * precision. The 256x256 warp reuses the shared BlazeFace warp primitive
 * (fiv_face_warp) with border_mode = BORDER_REPLICATE, as the landmark stage
 * does in MediaPipe.
 */

#include "fiv_landmark_geom.h"

#include "fiv_landmark.h"   /* FIV_LM_TENSOR_SIZE / FIV_LM_NUM_LANDMARKS */
#include "fiv_face_warp.h"  /* fiv_get_perspective_transform / fiv_warp_perspective_u8 */

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ivf32 fiv_lm_normalize_radians(ivf32 angle) {
    ivf32 a = angle;
    /* double arithmetic with M_PI (as in C++), result narrowed to float */
    return (ivf32)((ivf64)a - 2.0 * M_PI * floor(((ivf64)a + M_PI) / (2.0 * M_PI)));
}

fiv_lm_rect fiv_lm_detection_to_norm_rect(ivf32 xmin, ivf32 ymin, ivf32 w, ivf32 h,
                                          ivf32 k0x, ivf32 k0y, ivf32 k1x, ivf32 k1y,
                                          int img_w, int img_h, ivf32 target_angle) {
    ivf32 xc = xmin + w / 2.0f;
    ivf32 yc = ymin + h / 2.0f;
    ivf32 x0 = k0x * (ivf32)img_w, y0 = k0y * (ivf32)img_h;
    ivf32 x1 = k1x * (ivf32)img_w, y1 = k1y * (ivf32)img_h;
    ivf32 rot = fiv_lm_normalize_radians(target_angle - atan2f(-(y1 - y0), x1 - x0));
    fiv_lm_rect r = { xc, yc, w, h, rot };
    return r;
}

fiv_lm_rect fiv_lm_rect_transform(const fiv_lm_rect* r, int img_w, int img_h,
                                  ivf32 scale_x, ivf32 scale_y, int square_long) {
    ivf32 width = r->w, height = r->h;
    if (square_long) {
        ivf32 long_side = (width * (ivf32)img_w > height * (ivf32)img_h)
                              ? width * (ivf32)img_w : height * (ivf32)img_h;
        width = long_side / (ivf32)img_w;
        height = long_side / (ivf32)img_h;
    }
    fiv_lm_rect o = { r->xc, r->yc, width * scale_x, height * scale_y, r->rot };
    return o;
}

fiv_lm_rect fiv_lm_expanded_face_rect(ivf32 bxc, ivf32 byc, ivf32 bw, ivf32 bh,
                                      const ivf32 kps[6][2], int img_w, int img_h) {
    ivf32 xmin = bxc - bw / 2.0f;
    ivf32 ymin = byc - bh / 2.0f;
    fiv_lm_rect r0 = fiv_lm_detection_to_norm_rect(xmin, ymin, bw, bh,
                                                   kps[0][0], kps[0][1],
                                                   kps[1][0], kps[1][1],
                                                   img_w, img_h, 0.0f);
    /* keep caller centre verbatim (matches MediaPipe float32) */
    r0.xc = bxc;
    r0.yc = byc;
    return fiv_lm_rect_transform(&r0, img_w, img_h, 1.5f, 1.5f, 1);
}

void fiv_lm_get_roi(const fiv_lm_rect* r, int img_w, int img_h, ivf32 roi[5]) {
    roi[0] = r->xc * (ivf32)img_w;
    roi[1] = r->yc * (ivf32)img_h;
    roi[2] = r->w * (ivf32)img_w;
    roi[3] = r->h * (ivf32)img_h;
    roi[4] = r->rot;
}

ivf32 fiv_lm_rect_deg(ivf32 rot_rad) {
    return (ivf32)((ivf64)(rot_rad * 180.0f) / M_PI);
}

void fiv_lm_box_points(ivf32 cx, ivf32 cy, ivf32 w, ivf32 h, ivf32 angle_deg,
                       ivf32 pts[8]) {
    ivf64 _angle = (ivf64)angle_deg * M_PI / 180.0;
    ivf32 b = (ivf32)(cos(_angle) * 0.5f);
    ivf32 a = (ivf32)(sin(_angle) * 0.5f);
    ivf32 p0x = (ivf32)(cx - (ivf32)(a * h)) - (ivf32)(b * w);
    ivf32 p0y = (ivf32)(cy + (ivf32)(b * h)) - (ivf32)(a * w);
    ivf32 p1x = (ivf32)(cx + (ivf32)(a * h)) - (ivf32)(b * w);
    ivf32 p1y = (ivf32)(cy - (ivf32)(b * h)) - (ivf32)(a * w);
    pts[0] = p0x; pts[1] = p0y;
    pts[2] = p1x; pts[3] = p1y;
    pts[4] = (ivf32)(2.0f * cx) - p0x;
    pts[5] = (ivf32)(2.0f * cy) - p0y;
    pts[6] = (ivf32)(2.0f * cx) - p1x;
    pts[7] = (ivf32)(2.0f * cy) - p1y;
}

void fiv_lm_roi_to_tensor(const iv8u* img, int h, int w, int cn,
                          const fiv_lm_rect* rect, int ts,
                          iv8u* warped, ivf32* tensor) {
    ivf32 roi[5];
    fiv_lm_get_roi(rect, w, h, roi);
    ivf32 pts[8];
    fiv_lm_box_points(roi[0], roi[1], roi[2], roi[3], fiv_lm_rect_deg(roi[4]), pts);

    ivf32 src_pts[8];
    for (int i = 0; i < 8; i++) src_pts[i] = pts[i];
    ivf32 dst_pts[8] = {
        0.0f, (ivf32)ts,
        0.0f, 0.0f,
        (ivf32)ts, 0.0f,
        (ivf32)ts, (ivf32)ts
    };
    ivf32 H[9];
    fiv_get_perspective_transform(src_pts, dst_pts, H);

    /* border_mode = BORDER_REPLICATE (landmark stage default) */
    fiv_warp_perspective_u8(img, h, w, cn, H, ts, ts, warped, 1);

    /* convertTo(CV_32F, scale=(255-0)/255/255=1/255, offset=0) -> [0,1] */
    size_t n = (size_t)ts * ts * (size_t)cn;
    for (size_t i = 0; i < n; i++)
        tensor[i] = (ivf32)warped[i] * (1.0f / 255.0f);
}

void fiv_lm_sub_rect_to_rect_matrix(const ivf32 roi[5], int rect_w, int rect_h,
                                    int flip_horizontally, ivf32 m[16]) {
    ivf32 a = roi[2], b = roi[3];
    ivf32 fl = flip_horizontally ? -1.0f : 1.0f;
    ivf32 c = cosf(roi[4]), d = sinf(roi[4]);
    ivf32 e = roi[0], f = roi[1];
    ivf32 g = 1.0f / (ivf32)rect_w;
    ivf32 h = 1.0f / (ivf32)rect_h;
    for (int i = 0; i < 16; i++) m[i] = 0.0f;
    m[0] = (ivf32)(a * c * fl) * g;
    m[1] = (ivf32)(-(ivf32)(b * d)) * g;
    m[3] = ((ivf32)((ivf32)((ivf32)(-0.5f * a) * c) * fl) +
            (ivf32)((ivf32)(0.5f * b) * d) + e) * g;
    m[4] = (ivf32)(a * d * fl) * h;
    m[5] = (ivf32)(b * c) * h;
    m[7] = ((ivf32)((ivf32)(-0.5f * b) * c) -
            (ivf32)((ivf32)((ivf32)(0.5f * a) * d) * fl) + f) * h;
    m[10] = a * g;
    m[15] = 1.0f;
}

void fiv_lm_project_xy(ivf32 x, ivf32 y, ivf32 z, const ivf32 m[16],
                       ivf32* nx, ivf32* ny) {
    *nx = ((ivf32)((ivf32)((ivf32)(x * m[0]) + (ivf32)(y * m[1])) +
                   (ivf32)(z * m[2])) + m[3]);
    *ny = ((ivf32)((ivf32)((ivf32)(x * m[4]) + (ivf32)(y * m[5])) +
                   (ivf32)(z * m[6])) + m[7]);
}

ivf32 fiv_lm_calc_z_scale(const ivf32 m[16]) {
    ivf32 ax, ay, bx, by;
    fiv_lm_project_xy(0.0f, 0.0f, 0.0f, m, &ax, &ay);
    fiv_lm_project_xy(1.0f, 0.0f, 0.0f, m, &bx, &by);
    ivf32 dx = bx - ax, dy = by - ay;
    return (ivf32)sqrt((ivf64)dx * dx + (ivf64)dy * dy);
}

void fiv_lm_letterbox_removal(ivf32 lms[][3], ivf32 h_pad, ivf32 v_pad) {
    ivf32 left = h_pad, top = v_pad;
    ivf32 lr = h_pad + h_pad;
    ivf32 tb = v_pad + v_pad;
    ivf32 ix = 1.0f - lr, iy = 1.0f - tb;
    for (int i = 0; i < FIV_LM_NUM_LANDMARKS; i++) {
        lms[i][0] = (lms[i][0] - left) / ix;
        lms[i][1] = (lms[i][1] - top) / iy;
        lms[i][2] = lms[i][2] / ix;
    }
}
