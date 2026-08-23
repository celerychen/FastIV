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
 * BlazeFace ROI warp. Self-contained replacement of reference geom.c;
 * fixed-point bilinear warp matching OpenCV INTER_LINEAR / BORDER_CONSTANT.
 * No dependency on src/reference/.
 */

#include "fiv_face_warp.h"

#include "fiv_common.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define FIV_INTER_BITS  5
#define FIV_INTER_TAB   32          /* 1 << FIV_INTER_BITS */
#define FIV_COEF_BITS   15
#define FIV_COEF_SCALE  32768       /* 1 << FIV_COEF_BITS */
#define FIV_DELTA       16384       /* 1 << (FIV_COEF_BITS-1) */

/* 3x3 matrix inverse. */
static int fiv_invert3x3(const ivf32 src_matrix[9], ivf32 out_matrix[9]) {
    ivf32 m00 = (ivf32)src_matrix[0], m01 = (ivf32)src_matrix[1], m02 = (ivf32)src_matrix[2];
    ivf32 m10 = (ivf32)src_matrix[3], m11 = (ivf32)src_matrix[4], m12 = (ivf32)src_matrix[5];
    ivf32 m20 = (ivf32)src_matrix[6], m21 = (ivf32)src_matrix[7], m22 = (ivf32)src_matrix[8];
    ivf32 det = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) +
                m02 * (m10 * m21 - m11 * m20);
    if (det == 0.0f) return 0;
    ivf32 inv_det = 1.0f / det;
    ivf32 adj[9];
    adj[0] = (m11 * m22 - m12 * m21) * inv_det;
    adj[1] = (m02 * m21 - m01 * m22) * inv_det;
    adj[2] = (m01 * m12 - m02 * m11) * inv_det;
    adj[3] = (m12 * m20 - m10 * m22) * inv_det;
    adj[4] = (m00 * m22 - m02 * m20) * inv_det;
    adj[5] = (m02 * m10 - m00 * m12) * inv_det;
    adj[6] = (m10 * m21 - m11 * m20) * inv_det;
    adj[7] = (m01 * m20 - m00 * m21) * inv_det;
    adj[8] = (m00 * m11 - m01 * m10) * inv_det;
    for (int i = 0; i < 9; i++) out_matrix[i] = adj[i];
    return 1;
}

/* solve 8x8 linear system by Gaussian elimination (partial pivoting). */
static void fiv_gesv8(ivf32 aug_matrix[8][9], ivf32 solution[8]) {
    for (int col = 0; col < 8; col++) {
        int pivot_row = col;
        ivf32 best_val = (ivf32)fabsf(aug_matrix[col][col]);
        for (int row = col + 1; row < 8; row++) {
            ivf32 cand_val = (ivf32)fabsf(aug_matrix[row][col]);
            if (cand_val > best_val) { best_val = cand_val; pivot_row = row; }
        }
        if (pivot_row != col) {
            ivf32 tmp_row[9];
            for (int k = 0; k < 9; k++) tmp_row[k] = aug_matrix[col][k];
            for (int k = 0; k < 9; k++) aug_matrix[col][k] = aug_matrix[pivot_row][k];
            for (int k = 0; k < 9; k++) aug_matrix[pivot_row][k] = tmp_row[k];
        }
        ivf32 diag = aug_matrix[col][col];
        if (diag == 0.0f) diag = 1.0f;
        for (int k = col; k < 9; k++) aug_matrix[col][k] /= diag;
        for (int row = 0; row < 8; row++) {
            if (row == col) continue;
            ivf32 factor = aug_matrix[row][col];
            if (factor == 0.0f) continue;
            for (int k = col; k < 9; k++) aug_matrix[row][k] -= factor * aug_matrix[col][k];
        }
    }
    for (int i = 0; i < 8; i++) solution[i] = aug_matrix[i][8];
}

/* 4-point perspective transform -> 3x3 homography (row-major, h[8]=1). */
static void fiv_get_perspective_transform(const ivf32 src_pts[8],
                                          const ivf32 dst_pts[8],
                                          ivf32 homography[9]) {
    ivf32 aug_matrix[8][9];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 9; j++) aug_matrix[i][j] = 0.0f;
    for (int i = 0; i < 4; i++) {
        ivf32 src_x = src_pts[2 * i], src_y = src_pts[2 * i + 1];
        ivf32 dst_x = dst_pts[2 * i], dst_y = dst_pts[2 * i + 1];
        int row0 = 2 * i, row1 = 2 * i + 1;
        aug_matrix[row0][0] = src_x; aug_matrix[row0][1] = src_y; aug_matrix[row0][2] = 1.0f;
        aug_matrix[row0][6] = -src_x * dst_x; aug_matrix[row0][7] = -src_y * dst_x; aug_matrix[row0][8] = dst_x;
        aug_matrix[row1][3] = src_x; aug_matrix[row1][4] = src_y; aug_matrix[row1][5] = 1.0f;
        aug_matrix[row1][6] = -src_x * dst_y; aug_matrix[row1][7] = -src_y * dst_y; aug_matrix[row1][8] = dst_y;
    }
    ivf32 h[8];
    fiv_gesv8(aug_matrix, h);
    homography[0] = h[0]; homography[1] = h[1]; homography[2] = h[2];
    homography[3] = h[3]; homography[4] = h[4]; homography[5] = h[5];
    homography[6] = h[6]; homography[7] = h[7]; homography[8] = 1.0f;
}

void fiv_detection_warp_matrix(int img_w, int img_h, int tensor_size, ivf32 homography[9]) {
    ivf32 center_x = (ivf32)img_w * 0.5f, center_y = (ivf32)img_h * 0.5f;
    ivf32 roi_w = (ivf32)img_w, roi_h = (ivf32)img_h;
    ivf32 tensor_aspect = 1.0f, roi_aspect = roi_h / roi_w;
    if (tensor_aspect > roi_aspect)
        roi_h = roi_w * tensor_aspect;
    else
        roi_w = roi_h / tensor_aspect;

    ivf32 src_pts[8] = {
        center_x - 0.5f * roi_w, center_y + 0.5f * roi_h,
        center_x - 0.5f * roi_w, center_y - 0.5f * roi_h,
        center_x + 0.5f * roi_w, center_y - 0.5f * roi_h,
        center_x + 0.5f * roi_w, center_y + 0.5f * roi_h
    };
    ivf32 dst_pts[8] = {
        0.0f, (ivf32)tensor_size,
        0.0f, 0.0f,
        (ivf32)tensor_size, 0.0f,
        (ivf32)tensor_size, (ivf32)tensor_size
    };
    fiv_get_perspective_transform(src_pts, dst_pts, homography);
}

/* round to nearest, ties to even (banker's rounding). */
static ivf32 fiv_round_half_even(ivf32 value) {
    ivf32 r = (ivf32)floorf(value);
    ivf32 frac = value - r;
    if (frac < 0.5f) return r;
    if (frac > 0.5f) return r + 1.0f;
    if (((int)r) % 2 == 0) return r;
    return r + 1.0f;
}

void fiv_warp_perspective_u8(const iv8u* img, int h, int w, int cn,
                             const ivf32 fwd_matrix[9], int out_w, int out_h,
                             iv8u* out, int border_replicate) {
    ivf32 inv_matrix[9];
    if (!fiv_invert3x3(fwd_matrix, inv_matrix)) return;

    for (int y = 0; y < out_h; y++) {
        ivf32 yc = (ivf32)y;
        for (int x = 0; x < out_w; x++) {
            ivf32 xc = (ivf32)x;
            ivf32 x0d = inv_matrix[0] * xc + inv_matrix[1] * yc + inv_matrix[2];
            ivf32 y0d = inv_matrix[3] * xc + inv_matrix[4] * yc + inv_matrix[5];
            ivf32 w0  = inv_matrix[6] * xc + inv_matrix[7] * yc + inv_matrix[8];
            if (w0 == 0.0f) {
                for (int ch = 0; ch < cn; ch++)
                    out[((size_t)y * out_w + x) * cn + ch] = 0;
                continue;
            }
            ivf32 fx = (x0d / w0) * (ivf32)FIV_INTER_TAB;
            ivf32 fy = (y0d / w0) * (ivf32)FIV_INTER_TAB;
            int32_t vx = (int32_t)fiv_round_half_even(fx);
            int32_t vy = (int32_t)fiv_round_half_even(fy);
            int src_x = (int)(vx >> FIV_INTER_BITS);
            int src_y = (int)(vy >> FIV_INTER_BITS);
            int32_t kx = vx & (FIV_INTER_TAB - 1);
            int32_t ky = vy & (FIV_INTER_TAB - 1);

            int32_t w00 = (FIV_INTER_TAB - kx) * (FIV_INTER_TAB - ky) * FIV_INTER_TAB;
            int32_t w01 = kx * (FIV_INTER_TAB - ky) * FIV_INTER_TAB;
            int32_t w10 = (FIV_INTER_TAB - kx) * ky * FIV_INTER_TAB;
            int32_t w11 = kx * ky * FIV_INTER_TAB;

            int sx0 = src_x, sx1 = src_x + 1, sy0 = src_y, sy1 = src_y + 1;
            int ok_x0 = (sx0 >= 0 && sx0 < w), ok_x1 = (sx1 >= 0 && sx1 < w);
            int ok_y0 = (sy0 >= 0 && sy0 < h), ok_y1 = (sy1 >= 0 && sy1 < h);
            int cx0 = sx0 < 0 ? 0 : (sx0 >= w ? w - 1 : sx0);
            int cx1 = sx1 < 0 ? 0 : (sx1 >= w ? w - 1 : sx1);
            int cy0 = sy0 < 0 ? 0 : (sy0 >= h ? h - 1 : sy0);
            int cy1 = sy1 < 0 ? 0 : (sy1 >= h ? h - 1 : sy1);

            for (int ch = 0; ch < cn; ch++) {
                int32_t px00, px01, px10, px11;
                if (border_replicate) {
                    int bx0 = src_x < 0 ? 0 : (src_x >= w ? w - 1 : src_x);
                    int bx1 = (src_x + 1) < 0 ? 0 : ((src_x + 1) >= w ? w - 1 : src_x + 1);
                    int by0 = src_y < 0 ? 0 : (src_y >= h ? h - 1 : src_y);
                    int by1 = (src_y + 1) < 0 ? 0 : ((src_y + 1) >= h ? h - 1 : src_y + 1);
                    px00 = (int32_t)img[((size_t)by0 * w + bx0) * cn + ch];
                    px01 = (int32_t)img[((size_t)by0 * w + bx1) * cn + ch];
                    px10 = (int32_t)img[((size_t)by1 * w + bx0) * cn + ch];
                    px11 = (int32_t)img[((size_t)by1 * w + bx1) * cn + ch];
                } else {
                    px00 = ok_x0 && ok_y0 ? (int32_t)img[((size_t)cy0 * w + cx0) * cn + ch] : 0;
                    px01 = ok_x1 && ok_y0 ? (int32_t)img[((size_t)cy0 * w + cx1) * cn + ch] : 0;
                    px10 = ok_x0 && ok_y1 ? (int32_t)img[((size_t)cy1 * w + cx0) * cn + ch] : 0;
                    px11 = ok_x1 && ok_y1 ? (int32_t)img[((size_t)cy1 * w + cx1) * cn + ch] : 0;
                }
                int32_t acc = px00 * w00 + px01 * w01 + px10 * w10 + px11 * w11;
                int32_t val = (acc + FIV_DELTA) >> FIV_COEF_BITS;
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                out[((size_t)y * out_w + x) * cn + ch] = (iv8u)val;
            }
        }
    }
}
