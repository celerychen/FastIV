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
 * Black-box test of the fiv_face detector API. This test only talks to the
 * public entry points (fiv_create_face_detetor / fiv_face_detector_on_image /
 * fiv_release_face_detector) plus the image loader (fiv_create_image_from_file);
 * it reaches into no internal / reference implementation.
 */

#include "fiv_face.h"
#include "fiv_image.h"   /* fiv_create_image_from_file / fiv_release_image */
#include "fiv_common.h"  /* fiv_get_current_system_time */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

static void check(int cond, const char* msg) {
    if (cond) {
        g_pass++;
    } else {
        g_fail++;
        printf("  [FAIL] %s\n", msg);
    }
}

static int nearly_equal(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

/* draw helpers copied from test_landmark_e2e.c */
static void draw_point(fiv_mat* img, ivf32 nx, ivf32 ny, int r, iv8u cr, iv8u cg, iv8u cb) {
    int iw = (int)img->width, ih = (int)img->height;
    int cx = (int)(nx * (ivf32)iw);
    int cy = (int)(ny * (ivf32)ih);
    iv8u* p = img->data.ptr8u;
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int x = cx + dx, y = cy + dy;
            if (x < 0 || x >= iw || y < 0 || y >= ih) continue;
            size_t o = ((size_t)y * (size_t)iw + (size_t)x) * 3;
            p[o + 0] = cr; p[o + 1] = cg; p[o + 2] = cb;
        }
    }
}

static void draw_seg(fiv_mat* img, int x0, int y0, int x1, int y1,
                     int thick, iv8u cr, iv8u cg, iv8u cb) {
    int iw = (int)img->width, ih = (int)img->height;
    iv8u* p = img->data.ptr8u;
    if (x0 < 0) x0 = 0; if (x0 >= iw) x0 = iw - 1;
    if (x1 < 0) x1 = 0; if (x1 >= iw) x1 = iw - 1;
    if (y0 < 0) y0 = 0; if (y0 >= ih) y0 = ih - 1;
    if (y1 < 0) y1 = 0; if (y1 >= ih) y1 = ih - 1;
    int dx = abs(x1 - x0), dy = -abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (int ox = -thick / 2; ox <= thick / 2; ox++) {
        for (int oy = -thick / 2; oy <= thick / 2; oy++) {
            int x = x0, y = y0, e2 = err;
            for (;;) {
                int xx = x + ox, yy = y + oy;
                if (xx >= 0 && xx < iw && yy >= 0 && yy < ih) {
                    size_t o = ((size_t)yy * (size_t)iw + (size_t)xx) * 3;
                    p[o + 0] = cr; p[o + 1] = cg; p[o + 2] = cb;
                }
                if (x == x1 && y == y1) break;
                e2 = 2 * err;
                if (e2 >= dy) { err += dy; x += sx; }
                if (e2 <= dx) { err += dx; y += sy; }
            }
        }
    }
}

/* validate one detection against the API contract (no reference needed) */
static int detection_valid(const fiv_face_detection* d, int img_w, int img_h) {
    if (!(d->score >= 0.0f && d->score <= 1.0f)) return 0;
    if (!(d->w > 0.0f && d->h > 0.0f)) return 0;
    if (d->x < -1.0f || d->y < -1.0f) return 0;
    if (d->x + d->w > (float)img_w + 1.0f) return 0;
    if (d->y + d->h > (float)img_h + 1.0f) return 0;
    for (int k = 0; k < FIV_FACE_KEYPOINTS; k++) {
        if (!(d->kps[k][0] >= 0.0f && d->kps[k][0] <= 1.0f)) return 0;
        if (!(d->kps[k][1] >= 0.0f && d->kps[k][1] <= 1.0f)) return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    const char* img_path = argc > 1 ? argv[1]
        : "../src/reference/c_face_detect_release/15.png";
    const char* model_path = argc > 2 ? argv[2]
        : "../app/face/models/blazeface_weights.bin";

    /* load the source image through the public image API */
    double t_image_load = fiv_get_current_system_time();
    fiv_mat* image = fiv_create_image_from_file((char*)img_path, FIV_RGB24_CS);
    t_image_load = fiv_get_current_system_time() - t_image_load;
    check(image != NULL, "fiv_create_image_from_file returns non-NULL");
    if (!image) {
        printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
        return 1;
    }
    int img_w = (int)image->width;
    int img_h = (int)image->height;

    /* ---- interface 1: create ---- */
    double t_create = fiv_get_current_system_time();
    void* detector = fiv_create_face_detetor((char*)model_path);
    t_create = fiv_get_current_system_time() - t_create;
    check(detector != NULL, "fiv_create_face_detetor returns non-NULL");
    if (!detector) {
        fiv_release_image(image);
        printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
        return 1;
    }

    /* ---- bad-argument contract ---- */
    fiv_face_result r1;
    memset(&r1, 0, sizeof(r1));
    check(fiv_face_detector_on_image(NULL, NULL, detector) != FIV_RET_OK,
          "on_image(NULL,...) rejects bad args");
    check(fiv_face_detector_on_image(&r1, NULL, detector) != FIV_RET_OK,
          "on_image(img=NULL) rejects bad args");

    /* ---- interface 2: run on image ---- */
    fiv_face_result result;
    memset(&result, 0, sizeof(result));
    double t_detect = fiv_get_current_system_time();
    fiv_ret r = fiv_face_detector_on_image(&result, image, detector);
    t_detect = fiv_get_current_system_time() - t_detect;
    check(r == FIV_RET_OK, "fiv_face_detector_on_image returns FIV_RET_OK");
    check(result.count >= 0 && result.count <= FIV_FACE_MAX_DETS,
          "count within [0, FIV_FACE_MAX_DETS]");

    printf("fiv_face API: %d faces on %dx%d image\n", result.count, img_w, img_h);
    for (int i = 0; i < result.count; i++) {
        char msg[96];
        snprintf(msg, sizeof(msg), "det[%d] structure valid", i);
        check(detection_valid(&result.detections[i], img_w, img_h), msg);
    }
    if (result.count > 0) {
        const fiv_face_detection* d = &result.detections[0];
        printf("  det[0]: score=%.6f x=%.2f y=%.2f w=%.2f h=%.2f\n",
               d->score, d->x, d->y, d->w, d->h);
    }

    /* ---- determinism: same input -> identical output ---- */
    fiv_face_result result2;
    memset(&result2, 0, sizeof(result2));
    fiv_ret r2 = fiv_face_detector_on_image(&result2, image, detector);
    check(r2 == FIV_RET_OK, "second run returns FIV_RET_OK");
    check(result2.count == result.count, "deterministic count");
    int det_match = (result2.count == result.count);
    for (int i = 0; i < result.count && det_match; i++) {
        if (!nearly_equal(result2.detections[i].score, result.detections[i].score, 1e-6f) ||
            !nearly_equal(result2.detections[i].x, result.detections[i].x, 1e-6f) ||
            !nearly_equal(result2.detections[i].y, result.detections[i].y, 1e-6f) ||
            !nearly_equal(result2.detections[i].w, result.detections[i].w, 1e-6f) ||
            !nearly_equal(result2.detections[i].h, result.detections[i].h, 1e-6f))
            det_match = 0;
        for (int k = 0; k < FIV_FACE_KEYPOINTS && det_match; k++)
            if (!nearly_equal(result2.detections[i].kps[k][0], result.detections[i].kps[k][0], 1e-6f) ||
                !nearly_equal(result2.detections[i].kps[k][1], result.detections[i].kps[k][1], 1e-6f))
                det_match = 0;
    }
    check(det_match, "deterministic per-detection output");

    /* ---- steady-state detect timing: N iterations, time ONLY on_image ---- */
    const int bench_n = 200;
    double tiref = 0.0;                 /* sum of on_image durations only */
    double tlw = 0.0;                   /* whole-loop wall time (incl. loop) */
    double ta = fiv_get_current_system_time();
    fiv_face_result br;
    for (int bi = 0; bi < bench_n; bi++) {
        memset(&br, 0, sizeof(br));
        double ts = fiv_get_current_system_time();
        fiv_face_detector_on_image(&br, image, detector);
        tiref += fiv_get_current_system_time() - ts;
    }
    tlw = fiv_get_current_system_time() - ta;
    printf("detect_bench: %d iters  wall=%.3f ms  avg_on_image=%.3f ms/frame  %.1f FPS\n",
           bench_n, tlw, tiref / bench_n, 1000.0 * bench_n / tiref);

    /* ---- interface 3: release ---- */
    fiv_ret rr = fiv_release_face_detector(&detector);
    check(rr == FIV_RET_OK, "fiv_release_face_detector returns FIV_RET_OK");
    check(detector == NULL, "release sets *detector to NULL");

    /* double release must be a no-op / safe (NULL guard) */
    check(fiv_release_face_detector(&detector) != FIV_RET_OK ||
          detector == NULL, "release(NULL) guarded");

    /* ---- draw detections on the image and save (after bench, image is
       the clean source used by detection above) ---- */
    for (int i = 0; i < result.count; i++) {
        const fiv_face_detection* d = &result.detections[i];
        int x0 = (int)d->x, y0 = (int)d->y;
        int x1 = (int)(d->x + d->w), y1 = (int)(d->y + d->h);
        draw_seg(image, x0, y0, x1, y0, 2, 0, 255, 0);
        draw_seg(image, x1, y0, x1, y1, 2, 0, 255, 0);
        draw_seg(image, x1, y1, x0, y1, 2, 0, 255, 0);
        draw_seg(image, x0, y1, x0, y0, 2, 0, 255, 0);
        for (int k = 0; k < FIV_FACE_KEYPOINTS; k++)
            draw_point(image, d->kps[k][0], d->kps[k][1], 3, 255, 0, 0);
    }
    {
        char out_name[] = "blazeface_15_detect.png";
        fiv_ret wr = fiv_image_write(out_name, image);
        if (wr == FIV_RET_OK)
            printf("  saved annotated image -> %s\n", out_name);
    }

    /* release the loaded image through the public image API */
    fiv_release_image(image);

    printf("timing: image_load=%.3f ms  detector_create=%.3f ms  detect(on_image)=%.3f ms\n",
           t_image_load, t_create, t_detect);
    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
