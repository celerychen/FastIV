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
 * Black-box test of the YuNet detector API. This test only talks to the public
 * entry points (fiv_create_yunet_detector / fiv_yunet_detector_on_image /
 * fiv_release_yunet_detector) plus the image loader (fiv_create_image_from_file);
 * it reaches into no internal / reference implementation.
 */

#include "fiv_yunet_model.h"
#include "fiv_image.h"          /* fiv_create_image_from_file / fiv_release_image */
#include "fiv_common.h"         /* fiv_get_current_system_time */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIV_WORK_SIZE 320   /* longest side of the reduced working image */

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
static void draw_point(fiv_mat* img, int cx, int cy, int r, iv8u cr, iv8u cg, iv8u cb) {
    int iw = (int)img->width, ih = (int)img->height;
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
static int detection_valid(const fiv_yunet_result* r, int img_w, int img_h) {
    if (!(r->score >= 0.0f && r->score <= 1.0f)) return 0;
    if (!(r->w > 0 && r->h > 0)) return 0;
    if (r->x < -1 || r->y < -1) return 0;
    if (r->x + r->w > img_w + 1) return 0;
    if (r->y + r->h > img_h + 1) return 0;
    return 1;
}

int main(int argc, char** argv) {
    const char* img_path = argc > 1 ? argv[1]
        : "../src/test/15.png";
    const char* model_path = argc > 2 ? argv[2]
        : "../app/face/models/yunet_weights.bin";

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

    /* YuNet preprocess has no built-in downscale, so a full-HD frame is slow.
       Detect on a ratio-kept reduced image (longest side 320, pattern from
       test_yunet_resize.c) and map pixel results back to source coordinates. */
    const unsigned sw = image->width, sh = image->height;
    unsigned dw = FIV_WORK_SIZE, dh = FIV_WORK_SIZE;
    if (sw > sh) dh = (unsigned)((ivf64)FIV_WORK_SIZE * sh / sw + 0.5);
    else         dw = (unsigned)((ivf64)FIV_WORK_SIZE * sw / sh + 0.5);
    size_t s_work[2] = { dh, dw };
    fiv_mat* work = fiv_create_tensor2d(s_work, FIV_8U3);
    check(work != NULL, "create reduced working image");
    if (!work) {
        fiv_release_image(image);
        printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
        return 1;
    }
    double t_resize = fiv_get_current_system_time();
    check(fiv_image_resize(work, image, FIV_BILEAR_RESIZER) == FIV_RET_OK,
          "fiv_image_resize builds the working image");
    t_resize = fiv_get_current_system_time() - t_resize;
    const ivf32 fx = (ivf32)sw / (ivf32)dw, fy = (ivf32)sh / (ivf32)dh;

    /* ---- interface 1: create ---- */
    double t_create = fiv_get_current_system_time();
    void* detector = fiv_create_yunet_detector((char*)model_path);
    t_create = fiv_get_current_system_time() - t_create;
    check(detector != NULL, "fiv_create_yunet_detector returns non-NULL");
    if (!detector) {
        fiv_release_image(image);
        printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
        return 1;
    }

    /* ---- bad-argument contract ---- */
    fiv_yunet_model_result r1;
    memset(&r1, 0, sizeof(r1));
    check(fiv_yunet_detector_on_image(NULL, NULL, detector) != FIV_RET_OK,
          "on_image(NULL,...) rejects bad args");
    check(fiv_yunet_detector_on_image(&r1, NULL, detector) != FIV_RET_OK,
          "on_image(img=NULL) rejects bad args");

    /* ---- interface 2: run on image ---- */
    fiv_yunet_model_result result;
    memset(&result, 0, sizeof(result));
    double t_detect = fiv_get_current_system_time();
    fiv_ret r = fiv_yunet_detector_on_image(&result, work, detector);
    for (int mi = 0; mi < result.count; mi++) {   /* map back to source pixels */
        fiv_yunet_result* d = &result.detections[mi];
        d->x = (int)(d->x * fx); d->y = (int)(d->y * fy);
        d->w = (int)(d->w * fx); d->h = (int)(d->h * fy);
        for (int k = 0; k < 5; k++) {
            d->lm[2 * k] = (int)(d->lm[2 * k] * fx);
            d->lm[2 * k + 1] = (int)(d->lm[2 * k + 1] * fy);
        }
    }
    t_detect = fiv_get_current_system_time() - t_detect;
    check(r == FIV_RET_OK, "fiv_yunet_detector_on_image returns FIV_RET_OK");
    check(result.count >= 0 && result.count <= FIV_YUNET_MODEL_MAX_FACES,
          "count within [0, FIV_YUNET_MODEL_MAX_FACES]");

    printf("fiv_yunet API: %d faces on %dx%d image\n", result.count, img_w, img_h);
    for (int i = 0; i < result.count; i++) {
        char msg[96];
        snprintf(msg, sizeof(msg), "det[%d] structure valid", i);
        check(detection_valid(&result.detections[i], img_w, img_h), msg);
    }
    if (result.count > 0) {
        const fiv_yunet_result* d = &result.detections[0];
        printf("  det[0]: score=%.6f x=%d y=%d w=%d h=%d\n",
               d->score, d->x, d->y, d->w, d->h);
    }

    /* ---- determinism: same input -> identical output ---- */
    fiv_yunet_model_result result2;
    memset(&result2, 0, sizeof(result2));
    fiv_ret r2 = fiv_yunet_detector_on_image(&result2, work, detector);
    for (int mi = 0; mi < result2.count; mi++) {
        fiv_yunet_result* d = &result2.detections[mi];
        d->x = (int)(d->x * fx); d->y = (int)(d->y * fy);
        d->w = (int)(d->w * fx); d->h = (int)(d->h * fy);
        for (int k = 0; k < 5; k++) {
            d->lm[2 * k] = (int)(d->lm[2 * k] * fx);
            d->lm[2 * k + 1] = (int)(d->lm[2 * k + 1] * fy);
        }
    }
    check(r2 == FIV_RET_OK, "second run returns FIV_RET_OK");
    check(result2.count == result.count, "deterministic count");
    int det_match = (result2.count == result.count);
    for (int i = 0; i < result.count && det_match; i++) {
        if (!nearly_equal(result2.detections[i].score, result.detections[i].score, 1e-6f) ||
            result2.detections[i].x != result.detections[i].x ||
            result2.detections[i].y != result.detections[i].y ||
            result2.detections[i].w != result.detections[i].w ||
            result2.detections[i].h != result.detections[i].h)
            det_match = 0;
    }
    check(det_match, "deterministic per-detection output");

    /* ---- steady-state detect timing: N iterations, time ONLY on_image ---- */
    const int bench_n = 200;
    double tiref = 0.0;
    double tail = 0.0;
    double ta = fiv_get_current_system_time();
    fiv_yunet_model_result br;
    for (int bi = 0; bi < bench_n; bi++) {
        memset(&br, 0, sizeof(br));
        double ts = fiv_get_current_system_time();
        fiv_yunet_detector_on_image(&br, work, detector);
        tiref += fiv_get_current_system_time() - ts;
    }
    tail = fiv_get_current_system_time() - ta;
    printf("detect_bench: %d iters  wall=%.3f ms  avg_on_image=%.3f ms/frame  %.1f FPS\n",
           bench_n, tail, tiref / bench_n, 1000.0 * bench_n / tiref);

    /* ---- interface 3: release ---- */
    fiv_ret rr = fiv_release_yunet_detector(&detector);
    check(rr == FIV_RET_OK, "fiv_release_yunet_detector returns FIV_RET_OK");
    check(detector == NULL, "release sets *detector to NULL");

    /* double release must be a no-op / safe (NULL guard) */
    check(fiv_release_yunet_detector(&detector) != FIV_RET_OK ||
          detector == NULL, "release(NULL) guarded");

    /* ---- draw detections on the image and save (after bench, image is
       the clean source used by detection above) ---- */
    for (int i = 0; i < result.count; i++) {
        const fiv_yunet_result* d = &result.detections[i];
        draw_seg(image, d->x, d->y, d->x + d->w, d->y, 2, 0, 255, 0);
        draw_seg(image, d->x + d->w, d->y, d->x + d->w, d->y + d->h, 2, 0, 255, 0);
        draw_seg(image, d->x + d->w, d->y + d->h, d->x, d->y + d->h, 2, 0, 255, 0);
        draw_seg(image, d->x, d->y + d->h, d->x, d->y, 2, 0, 255, 0);
        for (int k = 0; k < 5; k++)
            draw_point(image, d->lm[2 * k], d->lm[2 * k + 1], 3, 255, 0, 0);
    }
    {
        char out_name[] = "yunet_15_detect.png";
        fiv_ret wr = fiv_image_write(out_name, image);
        if (wr == FIV_RET_OK)
            printf("  saved annotated image -> %s\n", out_name);
    }

    /* release the loaded image through the public image API */
    fiv_release_tensor((void**)&work);
    fiv_release_image(image);

    printf("timing: image_load=%.3f ms  detector_create=%.3f ms  resize(%ux%u)=%.3f ms  detect(on_image)=%.3f ms\n",
           t_image_load, t_create, dw, dh, t_resize, t_detect);
    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}