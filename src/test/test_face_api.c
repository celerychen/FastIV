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
    fiv_mat* image = fiv_create_image_from_file((char*)img_path, FIV_RGB24_CS);
    check(image != NULL, "fiv_create_image_from_file returns non-NULL");
    if (!image) {
        printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
        return 1;
    }
    int img_w = (int)image->width;
    int img_h = (int)image->height;

    /* ---- interface 1: create ---- */
    void* detector = fiv_create_face_detetor((char*)model_path);
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
    fiv_ret r = fiv_face_detector_on_image(&result, image, detector);
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

    /* ---- interface 3: release ---- */
    fiv_ret rr = fiv_release_face_detector(&detector);
    check(rr == FIV_RET_OK, "fiv_release_face_detector returns FIV_RET_OK");
    check(detector == NULL, "release sets *detector to NULL");

    /* double release must be a no-op / safe (NULL guard) */
    check(fiv_release_face_detector(&detector) != FIV_RET_OK ||
          detector == NULL, "release(NULL) guarded");

    /* release the loaded image through the public image API */
    fiv_release_image(image);

    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
