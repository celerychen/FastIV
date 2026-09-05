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
 * YuNet timing at a reduced working resolution that PRESERVES the source
 * aspect ratio (longest side = FIV_WORK_SIZE).
 *
 * The YuNet detached graph has no built-in resize: fiv_yunet_preprocess just
 * packs the input RGB and pads to a multiple of 32 at NATIVE resolution, so
 * running on a 1920x1080 frame is expensive. This test scales the frame down,
 * keeping the original aspect ratio, with the public fiv_image_resize()
 * interface first (for real-time face-tracking the boxes would then be mapped
 * back), then times the detector.
 */

#define FIV_WORK_SIZE 320   /* longest side of the reduced working image */

#include "fiv_yunet_model.h"
#include "fiv_image.h"     /* fiv_create_image_from_file / fiv_image_resize */
#include "fiv_ctensor.h"   /* fiv_create_tensor2d / fiv_release_tensor */
#include "fiv_common.h"    /* fiv_get_current_system_time */

#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    const char* img_path   = argc > 1 ? argv[1] : "../src/test/15.png";
    const char* model_path = argc > 2 ? argv[2] : "../app/face/models/yunet_weights.bin";

    /* ---- load the source frame ---- */
    fiv_mat* image = fiv_create_image_from_file((char*)img_path, FIV_RGB24_CS);
    if (!image) { printf("failed to load %s\n", img_path); return 1; }
    printf("source: %ux%u\n", (unsigned)image->width, (unsigned)image->height);

    /* ---- working image: longest side = FIV_WORK_SIZE, aspect ratio kept ---- */
    const unsigned sw = image->width, sh = image->height;
    unsigned dw = FIV_WORK_SIZE, dh = FIV_WORK_SIZE;
    if (sw > sh) dh = (unsigned)((ivf64)FIV_WORK_SIZE * sh / sw + 0.5);
    else         dw = (unsigned)((ivf64)FIV_WORK_SIZE * sw / sh + 0.5);
    size_t s_work[2] = { dh, dw };
    fiv_mat* work = fiv_create_tensor2d(s_work, FIV_8U3);
    if (!work) { fiv_release_image(image); return 1; }

    /* ---- create the detector once ---- */
    void* detector = fiv_create_yunet_detector((char*)model_path);
    if (!detector) { fiv_release_image(image); fiv_release_tensor((void**)&work); return 1; }

    fiv_yunet_model_result result;

    /* ---- steady-state: resize + detect ---- */
    const int bench_n = 200;
    ivf64 t_resize = 0.0, t_detect = 0.0;

    for (int bi = 0; bi < bench_n; bi++) {
        ivf64 ts = fiv_get_current_system_time();
        if (fiv_image_resize(work, image, FIV_BILEAR_RESIZER) != FIV_RET_OK) {
            printf("fiv_image_resize failed\n");
            return 1;
        }
        t_resize += fiv_get_current_system_time() - ts;

        memset(&result, 0, sizeof(result));
        ts = fiv_get_current_system_time();
        fiv_yunet_detector_on_image(&result, work, detector);
        t_detect += fiv_get_current_system_time() - ts;
    }

    printf("%ux%u (ratio-kept) benchmark (%d iters):\n", dw, dh, bench_n);
    printf("  resize  avg=%.3f ms/frame\n", t_resize / bench_n);
    printf("  detect  avg=%.3f ms/frame\n", t_detect / bench_n);
    printf("  total   avg=%.3f ms/frame  (%.1f FPS)\n",
           (t_resize + t_detect) / bench_n, 1000.0 * bench_n / (t_resize + t_detect));
    printf("  faces detected: %d\n", result.count);
    for (int i = 0; i < result.count; i++)
        printf("    det[%d]: score=%.4f x=%d y=%d w=%d h=%d\n",
               i, result.detections[i].score, result.detections[i].x,
               result.detections[i].y, result.detections[i].w, result.detections[i].h);

    fiv_release_yunet_detector(&detector);
    fiv_release_image(image);
    fiv_release_tensor((void**)&work);
    return 0;
}