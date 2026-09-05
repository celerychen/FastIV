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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fiv_image.h"
#include "fiv_ctensor.h"
#include "fiv_image_color_space.h"
#include "fiv_common.h"   /* fiv_get_current_system_time (returns ms as ivf64) */


/* Build a tightly-packed random RGB image (FIV_8U3). */
static fiv_mat* fiv_perf_make_rgb(int height, int width) {
    size_t      shape[4] = { (size_t)height, (size_t)width, 1, 1 };
    fiv_mat*    image    = fiv_create_tensor2d(shape, FIV_8U3);
    size_t      total    = (size_t)height * width * 3;
    size_t      k;

    if (image == NULL) return NULL;
    for (k = 0; k < total; k++) {
        image->data.ptr8u[k] = (iv8u)(rand() & 0xff);
    }
    image->color_space_type = FIV_RGB24_CS;
    return image;
}


/* Compare two grayscale buffers byte-by-byte. Returns 1 if identical. */
static int fiv_perf_gray_match(const iv8u* a, const iv8u* b, size_t n) {
    size_t k;
    for (k = 0; k < n; k++) {
        if (a[k] != b[k]) {
            printf("  mismatch at byte %zu: scalar=%d simd=%d\n", k, a[k], b[k]);
            return 0;
        }
    }
    return 1;
}


int main(int argc, char** argv) {
    int    height   = 1080;
    int    width    = 1920;
    int    iters    = 50;
    int    pass     = 1;

    (void)argc; (void)argv;

    if (height <= 0 || width <= 0) { height = 1080; width = 1920; }
    printf("color-space perf: %dx%d, %d iterations\n", width, height, iters);

    /* RGB source + two independent grayscale destinations. */
    fiv_mat* rgb   = fiv_perf_make_rgb(height, width);
    size_t   shape[4] = { (size_t)height, (size_t)width, 1, 1 };
    fiv_mat* g_simd  = fiv_create_tensor2d(shape, FIV_8U1);
    fiv_mat* g_scalar = fiv_create_tensor2d(shape, FIV_8U1);
    if (rgb == NULL || g_simd == NULL || g_scalar == NULL) {
        printf("FAIL: allocation error\n");
        pass = 0;
        goto cleanup;
    }

    /* ---- scalar baseline ---- */
    {
        double t0 = fiv_get_current_system_time();
        int i;
        for (i = 0; i < iters; i++) {
            fiv_cs_to_gray_scalar(g_scalar, rgb, 0, 1, 2);
        }
        double t1 = fiv_get_current_system_time();
        double us = (t1 - t0) / iters * 1000.0;
        printf("  scalar  : %8.2f us/iter\n", us);
    }

    /* ---- SIMD path (via public convertor; dispatches to AVX2/SSE/NEON) ---- */
    {
        double t0 = fiv_get_current_system_time();
        int i;
        for (i = 0; i < iters; i++) {
            fiv_image_color_space_convertor(g_simd, rgb, FIV_CS_RGB2GRAY);
        }
        double t1 = fiv_get_current_system_time();
        double us = (t1 - t0) / iters * 1000.0;
        printf("  simd    : %8.2f us/iter\n", us);
    }

    /* ---- bit-exact check: SIMD output must equal scalar output ---- */
    {
        size_t n = (size_t)height * width;
        if (fiv_perf_gray_match(g_scalar->data.ptr8u, g_simd->data.ptr8u, n)) {
            printf("  bit-exact: SIMD == scalar  (PASS)\n");
        } else {
            printf("  bit-exact: SIMD != scalar  (FAIL)\n");
            pass = 0;
        }
    }

    /* ---- BGR source: same bit-exact contract ---- */
    {
        fiv_mat* bgr = fiv_perf_make_rgb(height, width);
        if (bgr == NULL) { printf("FAIL: bgr alloc error\n"); pass = 0; goto cleanup; }
        bgr->color_space_type = FIV_BGR24_CS;
        fiv_cs_to_gray_scalar(g_scalar, bgr, 2, 1, 0);
        fiv_image_color_space_convertor(g_simd, bgr, FIV_CS_BGR2GRAY);
        size_t n = (size_t)height * width;
        if (fiv_perf_gray_match(g_scalar->data.ptr8u, g_simd->data.ptr8u, n)) {
            printf("  bit-exact (BGR): SIMD == scalar  (PASS)\n");
        } else {
            printf("  bit-exact (BGR): SIMD != scalar  (FAIL)\n");
            pass = 0;
        }
        fiv_release_tensor((void**)&bgr);
    }

    /* ---- R/B swap: scalar vs SIMD (NEON vld3/vst3) ---- */
    {
        fiv_mat* rgb_sw    = fiv_perf_make_rgb(height, width);
        fiv_mat* bgr_simd  = fiv_create_tensor2d(shape, FIV_8U3);
        fiv_mat* bgr_scalar = fiv_create_tensor2d(shape, FIV_8U3);
        if (rgb_sw == NULL || bgr_simd == NULL || bgr_scalar == NULL) {
            printf("FAIL: swap alloc error\n"); pass = 0; goto cleanup;
        }
        double t0, t1, us;
        int i;

        t0 = fiv_get_current_system_time();
        for (i = 0; i < iters; i++) fiv_cs_swap_rb_scalar(bgr_scalar, rgb_sw);
        t1 = fiv_get_current_system_time();
        us = (t1 - t0) / iters * 1000.0;
        printf("  swap scalar: %8.2f us/iter\n", us);

        t0 = fiv_get_current_system_time();
        for (i = 0; i < iters; i++)
            fiv_image_color_space_convertor(bgr_simd, rgb_sw, FIV_CS_RGB2BGR);
        t1 = fiv_get_current_system_time();
        us = (t1 - t0) / iters * 1000.0;
        printf("  swap simd  : %8.2f us/iter\n", us);

        size_t n = (size_t)height * width * 3;
        if (fiv_perf_gray_match(bgr_scalar->data.ptr8u, bgr_simd->data.ptr8u, n)) {
            printf("  bit-exact (swap): SIMD == scalar  (PASS)\n");
        } else {
            printf("  bit-exact (swap): SIMD != scalar  (FAIL)\n");
            pass = 0;
        }
        fiv_release_tensor((void**)&rgb_sw);
        fiv_release_tensor((void**)&bgr_simd);
        fiv_release_tensor((void**)&bgr_scalar);
    }

cleanup:
    if (rgb)      fiv_release_tensor((void**)&rgb);
    if (g_simd)   fiv_release_tensor((void**)&g_simd);
    if (g_scalar) fiv_release_tensor((void**)&g_scalar);

    printf(pass ? "PERF TEST: PASS\n" : "PERF TEST: FAIL\n");
    return pass ? 0 : 1;
}
