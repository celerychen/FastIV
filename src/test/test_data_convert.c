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
#include <math.h>

#include "fiv_image.h"
#include "fiv_ctensor.h"
#include "fiv_data_convert.h"

static int test_8u_to_32f(void) {
    int pass = 1;
    const size_t n = 1000;
    fiv_mat* src = fiv_create_tensor2d((size_t[2]){1, n}, FIV_8U1);
    fiv_mat* dst = fiv_create_tensor2d((size_t[2]){1, n}, FIV_32F1);
    if (src == NULL || dst == NULL) { printf("FAIL: alloc\n"); return 0; }

    iv8u* sp = src->data.ptr8u;
    for (size_t k = 0; k < n; k++) sp[k] = (iv8u)(k % 256);

    if (fiv_tensor_data_convert(dst, src, FIV_8U_TO_32F) != FIV_RET_OK) {
        printf("FAIL: FIV_8U_TO_32F status\n"); pass = 0;
    }
    ivf32* dp = dst->data.fl;
    for (size_t k = 0; k < n; k++) {
        if (dp[k] != (ivf32)sp[k]) { printf("FAIL: 8U->32F [%zu]=%g\n", k, dp[k]); pass = 0; break; }
    }
    /* element_bytes / dtype updated on dst */
    if (dst->element_bytes != 4) { printf("FAIL: dst element_bytes %d\n", dst->element_bytes); pass = 0; }

    fiv_release_tensor((void**)&src);
    fiv_release_tensor((void**)&dst);
    return pass;
}

static int test_8u_to_32f_norm(void) {
    int pass = 1;
    const size_t n = 256;
    fiv_mat* src = fiv_create_tensor2d((size_t[2]){1, n}, FIV_8U1);
    fiv_mat* dst = fiv_create_tensor2d((size_t[2]){1, n}, FIV_32F1);
    if (src == NULL || dst == NULL) return 0;
    iv8u* sp = src->data.ptr8u;
    for (size_t k = 0; k < n; k++) sp[k] = (iv8u)k;

    if (fiv_tensor_data_convert(dst, src, FIV_8U_TO_32F_NORM01) != FIV_RET_OK) {
        printf("FAIL: NORM01 status\n"); pass = 0;
    }
    ivf32* dp = dst->data.fl;
    for (size_t k = 0; k < n; k++) {
        ivf32 expected = (ivf32)((ivf32)k / 255.0f);
        if (fabsf(dp[k] - expected) > 1e-6f) { printf("FAIL: NORM01 [%zu]=%g exp %g\n", k, dp[k], expected); pass = 0; break; }
    }

    if (fiv_tensor_data_convert(dst, src, FIV_8U_TO_32F_NORM_N1_P1) != FIV_RET_OK) {
        printf("FAIL: NORM_N1_P1 status\n"); pass = 0;
    }
    for (size_t k = 0; k < n; k++) {
        ivf32 expected = (ivf32)k / 127.5f - 1.0f;
        if (fabsf(dp[k] - expected) > 1e-6f) { printf("FAIL: NORM_N1_P1 [%zu]=%g exp %g\n", k, dp[k], expected); pass = 0; break; }
    }

    if (fiv_tensor_data_convert(dst, src, FIV_8U_TO_32F_NORM_MU_SIGMA) != FIV_RET_ERR_NOT_SUPPORT) {
        printf("FAIL: MU_SIGMA should be NOT_SUPPORT (no param channel)\n"); pass = 0;
    }

    fiv_release_tensor((void**)&src);
    fiv_release_tensor((void**)&dst);
    return pass;
}

static int test_32s_to_32f_inplace_and_out(void) {
    int pass = 1;
    const size_t n = 500;
    fiv_mat* src = fiv_create_tensor2d((size_t[2]){1, n}, FIV_32S1);
    iv32s* sp = src->data.ptr32s;
    for (size_t k = 0; k < n; k++) sp[k] = (iv32s)((iv32s)k * 7 - 1000);

    /* out-of-place */
    fiv_mat* dst = fiv_create_tensor2d((size_t[2]){1, n}, FIV_32F1);
    if (fiv_tensor_data_convert(dst, src, FIV_32S_TO_32F) != FIV_RET_OK) {
        printf("FAIL: 32S->32F status\n"); pass = 0;
    }
    ivf32* dp = dst->data.fl;
    for (size_t k = 0; k < n; k++) {
        if (dp[k] != (ivf32)sp[k]) { printf("FAIL: 32S->32F out [%zu]=%g\n", k, dp[k]); pass = 0; break; }
    }
    fiv_release_tensor((void**)&dst);

    /* in-place: same tensor, element size matches (4 bytes) */
    if (fiv_tensor_data_convert(src, src, FIV_32S_TO_32F) != FIV_RET_OK) {
        printf("FAIL: 32S->32F in-place status\n"); pass = 0;
    }
    if (src->dtype != FIV_32F1) { printf("FAIL: in-place dtype not updated\n"); pass = 0; }
    ivf32* ip = src->data.fl;
    for (size_t k = 0; k < n; k++) {
        ivf32 expected = (ivf32)((iv32s)k * 7 - 1000);
        if (ip[k] != expected) { printf("FAIL: 32S->32F in-place [%zu]=%g exp %g\n", k, ip[k], expected); pass = 0; break; }
    }

    fiv_release_tensor((void**)&src);
    return pass;
}

/* 3-channel image conversion: scalar-wise, channel layout preserved. */
static int test_3channel(void) {
    int pass = 1;
    const size_t n = 64; /* 64 RGB pixels -> 192 scalars */
    fiv_mat* src = fiv_create_tensor2d((size_t[2]){n, 3}, FIV_8U3);
    fiv_mat* dst = fiv_create_tensor2d((size_t[2]){n, 3}, FIV_32F3);
    if (src == NULL || dst == NULL) return 0;
    iv8u* sp = src->data.ptr8u;
    for (size_t k = 0; k < n * 3; k++) sp[k] = (iv8u)(k % 256);

    if (fiv_tensor_data_convert(dst, src, FIV_8U_TO_32F) != FIV_RET_OK) {
        printf("FAIL: 8U3->32F3 status\n"); pass = 0;
    }
    if (dst->dtype != FIV_32F3) { printf("FAIL: 8U3->32F3 dtype not updated\n"); pass = 0; }
    ivf32* dp = dst->data.fl;
    for (size_t k = 0; k < n * 3; k++) {
        if (dp[k] != (ivf32)sp[k]) { printf("FAIL: 8U3->32F3 [%zu]=%g exp %u\n", k, dp[k], sp[k]); pass = 0; break; }
    }

    /* 32S3 -> 32F3 in-place (element size 12 bytes, matches) */
    fiv_mat* s3 = fiv_create_tensor2d((size_t[2]){n, 3}, FIV_32S3);
    iv32s* s3p = s3->data.ptr32s;
    for (size_t k = 0; k < n * 3; k++) s3p[k] = (iv32s)((iv32s)k * 3 - 500);
    if (fiv_tensor_data_convert(s3, s3, FIV_32S_TO_32F) != FIV_RET_OK) {
        printf("FAIL: 32S3->32F3 in-place status\n"); pass = 0;
    }
    if (s3->dtype != FIV_32F3) { printf("FAIL: 32S3 in-place dtype\n"); pass = 0; }
    ivf32* s3f = s3->data.fl;
    for (size_t k = 0; k < n * 3; k++) {
        ivf32 expected = (ivf32)((iv32s)k * 3 - 500);
        if (s3f[k] != expected) { printf("FAIL: 32S3->32F3 in-place [%zu]=%g exp %g\n", k, s3f[k], expected); pass = 0; break; }
    }

    fiv_release_tensor((void**)&src);
    fiv_release_tensor((void**)&dst);
    fiv_release_tensor((void**)&s3);
    return pass;
}

static int test_error_paths(void) {
    int pass = 1;
    fiv_mat* src = fiv_create_tensor2d((size_t[2]){1, 10}, FIV_32S1);
    fiv_mat* dst = fiv_create_tensor2d((size_t[2]){1, 10}, FIV_32F1);

    /* 8U->32F with same pointer must fail (element size differs, cannot alias) */
    if (fiv_tensor_data_convert(src, src, FIV_8U_TO_32F) != FIV_RET_ERR_PARA) {
        printf("FAIL: 8U->32F in-place alias should be ERR_PARA\n"); pass = 0;
    }

    /* dtype mismatch: 32S src but requested 8U->32F */
    if (fiv_tensor_data_convert(dst, src, FIV_8U_TO_32F) != FIV_RET_ERR_PARA) {
        printf("FAIL: src dtype mismatch should be ERR_PARA\n"); pass = 0;
    }

    /* dst element count mismatch */
    fiv_mat* short_dst = fiv_create_tensor2d((size_t[2]){1, 5}, FIV_32F1);
    if (fiv_tensor_data_convert(short_dst, src, FIV_32S_TO_32F) != FIV_RET_ERR_PARA) {
        printf("FAIL: dst count mismatch should be ERR_PARA\n"); pass = 0;
    }
    fiv_release_tensor((void**)&short_dst);

    if (fiv_tensor_data_convert(NULL, src, FIV_32S_TO_32F) != FIV_RET_ERR_PARA) {
        printf("FAIL: NULL dst should be ERR_PARA\n"); pass = 0;
    }

    fiv_release_tensor((void**)&src);
    fiv_release_tensor((void**)&dst);
    return pass;
}

int main(void) {
    int pass = 1;
    pass &= test_8u_to_32f();
    pass &= test_8u_to_32f_norm();
    pass &= test_32s_to_32f_inplace_and_out();
    pass &= test_3channel();
    pass &= test_error_paths();
    printf(pass ? "test_data_convert: PASS\n" : "test_data_convert: FAIL\n");
    return pass ? 0 : 1;
}
