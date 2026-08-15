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

/* Correctness tests for fiv_matrix_mul_vec (api/fiv_matrix.h):
 * dst = mat * vec (transpose == 0) and dst = mat^T * vec (transpose != 0),
 * plus the error paths (dtype, in-place, null args, short vector/dst). */

#include "fiv_matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static int g_pass = 0;
#define CHECK(c, msg)                                                           \
    do {                                                                        \
        if (!(c)) { printf("  [FAIL] %s @%d\n", (msg), __LINE__); g_fail++; }   \
        else       { g_pass++; }                                                \
    } while (0)
static float fabsf_local(float x) { return x < 0 ? -x : x; }

/* dst = mat * vec or dst = mat^T * vec, reference */
static void ref_mul_vec(int transpose, int rows, int cols,
                        const float* mat, const float* vec, float* dst)
{
    int out_len = transpose ? cols : rows;
    int k_len   = transpose ? rows : cols;
    for (int i = 0; i < out_len; i++) {
        float acc = 0.f;
        for (int k = 0; k < k_len; k++) {
            float m = transpose ? mat[k * cols + i] : mat[i * cols + k];
            acc += m * vec[k];
        }
        dst[i] = acc;
    }
}

static void run_mul_vec(int transpose, int rows, int cols, const char* name)
{
    int out_len = transpose ? cols : rows;
    int vec_len = transpose ? rows : cols;
    size_t sh_mat[2] = { (size_t)rows, (size_t)cols };
    fiv_mat* mat = fiv_create_tensor2d(sh_mat, FIV_32F1);
    fiv_vec* vec = fiv_create_tensor1d((size_t)vec_len, FIV_32F1);
    fiv_vec* dst = fiv_create_tensor1d((size_t)out_len, FIV_32F1);
    CHECK(mat != NULL && vec != NULL && dst != NULL, "alloc ok");

    unsigned s = 54321u;
    for (int i = 0; i < rows * cols; i++) {
        s = s * 1103515245u + 12345u;
        mat->data.fl[i] = (float)((s >> 8) & 0xffff) / 4096.f - 4.f;
    }
    for (int i = 0; i < vec_len; i++) {
        s = s * 1103515245u + 12345u;
        vec->data.fl[i] = (float)((s >> 8) & 0xffff) / 4096.f - 4.f;
    }

    float* exp = (float*)malloc((size_t)out_len * sizeof(float));
    ref_mul_vec(transpose, rows, cols, mat->data.fl, vec->data.fl, exp);

    fiv_ret r = fiv_matrix_mul_vec(dst, mat, vec, transpose);
    CHECK(r == FIV_RET_OK, "mul_vec returns OK");

    int bad = 0;
    for (int i = 0; i < out_len; i++) {
        if (fabsf_local(dst->data.fl[i] - exp[i]) > 1e-3f) bad++;
    }
    CHECK(bad == 0, "result matches reference");
    CHECK(dst->shapes[0] == (size_t)out_len, "dst length rewritten");

    free(exp);
    fiv_release_tensor2d(&mat);
    fiv_release_tensor1d(&vec);
    fiv_release_tensor1d(&dst);
    printf("  [ok] %s\n", name);
}

static void test_error_paths(void)
{
    size_t sh[2] = { 2, 3 };
    fiv_mat* mat = fiv_create_tensor2d(sh, FIV_32F1);
    fiv_vec* vec = fiv_create_tensor1d(3, FIV_32F1);
    fiv_vec* dst = fiv_create_tensor1d(2, FIV_32F1);
    CHECK(mat != NULL && vec != NULL && dst != NULL, "alloc ok");

    CHECK(fiv_matrix_mul_vec(NULL, mat, vec, 0) == FIV_RET_ERR_PARA, "null dst");
    CHECK(fiv_matrix_mul_vec(dst, NULL, vec, 0) == FIV_RET_ERR_PARA, "null mat");
    CHECK(fiv_matrix_mul_vec(dst, mat, NULL, 0) == FIV_RET_ERR_PARA, "null vec");
    CHECK(fiv_matrix_mul_vec(vec, mat, vec, 0) == FIV_RET_ERR_PARA, "in-place dst==vec");

    /* unsupported dtype: 8U mat */
    fiv_mat* u8 = fiv_create_tensor2d(sh, FIV_8U1);
    CHECK(fiv_matrix_mul_vec(dst, u8, vec, 0) == FIV_RET_ERR_NOT_SUPPORT, "8U mat not supported");
    fiv_release_tensor2d(&u8);

    /* vector too short for mat*vec (needs cols == 3) */
    fiv_vec* short_v = fiv_create_tensor1d(2, FIV_32F1);
    CHECK(fiv_matrix_mul_vec(dst, mat, short_v, 0) == FIV_RET_ERR_PARA, "vec too short");
    fiv_release_tensor1d(&short_v);

    /* dst too short for mat*vec (needs rows == 2) */
    fiv_vec* short_d = fiv_create_tensor1d(1, FIV_32F1);
    CHECK(fiv_matrix_mul_vec(short_d, mat, vec, 0) == FIV_RET_ERR_PARA, "dst too short");
    fiv_release_tensor1d(&short_d);

    fiv_release_tensor2d(&mat);
    fiv_release_tensor1d(&vec);
    fiv_release_tensor1d(&dst);
}

int main(void)
{
    printf("=== fiv_matrix_mul_vec ===\n");
    run_mul_vec(0, 2, 3, "mat(2x3)*vec(3) -> vec(2)");
    run_mul_vec(1, 2, 3, "mat(2x3)^T*vec(2) -> vec(3)");
    run_mul_vec(0, 4, 4, "mat(4x4)*vec(4)");
    run_mul_vec(1, 4, 4, "mat(4x4)^T*vec(4)");
    run_mul_vec(0, 1, 5, "mat(1x5)*vec(5)");
    test_error_paths();
    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
