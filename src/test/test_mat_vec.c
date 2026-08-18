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

static void test_matrix_add_vec(void)
{
    /* dim == 0: vector added to each row (broadcast along rows).
       src = [[1,2,3],[4,5,6]], vec = [10,20,30] (length 3 == cols). */
    {
        size_t sh[2] = { 2, 3 };
        fiv_mat* src = fiv_create_tensor2d(sh, FIV_32F1);
        fiv_mat* dst = fiv_create_tensor2d(sh, FIV_32F1);
        fiv_vec* vec = fiv_create_tensor1d(3, FIV_32F1);
        CHECK(src != NULL && dst != NULL && vec != NULL, "alloc ok");
        float m[6] = { 1, 2, 3, 4, 5, 6 };
        float v[3] = { 10, 20, 30 };
        memcpy(src->data.fl, m, sizeof(m));
        memcpy(vec->data.fl, v, sizeof(v));
        CHECK(fiv_matrix_add_vec(dst, src, vec, 0) == FIV_RET_OK, "add_vec dim0 OK");
        float exp[6] = { 11, 22, 33, 14, 25, 36 };
        int bad = 0;
        for (int k = 0; k < 6; k++) if (fabsf_local(dst->data.fl[k] - exp[k]) > 1e-6f) bad++;
        CHECK(bad == 0, "dim0: vec added per row");
        fiv_release_tensor2d(&src);
        fiv_release_tensor2d(&dst);
        fiv_release_tensor1d(&vec);
    }

    /* dim == 1: vector added to each column (broadcast along columns).
       vec = [100,200] (length 2 == rows): clearly distinct from dim0 above. */
    {
        size_t sh[2] = { 2, 3 };
        fiv_mat* src = fiv_create_tensor2d(sh, FIV_32F1);
        fiv_mat* dst = fiv_create_tensor2d(sh, FIV_32F1);
        fiv_vec* vec = fiv_create_tensor1d(2, FIV_32F1);
        CHECK(src != NULL && dst != NULL && vec != NULL, "alloc ok");
        float m[6] = { 1, 2, 3, 4, 5, 6 };
        float v[2] = { 100, 200 };
        memcpy(src->data.fl, m, sizeof(m));
        memcpy(vec->data.fl, v, sizeof(v));
        CHECK(fiv_matrix_add_vec(dst, src, vec, 1) == FIV_RET_OK, "add_vec dim1 OK");
        float exp[6] = { 101, 102, 103, 204, 205, 206 };
        int bad = 0;
        for (int k = 0; k < 6; k++) if (fabsf_local(dst->data.fl[k] - exp[k]) > 1e-6f) bad++;
        CHECK(bad == 0, "dim1: vec added per column");
        fiv_release_tensor2d(&src);
        fiv_release_tensor2d(&dst);
        fiv_release_tensor1d(&vec);
    }

    /* in-place: dst aliases src */
    {
        size_t sh[2] = { 2, 3 };
        fiv_mat* m = fiv_create_tensor2d(sh, FIV_32F1);
        fiv_vec* vec = fiv_create_tensor1d(3, FIV_32F1);
        CHECK(m != NULL && vec != NULL, "alloc ok");
        float a[6] = { 1, 2, 3, 4, 5, 6 };
        float v[3] = { 10, 20, 30 };
        memcpy(m->data.fl, a, sizeof(a));
        memcpy(vec->data.fl, v, sizeof(v));
        CHECK(fiv_matrix_add_vec(m, m, vec, 0) == FIV_RET_OK, "add_vec in-place OK");
        float exp[6] = { 11, 22, 33, 14, 25, 36 };
        int bad = 0;
        for (int k = 0; k < 6; k++) if (fabsf_local(m->data.fl[k] - exp[k]) > 1e-6f) bad++;
        CHECK(bad == 0, "in-place result correct");
        fiv_release_tensor2d(&m);
        fiv_release_tensor1d(&vec);
    }

    /* error paths: null args, length mismatch, bad dim */
    {
        size_t sh[2] = { 2, 3 };
        fiv_mat* src = fiv_create_tensor2d(sh, FIV_32F1);
        fiv_mat* dst = fiv_create_tensor2d(sh, FIV_32F1);
        fiv_vec* v3 = fiv_create_tensor1d(3, FIV_32F1);   /* length 3: OK for dim0, wrong for dim1 */
        fiv_vec* v2 = fiv_create_tensor1d(2, FIV_32F1);   /* length 2: OK for dim1, wrong for dim0 */
        CHECK(src != NULL && dst != NULL && v3 != NULL && v2 != NULL, "alloc ok");

        CHECK(fiv_matrix_add_vec(NULL, src, v3, 0) == FIV_RET_ERR_PARA, "null dst");
        CHECK(fiv_matrix_add_vec(dst, NULL, v3, 0) == FIV_RET_ERR_PARA, "null src");
        CHECK(fiv_matrix_add_vec(dst, src, NULL, 0) == FIV_RET_ERR_PARA, "null vec");
        /* dim0 needs vec length == cols(3); v2 has length 2 -> mismatch */
        CHECK(fiv_matrix_add_vec(dst, src, v2, 0) == FIV_RET_ERR_PARA, "dim0 vec length mismatch");
        /* dim1 needs vec length == rows(2); v3 has length 3 -> mismatch */
        CHECK(fiv_matrix_add_vec(dst, src, v3, 1) == FIV_RET_ERR_PARA, "dim1 vec length mismatch");
        /* dim1 with length-2 vec is valid */
        CHECK(fiv_matrix_add_vec(dst, src, v2, 1) == FIV_RET_OK, "dim1 with length-2 vec OK");
        /* illegal dim */
        CHECK(fiv_matrix_add_vec(dst, src, v3, 2) == FIV_RET_ERR_PARA, "bad dim value");

        fiv_release_tensor2d(&src);
        fiv_release_tensor2d(&dst);
        fiv_release_tensor1d(&v3);
        fiv_release_tensor1d(&v2);
    }

    printf("  [ok] fiv_matrix_add_vec (dim0 per-row / dim1 per-column / in-place / errors)\n");
}

static void test_matrix_reduce_sum(void)
{
    /* src = [[1,2,3],[4,5,6]] (2x3) */
    size_t sh[2] = { 2, 3 };
    fiv_mat* src = fiv_create_tensor2d(sh, FIV_32F1);
    CHECK(src != NULL, "alloc ok");
    float m[6] = { 1, 2, 3, 4, 5, 6 };
    memcpy(src->data.fl, m, sizeof(m));

    /* dim == 0: sum over rows -> per-column sums [1+4, 2+5, 3+6] = [5,7,9] */
    {
        fiv_vec* dst = fiv_create_tensor1d(3, FIV_32F1);
        CHECK(dst != NULL, "alloc dim0 dst");
        CHECK(fiv_matrix_reduce_sum(dst, src, 0, FIV_SCALAR_FP32(0.0f)) == FIV_RET_OK, "reduce dim0 OK");
        float exp[3] = { 5, 7, 9 };
        int bad = 0;
        for (int k = 0; k < 3; k++) if (fabsf_local(dst->data.fl[k] - exp[k]) > 1e-6f) bad++;
        CHECK(bad == 0, "dim0: column sums correct");
        fiv_release_tensor1d(&dst);
    }

    /* accumulation: beta = 1.0 accumulates onto a pre-seeded dst. This mirrors
       the bias-gradient path, where grad_bias is accumulated across multiple
       consumers of the same node. */
    {
        fiv_vec* dst = fiv_create_tensor1d(3, FIV_32F1);
        CHECK(dst != NULL, "alloc acc dst");
        dst->data.fl[0] = 1.0f; dst->data.fl[1] = 2.0f; dst->data.fl[2] = 3.0f;  /* seed */
        CHECK(fiv_matrix_reduce_sum(dst, src, 0, FIV_SCALAR_FP32(1.0f)) == FIV_RET_OK, "reduce dim0 acc OK");
        float exp[3] = { 6, 9, 12 };   /* seed [1,2,3] + column sums [5,7,9] */
        int bad = 0;
        for (int k = 0; k < 3; k++) if (fabsf_local(dst->data.fl[k] - exp[k]) > 1e-6f) bad++;
        CHECK(bad == 0, "dim0 acc: seed + column sums");
        fiv_release_tensor1d(&dst);
    }

    /* dim == 1: sum over cols -> per-row sums [1+2+3, 4+5+6] = [6,15] */
    {
        fiv_vec* dst = fiv_create_tensor1d(2, FIV_32F1);
        CHECK(dst != NULL, "alloc dim1 dst");
        CHECK(fiv_matrix_reduce_sum(dst, src, 1, FIV_SCALAR_FP32(0.0f)) == FIV_RET_OK, "reduce dim1 OK");
        float exp[2] = { 6, 15 };
        int bad = 0;
        for (int k = 0; k < 2; k++) if (fabsf_local(dst->data.fl[k] - exp[k]) > 1e-6f) bad++;
        CHECK(bad == 0, "dim1: row sums correct");
        fiv_release_tensor1d(&dst);
    }

    /* dim == -1: total sum = 21 (scalar) */
    {
        FIV_DECLAR_SCALAR_FP32(sc);
        CHECK(fiv_matrix_reduce_sum(&sc, src, -1, FIV_SCALAR_FP32(0.0f)) == FIV_RET_OK, "reduce dim-1 OK");
        CHECK(fabsf_local(sc.data.value_fp32 - 21.0f) < 1e-6f, "dim-1: total sum = 21 (scalar)");
    }

    /* error paths: null args, type/length mismatch, bad dim */
    {
        fiv_vec* v3 = fiv_create_tensor1d(3, FIV_32F1);   /* OK for dim0, wrong for dim1 */
        fiv_vec* v2 = fiv_create_tensor1d(2, FIV_32F1);   /* OK for dim1, wrong for dim0 */
        CHECK(v3 != NULL && v2 != NULL, "alloc error-path vecs");

        CHECK(fiv_matrix_reduce_sum(NULL, src, 0, FIV_SCALAR_FP32(0.0f)) == FIV_RET_ERR_PARA, "null dst");
        CHECK(fiv_matrix_reduce_sum(v3, NULL, 0, FIV_SCALAR_FP32(0.0f)) == FIV_RET_ERR_PARA, "null src");
        /* dim0 needs dst length == cols(3); v2 has length 2 -> mismatch */
        CHECK(fiv_matrix_reduce_sum(v2, src, 0, FIV_SCALAR_FP32(0.0f)) == FIV_RET_ERR_PARA, "dim0 dst length mismatch");
        /* dim1 needs dst length == rows(2); v3 has length 3 -> mismatch */
        CHECK(fiv_matrix_reduce_sum(v3, src, 1, FIV_SCALAR_FP32(0.0f)) == FIV_RET_ERR_PARA, "dim1 dst length mismatch");
        /* dim-1 requires a scalar dst; a fiv_vec is not accepted */
        CHECK(fiv_matrix_reduce_sum(v3, src, -1, FIV_SCALAR_FP32(0.0f)) == FIV_RET_ERR_PARA, "dim-1 dst is not scalar");
        /* illegal dim */
        CHECK(fiv_matrix_reduce_sum(v3, src, 2, FIV_SCALAR_FP32(0.0f)) == FIV_RET_ERR_PARA, "bad dim value");

        /* valid scalar dst for dim-1 */
        FIV_DECLAR_SCALAR_FP32(sc);
        CHECK(fiv_matrix_reduce_sum(&sc, src, -1, FIV_SCALAR_FP32(0.0f)) == FIV_RET_OK, "dim-1 scalar dst OK");

        /* beta must be an fp32 scalar; a non-fp32 scalar is rejected */
        FIV_DECLAR_SCALAR_INT32(bad_beta);
        bad_beta.data.value_int32 = 1;
        CHECK(fiv_matrix_reduce_sum(v3, src, 0, bad_beta) == FIV_RET_ERR_NOT_SUPPORT, "non-fp32 beta rejected");

        fiv_release_tensor1d(&v3);
        fiv_release_tensor1d(&v2);
    }

    fiv_release_tensor2d(&src);
    printf("  [ok] fiv_matrix_reduce_sum (dim0 col-sum / dim1 row-sum / dim-1 total / errors)\n");
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
    printf("=== fiv_matrix_add_vec ===\n");
    test_matrix_add_vec();
    printf("=== fiv_matrix_reduce_sum ===\n");
    test_matrix_reduce_sum();
    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
