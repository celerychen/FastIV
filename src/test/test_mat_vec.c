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
#include <math.h>

static int g_fail = 0;
static int g_pass = 0;
#define CHECK(c, msg)                                                           \
    do {                                                                        \
        if (!(c)) { printf("  [FAIL] %s @%d\n", (msg), __LINE__); g_fail++; }   \
        else       { g_pass++; }                                                \
    } while (0)
static float fabsf_local(float x) { return x < 0 ? -x : x; }
static double fabs_local_d(double x) { return x < 0 ? -x : x; }
static double ref_l1(const double* a, int n) { double s = 0; for (int i = 0; i < n; i++) s += (a[i] < 0 ? -a[i] : a[i]); return s; }
static double ref_l2(const double* a, int n) {
    double max_abs = 0;
    for (int i = 0; i < n; i++) { double t = a[i] < 0 ? -a[i] : a[i]; if (t > max_abs) max_abs = t; }
    if (max_abs == 0) return 0;
    double sum_sq = 0;
    for (int i = 0; i < n; i++) { double t = a[i] / max_abs; sum_sq += t * t; }
    return max_abs * sqrt(sum_sq);
}
static double ref_inf(const double* a, int n) {
    double m = 0;
    for (int i = 0; i < n; i++) { double t = a[i] < 0 ? -a[i] : a[i]; if (t > m) m = t; }
    return m;
}

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
    printf("  [ok] fiv_matrix_reduce_sum (dim0 col-sum / dim1 row-sum / dim-1 total / * errors)\n");
}

static void test_vec_norm(void)
{
    /* data {3,-4,12,0,-5}: L1 = 24, L2 = sqrt(194) ~ 13.928388 */
    double data[5] = { 3.0, -4.0, 12.0, 0.0, -5.0 };
    double l1_exp = ref_l1(data, 5);
    double l2_exp = ref_l2(data, 5);
    double inf_exp = ref_inf(data, 5);

    /* ivf32 path */
    {
        fiv_vec* vec = fiv_create_tensor1d(5, FIV_32F1);
        CHECK(vec != NULL, "alloc 32F vec");
        for (int i = 0; i < 5; i++) vec->data.fl[i] = (float)data[i];

        FIV_DECLAR_SCALAR_FP32(n1);
        CHECK(fiv_vec_norm(&n1, vec, FIV_L1_NORM) == FIV_RET_OK, "32F L1 OK");
        CHECK(fabsf_local(n1.data.value_fp32 - (float)l1_exp) < 1e-3f, "32F L1 value");

        FIV_DECLAR_SCALAR_FP32(n2);
        CHECK(fiv_vec_norm(&n2, vec, FIV_L2_NORM) == FIV_RET_OK, "32F L2 OK");
        CHECK(fabsf_local(n2.data.value_fp32 - (float)l2_exp) < 1e-3f, "32F L2 value");
        CHECK(n2.dtype == FIV_32F1 && n2.id == FIV_ID_SCALAR, "32F norm scalar dtype");

        FIV_DECLAR_SCALAR_FP32(n_inf);
        CHECK(fiv_vec_norm(&n_inf, vec, FIV_INF_NORM) == FIV_RET_OK, "32F INF OK");
        CHECK(fabsf_local(n_inf.data.value_fp32 - (float)inf_exp) < 1e-3f, "32F INF value");

        /* numerical stability for large magnitudes (float would overflow naively) */
        for (int i = 0; i < 5; i++) vec->data.fl[i] = (float)(data[i] * 1e20);
        FIV_DECLAR_SCALAR_FP32(n3);
        CHECK(fiv_vec_norm(&n3, vec, FIV_L2_NORM) == FIV_RET_OK, "32F L2 large OK");
        CHECK(fabsf_local(n3.data.value_fp32 - (float)(l2_exp * 1e20)) < 1e-2f, "32F L2 large stable");
        fiv_release_tensor1d(&vec);
    }

    /* ivf64 path */
    {
        fiv_vec* vec = fiv_create_tensor1d(5, FIV_64F1);
        CHECK(vec != NULL, "alloc 64F vec");
        for (int i = 0; i < 5; i++) vec->data.db[i] = data[i];

        FIV_DECLAR_SCALAR_FP64(n1);
        CHECK(fiv_vec_norm(&n1, vec, FIV_L1_NORM) == FIV_RET_OK, "64F L1 OK");
        CHECK(fabs_local_d(n1.data.value_fp64 - l1_exp) < 1e-9, "64F L1 value");

        FIV_DECLAR_SCALAR_FP64(n2);
        CHECK(fiv_vec_norm(&n2, vec, FIV_L2_NORM) == FIV_RET_OK, "64F L2 OK");
        CHECK(fabs_local_d(n2.data.value_fp64 - l2_exp) < 1e-9, "64F L2 value");
        CHECK(n2.dtype == FIV_64F1 && n2.id == FIV_ID_SCALAR, "64F norm scalar dtype");

        FIV_DECLAR_SCALAR_FP64(n_inf);
        CHECK(fiv_vec_norm(&n_inf, vec, FIV_INF_NORM) == FIV_RET_OK, "64F INF OK");
        CHECK(fabs_local_d(n_inf.data.value_fp64 - inf_exp) < 1e-9, "64F INF value");

        /* numerical stability for double magnitudes (~1e300, beyond float range) */
        for (int i = 0; i < 5; i++) vec->data.db[i] = data[i] * 1e300;
        FIV_DECLAR_SCALAR_FP64(n3);
        CHECK(fiv_vec_norm(&n3, vec, FIV_L2_NORM) == FIV_RET_OK, "64F L2 large OK");
        CHECK(fabs_local_d(n3.data.value_fp64 - (l2_exp * 1e300)) < 1e-6 * (l2_exp * 1e300), "64F L2 large stable");
        fiv_release_tensor1d(&vec);
    }

    /* all-zero vector -> both norms are 0 */
    {
        fiv_vec* z = fiv_create_tensor1d(4, FIV_32F1);
        CHECK(z != NULL, "alloc zero vec");
        for (int i = 0; i < 4; i++) z->data.fl[i] = 0.0f;
        FIV_DECLAR_SCALAR_FP32(nz);
        CHECK(fiv_vec_norm(&nz, z, FIV_L1_NORM) == FIV_RET_OK, "zero L1 OK");
        CHECK(fabsf_local(nz.data.value_fp32) < 1e-7f, "zero L1 == 0");
        FIV_DECLAR_SCALAR_FP32(nz2);
        CHECK(fiv_vec_norm(&nz2, z, FIV_L2_NORM) == FIV_RET_OK, "zero L2 OK");
        CHECK(fabsf_local(nz2.data.value_fp32) < 1e-7f, "zero L2 == 0");
        fiv_release_tensor1d(&z);
    }

    /* error paths */
    {
        fiv_vec* vec = fiv_create_tensor1d(3, FIV_32F1);
        CHECK(vec != NULL, "alloc err vec");
        FIV_DECLAR_SCALAR_FP32(n);
        CHECK(fiv_vec_norm(NULL, vec, FIV_L1_NORM) == FIV_RET_ERR_PARA, "null norm_value");
        CHECK(fiv_vec_norm(&n, NULL, FIV_L1_NORM) == FIV_RET_ERR_PARA, "null vec");
        /* unsupported dtype (8U) */
        fiv_vec* u8 = fiv_create_tensor1d(3, FIV_8U1);
        CHECK(fiv_vec_norm(&n, u8, FIV_L1_NORM) == FIV_RET_ERR_NOT_SUPPORT, "8U vec not supported");
        fiv_release_tensor1d(&u8);
        fiv_release_tensor1d(&vec);
    }

    printf("  [ok] fiv_vec_norm (L1/L2, 32F/64F, stability, errors)\n");
}

static void test_vec_axpy(void)
{
    /* ivf32: y = a*x + y, y=[1,2,3], x=[4,5,6], a=2 -> [9,12,15] */
    {
        fiv_vec* y = fiv_create_tensor1d(3, FIV_32F1);
        fiv_vec* x = fiv_create_tensor1d(3, FIV_32F1);
        CHECK(y != NULL && x != NULL, "alloc 32F axpy vecs");
        float yd[3] = { 1.0f, 2.0f, 3.0f };
        float xd[3] = { 4.0f, 5.0f, 6.0f };
        memcpy(y->data.fl, yd, sizeof(yd));
        memcpy(x->data.fl, xd, sizeof(xd));

        CHECK(fiv_vec_axpy(y, FIV_SCALAR_FP32(2.0f), x) == FIV_RET_OK, "32F axpy OK");
        float exp[3] = { 9.0f, 12.0f, 15.0f };
        int bad = 0;
        for (int k = 0; k < 3; k++) if (fabsf_local(y->data.fl[k] - exp[k]) > 1e-6f) bad++;
        CHECK(bad == 0, "32F axpy value");
        fiv_release_tensor1d(&y);
        fiv_release_tensor1d(&x);
    }

    /* ivf64: y = a*x + y, y=[1,2,3], x=[4,5,6], a=2 -> [9,12,15] */
    {
        fiv_vec* y = fiv_create_tensor1d(3, FIV_64F1);
        fiv_vec* x = fiv_create_tensor1d(3, FIV_64F1);
        CHECK(y != NULL && x != NULL, "alloc 64F axpy vecs");
        double yd[3] = { 1.0, 2.0, 3.0 };
        double xd[3] = { 4.0, 5.0, 6.0 };
        for (int i = 0; i < 3; i++) { y->data.db[i] = yd[i]; x->data.db[i] = xd[i]; }

        FIV_DECLAR_SCALAR_FP64(a64);
        a64.data.value_fp64 = 2.0;
        CHECK(fiv_vec_axpy(y, a64, x) == FIV_RET_OK, "64F axpy OK");
        double exp[3] = { 9.0, 12.0, 15.0 };
        int bad = 0;
        for (int k = 0; k < 3; k++) if (fabs_local_d(y->data.db[k] - exp[k]) > 1e-12) bad++;
        CHECK(bad == 0, "64F axpy value");
        fiv_release_tensor1d(&y);
        fiv_release_tensor1d(&x);
    }

    /* in-place: x and y aliased -> y = (a+1) * y */
    {
        fiv_vec* y = fiv_create_tensor1d(3, FIV_32F1);
        CHECK(y != NULL, "alloc in-place vec");
        float yd[3] = { 1.0f, 2.0f, 3.0f };
        memcpy(y->data.fl, yd, sizeof(yd));
        CHECK(fiv_vec_axpy(y, FIV_SCALAR_FP32(2.0f), y) == FIV_RET_OK, "in-place axpy OK");
        float exp[3] = { 3.0f, 6.0f, 9.0f };
        int bad = 0;
        for (int k = 0; k < 3; k++) if (fabsf_local(y->data.fl[k] - exp[k]) > 1e-6f) bad++;
        CHECK(bad == 0, "in-place axpy value");
        fiv_release_tensor1d(&y);
    }

    /* error paths */
    {
        fiv_vec* y = fiv_create_tensor1d(3, FIV_32F1);
        fiv_vec* x = fiv_create_tensor1d(3, FIV_32F1);
        fiv_vec* x64 = fiv_create_tensor1d(3, FIV_64F1);
        fiv_vec* x2 = fiv_create_tensor1d(2, FIV_32F1);
        CHECK(y != NULL && x != NULL && x64 != NULL && x2 != NULL, "alloc err vecs");

        CHECK(fiv_vec_axpy(NULL, FIV_SCALAR_FP32(1.0f), x) == FIV_RET_ERR_PARA, "null y");
        CHECK(fiv_vec_axpy(y, FIV_SCALAR_FP32(1.0f), NULL) == FIV_RET_ERR_PARA, "null x");
        CHECK(fiv_vec_axpy(y, FIV_SCALAR_FP32(1.0f), x64) == FIV_RET_ERR_PARA, "dtype mismatch");
        CHECK(fiv_vec_axpy(y, FIV_SCALAR_FP32(1.0f), x2) == FIV_RET_ERR_PARA, "length mismatch");
        fiv_release_tensor1d(&y);
        fiv_release_tensor1d(&x);
        fiv_release_tensor1d(&x64);
        fiv_release_tensor1d(&x2);

        /* unsupported dtype (8U) */
        fiv_vec* u8 = fiv_create_tensor1d(3, FIV_8U1);
        fiv_vec* u8b = fiv_create_tensor1d(3, FIV_8U1);
        CHECK(fiv_vec_axpy(u8, FIV_SCALAR_FP32(1.0f), u8b) == FIV_RET_ERR_NOT_SUPPORT, "8U not supported");
        fiv_release_tensor1d(&u8);
        fiv_release_tensor1d(&u8b);
    }

    printf("  [ok] fiv_vec_axpy (32F/64F, in-place, errors)\n");
}

static void test_vec_dot(void)
{
    /* ivf32: dot([1,2,3],[4,5,6]) = 1*4 + 2*5 + 3*6 = 32 */
    {
        fiv_vec* a = fiv_create_tensor1d(3, FIV_32F1);
        fiv_vec* b = fiv_create_tensor1d(3, FIV_32F1);
        CHECK(a != NULL && b != NULL, "alloc 32F dot vecs");
        float ad[3] = { 1.0f, 2.0f, 3.0f };
        float bd[3] = { 4.0f, 5.0f, 6.0f };
        memcpy(a->data.fl, ad, sizeof(ad));
        memcpy(b->data.fl, bd, sizeof(bd));

        FIV_DECLAR_SCALAR_FP32(d);
        CHECK(fiv_vec_dot(&d, a, b) == FIV_RET_OK, "32F dot OK");
        CHECK(fabsf_local(d.data.value_fp32 - 32.0f) < 1e-6f, "32F dot value");
        CHECK(d.dtype == FIV_32F1 && d.id == FIV_ID_SCALAR, "32F dot scalar dtype");
        fiv_release_tensor1d(&a);
        fiv_release_tensor1d(&b);
    }

    /* ivf64: mixed signs -> dot([1,-2,3],[4,-5,6]) = 4 + 10 + 18 = 32 */
    {
        fiv_vec* a = fiv_create_tensor1d(3, FIV_64F1);
        fiv_vec* b = fiv_create_tensor1d(3, FIV_64F1);
        CHECK(a != NULL && b != NULL, "alloc 64F dot vecs");
        double ad[3] = { 1.0, -2.0, 3.0 };
        double bd[3] = { 4.0, -5.0, 6.0 };
        for (int i = 0; i < 3; i++) { a->data.db[i] = ad[i]; b->data.db[i] = bd[i]; }

        FIV_DECLAR_SCALAR_FP64(d);
        CHECK(fiv_vec_dot(&d, a, b) == FIV_RET_OK, "64F dot OK");
        CHECK(fabs_local_d(d.data.value_fp64 - 32.0) < 1e-12, "64F dot value");
        CHECK(d.dtype == FIV_64F1 && d.id == FIV_ID_SCALAR, "64F dot scalar dtype");
        fiv_release_tensor1d(&a);
        fiv_release_tensor1d(&b);
    }

    /* sparse-ish: zero vector -> 0 */
    {
        fiv_vec* a = fiv_create_tensor1d(4, FIV_32F1);
        fiv_vec* b = fiv_create_tensor1d(4, FIV_32F1);
        CHECK(a != NULL && b != NULL, "alloc zero dot vecs");
        for (int i = 0; i < 4; i++) { a->data.fl[i] = 0.0f; b->data.fl[i] = (float)(i + 1); }
        FIV_DECLAR_SCALAR_FP32(d);
        CHECK(fiv_vec_dot(&d, a, b) == FIV_RET_OK, "zero dot OK");
        CHECK(fabsf_local(d.data.value_fp32) < 1e-7f, "zero dot == 0");
        fiv_release_tensor1d(&a);
        fiv_release_tensor1d(&b);
    }

    /* error paths */
    {
        fiv_vec* a = fiv_create_tensor1d(3, FIV_32F1);
        fiv_vec* b = fiv_create_tensor1d(3, FIV_32F1);
        fiv_vec* b64 = fiv_create_tensor1d(3, FIV_64F1);
        fiv_vec* b2 = fiv_create_tensor1d(2, FIV_32F1);
        CHECK(a != NULL && b != NULL && b64 != NULL && b2 != NULL, "alloc err vecs");

        FIV_DECLAR_SCALAR_FP32(d);
        CHECK(fiv_vec_dot(NULL, a, b) == FIV_RET_ERR_PARA, "null dot_value");
        CHECK(fiv_vec_dot(&d, NULL, b) == FIV_RET_ERR_PARA, "null a");
        CHECK(fiv_vec_dot(&d, a, NULL) == FIV_RET_ERR_PARA, "null b");
        CHECK(fiv_vec_dot(&d, a, b64) == FIV_RET_ERR_PARA, "dtype mismatch");
        CHECK(fiv_vec_dot(&d, a, b2) == FIV_RET_ERR_PARA, "length mismatch");
        fiv_release_tensor1d(&a);
        fiv_release_tensor1d(&b);
        fiv_release_tensor1d(&b64);
        fiv_release_tensor1d(&b2);

        /* unsupported dtype (8U) */
        fiv_vec* u8a = fiv_create_tensor1d(3, FIV_8U1);
        fiv_vec* u8b = fiv_create_tensor1d(3, FIV_8U1);
        CHECK(fiv_vec_dot(&d, u8a, u8b) == FIV_RET_ERR_NOT_SUPPORT, "8U not supported");
        fiv_release_tensor1d(&u8a);
        fiv_release_tensor1d(&u8b);
    }

    printf("  [ok] fiv_vec_dot (32F/64F, mixed, zero, errors)\n");
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
    printf("=== fiv_vec_norm ===\n");
    test_vec_norm();
    printf("=== fiv_vec_axpy ===\n");
    test_vec_axpy();
    printf("=== fiv_vec_dot ===\n");
    test_vec_dot();
    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
