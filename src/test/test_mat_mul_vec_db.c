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

/* Correctness tests for the generic matrix-vector API with 64-bit (FIV_64F1)
 * operands, exercised through api/fiv_matrix.h:
 *  - fiv_matrix_mul_vec: dst = mat * vec (transpose == 0) and dst = mat^T * vec
 *    (transpose != 0), plus error paths (dtype, in-place, null, short args);
 *  - fiv_matrix_add_vec: broadcast-add a vector over rows (dim == 0) or columns
 *    (dim == 1), in-place and error paths. */

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
static double fabs_local(double x) { return x < 0 ? -x : x; }

/* dst = mat * vec or dst = mat^T * vec, reference (double) */
static void ref_mul_vec(int transpose, int rows, int cols,
                        const double* mat, const double* vec, double* dst)
{
    int out_len = transpose ? cols : rows;
    int k_len   = transpose ? rows : cols;
    for (int i = 0; i < out_len; i++) {
        double acc = 0.0;
        for (int k = 0; k < k_len; k++) {
            double m = transpose ? mat[k * cols + i] : mat[i * cols + k];
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
    fiv_mat* mat = fiv_create_tensor2d(sh_mat, FIV_64F1);
    fiv_vec* vec = fiv_create_tensor1d((size_t)vec_len, FIV_64F1);
    fiv_vec* dst = fiv_create_tensor1d((size_t)out_len, FIV_64F1);
    CHECK(mat != NULL && vec != NULL && dst != NULL, "alloc ok");

    unsigned s = 54321u;
    for (int i = 0; i < rows * cols; i++) {
        s = s * 1103515245u + 12345u;
        mat->data.db[i] = (double)((s >> 8) & 0xffff) / 4096.0 - 4.0;
    }
    for (int i = 0; i < vec_len; i++) {
        s = s * 1103515245u + 12345u;
        vec->data.db[i] = (double)((s >> 8) & 0xffff) / 4096.0 - 4.0;
    }

    double* exp = (double*)malloc((size_t)out_len * sizeof(double));
    ref_mul_vec(transpose, rows, cols, mat->data.db, vec->data.db, exp);

    fiv_ret r = fiv_matrix_mul_vec(dst, mat, vec, transpose);
    CHECK(r == FIV_RET_OK, "mul_vec returns OK");

    int bad = 0;
    for (int i = 0; i < out_len; i++) {
        if (fabs_local(dst->data.db[i] - exp[i]) > 1e-9 * (1.0 + fabs_local(exp[i]))) bad++;
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
    fiv_mat* mat = fiv_create_tensor2d(sh, FIV_64F1);
    fiv_vec* vec = fiv_create_tensor1d(3, FIV_64F1);
    fiv_vec* dst = fiv_create_tensor1d(2, FIV_64F1);
    CHECK(mat != NULL && vec != NULL && dst != NULL, "alloc ok");

    CHECK(fiv_matrix_mul_vec(NULL, mat, vec, 0) == FIV_RET_ERR_PARA, "null dst");
    CHECK(fiv_matrix_mul_vec(dst, NULL, vec, 0) == FIV_RET_ERR_PARA, "null mat");
    CHECK(fiv_matrix_mul_vec(dst, mat, NULL, 0) == FIV_RET_ERR_PARA, "null vec");
    CHECK(fiv_matrix_mul_vec(vec, mat, vec, 0) == FIV_RET_ERR_PARA, "in-place dst==vec");

    /* unsupported dtype: 8U mat */
    fiv_mat* u8 = fiv_create_tensor2d(sh, FIV_8U1);
    CHECK(fiv_matrix_mul_vec(dst, u8, vec, 0) == FIV_RET_ERR_NOT_SUPPORT, "8U mat not supported");
    fiv_release_tensor2d(&u8);

    /* 64F dst/vec but 32F mat: rejected as non-fp64 operand */
    fiv_mat* m32 = fiv_create_tensor2d(sh, FIV_32F1);
    CHECK(fiv_matrix_mul_vec(dst, m32, vec, 0) == FIV_RET_ERR_NOT_SUPPORT, "32F mat not supported");
    fiv_release_tensor2d(&m32);

    /* vector too short for mat*vec (needs cols == 3) */
    fiv_vec* short_v = fiv_create_tensor1d(2, FIV_64F1);
    CHECK(fiv_matrix_mul_vec(dst, mat, short_v, 0) == FIV_RET_ERR_PARA, "vec too short");
    fiv_release_tensor1d(&short_v);

    /* dst too short for mat*vec (needs rows == 2) */
    fiv_vec* short_d = fiv_create_tensor1d(1, FIV_64F1);
    CHECK(fiv_matrix_mul_vec(short_d, mat, vec, 0) == FIV_RET_ERR_PARA, "dst too short");
    fiv_release_tensor1d(&short_d);

    fiv_release_tensor2d(&mat);
    fiv_release_tensor1d(&vec);
    fiv_release_tensor1d(&dst);
}

/* dst = src + broadcast(vec) over rows (dim == 0) or columns (dim == 1).
   Reference (double). */
static void ref_add_vec(int dim, int rows, int cols,
                        const double* src, const double* vec, double* dst)
{
    if (dim == 0) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                dst[i * cols + j] = src[i * cols + j] + vec[j];
    } else {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                dst[i * cols + j] = src[i * cols + j] + vec[i];
    }
}

static void run_add_vec(int dim, int rows, int cols, const char* name)
{
    size_t sh[2] = { (size_t)rows, (size_t)cols };
    fiv_mat* src = fiv_create_tensor2d(sh, FIV_64F1);
    fiv_mat* dst = fiv_create_tensor2d(sh, FIV_64F1);
    fiv_vec* vec = fiv_create_tensor1d((size_t)(dim == 0 ? cols : rows), FIV_64F1);
    CHECK(src != NULL && dst != NULL && vec != NULL, "alloc ok");

    unsigned s = 98765u;
    for (int i = 0; i < rows * cols; i++) {
        s = s * 1103515245u + 12345u;
        src->data.db[i] = (double)((s >> 8) & 0xffff) / 4096.0 - 4.0;
    }
    int vec_len = (dim == 0) ? cols : rows;
    for (int i = 0; i < vec_len; i++) {
        s = s * 1103515245u + 12345u;
        vec->data.db[i] = (double)((s >> 8) & 0xffff) / 4096.0 - 4.0;
    }

    double* exp = (double*)malloc((size_t)rows * cols * sizeof(double));
    ref_add_vec(dim, rows, cols, src->data.db, vec->data.db, exp);

    fiv_ret r = fiv_matrix_add_vec(dst, src, vec, dim);
    CHECK(r == FIV_RET_OK, "add_vec returns OK");

    int bad = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (fabs_local(dst->data.db[i] - exp[i]) > 1e-9 * (1.0 + fabs_local(exp[i]))) bad++;
    }
    CHECK(bad == 0, "result matches reference");

    free(exp);
    fiv_release_tensor2d(&src);
    fiv_release_tensor2d(&dst);
    fiv_release_tensor1d(&vec);
    printf("  [ok] %s\n", name);
}

static void test_add_vec_error_paths(void)
{
    size_t sh[2] = { 2, 3 };
    fiv_mat* src = fiv_create_tensor2d(sh, FIV_64F1);
    fiv_vec* vec = fiv_create_tensor1d(3, FIV_64F1);
    fiv_mat* dst = fiv_create_tensor2d(sh, FIV_64F1);
    CHECK(src != NULL && vec != NULL && dst != NULL, "alloc ok");

    CHECK(fiv_matrix_add_vec(NULL, src, vec, 0) == FIV_RET_ERR_PARA, "null dst");
    CHECK(fiv_matrix_add_vec(dst, NULL, vec, 0) == FIV_RET_ERR_PARA, "null src");
    CHECK(fiv_matrix_add_vec(dst, src, NULL, 0) == FIV_RET_ERR_PARA, "null vec");

    /* in-place: dst aliases src is allowed for add_vec */
    CHECK(fiv_matrix_add_vec(src, src, vec, 0) == FIV_RET_OK, "in-place dst==src OK");

    /* unsupported dtype: 32F src with 64F dst/vec */
    fiv_mat* src32 = fiv_create_tensor2d(sh, FIV_32F1);
    CHECK(fiv_matrix_add_vec(dst, src32, vec, 0) == FIV_RET_ERR_NOT_SUPPORT, "32F src not supported");
    fiv_release_tensor2d(&src32);

    /* dim0: vec length must equal cols (3) -> length-2 vec mismatches */
    fiv_vec* v2 = fiv_create_tensor1d(2, FIV_64F1);
    CHECK(fiv_matrix_add_vec(dst, src, v2, 0) == FIV_RET_ERR_PARA, "dim0 vec length mismatch");
    /* dim1: vec length must equal rows (2); default vec length 3 mismatches */
    CHECK(fiv_matrix_add_vec(dst, src, vec, 1) == FIV_RET_ERR_PARA, "dim1 vec length mismatch");
    /* dim1 with length-2 vec is valid (rows == 2) */
    CHECK(fiv_matrix_add_vec(dst, src, v2, 1) == FIV_RET_OK, "dim1 with length-2 vec OK");
    /* bad dim value */
    CHECK(fiv_matrix_add_vec(dst, src, vec, 2) == FIV_RET_ERR_PARA, "bad dim value");

    fiv_release_tensor1d(&v2);
    fiv_release_tensor2d(&src);
    fiv_release_tensor2d(&dst);
    fiv_release_tensor1d(&vec);
}

int main(void)
{
    printf("=== fiv_matrix_mul_vec (FIV_64F1, via generic api) ===\n");
    run_mul_vec(0, 2, 3, "mat(2x3)*vec(3) -> vec(2)");
    run_mul_vec(1, 2, 3, "mat(2x3)^T*vec(2) -> vec(3)");
    run_mul_vec(0, 4, 4, "mat(4x4)*vec(4)");
    run_mul_vec(1, 4, 4, "mat(4x4)^T*vec(4)");
    run_mul_vec(0, 1, 5, "mat(1x5)*vec(5)");
    run_mul_vec(0, 7, 9, "mat(7x9)*vec(9) odd");
    run_mul_vec(1, 9, 7, "mat(9x7)^T*vec(9) odd");
    run_mul_vec(0, 100, 50, "mat(100x50)*vec(50) large");
    test_error_paths();

    printf("=== fiv_matrix_add_vec (FIV_64F1, via generic api) ===\n");
    run_add_vec(0, 2, 3, "add_vec dim0 (broadcast over rows)");
    run_add_vec(1, 2, 3, "add_vec dim1 (broadcast over cols)");
    run_add_vec(0, 4, 4, "add_vec dim0 square");
    run_add_vec(1, 4, 4, "add_vec dim1 square");
    run_add_vec(0, 5, 7, "add_vec dim0 odd rows/cols");
    run_add_vec(1, 3, 6, "add_vec dim1 odd rows/cols");
    test_add_vec_error_paths();

    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
