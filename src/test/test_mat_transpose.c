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

/* Correctness tests for fiv_matrix_transpose (api/fiv_matrix.h).
 * Exercises rectangular/square transposes, metadata rewrite, and the error
 * paths (in-place, unsupported dtype, null args, non-contiguous, small dst). */

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

/* transpose: tr[j][i] = src[i][j], rows -> cols */
static void run_transpose(int rows, int cols, const char* name)
{
    size_t sh_src[2] = { (size_t)rows, (size_t)cols };
    size_t sh_dst[2] = { (size_t)cols, (size_t)rows };
    fiv_mat* src = fiv_create_tensor2d(sh_src, FIV_32F1);
    fiv_mat* dst = fiv_create_tensor2d(sh_dst, FIV_32F1);
    CHECK(src != NULL && dst != NULL, "alloc ok");

    unsigned s = 12345u;
    for (int i = 0; i < rows * cols; i++) {
        s = s * 1103515245u + 12345u;
        src->data.fl[i] = (float)((s >> 8) & 0xffff) / 4096.f - 4.f;
    }

    fiv_ret r = fiv_matrix_transpose(dst, src);
    CHECK(r == FIV_RET_OK, "transpose returns OK");

    int bad = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float got = dst->data.fl[j * rows + i];
            float exp = src->data.fl[i * cols + j];
            if (fabsf_local(got - exp) > 1e-3f) bad++;
        }
    }
    CHECK(bad == 0, "all elements transposed correctly");
    CHECK(dst->shapes[0] == (size_t)cols && dst->shapes[1] == (size_t)rows,
          "dst shape rewritten to cols x rows");

    fiv_release_tensor2d(&src);
    fiv_release_tensor2d(&dst);
    printf("  [ok] %s (%dx%d)\n", name, rows, cols);
}

static void test_error_paths(void)
{
    size_t sh[2] = { 3, 4 };
    fiv_mat* a = fiv_create_tensor2d(sh, FIV_32F1);
    fiv_mat* c = fiv_create_tensor2d(sh, FIV_32F1);
    CHECK(a != NULL && c != NULL, "alloc ok");

    CHECK(fiv_matrix_transpose(NULL, a) == FIV_RET_ERR_PARA, "null dst");
    CHECK(fiv_matrix_transpose(c, NULL) == FIV_RET_ERR_PARA, "null src");
    CHECK(fiv_matrix_transpose(a, a) == FIV_RET_ERR_PARA, "in-place (dst==src)");

    /* non-contiguous flagged tensors are rejected */
    fiv_mat* nc = fiv_create_tensor2d(sh, FIV_32F1);
    nc->data_continue = 0;
    CHECK(fiv_matrix_transpose(c, nc) == FIV_RET_ERR_PARA, "non-contiguous src");
    nc->data_continue = 1;

    /* unsupported dtype (8U) */
    fiv_mat* u8 = fiv_create_tensor2d(sh, FIV_8U1);
    fiv_mat* u8b = fiv_create_tensor2d(sh, FIV_8U1);
    CHECK(fiv_matrix_transpose(u8b, u8) == FIV_RET_ERR_NOT_SUPPORT, "8U not supported");
    fiv_release_tensor2d(&u8);
    fiv_release_tensor2d(&u8b);

    /* dst buffer too small for rows*cols */
    size_t sh_small[2] = { 2, 2 };   /* 4 elems < 12 needed */
    fiv_mat* small = fiv_create_tensor2d(sh_small, FIV_32F1);
    CHECK(fiv_matrix_transpose(small, a) == FIV_RET_ERR_PARA, "dst buffer too small");
    fiv_release_tensor2d(&small);

    fiv_release_tensor2d(&nc);
    fiv_release_tensor2d(&a);
    fiv_release_tensor2d(&c);
}

int main(void)
{
    printf("=== fiv_matrix_transpose ===\n");
    run_transpose(3, 4, "3x4 -> 4x3");
    run_transpose(5, 2, "5x2 -> 2x5");
    run_transpose(8, 8, "8x8 square");
    run_transpose(1, 7, "1x7 -> 7x1");
    test_error_paths();
    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
