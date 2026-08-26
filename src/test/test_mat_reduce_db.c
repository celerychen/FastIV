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

/* Correctness tests for fiv_matrix_reduce_sum with 64-bit (FIV_64F1) operands,
 * exercised through the generic fiv_matrix_reduce_sum (api/fiv_matrix.h):
 * dim 0 (column sums over rows), dim 1 (row sums over cols), dim -1 (total
 * scalar sum), beta accumulation, and the error paths. */

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

/* build an fp64 scalar with value v */
static fiv_scalar mk_db(double v)
{
    fiv_scalar s;
    s.id = FIV_ID_SCALAR;
    s.dtype = FIV_64F1;
    s.data.value_fp64 = v;
    return s;
}

static void test_matrix_reduce_sum(void)
{
    /* src = [[1,2,3],[4,5,6]] (2x3) */
    size_t sh[2] = { 2, 3 };
    fiv_mat* src = fiv_create_tensor2d(sh, FIV_64F1);
    CHECK(src != NULL, "alloc ok");
    double m[6] = { 1, 2, 3, 4, 5, 6 };
    memcpy(src->data.db, m, sizeof(m));

    /* dim == 0: sum over rows -> per-column sums [1+4, 2+5, 3+6] = [5,7,9] */
    {
        fiv_vec* dst = fiv_create_tensor1d(3, FIV_64F1);
        CHECK(dst != NULL, "alloc dim0 dst");
        CHECK(fiv_matrix_reduce_sum(dst, src, 0, mk_db(0.0)) == FIV_RET_OK, "reduce dim0 OK");
        double exp[3] = { 5, 7, 9 };
        int bad = 0;
        for (int k = 0; k < 3; k++) if (fabs_local(dst->data.db[k] - exp[k]) > 1e-9) bad++;
        CHECK(bad == 0, "dim0: column sums correct");
        fiv_release_tensor1d(&dst);
    }

    /* accumulation: beta = 1.0 accumulates onto a pre-seeded dst (bias-gradient path) */
    {
        fiv_vec* dst = fiv_create_tensor1d(3, FIV_64F1);
        CHECK(dst != NULL, "alloc acc dst");
        dst->data.db[0] = 1.0; dst->data.db[1] = 2.0; dst->data.db[2] = 3.0;  /* seed */
        CHECK(fiv_matrix_reduce_sum(dst, src, 0, mk_db(1.0)) == FIV_RET_OK, "reduce dim0 acc OK");
        double exp[3] = { 6, 9, 12 };   /* seed [1,2,3] + column sums [5,7,9] */
        int bad = 0;
        for (int k = 0; k < 3; k++) if (fabs_local(dst->data.db[k] - exp[k]) > 1e-9) bad++;
        CHECK(bad == 0, "dim0 acc: seed + column sums");
        fiv_release_tensor1d(&dst);
    }

    /* dim == 1: sum over cols -> per-row sums [1+2+3, 4+5+6] = [6,15] */
    {
        fiv_vec* dst = fiv_create_tensor1d(2, FIV_64F1);
        CHECK(dst != NULL, "alloc dim1 dst");
        CHECK(fiv_matrix_reduce_sum(dst, src, 1, mk_db(0.0)) == FIV_RET_OK, "reduce dim1 OK");
        double exp[2] = { 6, 15 };
        int bad = 0;
        for (int k = 0; k < 2; k++) if (fabs_local(dst->data.db[k] - exp[k]) > 1e-9) bad++;
        CHECK(bad == 0, "dim1: row sums correct");
        fiv_release_tensor1d(&dst);
    }

    /* dim == -1: total sum = 21 (scalar) */
    {
        fiv_scalar* sc = (fiv_scalar*)malloc(sizeof(fiv_scalar));
        sc->id = FIV_ID_SCALAR;
        sc->dtype = FIV_64F1;
        sc->data.value_fp64 = 0.0;
        CHECK(fiv_matrix_reduce_sum(sc, src, -1, mk_db(0.0)) == FIV_RET_OK, "reduce dim-1 OK");
        CHECK(fabs_local(sc->data.value_fp64 - 21.0) < 1e-9, "dim-1: total sum = 21 (scalar)");
        free(sc);
    }

    /* error paths: null args, type/length mismatch, bad dim */
    {
        fiv_vec* v3 = fiv_create_tensor1d(3, FIV_64F1);   /* OK for dim0, wrong for dim1 */
        fiv_vec* v2 = fiv_create_tensor1d(2, FIV_64F1);   /* OK for dim1, wrong for dim0 */
        CHECK(v3 != NULL && v2 != NULL, "alloc error-path vecs");

        CHECK(fiv_matrix_reduce_sum(NULL, src, 0, mk_db(0.0)) == FIV_RET_ERR_PARA, "null dst");
        CHECK(fiv_matrix_reduce_sum(v3, NULL, 0, mk_db(0.0)) == FIV_RET_ERR_PARA, "null src");
        /* dim0 needs dst length == cols(3); v2 has length 2 -> mismatch */
        CHECK(fiv_matrix_reduce_sum(v2, src, 0, mk_db(0.0)) == FIV_RET_ERR_PARA, "dim0 dst length mismatch");
        /* dim1 needs dst length == rows(2); v3 has length 3 -> mismatch */
        CHECK(fiv_matrix_reduce_sum(v3, src, 1, mk_db(0.0)) == FIV_RET_ERR_PARA, "dim1 dst length mismatch");
        /* dim-1 requires a scalar dst; a fiv_vec is not accepted */
        CHECK(fiv_matrix_reduce_sum(v3, src, -1, mk_db(0.0)) == FIV_RET_ERR_PARA, "dim-1 dst is not scalar");
        /* illegal dim */
        CHECK(fiv_matrix_reduce_sum(v3, src, 2, mk_db(0.0)) == FIV_RET_ERR_PARA, "bad dim value");

        /* valid scalar dst for dim-1 */
        fiv_scalar* sc = (fiv_scalar*)malloc(sizeof(fiv_scalar));
        sc->id = FIV_ID_SCALAR;
        sc->dtype = FIV_64F1;
        sc->data.value_fp64 = 0.0;
        CHECK(fiv_matrix_reduce_sum(sc, src, -1, mk_db(0.0)) == FIV_RET_OK, "dim-1 scalar dst OK");
        free(sc);

        /* beta must be an fp64 scalar; a non-fp64 (fp32) scalar is rejected */
        FIV_DECLAR_SCALAR_FP32(bad_beta);
        bad_beta.data.value_fp32 = 1.0f;
        CHECK(fiv_matrix_reduce_sum(v3, src, 0, bad_beta) == FIV_RET_ERR_NOT_SUPPORT, "non-fp64 beta rejected");

        fiv_release_tensor1d(&v3);
        fiv_release_tensor1d(&v2);
    }

    fiv_release_tensor2d(&src);
    printf("  [ok] fiv_matrix_reduce_sum (dim0 col-sum / dim1 row-sum / dim-1 total / errors)\n");
}

int main(void)
{
    printf("=== fiv_matrix_reduce_sum (FIV_64F1, via generic api) ===\n");
    test_matrix_reduce_sum();
    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
