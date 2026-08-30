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

/* Correctness tests for the PDLP vector/matrix operator layer
 * (api/fiv_lp_vec.h, api/fiv_lp_mat.h).
 * Every check compares the library result against an INDEPENDENT dense
 * reference (plain ivf64 math), so a wrong implementation fails rather than a
 * test that only re-checks its own logic. */

#include "fiv_lp_vec.h"
#include "fiv_lp_mat.h"
#include "fiv_sp_matrix.h"
#include "fiv_ctensor.h"   /* fiv_create_tensor1d / fiv_release_tensor1d */
#include "fiv_matrix.h"    /* fiv_create_tensor2d / fiv_release_tensor2d */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (cond) { g_pass++; }                                                 \
        else { g_fail++; printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); } \
    } while (0)

static int vec_close(const fiv_vec *result_vec, const ivf64 *reference, size_t length, ivf64 tol)
{
    if (result_vec->length != length) return 0;
    const ivf64 *data = result_vec->data.db;
    for (size_t index = 0; index < length; index++)
        if (fabs(data[index] - reference[index]) > tol * (1.0 + fabs(reference[index]))) return 0;
    return 1;
}


/* ---------- shared dense reference (ROWS x COLS) ---------- */
static const int ROWS = 5, COLS = 4;
/* nonzeros: (0,0)=1 (0,2)=2 (1,1)=3 (1,3)=4 (2,0)=5 (2,1)=6 (2,2)=7 (3,3)=8 */
static const int COO_ROW[] = {0,0, 1,1, 2,2,2, 3};
static const int COO_COL[] = {0,2, 1,3, 0,1,2, 3};
static const ivf64 COO_VAL[] = {1.0,2.0, 3.0,4.0, 5.0,6.0,7.0, 8.0};
static const size_t COO_NNZ = sizeof(COO_ROW) / sizeof(COO_ROW[0]);

static void build_dense(ivf64 dense[ROWS][COLS])
{
    for (int row = 0; row < ROWS; row++)
        for (int col = 0; col < COLS; col++)
            dense[row][col] = 0.0;
    for (size_t entry = 0; entry < COO_NNZ; entry++)
        dense[COO_ROW[entry]][COO_COL[entry]] = COO_VAL[entry];
}

static void dense_matvec(const ivf64 dense[ROWS][COLS], const ivf64 *input, ivf64 *output)
{
    for (int row = 0; row < ROWS; row++) {
        ivf64 sum = 0.0;
        for (int col = 0; col < COLS; col++) sum += dense[row][col] * input[col];
        output[row] = sum;
    }
}
static void dense_matvec_transpose(const ivf64 dense[ROWS][COLS], const ivf64 *input, ivf64 *output)
{
    for (int col = 0; col < COLS; col++) {
        ivf64 sum = 0.0;
        for (int row = 0; row < ROWS; row++) sum += dense[row][col] * input[row];
        output[col] = sum;
    }
}

static fiv_vec *make_vec(size_t length, const ivf64 *init)
{
    fiv_vec *vec = fiv_create_tensor1d(length, FIV_64F1);
    if (vec == NULL) return NULL;
    if (init) memcpy(vec->data.db, init, length * sizeof(ivf64));
    else memset(vec->data.db, 0, length * sizeof(ivf64));
    return vec;
}

/* write one element of a dense fiv_mat (byte strides) */
static void set_dense(fiv_mat *dense_matrix, int row, int col, ivf64 value)
{
    const size_t element_bytes = dense_matrix->element_bytes;
    const size_t flat = (size_t)row * dense_matrix->strides[0] / element_bytes
                      + (size_t)col * dense_matrix->strides[1] / element_bytes;
    dense_matrix->data.db[flat] = value;
}


int main(void)
{
    fiv_sparse_runtime_init();

    ivf64 dense[ROWS][COLS];
    build_dense(dense);

    /* ============================================================
     * 1. fiv_vec_clamp  (box projection, incl. unbounded sides)
     * ============================================================ */
    {
        const int n = 4;
        ivf64 xval[4]    = { -2.0, 0.5, 3.0, INFINITY };
        ivf64 lower[4]   = { -1.0, -1.0, -1.0, -INFINITY };
        ivf64 upper[4]   = {  1.0,  1.0,  1.0,  INFINITY };
        ivf64 ref[4];
        for (int i = 0; i < n; i++) {
            ivf64 v = xval[i];
            if (v < lower[i]) v = lower[i];
            if (v > upper[i]) v = upper[i];
            ref[i] = v;
        }
        fiv_vec *x = make_vec(n, xval);
        fiv_vec *lo = make_vec(n, lower);
        fiv_vec *hi = make_vec(n, upper);
        fiv_vec *dst = make_vec(n, NULL);
        fiv_ret rc = fiv_vec_clamp(dst, x, lo, hi);
        CHECK(rc == FIV_RET_OK, "clamp OK");
        CHECK(vec_close(dst, ref, n, 1e-15), "clamp matches reference (unbounded sides)");
        /* in-place: dst aliases x */
        fiv_vec *inplace = make_vec(n, xval);
        fiv_ret rc2 = fiv_vec_clamp(inplace, inplace, lo, hi);
        CHECK(rc2 == FIV_RET_OK, "clamp in-place OK");
        CHECK(vec_close(inplace, ref, n, 1e-15), "clamp in-place matches reference");
        fiv_release_tensor1d(&x); fiv_release_tensor1d(&lo); fiv_release_tensor1d(&hi);
        fiv_release_tensor1d(&dst); fiv_release_tensor1d(&inplace);
    }

    /* ============================================================
     * 2. fiv_vec_sum_finite_products  (skip non-finite in values)
     * ============================================================ */
    {
        const int n = 5;
        ivf64 values[5]     = { 1.0, INFINITY, 2.0, -INFINITY, NAN };
        ivf64 multipliers[5] = { 10.0, 20.0, 30.0, 40.0, 50.0 };
        ivf64 ref = 0.0;
        for (int i = 0; i < n; i++)
            if (isfinite(values[i])) ref += values[i] * multipliers[i];   /* 1*10 + 2*30 = 70 */
        fiv_vec *values_vec = make_vec(n, values);
        fiv_vec *multiplier_vec = make_vec(n, multipliers);
        ivf64 acc = -1.0;
        fiv_ret rc = fiv_vec_sum_finite_products(values_vec, multiplier_vec, &acc);
        CHECK(rc == FIV_RET_OK, "sum_finite_products OK");
        CHECK(fabs(acc - ref) < 1e-15, "sum_finite_products skips inf/nan");
        fiv_release_tensor1d(&values_vec); fiv_release_tensor1d(&multiplier_vec);
    }

    /* ============================================================
     * 3. LpMatrix matvec  (dense + sparse, both directions)
     * ============================================================ */
    {
        ivf64 input_x[COLS];
        for (int col = 0; col < COLS; col++) input_x[col] = 0.5 * col + 1.0;   /* 1,1.5,2,2.5 */
        ivf64 input_y[ROWS];
        for (int row = 0; row < ROWS; row++) input_y[row] = 0.7 * row - 0.3;

        ivf64 ref_y[ROWS], ref_yt[COLS];
        dense_matvec(dense, input_x, ref_y);
        dense_matvec_transpose(dense, input_y, ref_yt);

        /* --- dense backend --- */
        fiv_mat *dense_matrix = fiv_create_tensor2d((size_t[2]){ROWS, COLS}, FIV_64F1);
        CHECK(dense_matrix != NULL, "fiv_create_tensor2d OK");
        for (int row = 0; row < ROWS; row++)
            for (int col = 0; col < COLS; col++)
                set_dense(dense_matrix, row, col, dense[row][col]);
        fiv_lp_mat *M_dense = fiv_lp_mat_wrap_dense(dense_matrix);
        CHECK(M_dense != NULL, "wrap_dense OK");
        {
            fiv_vec *x = make_vec(COLS, input_x);
            fiv_vec *y = make_vec(ROWS, NULL);
            fiv_ret rc = fiv_lp_mat_matvec(y, M_dense, x, 0);
            CHECK(rc == FIV_RET_OK, "dense matvec(transpose=0) OK");
            CHECK(vec_close(y, ref_y, ROWS, 1e-12), "dense matvec matches reference");
            fiv_release_tensor1d(&x); fiv_release_tensor1d(&y);
        }
        {
            fiv_vec *y = make_vec(ROWS, input_y);
            fiv_vec *z = make_vec(COLS, NULL);
            fiv_ret rc = fiv_lp_mat_matvec(z, M_dense, y, 1);
            CHECK(rc == FIV_RET_OK, "dense matvec(transpose=1) OK");
            CHECK(vec_close(z, ref_yt, COLS, 1e-12), "dense matvec-transpose matches reference");
            fiv_release_tensor1d(&y); fiv_release_tensor1d(&z);
        }

        /* --- sparse backend (CSR) --- */
        fiv_sparse_mat *K = fiv_create_sp_matrix_from_coo(COO_ROW, COO_COL, COO_VAL,
                                                          FIV_64F1, COO_NNZ, ROWS, COLS);
        fiv_lp_mat *M_sparse = fiv_lp_mat_wrap_sparse(K);
        CHECK(M_sparse != NULL, "wrap_sparse OK");
        {
            fiv_vec *x = make_vec(COLS, input_x);
            fiv_vec *y = make_vec(ROWS, NULL);
            fiv_ret rc = fiv_lp_mat_matvec(y, M_sparse, x, 0);
            CHECK(rc == FIV_RET_OK, "sparse matvec(transpose=0) OK");
            CHECK(vec_close(y, ref_y, ROWS, 1e-12), "sparse matvec matches reference");
            fiv_release_tensor1d(&x); fiv_release_tensor1d(&y);
        }
        /* sparse transpose direction needs a CSC */
        fiv_sparse_mat *KT = NULL;
        fiv_ret rct = fiv_sparse_transpose(&KT, K);
        CHECK(rct == FIV_RET_OK && KT != NULL, "sparse transpose -> CSC OK");
        fiv_lp_mat *M_sparseT = fiv_lp_mat_wrap_sparse(KT);
        {
            fiv_vec *y = make_vec(ROWS, input_y);
            fiv_vec *z = make_vec(COLS, NULL);
            fiv_ret rc = fiv_lp_mat_matvec(z, M_sparseT, y, 1);
            CHECK(rc == FIV_RET_OK, "sparse matvec(transpose=1 via CSC) OK");
            CHECK(vec_close(z, ref_yt, COLS, 1e-12), "sparse matvec-transpose matches reference");
            fiv_release_tensor1d(&y); fiv_release_tensor1d(&z);
        }

        fiv_release_lp_mat(&M_dense);
        fiv_release_lp_mat(&M_sparse);
        fiv_release_lp_mat(&M_sparseT);
        fiv_release_tensor2d(&dense_matrix);
        fiv_release_sp_matrix(&K);
        fiv_release_sp_matrix(&KT);
        CHECK(M_dense == NULL && M_sparse == NULL && M_sparseT == NULL,
              "lp_mat release nulls pointers");
    }

    /* ============================================================
     * 4. LpMatrix reduce_abs_max  (dense + sparse, dim 0/1)
     * ============================================================ */
    {
        ivf64 ref_row_max[ROWS], ref_col_max[COLS];
        for (int row = 0; row < ROWS; row++) {
            ivf64 m = 0.0; for (int col = 0; col < COLS; col++) m = fmax(m, fabs(dense[row][col]));
            ref_row_max[row] = m;
        }
        for (int col = 0; col < COLS; col++) {
            ivf64 m = 0.0; for (int row = 0; row < ROWS; row++) m = fmax(m, fabs(dense[row][col]));
            ref_col_max[col] = m;
        }

        fiv_mat *dense_matrix = fiv_create_tensor2d((size_t[2]){ROWS, COLS}, FIV_64F1);
        for (int row = 0; row < ROWS; row++)
            for (int col = 0; col < COLS; col++) set_dense(dense_matrix, row, col, dense[row][col]);
        fiv_lp_mat *M_dense = fiv_lp_mat_wrap_dense(dense_matrix);

        fiv_sparse_mat *K = fiv_create_sp_matrix_from_coo(COO_ROW, COO_COL, COO_VAL,
                                                          FIV_64F1, COO_NNZ, ROWS, COLS);
        fiv_lp_mat *M_sparse = fiv_lp_mat_wrap_sparse(K);
        /* per-column (dim=1) reduction needs the column-major CSC view */
        fiv_sparse_mat *KT = NULL;
        fiv_sparse_transpose(&KT, K);
        fiv_lp_mat *M_sparseT = fiv_lp_mat_wrap_sparse(KT);

        fiv_vec *dst = make_vec(ROWS, NULL);
        fiv_ret rc = fiv_lp_mat_reduce_abs_max(dst, M_dense, 0);
        CHECK(rc == FIV_RET_OK, "dense reduce_abs_max dim=0 OK");
        CHECK(vec_close(dst, ref_row_max, ROWS, 1e-15), "dense reduce_abs_max per-row matches");
        fiv_vec *dstc = make_vec(COLS, NULL);
        fiv_ret rcc = fiv_lp_mat_reduce_abs_max(dstc, M_dense, 1);
        CHECK(rcc == FIV_RET_OK, "dense reduce_abs_max dim=1 OK");
        CHECK(vec_close(dstc, ref_col_max, COLS, 1e-15), "dense reduce_abs_max per-col matches");

        fiv_ret rcs = fiv_lp_mat_reduce_abs_max(dst, M_sparse, 0);
        CHECK(rcs == FIV_RET_OK, "sparse reduce_abs_max dim=0 OK");
        CHECK(vec_close(dst, ref_row_max, ROWS, 1e-15), "sparse reduce_abs_max per-row matches");
        fiv_ret rcs2 = fiv_lp_mat_reduce_abs_max(dstc, M_sparseT, 1);
        CHECK(rcs2 == FIV_RET_OK, "sparse reduce_abs_max dim=1 OK");
        CHECK(vec_close(dstc, ref_col_max, COLS, 1e-15), "sparse reduce_abs_max per-col matches");

        fiv_release_lp_mat(&M_dense);
        fiv_release_lp_mat(&M_sparse);
        fiv_release_lp_mat(&M_sparseT);
        fiv_release_tensor2d(&dense_matrix);
        fiv_release_sp_matrix(&K);
        fiv_release_sp_matrix(&KT);
        fiv_release_tensor1d(&dst); fiv_release_tensor1d(&dstc);
    }

    /* ============================================================
     * 5. LpMatrix reduce_abs_pow  (exponent 1 and 2; dense + sparse)
     * ============================================================ */
    {
        ivf64 ref_row1[ROWS], ref_col1[COLS], ref_row2[ROWS], ref_col2[COLS];
        for (int row = 0; row < ROWS; row++) {
            ivf64 s1 = 0.0, s2 = 0.0;
            for (int col = 0; col < COLS; col++) {
                ivf64 a = fabs(dense[row][col]);
                s1 += a; s2 += a * a;
            }
            ref_row1[row] = s1; ref_row2[row] = s2;
        }
        for (int col = 0; col < COLS; col++) {
            ivf64 s1 = 0.0, s2 = 0.0;
            for (int row = 0; row < ROWS; row++) {
                ivf64 a = fabs(dense[row][col]);
                s1 += a; s2 += a * a;
            }
            ref_col1[col] = s1; ref_col2[col] = s2;
        }

        fiv_mat *dense_matrix = fiv_create_tensor2d((size_t[2]){ROWS, COLS}, FIV_64F1);
        for (int row = 0; row < ROWS; row++)
            for (int col = 0; col < COLS; col++) set_dense(dense_matrix, row, col, dense[row][col]);
        fiv_lp_mat *M_dense = fiv_lp_mat_wrap_dense(dense_matrix);

        fiv_sparse_mat *K = fiv_create_sp_matrix_from_coo(COO_ROW, COO_COL, COO_VAL,
                                                          FIV_64F1, COO_NNZ, ROWS, COLS);
        fiv_lp_mat *M_sparse = fiv_lp_mat_wrap_sparse(K);
        fiv_sparse_mat *KT = NULL;
        fiv_sparse_transpose(&KT, K);
        fiv_lp_mat *M_sparseT = fiv_lp_mat_wrap_sparse(KT);

        fiv_vec *dst = make_vec(ROWS, NULL);
        fiv_vec *dstc = make_vec(COLS, NULL);

        /* exponent == 1 */
        fiv_ret rc = fiv_lp_mat_reduce_abs_pow(dst, M_dense, 0, 1.0);
        CHECK(rc == FIV_RET_OK, "dense reduce_abs_pow(1) dim=0 OK");
        CHECK(vec_close(dst, ref_row1, ROWS, 1e-12), "dense |a|^1 per-row matches");
        fiv_lp_mat_reduce_abs_pow(dstc, M_dense, 1, 1.0);
        CHECK(vec_close(dstc, ref_col1, COLS, 1e-12), "dense |a|^1 per-col matches");
        fiv_lp_mat_reduce_abs_pow(dst, M_sparse, 0, 1.0);
        CHECK(vec_close(dst, ref_row1, ROWS, 1e-12), "sparse |a|^1 per-row matches");
        fiv_lp_mat_reduce_abs_pow(dstc, M_sparseT, 1, 1.0);
        CHECK(vec_close(dstc, ref_col1, COLS, 1e-12), "sparse |a|^1 per-col matches");

        /* exponent == 2 (Pock-Chambolle uses 2-alpha / alpha) */
        fiv_lp_mat_reduce_abs_pow(dst, M_dense, 0, 2.0);
        CHECK(vec_close(dst, ref_row2, ROWS, 1e-12), "dense |a|^2 per-row matches");
        fiv_lp_mat_reduce_abs_pow(dstc, M_dense, 1, 2.0);
        CHECK(vec_close(dstc, ref_col2, COLS, 1e-12), "dense |a|^2 per-col matches");
        fiv_lp_mat_reduce_abs_pow(dst, M_sparse, 0, 2.0);
        CHECK(vec_close(dst, ref_row2, ROWS, 1e-12), "sparse |a|^2 per-row matches");
        fiv_lp_mat_reduce_abs_pow(dstc, M_sparseT, 1, 2.0);
        CHECK(vec_close(dstc, ref_col2, COLS, 1e-12), "sparse |a|^2 per-col matches");

        fiv_release_lp_mat(&M_dense);
        fiv_release_lp_mat(&M_sparse);
        fiv_release_lp_mat(&M_sparseT);
        fiv_release_tensor2d(&dense_matrix);
        fiv_release_sp_matrix(&K);
        fiv_release_sp_matrix(&KT);
        fiv_release_tensor1d(&dst); fiv_release_tensor1d(&dstc);
    }

    /* ============================================================
     * 6. fiv_create_lp_mat_from_coo owns its sparse matrix (release frees both)
     * ============================================================ */
    {
        fiv_lp_mat *M = fiv_create_lp_mat_from_coo(COO_ROW, COO_COL, COO_VAL,
                                                  FIV_64F1, COO_NNZ, ROWS, COLS);
        CHECK(M != NULL && M->owns_data == 1, "create_lp_mat_from_coo owns data");
        fiv_vec *x = make_vec(COLS, (ivf64[4]){1,1,1,1});
        fiv_vec *y = make_vec(ROWS, NULL);
        ivf64 ref_full[ROWS];   /* A * ones */
        ivf64 ones[COLS]; for (int c = 0; c < COLS; c++) ones[c] = 1.0;
        dense_matvec(dense, ones, ref_full);
        fiv_ret rc = fiv_lp_mat_matvec(y, M, x, 0);
        CHECK(rc == FIV_RET_OK, "owning lp_mat matvec OK");
        CHECK(vec_close(y, ref_full, ROWS, 1e-12), "owning lp_mat matvec matches");
        fiv_release_lp_mat(&M);
        CHECK(M == NULL, "owning lp_mat release nulls pointer");
        fiv_release_tensor1d(&x); fiv_release_tensor1d(&y);
    }

    printf("\nPASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
