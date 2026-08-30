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

/* Correctness tests for the sparse matrix module (api/fiv_sp_matrix.h).
 * Every check compares the sparse result against an INDEPENDENT dense
 * reference (plain ivf64[][] math), so a wrong implementation fails rather
 * than a test that only re-checks its own logic. */

#include "fiv_sp_matrix.h"
#include "fiv_ctensor.h"   /* fiv_create_tensor1d / fiv_release_tensor1d */
#include "fiv_common.h"      /* fiv_malloc / fiv_free */

#include <stdio.h>
#include <stdlib.h>          /* NULL */
#include <string.h>
#include <math.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (cond) { g_pass++; }                                            \
        else { g_fail++; printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); } \
    } while (0)

static int vec_close(const fiv_vec *result_vec, const ivf64 *reference, size_t length, ivf64 tol)
{
    if (result_vec->length != length) return 0;
    const ivf64 *data = result_vec->data.db;
    for (size_t index = 0; index < length; index++) {
        if (fabs(data[index] - reference[index]) > tol * (1.0 + fabs(reference[index]))) return 0;
    }
    return 1;
}


/* Reference dense matrix (ROW_COUNT x COL_COUNT) and its COO description. */
static const int ROW_COUNT = 6, COL_COUNT = 6;
/* nonzero pattern:
 *  row0: cols 0,1,2   (1 consecutive segment)
 *  row1: cols 0,3     (2 segments)
 *  row2: cols 4,5     (1 segment)
 *  row3: cols 1,2,3   (1 segment)
 *  row4: (empty)
 *  row5: cols 0,5     (2 segments)                                        */
static const int COO_ROW[] = {0,0,0, 1,1, 2,2, 3,3,3, 5,5};
static const int COO_COL[] = {0,1,2, 0,3, 4,5, 1,2,3, 0,5};
static const ivf64 COO_VAL[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
static const size_t COO_NNZ = sizeof(COO_ROW) / sizeof(COO_ROW[0]);

/* Build the dense reference dense_matrix[ROW_COUNT][COL_COUNT] by accumulating the
 * SAME COO triplets (including the appended duplicate) that from_coo consumes, so the
 * reference reflects the intended accumulation. */
static void build_dense(ivf64 dense_matrix[ROW_COUNT][COL_COUNT],
                        const int *coo_rows, const int *coo_cols,
                        const ivf64 *coo_values, size_t coo_count)
{
    for (int row = 0; row < ROW_COUNT; row++)
        for (int col = 0; col < COL_COUNT; col++)
            dense_matrix[row][col] = 0.0;
    for (size_t entry = 0; entry < coo_count; entry++)
        dense_matrix[coo_rows[entry]][coo_cols[entry]] += coo_values[entry];
}

static void dense_matvec(const ivf64 dense_matrix[ROW_COUNT][COL_COUNT], const ivf64 *input, ivf64 *output)
{
    for (int row = 0; row < ROW_COUNT; row++) {
        ivf64 sum_value = 0.0;
        for (int col = 0; col < COL_COUNT; col++) sum_value += dense_matrix[row][col] * input[col];
        output[row] = sum_value;
    }
}
static void dense_matvec_transpose(const ivf64 dense_matrix[ROW_COUNT][COL_COUNT], const ivf64 *input, ivf64 *output)
{
    for (int col = 0; col < COL_COUNT; col++) {
        ivf64 sum_value = 0.0;
        for (int row = 0; row < ROW_COUNT; row++) sum_value += dense_matrix[row][col] * input[row];
        output[col] = sum_value;
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


int main(void)
{
    fiv_sparse_runtime_init();
    const fiv_sparse_runtime *runtime_info = NULL;
    fiv_sparse_runtime_get(&runtime_info);
    printf("SIMD lanes = %d\n", runtime_info ? runtime_info->simd_lanes : 1);

    ivf64 dense_matrix[ROW_COUNT][COL_COUNT];

    /* ---- COO inputs (with one duplicated coordinate to test accumulation) ---- */
    int    *coo_rows = fiv_malloc((COO_NNZ + 1) * sizeof(int));
    int    *coo_cols = fiv_malloc((COO_NNZ + 1) * sizeof(int));
    ivf64  *coo_values = fiv_malloc((COO_NNZ + 1) * sizeof(ivf64));
    memcpy(coo_rows, COO_ROW, COO_NNZ * sizeof(int));
    memcpy(coo_cols, COO_COL, COO_NNZ * sizeof(int));
    memcpy(coo_values, COO_VAL, COO_NNZ * sizeof(ivf64));
    /* duplicate (row2,col4) with value -6.0 -> should accumulate with 6.0 to 0.0 */
    coo_rows[COO_NNZ] = 2; coo_cols[COO_NNZ] = 4; coo_values[COO_NNZ] = -6.0;

    /* Build the dense reference from the SAME COO (incl. the duplicate) so it matches
       the accumulated CSR exactly. */
    build_dense(dense_matrix, coo_rows, coo_cols, coo_values, COO_NNZ + 1);

    /* x and y vectors for spmv. */
    ivf64 input_x[COL_COUNT];
    for (int col = 0; col < COL_COUNT; col++) input_x[col] = 0.5 * col + 1.0;   /* 1,1.5,2,2.5,3,3.5 */
    ivf64 other_x[ROW_COUNT];
    for (int row = 0; row < ROW_COUNT; row++) other_x[row] = 0.7 * row - 0.3;

    ivf64 reference_y[ROW_COUNT], reference_yt[COL_COUNT];

    /* ---- build CSR from COO (factory fiv_create_sp_matrix_from_coo) ---- */
    fiv_sparse_mat *sparse_k = fiv_create_sp_matrix_from_coo(coo_rows, coo_cols, coo_values,
                                                            FIV_64F1, COO_NNZ + 1, ROW_COUNT, COL_COUNT);
    CHECK(sparse_k != NULL, "fiv_create_sp_matrix_from_coo produces a matrix");
    CHECK(sparse_k != NULL && fiv_sparse_get_fmt(sparse_k) == FIV_SPARSE_CSR, "from_coo -> CSR");
    CHECK(sparse_k != NULL && sparse_k->nnz == COO_NNZ, "duplicate (2,4) accumulated (nnz unchanged)");
    /* verify val at row2,col4 became 0 */
    if (sparse_k != NULL) {
        int row_start = sparse_k->indptr[2], row_end = sparse_k->indptr[3];
        ivf64 accumulated_value = -999.0;
        for (int entry = row_start; entry < row_end; entry++)
            if (sparse_k->indices[entry] == 4) accumulated_value = sparse_k->hdr.data.db[entry];
        CHECK(fabs(accumulated_value) < 1e-15, "accumulated (2,4) value is 0");
    }

    /* ---- CSR spmv: output = A * input ---- */
    {
        fiv_vec *input_vec = make_vec(COL_COUNT, input_x);
        fiv_vec *output_vec = make_vec(ROW_COUNT, NULL);
        dense_matvec(dense_matrix, input_x, reference_y);
        fiv_ret ret_code = fiv_sparse_matmul_vec(output_vec, sparse_k, input_vec, 0);
        CHECK(ret_code == FIV_RET_OK, "csr matmul_vec(transpose=0) OK");
        CHECK(vec_close(output_vec, reference_y, ROW_COUNT, 1e-12), "csr spmv matches dense reference");
        fiv_release_tensor1d(&input_vec);
        fiv_release_tensor1d(&output_vec);
    }

    /* ---- transpose -> CSC, then output = Aᵀ * input ---- */
    fiv_sparse_mat *sparse_kt = NULL;
    fiv_ret ret_code = fiv_sparse_transpose(&sparse_kt, sparse_k);
    CHECK(ret_code == FIV_RET_OK, "fiv_sparse_transpose returns OK");
    CHECK(sparse_kt != NULL && fiv_sparse_get_fmt(sparse_kt) == FIV_SPARSE_CSC, "transpose -> CSC");
    {
        fiv_vec *input_vec = make_vec(ROW_COUNT, other_x);   /* Aᵀ * input, input length = rows */
        fiv_vec *output_vec = make_vec(COL_COUNT, NULL);
        dense_matvec_transpose(dense_matrix, other_x, reference_yt);
        ret_code = fiv_sparse_matmul_vec(output_vec, sparse_kt, input_vec, 1);
        CHECK(ret_code == FIV_RET_OK, "csc matmul_vec(transpose=1) OK");
        CHECK(vec_close(output_vec, reference_yt, COL_COUNT, 1e-12), "csc spmv (Aᵀx) matches dense reference");
        fiv_release_tensor1d(&input_vec);
        fiv_release_tensor1d(&output_vec);
    }

    /* ---- error path: transpose=0 on a CSC must fail ---- */
    {
        fiv_vec *input_vec = make_vec(COL_COUNT, input_x);
        fiv_vec *output_vec = make_vec(ROW_COUNT, NULL);
        ret_code = fiv_sparse_matmul_vec(output_vec, sparse_kt, input_vec, 0);
        CHECK(ret_code == FIV_RET_ERR_PARA, "csr matvec on CSC rejected");
        fiv_release_tensor1d(&input_vec);
        fiv_release_tensor1d(&output_vec);
    }

    /* ---- reduce_abs_max: dim 0 (rows, CSR) ---- */
    {
        fiv_vec *result_vec = make_vec(ROW_COUNT, NULL);
        ivf64 reference[ROW_COUNT];
        for (int row = 0; row < ROW_COUNT; row++) {
            ivf64 max_abs = 0.0;
            for (int col = 0; col < COL_COUNT; col++) max_abs = fmax(max_abs, fabs(dense_matrix[row][col]));
            reference[row] = max_abs;
        }
        ret_code = fiv_sparse_reduce_abs_max(sparse_k, 0, result_vec);
        CHECK(ret_code == FIV_RET_OK, "reduce_abs_max dim=0 OK");
        CHECK(vec_close(result_vec, reference, ROW_COUNT, 1e-12), "reduce_abs_max rows matches reference");
        fiv_release_tensor1d(&result_vec);
    }

    /* ---- reduce_pow_abs_sum: dim 1 (cols, CSC) with exp=2 ---- */
    {
        fiv_vec *result_vec = make_vec(COL_COUNT, NULL);
        ivf64 reference[COL_COUNT];
        for (int col = 0; col < COL_COUNT; col++) {
            ivf64 sum_value = 0.0;
            for (int row = 0; row < ROW_COUNT; row++) sum_value += pow(fabs(dense_matrix[row][col]), 2.0);
            reference[col] = sum_value;
        }
        ret_code = fiv_sparse_reduce_pow_abs_sum(sparse_kt, 1, 2.0, result_vec);
        CHECK(ret_code == FIV_RET_OK, "reduce_pow_abs_sum dim=1 OK");
        CHECK(vec_close(result_vec, reference, COL_COUNT, 1e-12), "reduce_pow_abs_sum cols matches reference");
        fiv_release_tensor1d(&result_vec);
    }

    /* ---- reduce on wrong format/dim must fail ---- */
    {
        fiv_vec *result_vec = make_vec(ROW_COUNT, NULL);
        ret_code = fiv_sparse_reduce_abs_max(sparse_kt, 0, result_vec);   /* CSC + dim 0 -> error */
        CHECK(ret_code == FIV_RET_ERR_PARA, "reduce_abs_max CSC/dim0 rejected");
        fiv_release_tensor1d(&result_vec);
    }

    /* ---- CSRL packed view ---- */
    ret_code = fiv_sparse_build_packed(sparse_k, sparse_kt);
    CHECK(ret_code == FIV_RET_OK, "build_packed OK");
    CHECK(sparse_k->packed != NULL, "packed view built (avg seg len > 1.5)");
    if (sparse_k->packed != NULL) {
        ivf64 segment_avg = (ivf64)sparse_k->packed->csrl.nz / (ivf64)sparse_k->packed->csrl.nzseg;
        printf("  csrl nz=%zu nzseg=%zu avg=%.3f\n",
               sparse_k->packed->csrl.nz, sparse_k->packed->csrl.nzseg, segment_avg);

        /* packed transpose=0 vs CSR scalar vs dense */
        {
            fiv_vec *input_vec = make_vec(COL_COUNT, input_x);
            fiv_vec *output_vec = make_vec(ROW_COUNT, NULL);
            fiv_sparse_matmul_vec_packed(output_vec, sparse_k->packed, input_vec, 0);
            CHECK(vec_close(output_vec, reference_y, ROW_COUNT, 1e-12), "packed spmv(transpose=0) matches dense");
            fiv_release_tensor1d(&input_vec);
            fiv_release_tensor1d(&output_vec);
        }
        /* packed transpose=1 vs CSC scalar vs dense */
        {
            fiv_vec *input_vec = make_vec(ROW_COUNT, other_x);
            fiv_vec *output_vec = make_vec(COL_COUNT, NULL);
            fiv_sparse_matmul_vec_packed(output_vec, sparse_k->packed, input_vec, 1);
            CHECK(vec_close(output_vec, reference_yt, COL_COUNT, 1e-12), "packed spmv(transpose=1) matches dense");
            fiv_release_tensor1d(&input_vec);
            fiv_release_tensor1d(&output_vec);
        }
    }

    /* ---- degeneracy guard: isolated nonzeros -> packed stays NULL ---- */
    {
        int degenerate_rows[] = {0, 2, 4};
        int degenerate_cols[] = {1, 3, 5};
        ivf64 degenerate_vals[] = {1.0, 1.0, 1.0};
        fiv_sparse_mat *degenerate_mat = fiv_create_sp_matrix_from_coo(degenerate_rows, degenerate_cols,
                                                                       degenerate_vals, FIV_64F1, 3, 6, 6);
        ret_code = fiv_sparse_build_packed(degenerate_mat, NULL);
        CHECK(ret_code == FIV_RET_OK && degenerate_mat->packed == NULL, "degenerate matrix -> packed NULL");
        fiv_release_sp_matrix(&degenerate_mat);
    }

    /* ---- dense <-> sparse conversion ---- */
    {
        size_t dense_shape[2] = { ROW_COUNT, COL_COUNT };
        fiv_mat *dense_mat = fiv_create_tensor2d(dense_shape, FIV_64F1);
        CHECK(dense_mat != NULL, "fiv_create_tensor2d(FIV_64F1) OK");
        for (int row = 0; row < ROW_COUNT; row++)
            for (int col = 0; col < COL_COUNT; col++) {
                size_t dense_elem = ((size_t)row * dense_mat->strides[0]
                                     + (size_t)col * dense_mat->strides[1]) / dense_mat->element_bytes;
                dense_mat->data.db[dense_elem] = dense_matrix[row][col];
            }

        /* expected nonzeros: exact zeros are dropped (e.g. (2,4) accumulated to 0). */
        size_t expected_nnz = 0;
        for (int row = 0; row < ROW_COUNT; row++)
            for (int col = 0; col < COL_COUNT; col++)
                if (dense_matrix[row][col] != 0.0) expected_nnz++;

        fiv_sparse_mat *sparse_from_dense = fiv_create_sp_matrix_from_dense(dense_mat);
        CHECK(sparse_from_dense != NULL, "fiv_create_sp_matrix_from_dense OK");
        CHECK(fiv_sparse_get_fmt(sparse_from_dense) == FIV_SPARSE_CSR, "dense->sparse is CSR");
        CHECK(sparse_from_dense != NULL && sparse_from_dense->nnz == expected_nnz,
              "dense->sparse nnz drops zeros");

        /* spmv of the dense-derived CSR must match the dense matvec reference. */
        {
            fiv_vec *input_vec = make_vec(COL_COUNT, input_x);
            fiv_vec *output_vec = make_vec(ROW_COUNT, NULL);
            dense_matvec(dense_matrix, input_x, reference_y);
            ret_code = fiv_sparse_matmul_vec(output_vec, sparse_from_dense, input_vec, 0);
            CHECK(ret_code == FIV_RET_OK, "dense-derived csr spmv OK");
            CHECK(vec_close(output_vec, reference_y, ROW_COUNT, 1e-12),
                  "dense-derived csr spmv matches dense reference");
            fiv_release_tensor1d(&input_vec);
            fiv_release_tensor1d(&output_vec);
        }

        /* round-trip sparse -> dense must reproduce the original dense matrix. */
        fiv_mat *dense_roundtrip = fiv_create_dense_matrix_from_sp_matrix(sparse_from_dense);
        CHECK(dense_roundtrip != NULL, "fiv_create_dense_matrix_from_sp_matrix OK");
        CHECK(dense_roundtrip != NULL && dense_roundtrip->dtype == FIV_64F1, "sparse->dense is FIV_64F1");
        if (dense_roundtrip != NULL) {
            int mismatch = 0;
            for (int row = 0; row < ROW_COUNT && !mismatch; row++)
                for (int col = 0; col < COL_COUNT; col++) {
                    size_t dense_elem = ((size_t)row * dense_roundtrip->strides[0]
                                         + (size_t)col * dense_roundtrip->strides[1]) / dense_roundtrip->element_bytes;
                    if (fabs(dense_roundtrip->data.db[dense_elem] - dense_matrix[row][col]) > 1e-15)
                        mismatch = 1;
                }
            CHECK(mismatch == 0, "round-trip dense->sparse->dense preserves values");
        }

        /* reverse conversion must also work from a CSC (Kᵀ) and equal the reference. */
        fiv_mat *dense_from_csc = fiv_create_dense_matrix_from_sp_matrix(sparse_kt);
        CHECK(dense_from_csc != NULL, "sparse(CSC)->dense OK");
        if (dense_from_csc != NULL) {
            int mismatch = 0;
            for (int row = 0; row < ROW_COUNT && !mismatch; row++)
                for (int col = 0; col < COL_COUNT; col++) {
                    size_t dense_elem = ((size_t)row * dense_from_csc->strides[0]
                                         + (size_t)col * dense_from_csc->strides[1]) / dense_from_csc->element_bytes;
                    if (fabs(dense_from_csc->data.db[dense_elem] - dense_matrix[row][col]) > 1e-15)
                        mismatch = 1;
                }
            CHECK(mismatch == 0, "CSC->dense equals dense reference");
        }

        fiv_release_tensor2d(&dense_mat);
        fiv_release_sp_matrix(&sparse_from_dense);
        fiv_release_tensor2d(&dense_roundtrip);
        fiv_release_tensor2d(&dense_from_csc);
    }

    /* ---- dense->sparse widening from FIV_32F1 ---- */
    {
        size_t shape32[2] = { 2, 3 };
        fiv_mat *dense32 = fiv_create_tensor2d(shape32, FIV_32F1);
        CHECK(dense32 != NULL, "fiv_create_tensor2d(FIV_32F1) OK");
        ivf32 source32[2][3] = { {1.5f, 0.0f, 2.25f}, {0.0f, 3.0f, 0.0f} };
        for (int row = 0; row < 2; row++)
            for (int col = 0; col < 3; col++) {
                size_t dense_elem = ((size_t)row * dense32->strides[0]
                                     + (size_t)col * dense32->strides[1]) / dense32->element_bytes;
                dense32->data.fl[dense_elem] = source32[row][col];
            }

        fiv_sparse_mat *sparse32 = fiv_create_sp_matrix_from_dense(dense32);
        CHECK(sparse32 != NULL, "dense(FIV_32F1)->sparse OK");
        CHECK(sparse32 != NULL && sparse32->nnz == 3, "32F dense->sparse drops zeros (nnz=3)");
        if (sparse32 != NULL) {
            ivf64 expected = (ivf64)source32[0][0];   /* 1.5 widened to f64 */
            CHECK(fabs(sparse32->hdr.data.db[0] - expected) < 1e-15, "32F->64F value widened correctly");
            CHECK(sparse32->hdr.dtype == FIV_64F1, "dense->sparse carries FIV_64F1");
        }
        fiv_release_tensor2d(&dense32);
        fiv_release_sp_matrix(&sparse32);
    }

    /* ---- release ---- */
    fiv_release_sp_matrix(&sparse_k);
    fiv_release_sp_matrix(&sparse_kt);
    CHECK(sparse_k == NULL && sparse_kt == NULL, "release nulls the pointers");
    fiv_free(coo_rows); fiv_free(coo_cols); fiv_free(coo_values);

    printf("\nPASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
