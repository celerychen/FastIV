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

/* Blocked Cholesky factorization: fiv_matrix_cholesky (declared in api/fiv_matrix.h).
 *
 * Iterative block right-looking scheme (no recursion), row-major native:
 *
 *   for each NB-wide diagonal strip starting at column/row k:
 *     1) factor the NB x KB diagonal block in place        (fiv_potrf_lower_block)
 *     2) solve L11 * L21^T = A21 for the strip below it    (fiv_cholesky_strip_solve)
 *     3) trailing Schur update  A22 -= L21 * L21^T         (raw GEMM engine,
 *                                                             b operand used
 *                                                             transposed)
 *
 * Steps 1-2 cost O(n*NB^2) overall; step 3 carries essentially all of the
 * O(n^3/6) work and runs inside fiv_matrix_mul_real32's AVX2/NEON blocked
 * path, exactly like the OpenBLAS interface->driver->kernel split. The
 * trailing GEMM updates the full trailing rectangle, so the strictly-
 * unreferenced triangle of the parent matrix receives Schur leftovers and is
 * NOT preserved (documented in api/fiv_matrix.h).
 */

#include "fiv_matrix.h"
#include "fiv_mat_mul.h"
#include "fiv_linalg_kernels.h"

#include <math.h>

static int fiv_min_int(int lhs, int rhs)
{
    return lhs < rhs ? lhs : rhs;
}

/* Core blocked factorization over the lower triangle of the contiguous
   row-major n x n matrix 'mat_ptr'. Returns 0 on success. */
static int fiv_potrf_blocked_lower_real32(ivf32* mat_ptr, int n)
{
    const int rs = n;   /* contiguous row-major square: row stride == n */

    for (int k = 0; k < n; k += FIV_LINALG_NB) {
        const int kb = fiv_min_int(FIV_LINALG_NB, n - k);
        ivf32* p11 = mat_ptr + (size_t)k * rs + (size_t)k;

        if (fiv_potrf_lower_block_real32(p11, kb, rs) != 0) {
            return -1;
        }

        const int mb = n - k - kb;
        if (mb > 0) {
            ivf32* p21 = p11 + (size_t)kb * rs;

            fiv_cholesky_strip_solve_real32(p11, kb, rs, p21, mb, rs);

            /* A22 -= L21 * L21^T : both operands read the same strip, one
               used transposed; the GEMM engine only reads its inputs, so the
               aliasing is safe. p21 already anchors at (row k+kb, col k),
               so A22 anchors kb entries further along the same row; the
               result accumulates with beta == 1 over alpha == -1. */
            ivf32* p22 = p21 + kb;
            fiv_matrix_mul_real32(0, 1, mb, mb, kb,
                                  -1.0f, p21, rs, p21, rs,
                                  1.0f, p22, rs);
        }
    }
    return 0;
}

fiv_ret fiv_matrix_cholesky(fiv_mat* mat_a, int lower)
{
    if (mat_a == NULL || mat_a->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (mat_a->data_continue == 0)               return FIV_RET_ERR_PARA;
    if (mat_a->dtype != FIV_32F1)                return FIV_RET_ERR_NOT_SUPPORT;

    const size_t n_s = mat_a->shapes[0];
    if (n_s == 0 || mat_a->shapes[1] != n_s)     return FIV_RET_ERR_PARA;
    const int n = (int)n_s;
    if ((size_t)n * n > (size_t)(SIZE_MAX / sizeof(ivf32))) return FIV_RET_ERR_PARA;
    if (mat_a->total_bytes < n_s * n_s * (size_t)mat_a->element_bytes) {
        return FIV_RET_ERR_PARA;
    }

    ivf32* mat_ptr = (ivf32*)mat_a->data.ptr;
    int st;

    if (!lower) {
        /* A is symmetric (A = A^T), so its lower triangle already holds the
           data the lower-Cholesky kernel reads; a pre-transpose would be a
           no-op. Factor L in place, then a single transpose lifts L^T = U into
           the upper triangle so that A = U^T * U. */
        st = fiv_potrf_blocked_lower_real32(mat_ptr, n);
        if (fiv_matrix_transpose(mat_a, mat_a) != FIV_RET_OK) return FIV_RET_ERR_PARA;
    } else {
        st = fiv_potrf_blocked_lower_real32(mat_ptr, n);
    }

    return (st == 0) ? FIV_RET_OK : FIV_RET_ERR_NOT_POS_DEF;
}
