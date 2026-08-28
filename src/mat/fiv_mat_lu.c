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

/* Blocked LU factorization with partial (row) pivoting: fiv_matrix_lu
 * (declared in api/fiv_matrix.h).
 *
 * Iterative block right-looking scheme (no recursion), row-major native:
 *
 *   for each NB-wide panel starting at row/column k:
 *     1) unblocked partial-pivoting LU of the panel     (fiv_getrf_panel)
 *     2) apply the panel's row interchanges to the columns left of k and
 *        right of k+kb (one contiguous run per row each)
 *     3) solve L11 * U12 = A12 for the strip right of the panel
 *                                                      (fiv_lu_strip_solve)
 *     4) trailing update  A22 -= L21 * U12             (raw GEMM engine,
 *                                                        plain N*N form)
 *
 * Steps 1-3 cost O(m*n*NB) overall; step 4 carries essentially all of the
 * ~2*(min(m,n))^2*max(m,n)/3 work and runs inside fiv_matrix_mul_real32's
 * AVX2/NEON blocked path, the same interface->driver->kernel split as the
 * Cholesky driver. Unlike OpenBLAS's getrf, which defers the left-side
 * interchanges to a final batched pass, the interchanges are applied
 * immediately per panel (LAPACK's DGETRF order); in row-major storage each
 * interchange is a contiguous segment swap, so the extra passes stay cheap
 * relative to the GEMM.
 */

#include "fiv_matrix.h"
#include "fiv_mat_mul.h"
#include "fiv_linalg_kernels.h"

static int fiv_min_int(int lhs, int rhs)
{
    return lhs < rhs ? lhs : rhs;
}

static void fiv_swap_row_segments_real32(ivf32* row_a, ivf32* row_b, int n)
{
    for (int i = 0; i < n; i++) {
        ivf32 swap_tmp = row_a[i]; row_a[i] = row_b[i]; row_b[i] = swap_tmp;
    }
}

/* Core blocked factorization over the contiguous row-major m x n matrix 'mat_ptr'
   (row stride rs). piv must hold min(m, n) entries. Returns 0 on success,
   else the 1-based global column of the first zero pivot (the factorization
   is still completed, LAPACK INFO > 0 semantics). */
static int fiv_getrf_blocked_real32(ivf32* mat_ptr, int m, int n, int rs, int* piv)
{
    const int mn = fiv_min_int(m, n);
    int first_singular = 0;

    for (int k = 0; k < mn; k += FIV_LINALG_NB) {
        const int kb = fiv_min_int(FIV_LINALG_NB, mn - k);
        const int mrows = m - k;
        ivf32* p11 = mat_ptr + (size_t)k * rs + k;

        const int info = fiv_getrf_panel_real32(p11, kb, mrows, rs, piv + k, k);
        if (info && !first_singular) first_singular = k + info;

        /* apply the panel's interchanges outside the panel: the factored
           columns left of k and the trailing columns right of k+kb. The panel
           columns themselves were already swapped by the kernel. */
        const int nleft = k;
        const int nright = n - k - kb;
        for (int c = 0; c < kb; c++) {
            const int pivot_row_src = piv[k + c];
            if (pivot_row_src != k + c) {
                ivf32* row_c = mat_ptr + (size_t)(k + c) * rs;
                ivf32* row_r = mat_ptr + (size_t)pivot_row_src * rs;
                if (nleft > 0)
                    fiv_swap_row_segments_real32(row_c, row_r, nleft);
                if (nright > 0)
                    fiv_swap_row_segments_real32(row_c + k + kb,
                                                 row_r + k + kb, nright);
            }
        }

        if (nright > 0) {
            /* U12 = L11^-1 * A12, in place on the strip right of the panel */
            fiv_lu_strip_solve_real32(p11, kb, rs, p11 + kb, nright, rs);

            /* A22 -= L21 * U12 : plain non-transposed GEMM on the trailing
               rectangle. p21 anchors at (k+kb, k), p12 at (k, k+kb), and A22
               kb entries further along the same anchors; the result
               accumulates with beta == 1 over alpha == -1. */
            const int mb = m - k - kb;
            if (mb > 0) {
                ivf32* p21 = p11 + (size_t)kb * rs;
                ivf32* p12 = p11 + kb;
                ivf32* p22 = p21 + kb;
                fiv_matrix_mul_real32(0, 0, mb, nright, kb,
                                      -1.0f, p21, rs, p12, rs,
                                      1.0f, p22, rs);
            }
        }
    }
    return first_singular;
}

fiv_ret fiv_matrix_lu(fiv_mat* mat_a, int* piv)
{
    if (mat_a == NULL || mat_a->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (piv == NULL)                            return FIV_RET_ERR_PARA;
    if (mat_a->data_continue == 0)               return FIV_RET_ERR_PARA;
    if (mat_a->dtype != FIV_32F1)                return FIV_RET_ERR_NOT_SUPPORT;

    const size_t rows = mat_a->shapes[0];
    const size_t cols = mat_a->shapes[1];
    if (rows == 0 || cols == 0)                 return FIV_RET_ERR_PARA;
    const int m = (int)rows;
    const int n = (int)cols;
    if ((size_t)m * n > (size_t)(SIZE_MAX / sizeof(ivf32))) return FIV_RET_ERR_PARA;
    if (mat_a->total_bytes < rows * cols * (size_t)mat_a->element_bytes) {
        return FIV_RET_ERR_PARA;
    }

    const int st = fiv_getrf_blocked_real32((ivf32*)mat_a->data.ptr, m, n, n, piv);
    return (st == 0) ? FIV_RET_OK : FIV_RET_ERR_SINGULAR;
}
