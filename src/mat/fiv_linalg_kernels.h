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

#ifndef _FIV_LINALG_KERNELS_H_
#define _FIV_LINALG_KERNELS_H_

#include "fiv_data_typedefs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
   Internal building-block kernels for the blocked factorization drivers
   (Cholesky potrf, LU getrf, symmetric eig; see fiv_mat_cholesky.c and the
   later factorization files). They operate on RAW pointer anchors with an
   explicit row stride -- no tensor headers, no validation, no allocation --
   so the drivers can point them at arbitrary sub-blocks of a bigger matrix.

   The default tile size: the diagonal-block kernels below touch only one
   NB x NB float region (16 KB at the default), which stays L1/L2-resident.
   Override with -DFIV_LINALG_NB=<n> to tune for a specific cache.
   ======================================================================== */
#ifndef FIV_LINALG_NB
#define FIV_LINALG_NB 64
#endif

/* In-place unblocked (NB x KB) lower-triangular Cholesky factorization using
   the row-by-row Banachiewicz form, so every access walks the contiguous row
   direction. Element (i, j) of the block lives at p11[i*row_stride + j]
   (block columns start at p11 itself, i.e. indexing is block-local).
   Factor result: element (i, j) becomes L(i, j) for j <= i, for i > j the
   stored value is ignored/undefined for the caller's purposes (it holds the
   original Schur-complement entry until overwritten elsewhere).
   Returns 0 on success, -1 when a non-positive remainder is hit (the input
   block is not positive definite); partial results may already be written. */
int fiv_potrf_lower_block_real32(ivf32* p11, int nb, int row_stride);

/* Forward-substitution strip solve that produces L21 of the blocked Cholesky:
   given the factored NB x NB diagonal block L11 anchored at p11 (same layout
   as fiv_potrf_lower_block_real32), overwrite the mb x kb strip anchored at
   p21 (strip row r = rows k+kb+r of the parent matrix, strip column c in
   [0, kb) shares the diagonal block's column range) such that
       A21(r, c) on entry  ->  L21(r, c) = (A21 - sum_t<c L21(r,t)*L11(c,t)) / L11(c,c)
   satisfying A21 = L11 * L21^T. Row-major friendly: each output row is one
   contiguous run scanned left-to-right against contiguous segments of L11. */
void fiv_cholesky_strip_solve_real32(const ivf32* p11, int kb, int ldm11,
                                      ivf32* p21, int mb, int ldm21);

/* Unblocked partial-pivoting LU (GETF2 form) of a kb-wide panel: the panel
   anchors at p11 = A(row0, row0), spans mrows rows (the full column height
   below and including the panel top) and kb columns, element (r, c) living
   at p11[r*row_stride + c]. On return the panel holds U on and above its
   diagonal and the unit-lower multipliers strictly below it (unit diagonal
   implicit). piv[c] receives the absolute (parent-matrix) row index row0+p
   that row row0+c was interchanged with at elimination step c; piv must
   hold kb entries and interchanges must be applied in order. Pivot choice
   is the first row attaining max |A(r,c)| over r >= c (iSAMAX convention).
   A zero pivot does not abort: its (1-based, panel-local) column is recorded
   and the column is skipped LAPACK-style (no scaling, no update below it),
   so the caller still gets usable factors of a singular matrix. Returns 0
   when every pivot is nonzero, else the first such column (1-based). */
int fiv_getrf_panel_real32(ivf32* p11, int kb, int mrows, int row_stride,
                            int* piv, int row0);

/* Row-major forward substitution producing the U12 strip of the blocked LU:
   given the factored kb x kb diagonal block L11 (unit lower) at p11 and the
   kb x ncols strip A12 anchored at p12 (rows k..k+kb-1 of the parent, columns
   to the right of the panel), overwrite A12 in place with U12 = L11^-1 * A12.
   Row-wise AXPY form, so every step touches two contiguous runs: for each
   row i > 0, row_i <- row_i - sum_{t<i} L11(i,t) * row_t (row 0 is untouched
   because L11 carries the unit diagonal implicitly). */
void fiv_lu_strip_solve_real32(const ivf32* p11, int kb, int ldm11,
                                ivf32* p12, int ncols, int ld12);

/* In-place transpose of a square n x n contiguous row-major matrix via
   triangle swaps, O(n^2). Used by upper-mode wrappers to reuse the
   lower-triangle kernels untouched. */
void fiv_transpose_square_inplace_real32(ivf32* a, int n);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_LINALG_KERNELS_H_ */
