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

/* Thin singular value decomposition: fiv_matrix_svd (declared in
 * api/fiv_matrix.h), float32 only, row-major native.
 *
 * Port of Eigen 5.0.0's SVD pipeline (Eigen/src/SVD/UpperBidiagonalization.h)
 * with the classical Golub-Kahan QR iteration on the bidiagonal form:
 *
 *   Phase 1 - blocked Householder bidiagonalization A = U B V^T (Choi /
 *   Dongarra / Walker panel scheme, as in Eigen's
 *   upperbidiagonalization_inplace_blocked): each panel reduces its first bs
 *   columns/rows to bidiagonal form while accumulating the compact-WY update
 *   matrices X (brows x bs) and Y (bcols x bs), then applies the deferred
 *   rank-bs updates to the trailing block
 *       A22 -= A10 * Y_tail^T + X_tail * A01
 *   as two GEMM calls through fiv_matrix_mul_real32's blocked path -- the
 *   same engine the Cholesky/LU/eig drivers route their O(n^3) work through.
 *   Narrow remaining blocks finish with the per-column Householder recursion.
 *   The reflectors stay packed in A exactly like Eigen's m_householder
 *   format: left taus on the diagonal, right taus on the superdiagonal,
 *   essential parts in the strict lower / upper triangles.
 *
 *   Phase 2 - singular vector generation, dorgbr-style: the thin U (m x k)
 *   is built by applying the packed left reflectors in reverse order to the
 *   bottom-right corners of [I_k; 0]; V (k x k) likewise from the right
 *   reflectors. Both route their rank-1 updates through the GEMM engine.
 *
 *   Phase 3 - implicit-shift Golub-Kahan QR on the bidiagonal (Wilkinson
 *   shift from the trailing 2x2 of B^T B, bulge chasing with alternating
 *   right/left Givens rotations, zero-diagonal fixup), rotations accumulated
 *   into U / V; superdiagonals are deflated against an eps-scaled tolerance.
 *   Singular values are clamped non-negative and sorted descending with
 *   matching column swaps.
 *
 * Wide inputs (rows < cols) are transposed into an internal buffer and the
 * output matrices swap roles, so the bidiagonalization always sees
 * rows >= cols.
 */

#include "fiv_matrix.h"
#include "fiv_common.h"
#include "fiv_mat_mul.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <float.h>

/* Eigen's defaults: panel block size 32, unblocked threshold 48. */
#define FIV_SVD_MAX_BLOCK 32
#define FIV_SVD_UNBLOCKED_LIMIT 48
#define FIV_SVD_MAX_SWEEPS 40

/* =========================================================================
   makeHouseholder (Eigen Householder.h, real case) on the vector of 'len'
   entries anchored at 'head' with element stride 'stride': beta =
   -sign(x0)*||x||, tau = (beta - x0)/beta, essential_r = x_r/(x0 - beta)
   written back over entries 1..len-1. Returns beta, tau via *tau_out.
   A tail at float-min scale (or len == 1) yields tau == 0 (identity step).
   ========================================================================= */
static ivf32 fiv_make_householder_real32(ivf32* head, int len, int stride,
                                         ivf32* tau_out)
{
    ivf32 tail_sq = 0.0f;
    for (int idx = 1; idx < len; idx++) {
        const ivf32 val = head[(size_t)idx * stride];
        tail_sq += val * val;
    }
    const ivf32 head_val = head[0];

    if (len == 1 || tail_sq <= FLT_MIN) {
        *tau_out = 0.0f;
        return head_val;
    }
    ivf32 beta = sqrtf(head_val * head_val + tail_sq);
    if (head_val >= 0.0f) beta = -beta;
    *tau_out = (beta - head_val) / beta;
    const ivf32 denom = head_val - beta;
    for (int idx = 1; idx < len; idx++) {
        head[(size_t)idx * stride] /= denom;
    }
    return beta;
}

/* Givens rotation zeroing the second entry: cos*p + sin*q = 0 with unit
   norm, scaled so p^2 + q^2 cannot overflow. */
static void fiv_givens_zero_real32(ivf32 pval, ivf32 qval,
                                   ivf32* rot_cos, ivf32* rot_sin)
{
    const ivf32 mag_p = fabsf(pval);
    const ivf32 mag_q = fabsf(qval);
    const ivf32 scale = mag_p > mag_q ? mag_p : mag_q;
    if (scale == 0.0f) {
        *rot_cos = 1.0f;
        *rot_sin = 0.0f;
        return;
    }
    const ivf32 pval_s = pval / scale;
    const ivf32 qval_s = qval / scale;
    const ivf32 norm = sqrtf(pval_s * pval_s + qval_s * qval_s);
    *rot_cos = qval_s / norm;
    *rot_sin = -pval_s / norm;
}

/* =========================================================================
   Phase 1a: blocked panel (port of Eigen's
   upperbidiagonalization_blocked_helper). Reduces the first bs columns/rows
   of the brows x bcols block at 'mat' (row stride lda) to bidiagonal form
   and builds the compact-WY update matrices X (brows x bs, row stride xsd)
   and Y (bcols x bs, row stride ysd). diag / superdiag are indexed from the
   panel start. vec_v / vec_w: scratch with brows / bcols entries.
   ========================================================================= */
/* Symmetric rank-k helper for the panel: C(m x n, row stride ldc) -= 
   alpha * (L(k x n?) ...) expressed directly as loops over the compact-WY
   factors; the panel-internal products are O(bs^2) each and stay scalar. */

static void fiv_bidiag_panel_real32(ivf32* mat, int lda, int brows, int bcols,
                                    int bs, ivf32* diag, ivf32* superdiag,
                                    ivf32* xmat, int xsd, ivf32* ymat, int ysd,
                                    ivf32* vec_v, ivf32* vec_w,
                                    ivf32* vec_p, ivf32* vec_w2)
{
    ivf32 tau_u_prev = 0.0f;

    for (int col = 0; col < bs; col++) {
        const int remaining_rows = brows - col;
        const int remaining_cols = bcols - col - 1;
        ivf32* col_head = mat + (size_t)col * lda + col;

        /* 1 - v_k = A(:,k) - V_k1 * Y(k,:col)^T - X_k1 * A(:k,k):
               plain row-major loops over the k accumulated columns */
        if (col > 0) {
            for (int row = 0; row < remaining_rows; row++) {
                ivf32 acc = 0.0f;
                for (int j = 0; j < col; j++) {
                    acc += mat[(size_t)(col + row) * lda + j] *
                           ymat[(size_t)col * ysd + j];
                }
                col_head[(size_t)row * lda] -= acc;
                acc = 0.0f;
                for (int j = 0; j < col; j++) {
                    acc += xmat[(size_t)(col + row) * xsd + j] *
                           mat[(size_t)j * lda + col];
                }
                col_head[(size_t)row * lda] -= acc;
            }
        }

        /* 2 - left Householder on the column tail; essential stays packed
           in A(col+1.., col), beta is the diagonal entry */
        ivf32 tau_v;
        diag[col] = fiv_make_householder_real32(col_head, remaining_rows, lda, &tau_v);
        col_head[0] = 1.0f;
        vec_v[0] = 1.0f;
        for (int idx = 1; idx < remaining_rows; idx++) {
            vec_v[idx] = col_head[(size_t)idx * lda];
        }

        if (col + 1 < bcols) {
            ivf32* u_row = mat + (size_t)col * lda + (col + 1);

            /* 3 - y_k = tau_v * (A(k..,k+1..)^T v_k - Y_k1 * tmpY - U_k1^T * tmpX),
               mirroring Eigen term by term (Y_k(j,t) = Y(col+1+j, t)):
               tmpY(t) = Y(t, col) and tmpX(t) = X(t, col) live in the head
               rows of the two accumulated matrices. */
            /* 3 - y_k = tau_v * (A(k..,k+1..)^T v_k - Y_k * tmp - U_k1^T * tmp2),
               mirroring Eigen term by term. tmp / tmp2 are length-col VECTORS:
               tmp(t) = V_k1^T v_k and tmp2(t) = X_k1^T v_k for the already
               accumulated columns t = 0..col-1, each the inner product of the
               t-th column of the packed left / compact-WY matrices with v_k.
               They are NOT scalars -- the (wrong) prior form collapsed to
               sum_t v_k[t]^2 and broke every multi-panel reduction. */
            for (int t = 0; t < col; t++) {
                ivf32 dot_v = 0.0f;   /* tmp(t)  = V_k1^T v_k  */
                ivf32 dot_x = 0.0f;   /* tmp2(t) = X_k1^T v_k */
                for (int idx = 0; idx < remaining_rows; idx++) {
                    dot_v += mat[(size_t)(col + idx) * lda + t] * vec_v[idx];
                    dot_x += xmat[(size_t)(col + idx) * xsd + t] * vec_v[idx];
                }
                vec_p[t] = dot_v;
                vec_w2[t] = dot_x;
            }
            for (int j = 0; j < remaining_cols; j++) {
                ivf32 acc = 0.0f;
                for (int i = 0; i < remaining_rows; i++) {
                    acc += vec_v[i] * mat[(size_t)(col + i) * lda + (col + 1 + j)];
                }
                for (int t = 0; t < col; t++) {
                    acc -= ymat[(size_t)(col + 1 + j) * ysd + t] * vec_p[t];
                    acc -= mat[(size_t)t * lda + (col + 1 + j)] * vec_w2[t];
                }
                ymat[(size_t)(col + 1 + j) * ysd + col] = acc * tau_v;
            }

            /* 4 - u_k = A(k,k+1..) - Y_k * A(k,:k+1)^T - U_k1^T X(k,:k)^T */
            for (int j = 0; j < remaining_cols; j++) {
                ivf32 acc = u_row[j];
                for (int t = 0; t <= col; t++) {
                    acc -= ymat[(size_t)(col + 1 + j) * ysd + t] *
                           mat[(size_t)col * lda + t];
                }
                for (int t = 0; t < col; t++) {
                    acc -= mat[(size_t)t * lda + (col + 1 + j)] *
                           xmat[(size_t)col * xsd + t];
                }
                u_row[j] = acc;
            }

            /* 5 - right Householder on the row tail; essential packed in
               A(col, col+2..), beta is the superdiagonal entry */
            ivf32 tau_u;
            superdiag[col] = fiv_make_householder_real32(u_row, remaining_cols, 1, &tau_u);
            u_row[0] = 1.0f;
            vec_w[0] = 1.0f;
            for (int idx = 1; idx < remaining_cols; idx++) {
                vec_w[idx] = u_row[idx];
            }

            /* 6 - x_k = tau_u * (A(k+1..,k+1..) u_k - X_k1 * tmp0 - A * tmp1),
               mirroring Eigen term by term:
               tmp0(t) = U_k1(t,:) . u_k,  tmp1(t) = Y_k(t,:) . u_k */
            for (int t = 0; t <= col; t++) {
                ivf32 acc0 = 0.0f;
                ivf32 acc1 = 0.0f;
                for (int j = 0; j < remaining_cols; j++) {
                    acc0 += mat[(size_t)t * lda + (col + 1 + j)] * vec_w[j];
                    acc1 += ymat[(size_t)(col + 1 + j) * ysd + t] * vec_w[j];
                }
                vec_p[t] = acc0;
                vec_w2[t] = acc1;
            }
            for (int i = 0; i < remaining_rows - 1; i++) {
                ivf32 acc = 0.0f;
                for (int j = 0; j < remaining_cols; j++) {
                    acc += mat[(size_t)(col + 1 + i) * lda + (col + 1 + j)] *
                           vec_w[j];
                }
                for (int t = 0; t < col; t++) {
                    acc -= xmat[(size_t)(col + 1 + i) * xsd + t] * vec_p[t];
                }
                for (int t = 0; t <= col; t++) {
                    acc -= mat[(size_t)(col + 1 + i) * lda + t] * vec_w2[t];
                }
                xmat[(size_t)(col + 1 + i) * xsd + col] = acc * tau_u;
            }
            if (col > 0) {
                mat[(size_t)(col - 1) * lda + col] = tau_u_prev;
            }
            tau_u_prev = tau_u;
        } else if (col > 0) {
            mat[(size_t)(col - 1) * lda + col] = tau_u_prev;
        }
        col_head[0] = tau_v;
    }
    if (bs < bcols) {
        mat[(size_t)(bs - 1) * lda + bs] = tau_u_prev;
    }

    /* trailing rank-bs update A22 -= A10 * Y_tail^T + X_tail * A01: the
       O(brows * bcols * bs) bulk, routed through the blocked GEMM engine
       exactly like the Cholesky/LU/eig trailing updates */
    if (bcols > bs && brows > bs) {
        const int tail_rows = brows - bs;
        const int tail_cols = bcols - bs;
        ivf32* a01 = mat + (size_t)bs;         /* A(0..bs-1, bs..) */
        const ivf32 saved_head = a01[(size_t)(bs - 1) * lda];
        a01[(size_t)(bs - 1) * lda] = 1.0f;    /* head of the last u vector */

        fiv_matrix_mul_real32(0, 1, tail_rows, tail_cols, bs,
                              -1.0f, mat + (size_t)bs * lda, lda,
                              ymat + (size_t)bs * ysd, ysd,
                              1.0f, mat + (size_t)bs * lda + bs, lda);
        fiv_matrix_mul_real32(0, 0, tail_rows, tail_cols, bs,
                              -1.0f, xmat + (size_t)bs * xsd, xsd,
                              a01, lda,
                              1.0f, mat + (size_t)bs * lda + bs, lda);
        a01[(size_t)(bs - 1) * lda] = saved_head;
    }
}

/* =========================================================================
   Phase 1b: unblocked finish (port of Eigen's
   upperbidiagonalization_inplace_unblocked) for narrow remaining blocks.
   ========================================================================= */
static void fiv_bidiag_unblocked_real32(ivf32* mat, int lda, int brows, int bcols,
                                        ivf32* diag, ivf32* superdiag,
                                        ivf32* vec_v, ivf32* vec_w)
{
    for (int col = 0; col < bcols; col++) {
        const int remaining_rows = brows - col;
        const int remaining_cols = bcols - col - 1;
        ivf32* col_head = mat + (size_t)col * lda + col;

        /* left Householder + application on A(col.., col+1..) */
        ivf32 tau_v;
        diag[col] = fiv_make_householder_real32(col_head, remaining_rows, lda, &tau_v);
        if (remaining_cols > 0) {
            vec_v[0] = 1.0f;
            for (int idx = 1; idx < remaining_rows; idx++) {
                vec_v[idx] = col_head[(size_t)idx * lda];
            }
            fiv_matrix_mul_real32(0, 0, 1, remaining_cols, remaining_rows,
                                  1.0f, vec_v, 1, col_head + 1, lda,
                                  0.0f, vec_w, 1);
            fiv_matrix_mul_real32(0, 0, remaining_rows, remaining_cols, 1,
                                  -tau_v, vec_v, 1, vec_w, 1,
                                  1.0f, col_head + 1, lda);
        }
        col_head[0] = tau_v;

        if (remaining_cols <= 0) break;

        /* right Householder + application on A(col+1.., col+1..) */
        ivf32* row_tail = mat + (size_t)col * lda + (col + 1);
        ivf32 tau_u;
        superdiag[col] = fiv_make_householder_real32(row_tail, remaining_cols, 1, &tau_u);
        row_tail[0] = tau_u;               /* tau packed at the superdiagonal */
        vec_v[0] = 1.0f;
        for (int idx = 1; idx < remaining_cols; idx++) {
            vec_v[idx] = row_tail[idx];
        }
        fiv_matrix_mul_real32(0, 0, remaining_rows - 1, 1, remaining_cols,
                              1.0f, mat + (size_t)(col + 1) * lda + (col + 1), lda,
                              vec_v, 1, 0.0f, vec_w, 1);
        fiv_matrix_mul_real32(0, 0, remaining_rows - 1, remaining_cols, 1,
                              -tau_u, vec_w, 1, vec_v, 1,
                              1.0f, mat + (size_t)(col + 1) * lda + (col + 1), lda);
    }
}

/* =========================================================================
   Phase 1 driver: blocked bidiagonalization of the contiguous rows x cols
   matrix at 'mat' (row stride lda, rows >= cols). Produces diag[0..cols-1]
   and superdiag[0..cols-2]; the reflectors stay packed in 'mat'.
   xmat / ymat: rows x BS / cols x BS workspaces (row stride BS).
   ========================================================================= */
static void fiv_bidiagonalize_real32(ivf32* mat, int lda, int rows, int cols,
                                     ivf32* diag, ivf32* superdiag,
                                     ivf32* xmat, ivf32* ymat,
                                     ivf32* vec_v, ivf32* vec_w,
                                     ivf32* vec_p, ivf32* vec_w2)
{
    /* Blocked bidiagonalization (port of Eigen's
       upperbidiagonalization_inplace_blocked): walk the leading columns in
       panels of up to FIV_SVD_MAX_BLOCK. Each panel reduces its bs
       columns/rows and defers the O(brows*bcols*bs) trailing update through
       two GEMMs; a narrow remaining block (bcols < FIV_SVD_UNBLOCKED_LIMIT) or
       the final block finishes with the unblocked per-column recursion.
       xmat / ymat are reused across panels (local 0-based indexing) because
       each panel's deferred update has already consumed its own X/Y columns
       before the next panel starts. */
    const int max_block = FIV_SVD_MAX_BLOCK;
    int base = 0;
    for (base = 0; base < cols; base += max_block) {
        const int bs = (cols - base < max_block) ? (cols - base) : max_block;
        const int brows = rows - base;
        const int bcols = cols - base;
        ivf32* block = mat + (size_t)base * lda + base;

        if (base + bs == cols || bcols < FIV_SVD_UNBLOCKED_LIMIT) {
            fiv_bidiag_unblocked_real32(block, lda, brows, bcols,
                                        diag + base, superdiag + base,
                                        vec_v, vec_w);
            break;
        }
        fiv_bidiag_panel_real32(block, lda, brows, bcols, bs,
                                diag + base, superdiag + base,
                                xmat, max_block, ymat, max_block,
                                vec_v, vec_w, vec_p, vec_w2);
    }
}

/* =========================================================================
   Phase 2: singular vector generation from the packed reflectors. Thin U
   (rows x cols) = Q_left * [I; 0] and V (cols x cols) = Q_right * I, built
   by applying the reflectors in reverse to the bottom-right corners.
   ========================================================================= */
static void fiv_svd_build_u_real32(ivf32* mat, int lda, int rows, int cols,
                                   ivf32* umat, ivf32* vec_v, ivf32* vec_w)
{
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            umat[(size_t)row * cols + col] = (row == col) ? 1.0f : 0.0f;
        }
    }
    for (int col = cols - 1; col >= 0; col--) {
        const ivf32 tau = mat[(size_t)col * lda + col];
        if (tau == 0.0f) continue;
        const int corner_rows = rows - col;
        const int corner_cols = cols - col;
        ivf32* corner = umat + (size_t)col * cols + col;

        vec_v[0] = 1.0f;
        for (int idx = 1; idx < corner_rows; idx++) {
            vec_v[idx] = mat[(size_t)(col + idx) * lda + col];
        }
        /* w = corner^T * v (1 x corner_cols) */
        fiv_matrix_mul_real32(0, 0, 1, corner_cols, corner_rows,
                              1.0f, vec_v, 1, corner, cols,
                              0.0f, vec_w, 1);
        /* corner -= tau * v * w^T */
        fiv_matrix_mul_real32(0, 0, corner_rows, corner_cols, 1,
                              -tau, vec_v, 1, vec_w, 1,
                              1.0f, corner, cols);
    }
}

static void fiv_svd_build_v_real32(ivf32* mat, int lda, int cols,
                                   ivf32* vmat, ivf32* vec_v, ivf32* vec_w)
{
    for (int row = 0; row < cols; row++) {
        for (int col = 0; col < cols; col++) {
            vmat[(size_t)row * cols + col] = (row == col) ? 1.0f : 0.0f;
        }
    }
    for (int col = cols - 2; col >= 0; col--) {
        const ivf32 tau = mat[(size_t)col * lda + (col + 1)];
        if (tau == 0.0f) continue;
        const int corner = cols - col - 1;
        ivf32* corner_ptr = vmat + (size_t)(col + 1) * cols + (col + 1);

        vec_v[0] = 1.0f;
        for (int idx = 1; idx < corner; idx++) {
            vec_v[idx] = mat[(size_t)col * lda + (col + 1 + idx)];
        }
        fiv_matrix_mul_real32(0, 0, 1, corner, corner,
                              1.0f, vec_v, 1, corner_ptr, cols,
                              0.0f, vec_w, 1);
        fiv_matrix_mul_real32(0, 0, corner, corner, 1,
                              -tau, vec_v, 1, vec_w, 1,
                              1.0f, corner_ptr, cols);
    }
}

/* =========================================================================
   Phase 3: implicit-shift Golub-Kahan QR on the bidiagonal diag[0..dim-1],
   superdiag[0..dim-2], rotations accumulated into umat (urows x dim) and
   vmat (dim x dim), either optional. Returns 0 on success, -1 when the
   sweep budget is exhausted.

   State during the chase: d/e hold the bidiagonal band; at most one lower
   bulge (row j+1, col j) and one upper bulge (row j, col j+2) are alive.
   Rotation convention: R = [[c, s], [-s, c]]; right rotation B <- B*R
   (V <- V*R), left rotation B <- L*B with L = R^T ... applied as
   U <- U*L^T, i.e. U columns (j, j+1) mix with (+s, -s) respectively.
   ========================================================================= */
static void fiv_gk_left_rot_real32(ivf32* diag, ivf32* superdiag, int dim,
                                   int plane, ivf32 rot_cos, ivf32 rot_sin,
                                   ivf32* bulge_low, ivf32* bulge_up)
{
    const ivf32 dj = diag[plane];
    const ivf32 lb = *bulge_low;
    diag[plane] = rot_cos * dj + rot_sin * lb;
    *bulge_low = -rot_sin * dj + rot_cos * lb;   /* ~0 by construction */
    const ivf32 ej = superdiag[plane];
    const ivf32 dj1 = diag[plane + 1];
    superdiag[plane] = rot_cos * ej + rot_sin * dj1;
    diag[plane + 1] = -rot_sin * ej + rot_cos * dj1;
    if (plane + 1 < dim - 1) {
        const ivf32 ej2 = superdiag[plane + 1];
        const ivf32 ub = *bulge_up;
        superdiag[plane + 1] = rot_cos * ej2 - rot_sin * ub;  /* carry upper bulge */
        *bulge_up = rot_cos * ub + rot_sin * ej2;
    } else {
        *bulge_up = 0.0f;
    }
}

static void fiv_gk_right_rot_real32(ivf32* diag, ivf32* superdiag, int dim,
                                    int plane, ivf32 rot_cos, ivf32 rot_sin,
                                    ivf32* bulge_low, ivf32* bulge_up)
{
    (void)dim;
    if (plane > 0) {
        const ivf32 ej = superdiag[plane - 1];
        const ivf32 ub = *bulge_up;
        superdiag[plane - 1] = rot_cos * ej - rot_sin * ub;
        *bulge_up = rot_sin * ej + rot_cos * ub;   /* ~0 by construction */
    }
    const ivf32 dj = diag[plane];
    const ivf32 ej = superdiag[plane];
    diag[plane] = rot_cos * dj - rot_sin * ej;
    superdiag[plane] = rot_sin * dj + rot_cos * ej;
    const ivf32 dj1 = diag[plane + 1];
    const ivf32 lb = *bulge_low;
    *bulge_low = rot_cos * lb - rot_sin * dj1;   /* new lower bulge */
    diag[plane + 1] = rot_sin * lb + rot_cos * dj1;
    /* superdiag[plane+1] = B(plane+1, plane+2) sits in column plane+2,
       untouched by a rotation on columns (plane, plane+1) */
}

/* U columns (j, j+1) mix per the LEFT rotation L = [[c, s], [-s, c]]:
   U <- U * L^T: col_j <- c*col_j + s*col_{j+1}; col_{j+1} <- -s*col_j + c*col_{j+1}. */
static void fiv_rot_u_pair_real32(ivf32* umat, int urows, int ld, int cola, int colb,
                                  ivf32 rot_cos, ivf32 rot_sin)
{
    if (umat == NULL) return;
    for (int row = 0; row < urows; row++) {
        ivf32* rowp = umat + (size_t)row * ld;
        const ivf32 lead = rowp[cola];
        const ivf32 trail = rowp[colb];
        rowp[cola] = rot_cos * lead + rot_sin * trail;
        rowp[colb] = rot_cos * trail - rot_sin * lead;
    }
}

/* V columns (j, j+1) mix per the RIGHT rotation R = [[c, s], [-s, c]]:
   V <- V * R: col_j <- c*col_j - s*col_{j+1}; col_{j+1} <- s*col_j + c*col_{j+1}. */
static void fiv_rot_v_pair_real32(ivf32* vmat, int dim, int cola, int colb,
                                  ivf32 rot_cos, ivf32 rot_sin)
{
    if (vmat == NULL) return;
    for (int row = 0; row < dim; row++) {
        ivf32* rowp = vmat + (size_t)row * dim;
        const ivf32 lead = rowp[cola];
        const ivf32 trail = rowp[colb];
        rowp[cola] = rot_cos * lead - rot_sin * trail;
        rowp[colb] = rot_sin * lead + rot_cos * trail;
    }
}

static int fiv_bidiag_qr_real32(ivf32* diag, ivf32* superdiag, int dim,
                                int urows, ivf32* umat, ivf32* vmat)
{
    ivf32 anorm = 0.0f;
    for (int idx = 0; idx < dim; idx++) {
        anorm += fabsf(diag[idx]);
        if (idx + 1 < dim) anorm += fabsf(superdiag[idx]);
    }
    if (anorm == 0.0f) return 0;
    const ivf32 tol = FLT_EPSILON * anorm;


    int sweeps = 0;
    int end = dim - 1;
    while (end > 0) {
        while (end > 0 && (fabsf(superdiag[end - 1]) <= tol || fabsf(superdiag[end - 1]) < FLT_MIN)) {
            superdiag[end - 1] = 0.0f;
            end--;
        }
        if (end <= 0) break;

        int start = end - 1;
        while (start > 0 && fabsf(superdiag[start - 1]) > tol && fabsf(superdiag[start - 1]) >= FLT_MIN) {
            start--;
        }

        if (fabsf(diag[start]) <= tol || fabsf(diag[start]) < FLT_MIN) {
            /* zero-diagonal fixup: chase the stray B(start, start+1)
               rightward with left rotations in planes (start, target) */
            if (++sweeps > FIV_SVD_MAX_SWEEPS * dim) {
                return -1;
            }
            diag[start] = 0.0f;
            ivf32 stray = superdiag[start];
            superdiag[start] = 0.0f;
            for (int target = start + 1; target <= end; target++) {
                ivf32 rot_cos, rot_sin;
                fiv_givens_zero_real32(stray, diag[target], &rot_cos, &rot_sin);
                fiv_rot_u_pair_real32(umat, urows, dim, start, target, rot_cos, rot_sin);
                diag[target] = -rot_sin * stray + rot_cos * diag[target];
                if (target < end) {
                    /* note: the plane (start, target) rotation also mixes
                       e[target] into the next stray */
                    const ivf32 et = superdiag[target];
                    superdiag[target] = rot_cos * et;
                    stray = rot_sin * et;
                }
            }
            continue;
        }

        if (++sweeps > FIV_SVD_MAX_SWEEPS * dim) {
            return -1;
        }

        /* Wilkinson shift from the trailing 2x2 of T = B^T B */
        if (diag[start] != diag[start] || superdiag[start] != superdiag[start] ||
            diag[end] != diag[end] || superdiag[end - 1] != superdiag[end - 1]) {
            return -1;
        }
        const ivf32 alpha = diag[end - 1] * diag[end - 1] +
                            (end - 1 > start ? superdiag[end - 2] * superdiag[end - 2] : 0.0f);
        const ivf32 beta_t = diag[end - 1] * superdiag[end - 1];
        const ivf32 gamma = superdiag[end - 1] * superdiag[end - 1] + diag[end] * diag[end];
        const ivf32 delta = (alpha - gamma) * 0.5f;
        const ivf32 denom = delta + copysignf(sqrtf(delta * delta + beta_t * beta_t), delta);
        const ivf32 shift = (denom != 0.0f) ? gamma - beta_t * beta_t / denom : gamma;

        /* first right rotation from the implicit-Q condition on
           (T(start,start) - shift, T(start,start+1)): sin*x + cos*z = 0 */
        ivf32 bulge_low = 0.0f;
        ivf32 bulge_up = 0.0f;
        {
            ivf32 rot_cos, rot_sin;
            const ivf32 xval = diag[start] * diag[start] - shift;
            const ivf32 zval = diag[start] * superdiag[start];
            fiv_givens_zero_real32(zval, xval, &rot_cos, &rot_sin);
            fiv_gk_right_rot_real32(diag, superdiag, dim, start, rot_cos, rot_sin,
                                    &bulge_low, &bulge_up);
            fiv_rot_v_pair_real32(vmat, dim, start, start + 1, rot_cos, rot_sin);
        }

        /* bulge chase */
        for (int plane = start; plane < end; plane++) {
            ivf32 rot_cos, rot_sin;
            /* left rotation zeroing the lower bulge at (plane+1, plane):
               -sin*d + cos*l = 0 */
            fiv_givens_zero_real32(bulge_low, -diag[plane], &rot_cos, &rot_sin);
            fiv_gk_left_rot_real32(diag, superdiag, dim, plane, rot_cos, rot_sin,
                                   &bulge_low, &bulge_up);
            fiv_rot_u_pair_real32(umat, urows, dim, plane, plane + 1, rot_cos, rot_sin);

            if (plane + 1 < end) {
                /* right rotation zeroing the upper bulge at (plane, plane+2):
                   cos*ub + sin*e = 0 */
                fiv_givens_zero_real32(bulge_up, superdiag[plane], &rot_cos, &rot_sin);
                fiv_gk_right_rot_real32(diag, superdiag, dim, plane + 1,
                                        rot_cos, rot_sin, &bulge_low, &bulge_up);
                fiv_rot_v_pair_real32(vmat, dim, plane + 1, plane + 2, rot_cos, rot_sin);
            }
        }
    }
    return 0;
}

/* descending sort of the singular values with matching column swaps */
static void fiv_svd_sort_real32(ivf32* sing_vals, int dim,
                                ivf32* umat, int urows, ivf32* vmat)
{
    for (int slot = 0; slot < dim - 1; slot++) {
        int best = slot;
        for (int probe = slot + 1; probe < dim; probe++) {
            if (sing_vals[probe] > sing_vals[best]) best = probe;
        }
        if (best != slot) {
            const ivf32 swap_val = sing_vals[slot];
            sing_vals[slot] = sing_vals[best];
            sing_vals[best] = swap_val;
            if (umat != NULL) {
                for (int row = 0; row < urows; row++) {
                    ivf32* rowp = umat + (size_t)row * dim;
                    const ivf32 lead = rowp[slot];
                    rowp[slot] = rowp[best];
                    rowp[best] = lead;
                }
            }
            if (vmat != NULL) {
                for (int row = 0; row < dim; row++) {
                    ivf32* rowp = vmat + (size_t)row * dim;
                    const ivf32 lead = rowp[slot];
                    rowp[slot] = rowp[best];
                    rowp[best] = lead;
                }
            }
        }
    }
}

fiv_ret fiv_matrix_svd(fiv_mat* mat_a, ivf32* sing_vals, fiv_mat* mat_u, fiv_mat* mat_v)
{
    if (mat_a == NULL || mat_a->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (sing_vals == NULL)                        return FIV_RET_ERR_PARA;
    if (mat_a->data_continue == 0)                return FIV_RET_ERR_PARA;
    if (mat_a->dtype != FIV_32F1)                 return FIV_RET_ERR_NOT_SUPPORT;

    const size_t rows_s = mat_a->shapes[0];
    const size_t cols_s = mat_a->shapes[1];
    if (rows_s == 0 || cols_s == 0)               return FIV_RET_ERR_PARA;
    const int rows = (int)rows_s;
    const int cols = (int)cols_s;
    const int dim = rows < cols ? rows : cols;
    if ((size_t)rows * cols > (size_t)(SIZE_MAX / sizeof(ivf32))) return FIV_RET_ERR_PARA;
    if (mat_a->total_bytes < rows_s * cols_s * (size_t)mat_a->element_bytes) {
        return FIV_RET_ERR_PARA;
    }

    fiv_mat* out_mats[2];
    out_mats[0] = mat_u;
    out_mats[1] = mat_v;
    /* expected shapes: U rows x dim, V cols x dim */
    const size_t want_rows[2] = { rows_s, cols_s };
    for (int idx = 0; idx < 2; idx++) {
        const fiv_mat* out_mat = out_mats[idx];
        if (out_mat == NULL) continue;
        if (out_mat->data.ptr == NULL || out_mat->data_continue == 0) return FIV_RET_ERR_PARA;
        if (out_mat->dtype != FIV_32F1)               return FIV_RET_ERR_NOT_SUPPORT;
        if (out_mat->shapes[0] != want_rows[idx] || out_mat->shapes[1] != (size_t)dim) {
            return FIV_RET_ERR_PARA;
        }
        if (out_mat->total_bytes <
                want_rows[idx] * (size_t)dim * (size_t)out_mat->element_bytes) {
            return FIV_RET_ERR_PARA;
        }
        if (out_mat->data.ptr == mat_a->data.ptr)     return FIV_RET_ERR_PARA;
    }
    if (mat_u != NULL && mat_v != NULL && mat_u->data.ptr == mat_v->data.ptr) {
        return FIV_RET_ERR_PARA;
    }

    /* work matrix: always a private copy so mat_a is preserved in every shape.
       Tall/square copies A verbatim; wide copies A^T (work_rows x work_cols,
       ld = work_cols). work never aliases mat_a. */
    const int work_rows = rows > cols ? rows : cols;
    const int work_cols = dim;
    const int work_ld = work_cols;
    ivf32* work = (ivf32*)fiv_malloc(sizeof(ivf32) * (size_t)rows * cols);
    if (work == NULL) return FIV_RET_ERR_MEM;
    if (rows >= cols) {
        memcpy(work, (const void*)mat_a->data.ptr,
               (size_t)rows * cols * sizeof(ivf32));
    } else {
        /* wide: copy A into work as A^T (work rows x cols = cols x rows, ld = rows) */
        fiv_mat work_mat = *mat_a;          /* inherit dtype / element_bytes / data_continue */
        work_mat.data.ptr    = work;
        work_mat.shapes[0]   = (size_t)work_rows;   /* dst rows = cols */
        work_mat.shapes[1]   = (size_t)work_cols;   /* dst cols = rows */
        work_mat.strides[0]  = (size_t)work_cols * mat_a->element_bytes;
        work_mat.strides[1]  = (size_t)mat_a->element_bytes;
        work_mat.total_bytes = (size_t)rows * cols * mat_a->element_bytes;
        if (fiv_matrix_transpose(&work_mat, mat_a) != FIV_RET_OK) {
            fiv_free(work);
            return FIV_RET_ERR_UNKNOWN;
        }
    }
    /* the tall SVD of A^T swaps the output roles */
    ivf32* umat = (rows >= cols) ? (mat_u ? (ivf32*)mat_u->data.ptr : NULL)
                                 : (mat_v ? (ivf32*)mat_v->data.ptr : NULL);
    ivf32* vmat = (rows >= cols) ? (mat_v ? (ivf32*)mat_v->data.ptr : NULL)
                                 : (mat_u ? (ivf32*)mat_u->data.ptr : NULL);

    /* scratch: bidiagonal band, three reflector vectors, X/Y compact-WY */
    const size_t big = (size_t)work_rows;
    ivf32* scratch_fl = (ivf32*)fiv_malloc(sizeof(ivf32) * (2 * (size_t)dim + 4 * big));
    ivf32* xmat = (ivf32*)fiv_malloc(sizeof(ivf32) * big * FIV_SVD_MAX_BLOCK);
    ivf32* ymat = (ivf32*)fiv_malloc(sizeof(ivf32) * (size_t)work_cols * FIV_SVD_MAX_BLOCK);
    if (scratch_fl == NULL || xmat == NULL || ymat == NULL) {
        fiv_free(ymat);
        fiv_free(xmat);
        fiv_free(scratch_fl);
        fiv_free(work);
        return FIV_RET_ERR_MEM;
    }
    ivf32* diag = scratch_fl;
    ivf32* superdiag = scratch_fl + dim;
    ivf32* vec_v = scratch_fl + 2 * dim;
    ivf32* vec_w = scratch_fl + 2 * dim + big;
    ivf32* vec_p = scratch_fl + 2 * dim + 2 * big;
    ivf32* vec_w2 = scratch_fl + 2 * dim + 3 * big;

    fiv_bidiagonalize_real32(work, work_ld, work_rows, work_cols,
                             diag, superdiag, xmat, ymat, vec_v, vec_w,
                             vec_p, vec_w2);

    if (umat != NULL || vmat != NULL) {
        if (umat != NULL) {
            fiv_svd_build_u_real32(work, work_ld, work_rows, work_cols,
                                   umat, vec_v, vec_w);
        }
        if (vmat != NULL) {
            fiv_svd_build_v_real32(work, work_ld, work_cols,
                                   vmat, vec_v, vec_w);
        }
    }

    const int qr_st = fiv_bidiag_qr_real32(diag, superdiag, work_cols,
                                           work_rows, umat, vmat);

    if (qr_st == 0) {
        /* the QR phase can leave a singular value negative: fold the sign
           into the corresponding U column, then sort descending */
        for (int idx = 0; idx < work_cols; idx++) {
            if (diag[idx] < 0.0f) {
                diag[idx] = -diag[idx];
                if (umat != NULL) {
                    for (int row = 0; row < work_rows; row++) {
                        umat[(size_t)row * work_cols + idx] =
                            -umat[(size_t)row * work_cols + idx];
                    }
                }
            }
        }
        fiv_svd_sort_real32(diag, work_cols, umat, work_rows, vmat);
        memcpy(sing_vals, diag, sizeof(ivf32) * (size_t)work_cols);
    }

    fiv_free(ymat);
    fiv_free(xmat);
    fiv_free(scratch_fl);
    fiv_free(work);
    return (qr_st == 0) ? FIV_RET_OK : FIV_RET_ERR_UNKNOWN;
}
