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

/* Symmetric eigen decomposition: fiv_matrix_eig_sym (declared in
 * api/fiv_matrix.h).
 *
 * Faithful row-major port of Eigen 5.0.0's SelfAdjointEigenSolver
 * (Eigen/src/Eigenvalues/{Tridiagonalization,SelfAdjointEigenSolver}.h):
 *
 *   Phase 1 - tridiagonalization_inplace: per column i a Householder
 *   reflector H_i = I - tau*v*v^T is built from the column tail below the
 *   diagonal (makeHouseholder convention: beta = -sign(c0)*||x||,
 *   v = [1, tail/(c0-beta)], tau = (beta-c0)/beta), then the similarity
 *       A22 <- A22 - v*w^T - w*v^T,  w = tau*(A22*v - tau/2*(v^T*A22*v)*v)
 *   is applied. The O(n^3) matvec + rank-2 are issued to
 *   fiv_matrix_mul_real32's blocked path, the same engine the Cholesky/LU
 *   drivers route their trailing updates through. The reflector product
 *   Q = H_0*H_1*... is accumulated on the fly into the caller's eigenvector
 *   matrix (V <- V*H_i), touching only V[:, i+1:].
 *
 *   Phase 2 - computeFromTridiagonal_impl: Francis implicit QR steps with
 *   the Wilkinson shift (tridiagonal_qr_step), Givens rotations built by
 *   the real makeGivens, rotations applied on the right of Q (columns
 *   k, k+1 stay one contiguous float pair per row in row-major storage),
 *   the Eigen convergence test (|e|^2 <= eps^2*(|d_i|+|d_i+1|), scaled to
 *   dodge underflow), and the final selection-sort of eigenvalues with
 *   matching column swaps.
 */

#include "fiv_matrix.h"
#include "fiv_common.h"
#include "fiv_mat_mul.h"
#include "fiv_linalg_kernels.h"

#include <math.h>
#include <string.h>
#include <float.h>

/* Eigen's defaults: m_maxIterations = 30, budget = maxIterations * n. */
#define FIV_EIG_MAX_ITERATIONS 30

/* Real-scalar JacobiRotation::makeGivens (Eigen Jacobi.h): produces the
   rotation (cos, sin) with [c s; -s c]^T * (p, q)^T = (r, 0)^T. */
static void fiv_make_givens_real32(ivf32 pval, ivf32 qval, ivf32* rot_cos, ivf32* rot_sin)
{
    if (qval == 0.0f) {
        *rot_cos = (pval < 0.0f) ? -1.0f : 1.0f;
        *rot_sin = 0.0f;
    } else if (pval == 0.0f) {
        *rot_cos = 0.0f;
        *rot_sin = (qval < 0.0f) ? 1.0f : -1.0f;
    } else if (fabsf(pval) > fabsf(qval)) {
        const ivf32 tang = qval / pval;
        ivf32 uval = sqrtf(1.0f + tang * tang);
        if (pval < 0.0f) uval = -uval;
        *rot_cos = 1.0f / uval;
        *rot_sin = -tang * (*rot_cos);
    } else {
        const ivf32 tang = pval / qval;
        ivf32 uval = sqrtf(1.0f + tang * tang);
        if (qval < 0.0f) uval = -uval;
        *rot_sin = -1.0f / uval;
        *rot_cos = -tang * (*rot_sin);
    }
}

/* Row-major port of Eigen's tridiagonalization_inplace over the contiguous
   symmetric dim x dim matrix 'mat_ptr' (row stride == dim). On return
   diag[0..dim-1] / sub[0..dim-2] hold the tridiagonal T with
   A = Q * T * Q^T; sub[dim-1] is scratch. When 'vmat' is non-NULL the
   reflector product Q is accumulated into it on top of its entry contents
   (callers pass the identity). vec_v / vec_w are dim-entry scratch. */
static void fiv_tridiagonalize_real32(ivf32* mat_ptr, int dim, ivf32* diag,
                                      ivf32* sub, ivf32* vmat,
                                      ivf32* vec_v, ivf32* vec_w)
{
    for (int col = 0; col + 1 < dim; col++) {
        const int remaining = dim - col - 1;
        ivf32* col_ptr = mat_ptr + (size_t)col;      /* A[row][col] */

        /* makeHouseholder on x = A[col+1:][col] (strided column reads) */
        ivf32 tail_sq = 0.0f;
        for (int idx = 1; idx < remaining; idx++) {
            const ivf32 xval = col_ptr[(size_t)(col + 1 + idx) * dim];
            tail_sq += xval * xval;
        }
        const ivf32 head = col_ptr[(size_t)(col + 1) * dim];

        ivf32 beta, tau;
        if (remaining == 1 || tail_sq <= FLT_MIN) {
            tau = 0.0f;
            beta = head;
        } else {
            beta = sqrtf(head * head + tail_sq);
            if (head >= 0.0f) beta = -beta;
            tau = (beta - head) / beta;

            /* v = [1, tail/(head-beta)] lives in vec_v[0..remaining-1] */
            vec_v[0] = 1.0f;
            const ivf32 denom = head - beta;
            for (int idx = 1; idx < remaining; idx++) {
                vec_v[idx] = col_ptr[(size_t)(col + 1 + idx) * dim] / denom;
            }
            col_ptr[(size_t)(col + 1) * dim] = 1.0f;

            /* w = tau*(A22*v); then w += (-tau/2*(w.v))*v; then the symmetric
               rank-2 A22 -= v*w^T + w*v^T. A22 anchors at (col+1, col+1). */
            ivf32* trail = mat_ptr + (size_t)(col + 1) * dim + (col + 1);
            fiv_matrix_mul_real32(0, 0, remaining, 1, remaining,
                                  tau, trail, dim, vec_v, 1, 0.0f, vec_w, 1);
            ivf32 proj = 0.0f;
            for (int idx = 0; idx < remaining; idx++) {
                proj += vec_w[idx] * vec_v[idx];
            }
            const ivf32 half_proj = -0.5f * tau * proj;
            for (int idx = 0; idx < remaining; idx++) {
                vec_w[idx] += half_proj * vec_v[idx];
            }
            fiv_matrix_mul_real32(0, 0, remaining, remaining, 1,
                                  -1.0f, vec_v, 1, vec_w, 1, 1.0f, trail, dim);
            fiv_matrix_mul_real32(0, 0, remaining, remaining, 1,
                                  -1.0f, vec_w, 1, vec_v, 1, 1.0f, trail, dim);

            /* V <- V*H_col = V - tau*(V*v)*v^T on the tail columns; note
               (V*v) is unscaled here, unlike the pre-scaled w above */
            if (vmat != NULL) {
                ivf32* vtail = vmat + (col + 1);      /* V[:, col+1:] */
                fiv_matrix_mul_real32(0, 0, dim, 1, remaining,
                                      1.0f, vtail, dim, vec_v, 1, 0.0f, vec_w, 1);
                fiv_matrix_mul_real32(0, 0, dim, remaining, 1,
                                      -tau, vec_w, 1, vec_v, 1, 1.0f, vtail, dim);
            }
        }
        col_ptr[(size_t)(col + 1) * dim] = beta;      /* subdiagonal value */
        diag[col] = mat_ptr[(size_t)col * dim + col];
        sub[col] = beta;
    }
    diag[dim - 1] = mat_ptr[(size_t)(dim - 1) * dim + (dim - 1)];
    sub[dim - 1] = 0.0f;
}

/* Row-major port of Eigen's tridiagonal_qr_step (Francis implicit QR with
   the Wilkinson shift), rotations applied on the right of the optional
   dim x dim row-major 'vmat' (columns k / k+1 are adjacent per row). */
static void fiv_tridiagonal_qr_step_real32(ivf32* diag, ivf32* sub,
                                           int start, int end,
                                           ivf32* vmat, int dim)
{
    /* Wilkinson shift */
    ivf32 td = (diag[end - 1] - diag[end]) * 0.5f;
    const ivf32 eval = sub[end - 1];
    ivf32 mu = diag[end];
    if (td == 0.0f) {
        mu -= fabsf(eval);
    } else if (eval != 0.0f) {
        const ivf32 esq = eval * eval;
        const ivf32 hyp = hypotf(td, eval);
        if (esq == 0.0f) {
            mu -= eval / ((td + (td > 0.0f ? hyp : -hyp)) / eval);
        } else {
            mu -= esq / (td + (td > 0.0f ? hyp : -hyp));
        }
    }

    ivf32 xval = diag[start] - mu;
    ivf32 zval = sub[start];
    for (int step = start; step < end && zval != 0.0f; step++) {
        ivf32 rot_cos, rot_sin;
        fiv_make_givens_real32(xval, zval, &rot_cos, &rot_sin);

        /* T = G^T * T * G, chasing the bulge back to tridiagonal form */
        const ivf32 sdk = rot_sin * diag[step] + rot_cos * sub[step];
        const ivf32 dkp1 = rot_sin * sub[step] + rot_cos * diag[step + 1];
        diag[step] = rot_cos * (rot_cos * diag[step] - rot_sin * sub[step]) -
                     rot_sin * (rot_cos * sub[step] - rot_sin * diag[step + 1]);
        diag[step + 1] = rot_sin * sdk + rot_cos * dkp1;
        sub[step] = rot_cos * sdk - rot_sin * dkp1;
        if (step > start) {
            sub[step - 1] = rot_cos * sub[step - 1] - rot_sin * zval;
        }
        xval = sub[step];
        if (step < end - 1) {
            zval = -rot_sin * sub[step + 1];
            sub[step + 1] = rot_cos * sub[step + 1];
        }

        /* Q = Q * G: columns step / step+1, one contiguous pair per row */
        if (vmat != NULL) {
            for (int row = 0; row < dim; row++) {
                ivf32* rowp = vmat + (size_t)row * dim;
                const ivf32 lead = rowp[step];
                const ivf32 trail = rowp[step + 1];
                rowp[step] = rot_cos * lead - rot_sin * trail;
                rowp[step + 1] = rot_sin * lead + rot_cos * trail;
            }
        }
    }
}

/* Row-major port of Eigen's computeFromTridiagonal_impl: convergence sweep,
   unreduced-block search, QR steps, then selection sort of the eigenvalues
   with matching column swaps. Returns 0 on success, -1 on no convergence. */
static int fiv_eig_from_tridiagonal_real32(ivf32* diag, ivf32* sub, int dim,
                                           ivf32* vmat)
{
    const ivf32 consider_as_zero = FLT_MIN;
    const ivf32 precision_inv = 1.0f / FLT_EPSILON;

    int end = dim - 1;
    int start = 0;
    int iter = 0;
    while (end > 0) {
        for (int idx = start; idx < end; idx++) {
            if (fabsf(sub[idx]) < consider_as_zero) {
                sub[idx] = 0.0f;
            } else {
                /* |sub|^2 <= eps^2 * (|diag| + |diag|), scaled vs underflow */
                const ivf32 scaled = precision_inv * sub[idx];
                if (scaled * scaled <= fabsf(diag[idx]) + fabsf(diag[idx + 1])) {
                    sub[idx] = 0.0f;
                }
            }
        }

        while (end > 0 && sub[end - 1] == 0.0f) end--;
        if (end <= 0) break;

        iter++;
        if (iter > FIV_EIG_MAX_ITERATIONS * dim) return -1;

        start = end - 1;
        while (start > 0 && sub[start - 1] != 0.0f) start--;

        fiv_tridiagonal_qr_step_real32(diag, sub, start, end, vmat, dim);
    }

    /* ascending sort with matching column swaps (Eigen's selection sort) */
    for (int slot = 0; slot < dim - 1; slot++) {
        int best = 0;
        ivf32 best_val = diag[slot];
        for (int probe = 1; probe < dim - slot; probe++) {
            if (diag[slot + probe] < best_val) {
                best_val = diag[slot + probe];
                best = probe;
            }
        }
        if (best > 0) {
            const ivf32 swap_val = diag[slot];
            diag[slot] = diag[slot + best];
            diag[slot + best] = swap_val;
            if (vmat != NULL) {
                for (int row = 0; row < dim; row++) {
                    ivf32* rowp = vmat + (size_t)row * dim;
                    const ivf32 lead = rowp[slot];
                    rowp[slot] = rowp[slot + best];
                    rowp[slot + best] = lead;
                }
            }
        }
    }
    return 0;
}

fiv_ret fiv_matrix_eig_sym(fiv_mat* mat_a, ivf32* evals, fiv_mat* mat_evec, int upper)
{
    if (mat_a == NULL || mat_a->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (evals == NULL)                            return FIV_RET_ERR_PARA;
    if (mat_a->data_continue == 0)                return FIV_RET_ERR_PARA;
    if (mat_a->dtype != FIV_32F1)                 return FIV_RET_ERR_NOT_SUPPORT;

    const size_t dim_s = mat_a->shapes[0];
    if (dim_s == 0 || mat_a->shapes[1] != dim_s)  return FIV_RET_ERR_PARA;
    const int dim = (int)dim_s;
    if ((size_t)dim * dim > (size_t)(SIZE_MAX / sizeof(ivf32))) return FIV_RET_ERR_PARA;
    if (mat_a->total_bytes < dim_s * dim_s * (size_t)mat_a->element_bytes) {
        return FIV_RET_ERR_PARA;
    }

    if (mat_evec != NULL) {
        if (mat_evec->data.ptr == NULL || mat_evec->data_continue == 0) {
            return FIV_RET_ERR_PARA;
        }
        if (mat_evec->dtype != FIV_32F1)          return FIV_RET_ERR_NOT_SUPPORT;
        if (mat_evec->shapes[0] != dim_s || mat_evec->shapes[1] != dim_s) {
            return FIV_RET_ERR_PARA;
        }
        if (mat_evec->total_bytes <
                dim_s * dim_s * (size_t)mat_evec->element_bytes) {
            return FIV_RET_ERR_PARA;
        }
        /* A is consumed as scratch; aliasing the eigenvector output would
           destroy the input mid-reduction */
        if (mat_evec->data.ptr == mat_a->data.ptr) return FIV_RET_ERR_PARA;
    }

    ivf32* mat_ptr = (ivf32*)mat_a->data.ptr;

    /* the GEMM updates walk the full trailing rectangle, so both triangles
       must hold the symmetric data before the reduction starts */
    if (upper) {
        for (int row = 1; row < dim; row++) {
            for (int col = 0; col < row; col++) {
                mat_ptr[(size_t)row * dim + col] = mat_ptr[(size_t)col * dim + row];
            }
        }
    } else {
        for (int row = 1; row < dim; row++) {
            for (int col = 0; col < row; col++) {
                mat_ptr[(size_t)col * dim + row] = mat_ptr[(size_t)row * dim + col];
            }
        }
    }

    ivf32* vmat = NULL;
    if (mat_evec != NULL) {
        vmat = (ivf32*)mat_evec->data.ptr;
        for (int row = 0; row < dim; row++) {
            for (int col = 0; col < dim; col++) {
                vmat[(size_t)row * dim + col] = (row == col) ? 1.0f : 0.0f;
            }
        }
    }

    /* scratch: tridiagonal d/e plus the reflector and update vectors */
    const size_t entries = (size_t)dim;
    ivf32* scratch_fl = (ivf32*)fiv_malloc(sizeof(ivf32) * entries * 4);
    if (scratch_fl == NULL) return FIV_RET_ERR_MEM;
    ivf32* diag = scratch_fl;
    ivf32* sub = scratch_fl + entries;
    ivf32* vec_v = scratch_fl + entries * 2;
    ivf32* vec_w = scratch_fl + entries * 3;

    fiv_tridiagonalize_real32(mat_ptr, dim, diag, sub, vmat, vec_v, vec_w);

    if (fiv_eig_from_tridiagonal_real32(diag, sub, dim, vmat) != 0) {
        fiv_free(scratch_fl);
        return FIV_RET_ERR_UNKNOWN;      /* QR sweep budget exhausted */
    }

    memcpy(evals, diag, sizeof(ivf32) * entries);
    fiv_free(scratch_fl);
    return FIV_RET_OK;
}
