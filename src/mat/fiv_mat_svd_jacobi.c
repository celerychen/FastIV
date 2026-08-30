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

/* Thin singular value decomposition via one-sided Jacobi rotations, output
 * convention aligned with OpenCV SVD::compute: fiv_matrix_svd_jacobi
 * (declared in api/fiv_matrix.h), float32 only, row-major native.
 *
 * Faithful port of OpenCV's JacobiSVDImpl_ (modules/core/src/lapack.cpp):
 *
 *   Setup   - work = A^T as a dim x rows row-major buffer (row i of work is
 *           column i of A, length = rows of A); the input A is preserved.
 *           The sweeps orthogonalize the dim rows of work, which are exactly
 *           the columns of A for tall inputs and the rows of A for wide
 *           inputs (OpenCV's `at` swap). A dim x dim Vt buffer accumulates
 *           the right rotations from the identity.
 *
 *   Sweeps  - for every row pair (i, j): p = dot(row_i, row_j); skip when
 *           |p| <= eps * sqrt(W[i] * W[j]) with eps = 2 * FLT_EPSILON;
 *           otherwise apply OpenCV's exact rotation
 *               beta = W[i] - W[j],  gamma = hypot(2p, beta)
 *               (beta < 0:  s = sqrt((gamma-beta)/2gamma), c = p/(2 gamma s))
 *               (else:      c = sqrt((gamma+beta)/2gamma), s = p/(2 gamma c))
 *           to both work rows and both Vt rows; W entries carry the running
 *           squared norms in double precision. max_iter = max(rows, 30).
 *
 *   Finish  - W = row norms; descending sort with row swaps of work and Vt;
 *           U = work rows normalized (zero singular values are completed
 *           with deterministic random vectors run through two Gram-Schmidt
 *           passes, OpenCV's scheme); output mapping follows OpenCV's `at`
 *           swap: tall input -> mat_u = work^T, mat_vt = Vt; wide input ->
 *           mat_u = Vt^T, mat_vt = work.
 */

#include "fiv_matrix.h"
#include "fiv_common.h"

#include <math.h>
#include <string.h>
#include <float.h>

#define FIV_SVDJ_MAX_ITER_SCALE 30

/* deterministic replacement for OpenCV's RNG(0x12345678) used to complete
   zero singular values; any fixed-sequence source works for the tests */
static ivf32 fiv_svdj_rand_pm1(unsigned* rng_state, int dim)
{
    *rng_state = *rng_state * 1103515245u + 12345u;
    return ((*rng_state >> 16) & 1u) ? 1.0f / (ivf32)dim
                                     : -1.0f / (ivf32)dim;
}

fiv_ret fiv_matrix_svd_jacobi(fiv_mat* mat_a, ivf32* sing_vals,
                              fiv_mat* mat_u, fiv_mat* mat_vt)
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

    /* mat_u: rows x dim; mat_vt: dim x cols */
    fiv_mat* out_mats[2];
    out_mats[0] = mat_u;
    out_mats[1] = mat_vt;
    const size_t want_shapes[2][2] = { { rows_s, (size_t)dim },
                                       { (size_t)dim, cols_s } };
    for (int idx = 0; idx < 2; idx++) {
        const fiv_mat* out_mat = out_mats[idx];
        if (out_mat == NULL) continue;
        if (out_mat->data.ptr == NULL || out_mat->data_continue == 0) return FIV_RET_ERR_PARA;
        if (out_mat->dtype != FIV_32F1)               return FIV_RET_ERR_NOT_SUPPORT;
        if (out_mat->shapes[0] != want_shapes[idx][0] ||
            out_mat->shapes[1] != want_shapes[idx][1]) {
            return FIV_RET_ERR_PARA;
        }
        if (out_mat->total_bytes <
                want_shapes[idx][0] * want_shapes[idx][1] *
                (size_t)out_mat->element_bytes) {
            return FIV_RET_ERR_PARA;
        }
        if (out_mat->data.ptr == mat_a->data.ptr)     return FIV_RET_ERR_PARA;
    }
    if (mat_u != NULL && mat_vt != NULL && mat_u->data.ptr == mat_vt->data.ptr) {
        return FIV_RET_ERR_PARA;
    }

    const int wide = rows < cols;
    /* OpenCV: m = max(rows, cols), n = dim after the `at` swap */
    const int opencv_m = rows > cols ? rows : cols;

    /* work rows = the vectors to orthogonalize, one per singular value:
       tall input -> A^T (work row i = column i of A, length rows);
       wide input -> A itself (work row i = row i of A, length cols), which
       is OpenCV's `at` swap (for wide A the SVD of A^T is computed from the
       rows of A). Row length of the work buffer is row_len. */
    const int row_len = wide ? cols : rows;
    ivf32* work = (ivf32*)fiv_malloc(sizeof(ivf32) * (size_t)dim * row_len);
    ivf32* wbuf = (ivf32*)fiv_malloc(sizeof(ivf32) * (size_t)dim);      /* norms^2 */
    ivf32* vt = (ivf32*)fiv_malloc(sizeof(ivf32) * (size_t)dim * dim);
    if (work == NULL || wbuf == NULL || vt == NULL) {
        fiv_free(vt);
        fiv_free(wbuf);
        fiv_free(work);
        return FIV_RET_ERR_MEM;
    }
    if (wide) {
        memcpy(work, (const void*)mat_a->data.ptr,
               (size_t)rows * cols * sizeof(ivf32));
    } else {
        /* work = A^T through the transpose API (row stride = rows) */
        fiv_mat work_mat = *mat_a;
        work_mat.data.ptr    = work;
        work_mat.shapes[0]   = (size_t)dim;
        work_mat.shapes[1]   = (size_t)rows;
        work_mat.strides[0]  = (size_t)rows * mat_a->element_bytes;
        work_mat.strides[1]  = mat_a->element_bytes;
        work_mat.total_bytes = (size_t)dim * rows * mat_a->element_bytes;
        if (fiv_matrix_transpose(&work_mat, mat_a) != FIV_RET_OK) {
            fiv_free(vt);
            fiv_free(wbuf);
            fiv_free(work);
            return FIV_RET_ERR_UNKNOWN;
        }
    }

    const int compute_uv = (mat_u != NULL || mat_vt != NULL);
    if (compute_uv) {
        for (int i = 0; i < dim; i++) {
            for (int k = 0; k < dim; k++) {
                vt[(size_t)i * dim + k] = (i == k) ? 1.0f : 0.0f;
            }
        }
    }

    /* W[i] = squared norm of work row i (double accumulator, like OpenCV) */
    for (int i = 0; i < dim; i++) {
        double sd = 0.0;
        for (int k = 0; k < row_len; k++) {
            const double t = work[(size_t)i * row_len + k];
            sd += t * t;
        }
        wbuf[i] = (ivf32)sd;
    }

    /* Jacobi sweeps */
    const int max_iter = opencv_m > FIV_SVDJ_MAX_ITER_SCALE
                       ? opencv_m : FIV_SVDJ_MAX_ITER_SCALE;
    const ivf32 eps = FLT_EPSILON * 2.0f;
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        int changed = 0;
        for (int i = 0; i < dim - 1; i++) {
            for (int j = i + 1; j < dim; j++) {
                ivf32* wi = work + (size_t)i * row_len;
                ivf32* wj = work + (size_t)j * row_len;
                double a = wbuf[i], b = wbuf[j], p = 0.0;
                for (int k2 = 0; k2 < row_len; k2++) {
                    p += (double)wi[k2] * (double)wj[k2];
                }
                if (fabs(p) <= (double)eps * sqrt(a * b)) continue;

                const double p2 = p * 2.0;
                const double beta = a - b;
                const double gamma = hypot(p2, beta);
                ivf32 c_rot, s_rot;
                if (beta < 0.0) {
                    const double delta = (gamma - beta) * 0.5;
                    s_rot = (ivf32)sqrt(delta / gamma);
                    c_rot = (ivf32)(p2 / (gamma * s_rot * 2.0));
                } else {
                    c_rot = (ivf32)sqrt((gamma + beta) / (gamma * 2.0));
                    s_rot = (ivf32)(p2 / (gamma * c_rot * 2.0));
                }

                double na = 0.0, nb = 0.0;
                for (int k2 = 0; k2 < row_len; k2++) {
                    const ivf32 t0 = c_rot * wi[k2] + s_rot * wj[k2];
                    const ivf32 t1 = -s_rot * wi[k2] + c_rot * wj[k2];
                    wi[k2] = t0;
                    wj[k2] = t1;
                    na += (double)t0 * t0;
                    nb += (double)t1 * t1;
                }
                wbuf[i] = (ivf32)na;
                wbuf[j] = (ivf32)nb;
                changed = 1;

                if (compute_uv) {
                    ivf32* vi = vt + (size_t)i * dim;
                    ivf32* vj = vt + (size_t)j * dim;
                    for (int k2 = 0; k2 < dim; k2++) {
                        const ivf32 t0 = c_rot * vi[k2] + s_rot * vj[k2];
                        const ivf32 t1 = -s_rot * vi[k2] + c_rot * vj[k2];
                        vi[k2] = t0;
                        vj[k2] = t1;
                    }
                }
            }
        }
        if (!changed) break;
    }
    if (iter >= max_iter) {
        fiv_free(vt);
        fiv_free(wbuf);
        fiv_free(work);
        return FIV_RET_ERR_UNKNOWN;
    }

    /* W = row norms */
    for (int i = 0; i < dim; i++) {
        double sd = 0.0;
        for (int k = 0; k < row_len; k++) {
            const double t = work[(size_t)i * row_len + k];
            sd += t * t;
        }
        wbuf[i] = (ivf32)sqrt(sd);
    }

    /* descending sort with row swaps of work and vt */
    for (int i = 0; i < dim - 1; i++) {
        int best = i;
        for (int k = i + 1; k < dim; k++) {
            if (wbuf[best] < wbuf[k]) best = k;
        }
        if (best != i) {
            const ivf32 swap_w = wbuf[i];
            wbuf[i] = wbuf[best];
            wbuf[best] = swap_w;
            for (int k = 0; k < row_len; k++) {
                const ivf32 swap_t = work[(size_t)i * row_len + k];
                work[(size_t)i * row_len + k] = work[(size_t)best * row_len + k];
                work[(size_t)best * row_len + k] = swap_t;
            }
            if (compute_uv) {
                for (int k = 0; k < dim; k++) {
                    const ivf32 swap_v = vt[(size_t)i * dim + k];
                    vt[(size_t)i * dim + k] = vt[(size_t)best * dim + k];
                    vt[(size_t)best * dim + k] = swap_v;
                }
            }
        }
    }
    memcpy(sing_vals, wbuf, sizeof(ivf32) * (size_t)dim);

    /* normalize the work rows into the vector outputs; zero singular values
       are completed with random + Gram-Schmidt vectors (OpenCV's scheme) */
    if (compute_uv) {
        unsigned rng_state = 0x12345678u;
        const ivf32 minval = FLT_MIN;
        for (int i = 0; i < dim; i++) {
            ivf32 sd = wbuf[i];
            if (sd <= minval) {
                for (int attempt = 0; attempt < 100 && sd <= minval; attempt++) {
                    for (int k = 0; k < row_len; k++) {
                        work[(size_t)i * row_len + k] =
                            fiv_svdj_rand_pm1(&rng_state, row_len);
                    }
                    for (int gs_iter = 0; gs_iter < 2; gs_iter++) {
                        for (int j = 0; j < i; j++) {
                            ivf32 dot = 0.0f, asum = 0.0f;
                            for (int k = 0; k < row_len; k++) {
                                dot += work[(size_t)i * row_len + k] *
                                       work[(size_t)j * row_len + k];
                            }
                            for (int k = 0; k < row_len; k++) {
                                const ivf32 t = work[(size_t)i * row_len + k] -
                                                dot * work[(size_t)j * row_len + k];
                                work[(size_t)i * row_len + k] = t;
                                asum += fabsf(t);
                            }
                            const ivf32 scale = asum > eps * 100.0f
                                              ? 1.0f / asum : 0.0f;
                            for (int k = 0; k < rows; k++) {
                                work[(size_t)i * row_len + k] *= scale;
                            }
                        }
                    }
                    double sd2 = 0.0;
                    for (int k = 0; k < rows; k++) {
                        const double t = work[(size_t)i * row_len + k];
                        sd2 += t * t;
                    }
                    sd = (ivf32)sqrt(sd2);
                }
            }
            const ivf32 scale = sd > minval ? 1.0f / sd : 0.0f;
            for (int k = 0; k < row_len; k++) {
                work[(size_t)i * row_len + k] *= scale;
            }
        }

        /* output mapping with OpenCV's `at` swap; both U outputs are
           transposes of the internal buffers via the transpose API */
        fiv_mat work_view = *mat_a;
        work_view.data.ptr    = work;
        work_view.shapes[0]   = (size_t)dim;
        work_view.shapes[1]   = (size_t)row_len;
        work_view.strides[0]  = (size_t)row_len * mat_a->element_bytes;
        work_view.strides[1]  = mat_a->element_bytes;
        work_view.total_bytes = (size_t)dim * row_len * mat_a->element_bytes;

        if (!wide) {
            /* mat_u = work^T (rows x dim), mat_vt = vt (dim x cols) */
            if (mat_u != NULL &&
                fiv_matrix_transpose(mat_u, &work_view) != FIV_RET_OK) {
                fiv_free(vt);
                fiv_free(wbuf);
                fiv_free(work);
                return FIV_RET_ERR_UNKNOWN;
            }
            if (mat_vt != NULL) {
                memcpy(mat_vt->data.ptr, vt,
                       sizeof(ivf32) * (size_t)dim * cols);
            }
        } else {
            /* mat_u = vt^T (rows x dim), mat_vt = work (dim x cols) */
            if (mat_u != NULL) {
                fiv_mat vt_view = *mat_a;
                vt_view.data.ptr    = vt;
                vt_view.shapes[0]   = (size_t)dim;
                vt_view.shapes[1]   = (size_t)dim;
                vt_view.strides[0]  = (size_t)dim * mat_a->element_bytes;
                vt_view.strides[1]  = mat_a->element_bytes;
                vt_view.total_bytes = (size_t)dim * dim * mat_a->element_bytes;
                if (fiv_matrix_transpose(mat_u, &vt_view) != FIV_RET_OK) {
                    fiv_free(vt);
                    fiv_free(wbuf);
                    fiv_free(work);
                    return FIV_RET_ERR_UNKNOWN;
                }
            }
            if (mat_vt != NULL) {
                memcpy(mat_vt->data.ptr, work,
                       sizeof(ivf32) * (size_t)dim * cols);
            }
        }
    }

    fiv_free(vt);
    fiv_free(wbuf);
    fiv_free(work);
    return FIV_RET_OK;
}
