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

#include "fiv_lp_solve.h"
#include "fiv_lp_mat.h"    /* fiv_lp_mat, fiv_lp_mat_matvec, fiv_lp_mat_build_transpose, fiv_lp_mat_reduce_abs_pow */
#include "fiv_lp_vec.h"    /* fiv_vec_clamp / fiv_vec_sum_finite_products */
#include "fiv_lp_rescale.h"/* fiv_lp_rescaling / fiv_lp_rescale_solve / fiv_lp_unscale_* */
#include "fiv_lp_pdhg.h"   /* fiv_lp_adaptive_step_pdhg / fiv_lp_primal_weight_update / fiv_lp_compute_lambda_box / fiv_lp_kkt_error_sq */
#include "fiv_matrix.h"    /* fiv_vec_norm / fiv_vec_dot / fiv_vec_axpy / fiv_vec_scale */
#include "fiv_ctensor.h"   /* fiv_create_tensor1d / fiv_release_tensor1d / fiv_tensor_sub */
#include "fiv_sp_matrix.h" /* fiv_sparse_mat (struct layout for clone) */
#include "fiv_common.h"    /* fiv_malloc / fiv_calloc / fiv_free */

#include <math.h>          /* fabs, sqrt, pow, isfinite, INFINITY */
#include <string.h>        /* memcpy */
#include <stdlib.h>        /* NULL */
#include <time.h>          /* clock */

/* FastIV's read-only vec operands take a non-const fiv_vec*; documented
 * boundary cast used throughout M3 and here. */
#define FIV_VEC_NC(v) ((fiv_vec *)(v))

static void release_temp(fiv_vec **temp_vector)
{
    if (temp_vector != NULL && *temp_vector != NULL)
        fiv_release_tensor1d(temp_vector);
}

/* ---- L2 / Inf vector norms (FastIV fiv_vec_norm reads operand non-const) ---- */
static ivf64 vec_l2(const fiv_vec *vec)
{
    fiv_scalar result;
    fiv_vec_norm(&result, FIV_VEC_NC(vec), FIV_L2_NORM);
    return result.data.value_fp64;
}
static ivf64 vec_inf(const fiv_vec *vec)
{
    fiv_scalar result;
    fiv_vec_norm(&result, FIV_VEC_NC(vec), FIV_INF_NORM);
    return result.data.value_fp64;
}
static ivf64 vec_dot(const fiv_vec *a, const fiv_vec *b)
{
    fiv_scalar result;
    fiv_vec_dot(&result, a, b);
    return result.data.value_fp64;
}

/* Deep-copy a vector. */
static fiv_vec *clone_vec(const fiv_vec *src)
{
    fiv_vec *dst = fiv_create_tensor1d(src->length, FIV_64F1);
    if (dst != NULL) memcpy(dst->data.db, src->data.db, src->length * sizeof(ivf64));
    return dst;
}

/* Deep-copy a matrix, keeping the SAME kind (DENSE -> DENSE, SPARSE CSR -> CSR).
 * For DENSE the returned wrapper is non-owning over a freshly allocated dense;
 * the caller must release the underlying dense separately (tracked via
 * *out_dense). For SPARSE the returned wrapper OWNS the clone (from COO). */
static fiv_lp_mat *clone_lp_mat(const fiv_lp_mat *src, fiv_mat **out_dense)
{
    *out_dense = NULL;
    if (src->kind == FIV_LP_MAT_DENSE) {
        fiv_mat *dense = fiv_create_tensor2d((size_t[2]){src->rows, src->cols}, FIV_64F1);
        if (dense == NULL) return NULL;
        const fiv_mat *s = src->as.dense;
        const size_t element_bytes = s->element_bytes;
        const size_t stride_s_row = s->strides[0] / element_bytes;
        const size_t stride_s_col = s->strides[1] / element_bytes;
        const size_t stride_d_row = dense->strides[0] / element_bytes;
        const size_t stride_d_col = dense->strides[1] / element_bytes;
        for (size_t row = 0; row < src->rows; row++)
            for (size_t col = 0; col < src->cols; col++)
                dense->data.db[row * stride_d_row + col * stride_d_col] =
                    s->data.db[row * stride_s_row + col * stride_s_col];
        *out_dense = dense;
        return fiv_lp_mat_wrap_dense(dense);
    }
    /* SPARSE: extract COO from the CSR, then rebuild an owning CSR. */
    const fiv_sparse_mat *sp = src->as.sparse;
    const size_t nnz = sp->nnz;
    int *coo_row = (int *)fiv_malloc(nnz * sizeof(int));
    int *coo_col = (int *)fiv_malloc(nnz * sizeof(int));
    ivf64 *coo_val = (ivf64 *)fiv_malloc(nnz * sizeof(ivf64));
    if (coo_row == NULL || coo_col == NULL || coo_val == NULL) {
        fiv_free(coo_row); fiv_free(coo_col); fiv_free(coo_val);
        return NULL;
    }
    const ivf64 *val_ptr = (const ivf64 *)sp->hdr.data.ptr;
    size_t entry = 0;
    for (size_t row = 0; row < sp->rows; row++) {
        for (int p = sp->indptr[row]; p < sp->indptr[row + 1]; p++) {
            coo_row[entry] = (int)row;
            coo_col[entry] = sp->indices[p];
            coo_val[entry] = val_ptr[p];
            entry++;
        }
    }
    fiv_lp_mat *out = fiv_create_lp_mat_from_coo(coo_row, coo_col, coo_val,
                                                FIV_64F1, nnz, sp->rows, sp->cols);
    fiv_free(coo_row); fiv_free(coo_col); fiv_free(coo_val);
    return out;
}

/* Dual objective on the UNSCALED problem: q^T y + l^T lam+ - u^T lam-, where lam
 * is the box normal-cone component at x (using g = c - K^T y). KT_orig_y, when
 * supplied, is K_orig^T @ y and avoids a recomputation. */
static fiv_ret compute_dual_objective(ivf64 *dual_obj_out,
                                      const fiv_lp_mat *K_orig,
                                      const fiv_vec *c_orig, const fiv_vec *l_orig,
                                      const fiv_vec *u_orig, const fiv_vec *q_orig,
                                      const fiv_vec *x_unscaled, const fiv_vec *y_unscaled,
                                      const fiv_vec *KT_orig_y, ivf64 eps_tol)
{
    const size_t num_variables = c_orig->length;
    fiv_vec *gradient = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *lambda   = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *lam_pos  = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *lam_neg  = fiv_create_tensor1d(num_variables, FIV_64F1);
    if (gradient == NULL || lambda == NULL || lam_pos == NULL || lam_neg == NULL) {
        release_temp(&gradient); release_temp(&lambda);
        release_temp(&lam_pos);   release_temp(&lam_neg);
        return FIV_RET_ERR_MEM;
    }

    if (KT_orig_y != NULL) {
        fiv_vec_scale(gradient, FIV_VEC_NC(c_orig), FIV_SCALAR_FP64(1.0));
        fiv_vec_axpy(gradient, FIV_SCALAR_FP64(-1.0), FIV_VEC_NC(KT_orig_y));
    } else {
        fiv_vec *KT = fiv_create_tensor1d(num_variables, FIV_64F1);
        fiv_lp_mat_matvec(KT, K_orig, y_unscaled, 1);
        fiv_vec_scale(gradient, FIV_VEC_NC(c_orig), FIV_SCALAR_FP64(1.0));
        fiv_vec_axpy(gradient, FIV_SCALAR_FP64(-1.0), KT);
        release_temp(&KT);
    }

    fiv_lp_compute_lambda_box(lambda, x_unscaled, gradient, l_orig, u_orig, eps_tol);

    const ivf64 *lam = lambda->data.db;
    ivf64 *lp = lam_pos->data.db, *ln = lam_neg->data.db;
    for (size_t i = 0; i < num_variables; i++) {
        const ivf64 value = lam[i];
        lp[i] = (value > 0.0) ? value : 0.0;
        ln[i] = (value < 0.0) ? -value : 0.0;
    }
    ivf64 l_term = 0.0, u_term = 0.0;
    fiv_vec_sum_finite_products(l_orig, lam_pos, &l_term);
    fiv_vec_sum_finite_products(u_orig, lam_neg, &u_term);

    const ivf64 qy = vec_dot(q_orig, y_unscaled);
    *dual_obj_out = qy + l_term - u_term;

    release_temp(&gradient); release_temp(&lambda);
    release_temp(&lam_pos);   release_temp(&lam_neg);
    return FIV_RET_OK;
}

/* Termination criteria on the UNSCALED problem (pdlp.py termination_criteria).
 * Returns 1 and fills (*out_status, out_info) when a terminal status is found,
 * 0 to keep iterating. K_orig must carry a materialized transpose_view. */
static int termination_criteria(fiv_lp_status *out_status, fiv_lp_solve_info *out_info,
                                const fiv_lp_mat *K_orig,
                                const fiv_vec *c_orig, const fiv_vec *l_orig,
                                const fiv_vec *u_orig, const fiv_vec *q_orig,
                                size_t num_inequality,
                                const fiv_vec *x_unscaled, const fiv_vec *y_unscaled,
                                double eps_tol, double eps_primal_infeasible,
                                double eps_dual_infeasible)
{
    const size_t num_variables = c_orig->length;
    const size_t num_constraints = q_orig->length;
    const size_t m1 = num_inequality;
    const ivf64 eps_zero = 1e-12;

    /* KT_orig_y = K_orig^T @ y ; dual/primal objectives */
    fiv_vec *KT_orig_y = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *Kx_orig   = fiv_create_tensor1d(num_constraints, FIV_64F1);
    if (KT_orig_y == NULL || Kx_orig == NULL) {
        release_temp(&KT_orig_y); release_temp(&Kx_orig);
        return 0;
    }
    fiv_lp_mat_matvec(KT_orig_y, K_orig, y_unscaled, 1);
    fiv_lp_mat_matvec(Kx_orig,   K_orig, x_unscaled, 0);

    ivf64 dual_obj = 0.0;
    compute_dual_objective(&dual_obj, K_orig, c_orig, l_orig, u_orig, q_orig,
                           x_unscaled, y_unscaled, KT_orig_y, eps_tol);
    const ivf64 primal_obj = vec_dot(c_orig, x_unscaled);
    out_info->primal_obj = primal_obj;
    out_info->dual_obj   = dual_obj;

    int terminated = 0;

    /* (1) primal infeasibility: Farkas certificate via dual ray y */
    const ivf64 dual_norm_inf = vec_inf(y_unscaled);
    if (dual_norm_inf > eps_zero) {
        fiv_vec *y_ray = fiv_create_tensor1d(num_constraints, FIV_64F1);
        fiv_vec_scale(y_ray, FIV_VEC_NC(y_unscaled), FIV_SCALAR_FP64(1.0 / dual_norm_inf));
        const ivf64 dual_ray_obj = vec_dot(q_orig, y_ray);
        if (dual_ray_obj > 0.0) {
            fiv_vec *KTy_ray = fiv_create_tensor1d(num_variables, FIV_64F1);
            fiv_lp_mat_matvec(KTy_ray, K_orig, y_ray, 1);
            const ivf64 dual_residual = vec_inf(KTy_ray);
            const ivf64 relative = (dual_ray_obj > 0.0)
                                 ? dual_residual / dual_ray_obj : INFINITY;
            if (relative < eps_primal_infeasible) {
                *out_status = FIV_LP_STATUS_PRIMAL_INFEASIBLE;
                out_info->certificate_quality = relative;
                out_info->dual_residual = dual_residual;
                terminated = 1;
            }
            release_temp(&KTy_ray);
        }
        release_temp(&y_ray);
    }

    /* (2) dual infeasibility: unbounded via primal ray x */
    if (!terminated) {
        const ivf64 primal_norm_inf = vec_inf(x_unscaled);
        if (primal_norm_inf > eps_zero) {
            fiv_vec *x_ray = fiv_create_tensor1d(num_variables, FIV_64F1);
            fiv_vec_scale(x_ray, FIV_VEC_NC(x_unscaled), FIV_SCALAR_FP64(1.0 / primal_norm_inf));
            const ivf64 primal_ray_obj = vec_dot(c_orig, x_ray);
            if (primal_ray_obj < 0.0) {
                fiv_vec *Kx_ray = fiv_create_tensor1d(num_constraints, FIV_64F1);
                fiv_lp_mat_matvec(Kx_ray, K_orig, x_ray, 0);
                const ivf64 *kx = Kx_ray->data.db;
                ivf64 max_primal_residual = 0.0;
                for (size_t i = 0; i < num_constraints; i++) {
                    const ivf64 value = (i < m1) ? ((kx[i] < 0.0) ? -kx[i] : 0.0) : kx[i];
                    if (fabs(value) > max_primal_residual) max_primal_residual = fabs(value);
                }
                const ivf64 relative = (-primal_ray_obj > 0.0)
                                     ? max_primal_residual / (-primal_ray_obj) : INFINITY;
                if (relative < eps_dual_infeasible) {
                    *out_status = FIV_LP_STATUS_DUAL_INFEASIBLE;
                    out_info->certificate_quality = relative;
                    out_info->primal_residual = max_primal_residual;
                    terminated = 1;
                }
                release_temp(&Kx_ray);
            }
            release_temp(&x_ray);
        }
    }

    if (terminated) {
        release_temp(&KT_orig_y); release_temp(&Kx_orig);
        return 1;
    }

    /* (3) optimality: relative gap + primal feasibility + stationarity */
    const ivf64 gap_num = fabs(dual_obj - primal_obj);
    const ivf64 gap_den = 1.0 + fabs(dual_obj) + fabs(primal_obj);
    const int gap_ok = (gap_num <= eps_tol * gap_den);

    /* primal feasibility residual: r_eq = b - A x, r_ineq = clamp(h - G x, 0) */
    const ivf64 *kxo = Kx_orig->data.db;
    const ivf64 *qo  = q_orig->data.db;
    ivf64 r_eq_sq = 0.0, r_ineq_sq = 0.0;
    for (size_t i = 0; i < num_constraints; i++) {
        if (i < m1) {
            const ivf64 residual = qo[i] - kxo[i];
            if (residual > 0.0) r_ineq_sq += residual * residual;
        } else {
            const ivf64 residual = kxo[i] - qo[i];
            r_eq_sq += residual * residual;
        }
    }
    const ivf64 feas = sqrt(r_eq_sq + r_ineq_sq);
    const ivf64 q_norm = vec_l2(q_orig);
    const int feas_ok = (feas <= eps_tol * (1.0 + q_norm));

    /* stationarity: ||g - lambda||, g = c - K^T y */
    fiv_vec *g_orig = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *lam_s  = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec_scale(g_orig, FIV_VEC_NC(c_orig), FIV_SCALAR_FP64(1.0));
    fiv_vec_axpy(g_orig, FIV_SCALAR_FP64(-1.0), KT_orig_y);
    fiv_lp_compute_lambda_box(lam_s, x_unscaled, g_orig, l_orig, u_orig, eps_tol);
    fiv_vec_axpy(g_orig, FIV_SCALAR_FP64(-1.0), lam_s);
    const ivf64 stat = vec_l2(g_orig);
    const ivf64 c_norm = vec_l2(c_orig);
    const int stat_ok = (stat <= eps_tol * (1.0 + c_norm));

    release_temp(&g_orig); release_temp(&lam_s);
    release_temp(&KT_orig_y); release_temp(&Kx_orig);

    if (gap_ok && feas_ok && stat_ok) {
        *out_status = FIV_LP_STATUS_OPTIMAL;
        return 1;
    }
    return 0;
}

/* =========================================================================
 * fiv_lp_solve  (pdlp.py::solve, lines 4-575)
 * ========================================================================= */
fiv_lp_status fiv_lp_solve(const fiv_lp_mat *K_orig, const fiv_vec *c_orig,
                           const fiv_vec *l_orig, const fiv_vec *u_orig,
                           const fiv_vec *q_orig, size_t num_inequality,
                           const fiv_lp_solve_params *params,
                           fiv_vec *x_out, fiv_vec *y_out, fiv_lp_solve_info *info)
{
    const ivf64 eps_zero = 1e-12;
    const size_t num_variables = c_orig->length;
    const size_t num_constraints = q_orig->length;
    const size_t m1 = num_inequality;
    const ivf64 eps_tol = (params != NULL) ? params->eps_tol : 1e-4;
    const ivf64 eps_primal_infeasible = (params != NULL) ? params->eps_primal_infeasible : 1e-8;
    const ivf64 eps_dual_infeasible   = (params != NULL) ? params->eps_dual_infeasible : 1e-8;
    const int ruiz_iterations = (params != NULL) ? params->ruiz_iterations : 10;
    const ivf64 pock_chambolle_alpha = (params != NULL) ? params->pock_chambolle_alpha : 1.0;
    const ivf64 smoothing = (params != NULL) ? params->primal_weight_smoothing : 0.5;
    const long iteration_limit = (params != NULL && params->iteration_limit > 0)
                               ? (long)params->iteration_limit : 10000;
    const double time_sec_limit = (params != NULL) ? params->time_sec_limit : INFINITY;
    const int max_inner_iters = 1000;
    const double beta_sufficient = 0.2, beta_necessary = 0.8, beta_artificial = 0.36;
    fiv_lp_status final_status = FIV_LP_STATUS_ITERATION_LIMIT;

    if (info != NULL) memset(info, 0, sizeof(*info));
    if (x_out != NULL) memset(x_out->data.db, 0, num_variables * sizeof(ivf64));
    if (y_out != NULL) memset(y_out->data.db, 0, num_constraints * sizeof(ivf64));

    /* ----------------------- trivial cases ----------------------- */
    if (num_variables == 0) {
        int feasible = 1;
        const ivf64 *qd = q_orig->data.db;
        for (size_t i = 0; i < num_constraints; i++) {
            if (i < m1) { if (qd[i] > eps_zero) feasible = 0; }
            else { if (fabs(qd[i]) > eps_zero) feasible = 0; }
        }
        if (info != NULL) { info->iterations = 0; info->solve_time_sec = 0.0; }
        if (feasible) {
            if (info != NULL) { info->primal_obj = 0.0; info->dual_obj = 0.0; }
            return FIV_LP_STATUS_OPTIMAL;
        }
        /* Farkas certificate y_ray */
        fiv_vec *y_ray = fiv_create_tensor1d(num_constraints, FIV_64F1);
        for (size_t i = 0; i < num_constraints; i++) {
            ivf64 v = 0.0;
            if (i < m1) v = (qd[i] > eps_zero) ? 1.0 : 0.0;
            else        v = (fabs(qd[i]) > eps_zero) ? 1.0 : 0.0;
            y_ray->data.db[i] = v;
        }
        const ivf64 norm = vec_inf(y_ray);
        if (norm > eps_zero) fiv_vec_scale(y_ray, FIV_VEC_NC(y_ray), FIV_SCALAR_FP64(1.0 / norm));
        if (y_out != NULL) memcpy(y_out->data.db, y_ray->data.db, num_constraints * sizeof(ivf64));
        release_temp(&y_ray);
        if (info != NULL) info->certificate_quality = 0.0;
        return FIV_LP_STATUS_PRIMAL_INFEASIBLE;
    }
    if (num_constraints == 0) {
        fiv_vec *x_sol = fiv_create_tensor1d(num_variables, FIV_64F1);
        int unbounded = 0;
        for (size_t i = 0; i < num_variables; i++) {
            const ivf64 ci = c_orig->data.db[i];
            const ivf64 li = l_orig->data.db[i], ui = u_orig->data.db[i];
            if (ci < -eps_zero)      x_sol->data.db[i] = ui;   /* minimize: x=u if c<0 */
            else if (ci > eps_zero)  x_sol->data.db[i] = li;   /* x=l if c>0 */
            else                     x_sol->data.db[i] = li;
            if (((ci < -eps_zero) && !isfinite(ui)) || ((ci > eps_zero) && !isfinite(li)))
                unbounded = 1;
        }
        if (!unbounded) {
            ivf64 obj = 0.0;
            for (size_t i = 0; i < num_variables; i++) obj += c_orig->data.db[i] * x_sol->data.db[i];
            if (x_out != NULL) memcpy(x_out->data.db, x_sol->data.db, num_variables * sizeof(ivf64));
            release_temp(&x_sol);
            if (info != NULL) { info->iterations = 0; info->solve_time_sec = 0.0;
                                info->primal_obj = obj; info->dual_obj = obj; }
            return FIV_LP_STATUS_OPTIMAL;
        }
        /* primal ray (dual unbounded certificate) */
        fiv_vec *x_ray = fiv_create_tensor1d(num_variables, FIV_64F1);
        for (size_t i = 0; i < num_variables; i++) x_ray->data.db[i] = 0.0;
        for (size_t i = 0; i < num_variables; i++) {
            const ivf64 ci = c_orig->data.db[i];
            if (((ci < -eps_zero) && !isfinite(u_orig->data.db[i])) ||
                ((ci > eps_zero) && !isfinite(l_orig->data.db[i]))) {
                x_ray->data.db[i] = (ci < 0.0) ? 1.0 : -1.0;
                break;
            }
        }
        if (x_out != NULL) memcpy(x_out->data.db, x_sol->data.db, num_variables * sizeof(ivf64));
        release_temp(&x_sol); release_temp(&x_ray);
        if (info != NULL) info->certificate_quality = 0.0;
        return FIV_LP_STATUS_DUAL_INFEASIBLE;
    }

    /* ----------------------- working copies ----------------------- */
    fiv_mat *work_dense = NULL;
    fiv_lp_mat *K_work = clone_lp_mat(K_orig, &work_dense);
    fiv_vec *c_work = clone_vec(c_orig);
    fiv_vec *l_work = clone_vec(l_orig);
    fiv_vec *u_work = clone_vec(u_orig);
    fiv_vec *q_work = clone_vec(q_orig);
    fiv_lp_rescaling *rescaling = fiv_create_lp_rescaling(num_variables, num_constraints);
    if (K_work == NULL || c_work == NULL || l_work == NULL || u_work == NULL ||
        q_work == NULL || rescaling == NULL) {
        fiv_release_lp_mat(&K_work);
        if (work_dense != NULL) fiv_release_tensor2d(&work_dense);
        release_temp(&c_work); release_temp(&l_work); release_temp(&u_work); release_temp(&q_work);
        fiv_release_lp_rescaling(&rescaling);
        return FIV_LP_STATUS_ITERATION_LIMIT; /* MEM failure; not a real status */
    }

    /* transposed view on the ORIGINAL (for termination checks). */
    fiv_lp_mat K_orig_local = *K_orig;
    K_orig_local.owns_data = 0;
    fiv_lp_mat_build_transpose(&K_orig_local);

    /* rescale the working problem (mutates K_work / c/l/u/q in place). */
    fiv_lp_rescale_solve(rescaling, K_work, c_work, l_work, u_work, q_work,
                          ruiz_iterations, pock_chambolle_alpha, eps_zero);
    /* CSC transposed view on the WORKING matrix (now scaled). */
    fiv_lp_mat_build_transpose(K_work);

    /* ----------------------- initializations ----------------------- */
    fiv_vec *x = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *y = fiv_create_tensor1d(num_constraints, FIV_64F1);
    /* x0 = proj_X(0) = clamp(0, l, u); y0 = 0 (pdlp.py initial dual iterate).
     * fiv_create_tensor1d does NOT zero its buffer, so both must be set here. */
    for (size_t i = 0; i < num_variables; i++) x->data.db[i] = 0.0;
    for (size_t i = 0; i < num_constraints; i++) y->data.db[i] = 0.0;
    fiv_vec_clamp(x, x, l_work, u_work);

    fiv_vec *row_sum = fiv_create_tensor1d(num_constraints, FIV_64F1);
    fiv_lp_mat_reduce_abs_pow(row_sum, K_work, 0, 1.0);  /* L1 row norm */
    ivf64 max_row = 0.0;
    for (size_t i = 0; i < num_constraints; i++)
        if (row_sum->data.db[i] > max_row) max_row = row_sum->data.db[i];
    ivf64 eta_hat = (max_row > eps_zero) ? 1.0 / max_row : 1.0;
    release_temp(&row_sum);

    const ivf64 c_norm = vec_l2(c_work);
    const ivf64 q_norm = vec_l2(q_work);
    ivf64 primal_weight = (c_norm > eps_zero && q_norm > eps_zero) ? (c_norm / q_norm) : 1.0;

    fiv_vec *x_prev = clone_vec(x);
    fiv_vec *y_prev = clone_vec(y);
    fiv_vec *x_c   = clone_vec(x);
    fiv_vec *y_c   = clone_vec(y);
    fiv_vec *x_bar = clone_vec(x);
    fiv_vec *y_bar = clone_vec(y);

    /* persistent scratch for the inner loop */
    fiv_vec *x_projected = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *y_projected = fiv_create_tensor1d(num_constraints, FIV_64F1);
    fiv_vec *diff_x = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *diff_y = fiv_create_tensor1d(num_constraints, FIV_64F1);
    fiv_vec *Kx_cur = fiv_create_tensor1d(num_constraints, FIV_64F1);
    fiv_vec *KTy_cur = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *Kx_bar = fiv_create_tensor1d(num_constraints, FIV_64F1);
    fiv_vec *KTy_bar = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *x_uns  = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *y_uns  = fiv_create_tensor1d(num_constraints, FIV_64F1);
    fiv_vec *Kx_orig_t = fiv_create_tensor1d(num_constraints, FIV_64F1);
    fiv_vec *KT_orig_y_t = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *x_c_new = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *y_c_new = fiv_create_tensor1d(num_constraints, FIV_64F1);
    fiv_vec *x_unscaled_last = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *y_unscaled_last = fiv_create_tensor1d(num_constraints, FIV_64F1);
    if (x == NULL || y == NULL || x_prev == NULL || y_prev == NULL || x_c == NULL ||
        y_c == NULL || x_bar == NULL || y_bar == NULL || x_projected == NULL ||
        y_projected == NULL || diff_x == NULL || diff_y == NULL || Kx_cur == NULL ||
        KTy_cur == NULL || Kx_bar == NULL || KTy_bar == NULL || x_uns == NULL ||
        y_uns == NULL || Kx_orig_t == NULL || KT_orig_y_t == NULL || x_c_new == NULL ||
        y_c_new == NULL || x_unscaled_last == NULL || y_unscaled_last == NULL) {
        goto cleanup;
    }

    const clock_t start_time = clock();
    long n_iterations = 0;
    int have_status = 0;

    while (1) {
        /* KKT of last restart point (working problem) */
        ivf64 kkt_last = 0.0;
        fiv_lp_kkt_error_sq(&kkt_last, K_work, c_work, l_work, u_work, q_work, m1,
                            x, y, primal_weight, eps_tol, eps_zero, NULL, NULL);

        /* save unscaled iterate for final return */
        fiv_lp_unscale_primal(x_unscaled_last, rescaling, x);
        fiv_lp_unscale_dual(y_unscaled_last, rescaling, y);

        ivf64 eta_sum = 0.0;
        fiv_vec_scale(x_bar, x, FIV_SCALAR_FP64(1.0));
        fiv_vec_scale(y_bar, y, FIV_SCALAR_FP64(1.0));
        ivf64 kkt_c_prev = kkt_last;
        ivf64 kkt_c_new = 0.0;

        int restart_triggered = 0;
        for (size_t t = 0; t < (size_t)max_inner_iters; t++) {
            fiv_lp_step_result step;
            fiv_lp_adaptive_step_pdhg(x, y, x_projected, y_projected, K_work,
                                      c_work, l_work, u_work, q_work, m1,
                                      primal_weight, eta_hat, (int)n_iterations,
                                      eps_zero, &step);
            fiv_vec_scale(x, x_projected, FIV_SCALAR_FP64(1.0));
            fiv_vec_scale(y, y_projected, FIV_SCALAR_FP64(1.0));
            eta_hat = step.eta_hat_next;

            eta_sum += step.eta_used;
            const ivf64 alpha = step.eta_used / eta_sum;
            fiv_tensor_sub(diff_x, x, x_bar);
            fiv_vec_axpy(x_bar, FIV_SCALAR_FP64(alpha), diff_x);
            fiv_tensor_sub(diff_y, y, y_bar);
            fiv_vec_axpy(y_bar, FIV_SCALAR_FP64(alpha), diff_y);

            n_iterations++;
            if (n_iterations >= iteration_limit) {
                final_status = FIV_LP_STATUS_ITERATION_LIMIT;
                fiv_lp_unscale_primal(x_unscaled_last, rescaling, x_bar);
                fiv_lp_unscale_dual(y_unscaled_last, rescaling, y_bar);
                have_status = 1; break;
            }
            if ((double)(clock() - start_time) / CLOCKS_PER_SEC >= time_sec_limit) {
                final_status = FIV_LP_STATUS_TIME_LIMIT;
                fiv_lp_unscale_primal(x_unscaled_last, rescaling, x_bar);
                fiv_lp_unscale_dual(y_unscaled_last, rescaling, y_bar);
                have_status = 1; break;
            }

            if (n_iterations <= 10 || n_iterations % 50 == 0) {
                fiv_lp_mat_matvec(Kx_cur, K_work, x, 0);
                fiv_lp_mat_matvec(KTy_cur, K_work, y, 1);
                fiv_lp_mat_matvec(Kx_bar, K_work, x_bar, 0);
                fiv_lp_mat_matvec(KTy_bar, K_work, y_bar, 1);
                fiv_lp_unscale_primal(x_uns, rescaling, x_bar);
                fiv_lp_unscale_dual(y_uns, rescaling, y_bar);
                fiv_lp_mat_matvec(Kx_orig_t, &K_orig_local, x_uns, 0);
                fiv_lp_mat_matvec(KT_orig_y_t, &K_orig_local, y_uns, 1);

                ivf64 kkt_cur = 0.0, kkt_avg = 0.0;
                fiv_lp_kkt_error_sq(&kkt_cur, K_work, c_work, l_work, u_work, q_work, m1,
                                    x, y, primal_weight, eps_tol, eps_zero, Kx_cur, KTy_cur);
                fiv_lp_kkt_error_sq(&kkt_avg, K_work, c_work, l_work, u_work, q_work, m1,
                                    x_bar, y_bar, primal_weight, eps_tol, eps_zero, Kx_bar, KTy_bar);
                if (kkt_cur < kkt_avg) {
                    fiv_vec_scale(x_c_new, x, FIV_SCALAR_FP64(1.0));
                    fiv_vec_scale(y_c_new, y, FIV_SCALAR_FP64(1.0));
                    kkt_c_new = kkt_cur;
                } else {
                    fiv_vec_scale(x_c_new, x_bar, FIV_SCALAR_FP64(1.0));
                    fiv_vec_scale(y_c_new, y_bar, FIV_SCALAR_FP64(1.0));
                    kkt_c_new = kkt_avg;
                }

                fiv_lp_status term_status;
                fiv_lp_solve_info term_info;
                memset(&term_info, 0, sizeof(term_info));
                int term = termination_criteria(&term_status, &term_info, &K_orig_local,
                                                c_orig, l_orig, u_orig, q_orig, m1,
                                                x_uns, y_uns, eps_tol,
                                                eps_primal_infeasible, eps_dual_infeasible);
                if (term) {
                    if (n_iterations < 10 &&
                        (term_status == FIV_LP_STATUS_PRIMAL_INFEASIBLE ||
                         term_status == FIV_LP_STATUS_DUAL_INFEASIBLE)) {
                        /* ignore infeasibility before warm-up */
                    } else {
                        final_status = term_status;
                        if (info != NULL) *info = term_info;
                        fiv_vec_scale(x_unscaled_last, x_uns, FIV_SCALAR_FP64(1.0));
                        fiv_vec_scale(y_unscaled_last, y_uns, FIV_SCALAR_FP64(1.0));
                        have_status = 1; break;
                    }
                }

                const int cond_i   = (kkt_c_new <= beta_sufficient * beta_sufficient * kkt_last);
                const int cond_ii  = (kkt_c_new <= beta_necessary * beta_necessary * kkt_last) &&
                                     (t > 0) && (kkt_c_new > kkt_c_prev);
                const int cond_iii = (t >= (size_t)(beta_artificial * (double)n_iterations));
                if (cond_i || cond_ii || cond_iii) {
                    fiv_vec_scale(x_c, x_c_new, FIV_SCALAR_FP64(1.0));
                    fiv_vec_scale(y_c, y_c_new, FIV_SCALAR_FP64(1.0));
                    restart_triggered = 1; break;
                }
                kkt_c_prev = kkt_c_new;
            }
        }

        if (have_status) break;

        if (!restart_triggered) {
            fiv_vec_scale(x_c, x_c_new, FIV_SCALAR_FP64(1.0));
            fiv_vec_scale(y_c, y_c_new, FIV_SCALAR_FP64(1.0));
        }

        /* restart from candidate */
        fiv_vec_scale(x, x_c, FIV_SCALAR_FP64(1.0));
        fiv_vec_scale(y, y_c, FIV_SCALAR_FP64(1.0));
        primal_weight = fiv_lp_primal_weight_update(x, y, x_prev, y_prev,
                                                    primal_weight, smoothing, eps_zero);
        fiv_vec_scale(x_prev, x, FIV_SCALAR_FP64(1.0));
        fiv_vec_scale(y_prev, y, FIV_SCALAR_FP64(1.0));
    }

    /* ----------------------- final statistics ----------------------- */
    if (info != NULL) {
        info->iterations = n_iterations;
        info->solve_time_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        if (final_status == FIV_LP_STATUS_OPTIMAL ||
            final_status == FIV_LP_STATUS_ITERATION_LIMIT ||
            final_status == FIV_LP_STATUS_TIME_LIMIT) {
            ivf64 kkt_sq = 0.0;
            fiv_lp_kkt_error_sq(&kkt_sq, K_work, c_work, l_work, u_work, q_work, m1,
                                x, y, primal_weight, eps_tol, eps_zero, NULL, NULL);
            info->kkt_error_sq = kkt_sq;
            const ivf64 primal_obj = vec_dot(c_orig, x_unscaled_last);
            ivf64 dual_obj = 0.0;
            compute_dual_objective(&dual_obj, &K_orig_local, c_orig, l_orig, u_orig, q_orig,
                                   x_unscaled_last, y_unscaled_last, NULL, eps_tol);
            info->primal_obj = primal_obj;
            info->dual_obj = dual_obj;
            info->duality_gap = fabs(primal_obj - dual_obj);
            info->relative_gap = info->duality_gap /
                                 (1.0 + fabs(primal_obj) + fabs(dual_obj));

            /* primal residual on unscaled problem */
            fiv_lp_mat_matvec(Kx_orig_t, &K_orig_local, x_unscaled_last, 0);
            const ivf64 *kxo = Kx_orig_t->data.db;
            const ivf64 *qo = q_orig->data.db;
            ivf64 r_eq_sq = 0.0, r_ineq_sq = 0.0;
            for (size_t i = 0; i < num_constraints; i++) {
                if (i < m1) {
                    const ivf64 r = qo[i] - kxo[i];
                    if (r > 0.0) r_ineq_sq += r * r;
                } else {
                    const ivf64 r = kxo[i] - qo[i];
                    r_eq_sq += r * r;
                }
            }
            info->primal_residual = sqrt(r_eq_sq + r_ineq_sq);

            /* dual residual = ||c - K^T y - lambda|| on unscaled problem */
            fiv_lp_mat_matvec(KT_orig_y_t, &K_orig_local, y_unscaled_last, 1);
            fiv_vec *g_final = fiv_create_tensor1d(num_variables, FIV_64F1);
            fiv_vec *lam_final = fiv_create_tensor1d(num_variables, FIV_64F1);
            fiv_vec_scale(g_final, FIV_VEC_NC(c_orig), FIV_SCALAR_FP64(1.0));
            fiv_vec_axpy(g_final, FIV_SCALAR_FP64(-1.0), KT_orig_y_t);
            fiv_lp_compute_lambda_box(lam_final, x_unscaled_last, g_final, l_orig, u_orig, eps_tol);
            fiv_vec_axpy(g_final, FIV_SCALAR_FP64(-1.0), lam_final);
            info->dual_residual = vec_l2(g_final);
            release_temp(&g_final); release_temp(&lam_final);
        }
    }

    if (x_out != NULL) memcpy(x_out->data.db, x_unscaled_last->data.db, num_variables * sizeof(ivf64));
    if (y_out != NULL) memcpy(y_out->data.db, y_unscaled_last->data.db, num_constraints * sizeof(ivf64));

cleanup:
    fiv_release_lp_mat(&K_work);
    if (work_dense != NULL) fiv_release_tensor2d(&work_dense);
    fiv_release_lp_rescaling(&rescaling);
    release_temp(&c_work); release_temp(&l_work); release_temp(&u_work); release_temp(&q_work);
    if (K_orig_local.transpose_view != NULL) fiv_release_sp_matrix(&K_orig_local.transpose_view);
    release_temp(&x); release_temp(&y); release_temp(&x_prev); release_temp(&y_prev);
    release_temp(&x_c); release_temp(&y_c); release_temp(&x_bar); release_temp(&y_bar);
    release_temp(&x_projected); release_temp(&y_projected); release_temp(&diff_x); release_temp(&diff_y);
    release_temp(&Kx_cur); release_temp(&KTy_cur); release_temp(&Kx_bar); release_temp(&KTy_bar);
    release_temp(&x_uns); release_temp(&y_uns); release_temp(&Kx_orig_t); release_temp(&KT_orig_y_t);
    release_temp(&x_c_new); release_temp(&y_c_new);
    release_temp(&x_unscaled_last); release_temp(&y_unscaled_last);

    return final_status;
}
