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

#include "fiv_lp_pdhg.h"
#include "fiv_lp_mat.h"    /* fiv_lp_mat_matvec / fiv_lp_mat_build_transpose */
#include "fiv_lp_vec.h"    /* fiv_vec_clamp */
#include "fiv_matrix.h"    /* fiv_vec_norm / fiv_vec_dot / fiv_vec_axpy / fiv_vec_scale */
#include "fiv_ctensor.h"   /* fiv_create_tensor1d / fiv_release_tensor1d */

#include <math.h>          /* fabs, pow, isfinite, INFINITY */

/* Local helper: release a possibly-NULL temp vector. */
static void release_temp(fiv_vec **temp_vector)
{
    if (temp_vector != NULL && *temp_vector != NULL)
        fiv_release_tensor1d(temp_vector);
}

/* FastIV's fiv_vec_scale / fiv_vec_axpy / fiv_vec_dot / fiv_vec_norm take a
 * non-const fiv_vec* for their read-only operands. Our PDHG API keeps those
 * operands const (they are never mutated); the cast below is the documented
 * boundary crossing and is safe because those kernels only read the operand. */
#define FIV_VEC_NC(v) ((fiv_vec *)(v))

/* ----------------------------------------------------------------------------
 * adaptive_step_pdhg  (pdlp.py lines 386-421)
 * ------------------------------------------------------------------------- */
fiv_ret fiv_lp_adaptive_step_pdhg(const fiv_vec *x, const fiv_vec *y,
                                  fiv_vec *x_projected, fiv_vec *y_projected,
                                  const fiv_lp_mat *K, const fiv_vec *c,
                                  const fiv_vec *l, const fiv_vec *u, const fiv_vec *q,
                                  size_t num_inequality,
                                  ivf64 primal_weight, ivf64 eta_hat,
                                  int iteration_index, ivf64 eps_zero,
                                  fiv_lp_step_result *result)
{
    if (x == NULL || y == NULL || x_projected == NULL || y_projected == NULL ||
        K == NULL || c == NULL || l == NULL || u == NULL || q == NULL || result == NULL)
        return FIV_RET_ERR_PARA;
    const size_t num_variables = c->length;
    const size_t num_constraints = q->length;
    if (x->length != num_variables || y->length != num_constraints ||
        x_projected->length != num_variables || y_projected->length != num_constraints ||
        l->length != num_variables || u->length != num_variables ||
        K->rows != num_constraints || K->cols != num_variables)
        return FIV_RET_ERR_PARA;

    const ivf64 step_weight = (primal_weight > eps_zero) ? primal_weight : eps_zero;
    ivf64 eta = (eta_hat > eps_zero) ? eta_hat : eps_zero;

    const ivf64 kp1 = (ivf64)(iteration_index + 1);
    const ivf64 fac1 = (iteration_index == 0) ? 1.0 : (1.0 - pow(kp1, -0.3));
    const ivf64 fac2 = 1.0 + pow(kp1, -0.6);

    /* scratch buffers (reused every loop iteration) */
    fiv_vec *KTy   = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *grad_x = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *t_vec  = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *Kt    = fiv_create_tensor1d(num_constraints, FIV_64F1);
    fiv_vec *grad_y = fiv_create_tensor1d(num_constraints, FIV_64F1);
    fiv_vec *y_raw  = fiv_create_tensor1d(num_constraints, FIV_64F1);
    fiv_vec *dx    = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *dy    = fiv_create_tensor1d(num_constraints, FIV_64F1);
    fiv_vec *Kdx   = fiv_create_tensor1d(num_constraints, FIV_64F1);
    if (KTy == NULL || grad_x == NULL || t_vec == NULL || Kt == NULL ||
        grad_y == NULL || y_raw == NULL || dx == NULL || dy == NULL || Kdx == NULL) {
        release_temp(&KTy); release_temp(&grad_x); release_temp(&t_vec);
        release_temp(&Kt);  release_temp(&grad_y); release_temp(&y_raw);
        release_temp(&dx);  release_temp(&dy);     release_temp(&Kdx);
        return FIV_RET_ERR_MEM;
    }

    fiv_scalar scalar_acc;
    ivf64 eta_p = eta;
    int safety_iterations = 0;

    while (1) {
        /* KTy = Kᵀ y ;  grad_x = c - KTy */
        fiv_lp_mat_matvec(KTy, K, y, 1);
        fiv_vec_scale(grad_x, FIV_VEC_NC(c), FIV_SCALAR_FP64(1.0));
        fiv_vec_axpy(grad_x, FIV_SCALAR_FP64(-1.0), KTy);

        /* x_p = proj_X( x - (eta / w) * grad_x ) */
        const ivf64 step_x = eta / step_weight;
        fiv_vec_scale(x_projected, FIV_VEC_NC(x), FIV_SCALAR_FP64(1.0));
        fiv_vec_axpy(x_projected, FIV_SCALAR_FP64(-step_x), grad_x);
        fiv_vec_clamp(x_projected, x_projected, l, u);   /* proj_X (box) */

        /* t = 2 x_p - x ;  Kt = K t */
        fiv_vec_scale(t_vec, x_projected, FIV_SCALAR_FP64(2.0));
        fiv_vec_axpy(t_vec, FIV_SCALAR_FP64(-1.0), FIV_VEC_NC(x));
        fiv_lp_mat_matvec(Kt, K, t_vec, 0);

        /* grad_y = q - Kt ;  y_raw = y + (eta * w) * grad_y */
        fiv_vec_scale(grad_y, FIV_VEC_NC(q), FIV_SCALAR_FP64(1.0));
        fiv_vec_axpy(grad_y, FIV_SCALAR_FP64(-1.0), Kt);
        fiv_vec_scale(y_raw, FIV_VEC_NC(y), FIV_SCALAR_FP64(1.0));
        fiv_vec_axpy(y_raw, FIV_SCALAR_FP64(eta * step_weight), grad_y);

        /* y_p = proj_Y(y_raw): first m1 components clamped to >= 0, rest free. */
        const ivf64 *raw_data = y_raw->data.db;
        ivf64 *projected_data = y_projected->data.db;
        for (size_t index = 0; index < num_constraints; index++) {
            ivf64 value = raw_data[index];
            if (index < num_inequality && value < 0.0) value = 0.0;
            projected_data[index] = value;
        }

        /* dx = x_p - x ;  dy = y_p - y */
        fiv_vec_scale(dx, x_projected, FIV_SCALAR_FP64(1.0));
        fiv_vec_axpy(dx, FIV_SCALAR_FP64(-1.0), FIV_VEC_NC(x));
        fiv_vec_scale(dy, y_projected, FIV_SCALAR_FP64(1.0));
        fiv_vec_axpy(dy, FIV_SCALAR_FP64(-1.0), FIV_VEC_NC(y));

        /* num = w * ||dx||^2 + ||dy||^2 / w ;  denom = 2 |dyᵀ K dx| */
        fiv_vec_dot(&scalar_acc, dx, dx);
        const ivf64 dx_sq = scalar_acc.data.value_fp64;
        fiv_vec_dot(&scalar_acc, dy, dy);
        const ivf64 dy_sq = scalar_acc.data.value_fp64;
        const ivf64 movement = step_weight * dx_sq + dy_sq / step_weight;

        fiv_lp_mat_matvec(Kdx, K, dx, 0);
        fiv_vec_dot(&scalar_acc, dy, Kdx);
        const ivf64 interaction = 2.0 * fabs(scalar_acc.data.value_fp64);

        const ivf64 bar_eta = (interaction <= eps_zero) ? INFINITY : (movement / interaction);
        eta_p = fac1 * bar_eta;
        const ivf64 capped = fac2 * eta;
        if (eta_p > capped) eta_p = capped;
        if (eta_p < eps_zero) eta_p = eps_zero;

        if (eta <= bar_eta) {
            result->eta_used = eta;
            result->eta_hat_next = eta_p;
            release_temp(&KTy); release_temp(&grad_x); release_temp(&t_vec);
            release_temp(&Kt);  release_temp(&grad_y); release_temp(&y_raw);
            release_temp(&dx);  release_temp(&dy);     release_temp(&Kdx);
            return FIV_RET_OK;
        }
        eta = eta_p;

        if (++safety_iterations > 10000) {   /* theory guarantees termination */
            release_temp(&KTy); release_temp(&grad_x); release_temp(&t_vec);
            release_temp(&Kt);  release_temp(&grad_y); release_temp(&y_raw);
            release_temp(&dx);  release_temp(&dy);     release_temp(&Kdx);
            return FIV_RET_ERR_UNKNOWN;
        }
    }
}


/* ----------------------------------------------------------------------------
 * primal_weight_update  (pdlp.py lines 219-231)
 * ------------------------------------------------------------------------- */
ivf64 fiv_lp_primal_weight_update(const fiv_vec *x_new, const fiv_vec *y_new,
                                  const fiv_vec *x_old, const fiv_vec *y_old,
                                  ivf64 primal_weight_old, ivf64 smoothing,
                                  ivf64 eps_zero)
{
    if (x_new == NULL || y_new == NULL || x_old == NULL || y_old == NULL)
        return primal_weight_old;

    fiv_vec *delta_x = fiv_create_tensor1d(x_new->length, FIV_64F1);
    fiv_vec *delta_y = fiv_create_tensor1d(y_new->length, FIV_64F1);
    if (delta_x == NULL || delta_y == NULL) {
        release_temp(&delta_x); release_temp(&delta_y);
        return primal_weight_old;
    }

    fiv_scalar norm_scalar;
    fiv_vec_scale(delta_x, FIV_VEC_NC(x_new), FIV_SCALAR_FP64(1.0));
    fiv_vec_axpy(delta_x, FIV_SCALAR_FP64(-1.0), FIV_VEC_NC(x_old));
    fiv_vec_norm(&norm_scalar, delta_x, FIV_L2_NORM);
    const ivf64 delta_x_norm = norm_scalar.data.value_fp64;

    fiv_vec_scale(delta_y, FIV_VEC_NC(y_new), FIV_SCALAR_FP64(1.0));
    fiv_vec_axpy(delta_y, FIV_SCALAR_FP64(-1.0), FIV_VEC_NC(y_old));
    fiv_vec_norm(&norm_scalar, delta_y, FIV_L2_NORM);
    const ivf64 delta_y_norm = norm_scalar.data.value_fp64;

    release_temp(&delta_x); release_temp(&delta_y);

    if (delta_x_norm > eps_zero && delta_y_norm > eps_zero) {
        const ivf64 ratio = delta_y_norm / delta_x_norm;
        return pow(ratio, smoothing) * pow(primal_weight_old, 1.0 - smoothing);
    }
    return primal_weight_old;
}


/* ----------------------------------------------------------------------------
 * compute_lambda_for_box  (pdlp.py lines 234-246)
 * ------------------------------------------------------------------------- */
fiv_ret fiv_lp_compute_lambda_box(fiv_vec *lambda_out, const fiv_vec *x,
                                  const fiv_vec *gradient, const fiv_vec *lower,
                                  const fiv_vec *upper, ivf64 eps_tol)
{
    if (lambda_out == NULL || x == NULL || gradient == NULL ||
        lower == NULL || upper == NULL)
        return FIV_RET_ERR_PARA;
    if (lambda_out->dtype != FIV_64F1 || x->dtype != FIV_64F1 ||
        gradient->dtype != FIV_64F1 || lower->dtype != FIV_64F1 ||
        upper->dtype != FIV_64F1)
        return FIV_RET_ERR_PARA;
    if (lambda_out->length != x->length || x->length != gradient->length ||
        gradient->length != lower->length || lower->length != upper->length)
        return FIV_RET_ERR_PARA;

    const ivf64 *x_data = x->data.db;
    const ivf64 *grad_data = gradient->data.db;
    const ivf64 *lower_data = lower->data.db;
    const ivf64 *upper_data = upper->data.db;
    ivf64 *lambda_data = lambda_out->data.db;

    for (size_t index = 0; index < x->length; index++) {
        const ivf64 xi = x_data[index];
        const ivf64 gi = grad_data[index];
        const ivf64 li = lower_data[index];
        const ivf64 ui = upper_data[index];

        int at_lower = (isfinite(li) && (xi <= li + eps_tol)) ? 1 : 0;
        int at_upper = (isfinite(ui) && (xi >= ui - eps_tol)) ? 1 : 0;

        /* numerically-tight box where both flags fire: assign to the nearer side */
        if (at_lower && at_upper) {
            const ivf64 dist_lower = fabs(xi - li);
            const ivf64 dist_upper = fabs(ui - xi);
            at_lower = (dist_lower <= dist_upper) ? 1 : 0;
            at_upper = (dist_upper < dist_lower) ? 1 : 0;
        }

        ivf64 value = 0.0;
        if (at_lower)
            value = (gi > 0.0) ? gi : 0.0;       /* clamp(g, min=0) */
        else if (at_upper)
            value = (gi < 0.0) ? gi : 0.0;        /* clamp(g, max=0) */
        lambda_data[index] = value;
    }
    return FIV_RET_OK;
}


/* ----------------------------------------------------------------------------
 * kkt_error_sq  (pdlp.py lines 267-303, on the WORKING/scaled problem)
 * ------------------------------------------------------------------------- */
fiv_ret fiv_lp_kkt_error_sq(ivf64 *kkt_sq_out,
                             const fiv_lp_mat *K, const fiv_vec *c,
                             const fiv_vec *l, const fiv_vec *u, const fiv_vec *q,
                             size_t num_inequality, const fiv_vec *x, const fiv_vec *y,
                             ivf64 primal_weight, ivf64 eps_tol, ivf64 eps_zero,
                             const fiv_vec *Kx_cached, const fiv_vec *KTy_cached)
{
    if (kkt_sq_out == NULL || K == NULL || c == NULL || l == NULL || u == NULL ||
        q == NULL || x == NULL || y == NULL)
        return FIV_RET_ERR_PARA;
    const size_t num_variables = c->length;
    const size_t num_constraints = q->length;
    if (x->length != num_variables || y->length != num_constraints ||
        l->length != num_variables || u->length != num_variables ||
        K->rows != num_constraints || K->cols != num_variables)
        return FIV_RET_ERR_PARA;

    const ivf64 step_weight = (primal_weight > eps_zero) ? primal_weight : eps_zero;

    /* Kx (length m) and KTy (length n); reuse caller cache when supplied. */
    fiv_vec *owned_Kx = NULL;
    fiv_vec *owned_KTy = NULL;
    fiv_vec *Kx = (Kx_cached != NULL) ? (fiv_vec *)Kx_cached
                                      : fiv_create_tensor1d(num_constraints, FIV_64F1);
    fiv_vec *KTy = (KTy_cached != NULL) ? (fiv_vec *)KTy_cached
                                       : fiv_create_tensor1d(num_variables, FIV_64F1);
    if (Kx == NULL || KTy == NULL) {
        release_temp(&Kx); release_temp(&KTy);
        return FIV_RET_ERR_MEM;
    }
    if (Kx_cached == NULL) { fiv_lp_mat_matvec(Kx, K, x, 0); owned_Kx = Kx; }
    if (KTy_cached == NULL) { fiv_lp_mat_matvec(KTy, K, y, 1); owned_KTy = KTy; }

    /* term1 = w^2 ( ||r_eq||^2 + ||r_ineq||^2 ).
       Kx = [Gx ; Ax];  q = [h ; b].  r_ineq = clamp(h - Gx, 0); r_eq = Ax - b. */
    const ivf64 *Kx_data = Kx->data.db;
    const ivf64 *q_data = q->data.db;
    ivf64 r_eq_sq = 0.0, r_ineq_sq = 0.0;
    for (size_t index = 0; index < num_constraints; index++) {
        if (index < num_inequality) {
            const ivf64 residual = q_data[index] - Kx_data[index];
            if (residual > 0.0) r_ineq_sq += residual * residual;
        } else {
            const ivf64 residual = Kx_data[index] - q_data[index];
            r_eq_sq += residual * residual;
        }
    }
    const ivf64 term1 = step_weight * step_weight * (r_eq_sq + r_ineq_sq);

    /* g = c - KTy ; lambda = box(g) ; rs = g - lambda (reuse g buffer) */
    fiv_vec *gradient = fiv_create_tensor1d(num_variables, FIV_64F1);
    fiv_vec *lambda = fiv_create_tensor1d(num_variables, FIV_64F1);
    if (gradient == NULL || lambda == NULL) {
        release_temp(&gradient); release_temp(&lambda);
        release_temp(&owned_Kx); release_temp(&owned_KTy);
        return FIV_RET_ERR_MEM;
    }
    fiv_vec_scale(gradient, FIV_VEC_NC(c), FIV_SCALAR_FP64(1.0));
    fiv_vec_axpy(gradient, FIV_SCALAR_FP64(-1.0), KTy);
    fiv_lp_compute_lambda_box(lambda, x, gradient, l, u, eps_tol);
    fiv_vec_axpy(gradient, FIV_SCALAR_FP64(-1.0), lambda);   /* gradient := rs */

    fiv_scalar scalar_acc;
    fiv_vec_dot(&scalar_acc, gradient, gradient);
    const ivf64 term2 = (1.0 / (step_weight * step_weight)) * scalar_acc.data.value_fp64;

    /* l_term / u_term via sum_finite_products(l, lam_pos) / sum_finite_products(u, lam_neg). */
    const ivf64 *lambda_data = lambda->data.db;
    const ivf64 *lower_data = l->data.db;
    const ivf64 *upper_data = u->data.db;
    ivf64 l_term = 0.0, u_term = 0.0;
    for (size_t index = 0; index < num_variables; index++) {
        const ivf64 lam_i = lambda_data[index];
        const ivf64 lam_pos = (lam_i > 0.0) ? lam_i : 0.0;
        const ivf64 lam_neg = (lam_i < 0.0) ? -lam_i : 0.0;
        if (isfinite(lower_data[index])) l_term += lower_data[index] * lam_pos;
        if (isfinite(upper_data[index])) u_term += upper_data[index] * lam_neg;
    }

    /* scalar = qᵀ y + l_term - u_term - cᵀ x ;  term3 = scalar^2 */
    fiv_vec_dot(&scalar_acc, FIV_VEC_NC(q), y);
    const ivf64 qy = scalar_acc.data.value_fp64;
    fiv_vec_dot(&scalar_acc, FIV_VEC_NC(c), x);
    const ivf64 cx = scalar_acc.data.value_fp64;
    const ivf64 scalar = qy + l_term - u_term - cx;
    const ivf64 term3 = scalar * scalar;

    *kkt_sq_out = term1 + term2 + term3;

    release_temp(&gradient); release_temp(&lambda);
    release_temp(&owned_Kx); release_temp(&owned_KTy);
    return FIV_RET_OK;
}
