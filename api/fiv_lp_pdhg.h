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

#ifndef _FIV_LP_PDHG_H_
#define _FIV_LP_PDHG_H_

#include "fiv_ctensor.h"   /* fiv_vec, fiv_ret, ivf64 */
#include "fiv_lp_mat.h"    /* fiv_lp_mat */

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * PDHG core for the PDLP linear-programming solver (M3).
 *
 * Faithful port of the subprocedures in pdlp.py:
 *   - adaptive_step_pdhg  (pdlp.py lines 386-421)
 *   - primal_weight_update (lines 219-231)
 *   - compute_lambda_for_box (lines 234-246)
 *   - kkt_error_sq          (lines 267-303)
 *
 * Every function consumes the unified fiv_lp_mat / fiv_vec layer (built in M1)
 * so the algorithm core never sees DENSE vs SPARSE. The KKT / stationarity math
 * operates on the WORKING (scaled) problem -- pdlp.py's kkt_error_sq uses the
 * scaled K, c, l, u, q -- while termination_criteria (M4) works on the unscaled
 * problem. num_inequality (m1) splits the constraint matrix q = [h; b] and the
 * dual projection: the first m1 dual components are non-negative, the rest free.
 *
 * Naming rule: only fiv_create_* returns a pointer; every entry here returns
 * fiv_ret (or a plain ivf64 value for primal_weight_update).
 * ========================================================================= */


/* Result of one adaptive PDHG step: the step size actually used (eta_used) and
 * the next candidate step-size bound (eta_hat_next, = eta_p in pdlp.py). */
typedef struct {
    ivf64 eta_used;
    ivf64 eta_hat_next;
} fiv_lp_step_result;


/* One adaptive PDHG step (pdlp.py adaptive_step_pdhg).
 *
 *   x_p = proj_X(x - (eta / w) * (c - Kᵀ y))
 *   y_p = proj_Y(y + (eta * w) * (q - K (2 x_p - x)))
 *   ... adaptive acceptance / eta update (while loop) ...
 *
 * The current iterate is (x, y); the projected candidate is written to
 * (*x_projected, *y_projected), which MUST be distinct, caller-owned vectors of
 * length n and m respectively (they must NOT alias x / y). The accepted step
 * sizes are written to *result.
 *
 * x_projected / y_projected / K / c / l / u / q must be FIV_64F1 and correctly
 * dimensioned (K is m x n, c/l/u length n, q length m). */
fiv_ret fiv_lp_adaptive_step_pdhg(const fiv_vec *x, const fiv_vec *y,
                                  fiv_vec *x_projected, fiv_vec *y_projected,
                                  const fiv_lp_mat *K, const fiv_vec *c,
                                  const fiv_vec *l, const fiv_vec *u, const fiv_vec *q,
                                  size_t num_inequality,
                                  ivf64 primal_weight, ivf64 eta_hat,
                                  int iteration_index, ivf64 eps_zero,
                                  fiv_lp_step_result *result);


/* Primal-weight update (pdlp.py primal_weight_update).
 *   dx = ||x_new - x_old||, dy = ||y_new - y_old||
 *   if dx, dy > eps_zero:  return (dy / dx)^smoothing * w_old^(1 - smoothing)
 *   else:                  return w_old
 * Pure value (no pointer), so it returns ivf64 directly (not a fiv_ret). */
ivf64 fiv_lp_primal_weight_update(const fiv_vec *x_new, const fiv_vec *y_new,
                                  const fiv_vec *x_old, const fiv_vec *y_old,
                                  ivf64 primal_weight_old, ivf64 smoothing,
                                  ivf64 eps_zero);


/* Box normal-cone component lambda (pdlp.py compute_lambda_for_box).
 *   g = c - Kᵀ y ;  lambda[i] = clamp(g[i], min=0) if x[i] at lower bound,
 *                            clamp(g[i], max=0) if x[i] at upper bound, else 0.
 * Handles the numerically-tight case where x[i] is within eps_tol of BOTH
 * bounds (e.g. a fixed variable): it is assigned to the nearer side.
 * lower / upper may carry +-INF (free side). lambda_out length == n. */
fiv_ret fiv_lp_compute_lambda_box(fiv_vec *lambda_out, const fiv_vec *x,
                                  const fiv_vec *gradient, const fiv_vec *lower,
                                  const fiv_vec *upper, ivf64 eps_tol);


/* Squared KKT error (pdlp.py kkt_error_sq, equation (5)) on the WORKING problem.
 *
 *   r_eq = Ax - b ;  r_ineq = clamp(h - Gx, min=0)
 *   term1 = w^2 (||r_eq||^2 + ||r_ineq||^2)
 *   g = c - Kᵀ y ;  lambda = box(g) ;  rs = g - lambda
 *   term2 = (1 / w^2) ||rs||^2
 *   lam_pos = clamp(lambda, min=0) ; lam_neg = clamp(-lambda, min=0)
 *   l_term = sum_finite(l, lam_pos) ; u_term = sum_finite(u, lam_neg)
 *   scalar = qᵀ y + l_term - u_term - cᵀ x
 *   term3 = scalar^2
 *   return term1 + term2 + term3
 *
 * K must support BOTH K@x (CSR via fiv_lp_mat_matvec transpose==0) and Kᵀ y
 * (CSC via transpose_view, see fiv_lp_mat_build_transpose). num_inequality (m1)
 * splits q = [h; b] (first m1 entries are h, the rest b). Kx_cached / KTy_cached
 * are optional caller-supplied matvec results (length m / n); pass NULL to have
 * them computed internally. eps_tol drives the box slack in lambda; eps_zero
 * guards the w clamp. *kkt_sq_out receives the result. */
fiv_ret fiv_lp_kkt_error_sq(ivf64 *kkt_sq_out,
                             const fiv_lp_mat *K, const fiv_vec *c,
                             const fiv_vec *l, const fiv_vec *u, const fiv_vec *q,
                             size_t num_inequality, const fiv_vec *x, const fiv_vec *y,
                             ivf64 primal_weight, ivf64 eps_tol, ivf64 eps_zero,
                             const fiv_vec *Kx_cached, const fiv_vec *KTy_cached);


#ifdef __cplusplus
}
#endif

#endif  /* _FIV_LP_PDHG_H_ */
