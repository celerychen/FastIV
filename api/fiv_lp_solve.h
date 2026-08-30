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

#ifndef _FIV_LP_SOLVE_H_
#define _FIV_LP_SOLVE_H_

#include "fiv_ctensor.h"   /* fiv_vec, fiv_ret, ivf64 */
#include "fiv_lp_mat.h"    /* fiv_lp_mat */

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * PDLP solve() driver (M4).
 *
 * Faithful port of pdlp.py::solve (lines 4-575): rescale (M2) -> initialize ->
 * restart loop (adaptive_step_pdhg + primal_weight_update, M3) -> termination
 * criteria (operates on the UNSCALED problem) -> unscale -> return.
 *
 * Orchestrates the M1 (fiv_lp_mat / fiv_lp_vec) + M2 (fiv_lp_rescale) + M3
 * (fiv_lp_pdhg) layers. The algorithm core never sees DENSE vs SPARSE: both
 * the original and the working constraint matrix are fiv_lp_mat, and the
 * matvec/reduce calls dispatch internally. termination_criteria is a private
 * static helper here (it works on the unscaled problem and is tightly coupled
 * to solve()), so it is NOT part of the public pdhg.h surface.
 *
 * Problem convention (matches pdlp.py and M1/M2/M3 stacked representation):
 *   minimize    c^T x
 *   subject to  G x >= h        (first num_inequality rows of K / q)
 *              A x  = b        (remaining rows)
 *              l <= x <= u
 *   K is the STACKED matrix [G ; A] (m x n); q is [h ; b] (length m).
 *
 * Naming rule: solve() returns an enum (not a pointer), so it is NOT a
 * fiv_create_*; it writes the unscaled solutions into caller-owned x_out / y_out.
 * ========================================================================= */

typedef enum {
    FIV_LP_STATUS_OPTIMAL = 0,
    FIV_LP_STATUS_PRIMAL_INFEASIBLE,
    FIV_LP_STATUS_DUAL_INFEASIBLE,
    FIV_LP_STATUS_ITERATION_LIMIT,
    FIV_LP_STATUS_TIME_LIMIT
} fiv_lp_status;

typedef struct {
    int    iteration_limit;          /* 10000 */
    int    ruiz_iterations;          /* 10 */
    double pock_chambolle_alpha;     /* 1.0 (0 = disable) */
    double primal_weight_smoothing;  /* 0.5 */
    double eps_tol;                  /* 1e-4 */
    double eps_primal_infeasible;    /* 1e-8 */
    double eps_dual_infeasible;      /* 1e-8 */
    double time_sec_limit;           /* INFINITY = unlimited */
    int    verbose;
} fiv_lp_solve_params;

typedef struct {
    double solve_time_sec;
    long   iterations;
    double primal_obj, dual_obj;
    double duality_gap, relative_gap;
    double primal_residual, dual_residual;
    double kkt_error_sq;
    double certificate_quality;     /* infeasible / unbounded certificate */
} fiv_lp_solve_info;

/* Solve the LP described by (K_orig, c_orig, l_orig, u_orig, q_orig,
 * num_inequality). Writes the UNSCALED primal solution into x_out (length n)
 * and dual solution into y_out (length m), and statistics into info.
 * Returns the termination status. K_orig is read-only (a transposed view is
 * materialized internally for the termination checks). */
fiv_lp_status fiv_lp_solve(const fiv_lp_mat *K_orig, const fiv_vec *c_orig,
                           const fiv_vec *l_orig, const fiv_vec *u_orig,
                           const fiv_vec *q_orig, size_t num_inequality,
                           const fiv_lp_solve_params *params,
                           fiv_vec *x_out, fiv_vec *y_out, fiv_lp_solve_info *info);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_LP_SOLVE_H_ */
