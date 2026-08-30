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

#ifndef _FIV_LP_RESCALE_H_
#define _FIV_LP_RESCALE_H_

#include "fiv_ctensor.h"   /* fiv_vec, fiv_ret, ivf64 */
#include "fiv_lp_mat.h"    /* fiv_lp_mat */

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * LP preconditioning: Ruiz equilibration + Pock-Chambolle rescaling.
 *
 * Faithful port of pdlp.py lines 160-198. The rescaling transforms the
 * constraint matrix into K_scaled = D_row^{-1} * K * D_col^{-1} (with D_row =
 * diag(con_rescale), D_col = diag(var_rescale)) and the bounds/rhs accordingly,
 * so the PDHG iteration converges faster. The two scaling vectors are
 * accumulated in an fiv_lp_rescaling and later used to unscale the solution.
 *
 * All apply steps mutate the working problem (K, c, l, u, q) IN PLACE -- this
 * matches pdlp.py, where the scaled problem becomes the working problem.
 *
 * Naming rule: only fiv_create_lp_rescaling returns a pointer; every other
 * entry point returns fiv_ret.
 * ========================================================================= */

/* Accumulated scaling factors. var_rescale == product of column rescalings
 * (length == #variables n), con_rescale == product of row rescalings
 * (length == #constraints m). Both allocated by fiv_create_lp_rescale and
 * freed by fiv_release_lp_rescale. */
typedef struct {
    fiv_vec *variable_rescaling;   /* diag(D_col), length n */
    fiv_vec *constraint_rescaling; /* diag(D_row), length m */
} fiv_lp_rescaling;


/* Create an identity rescaling (both factors all ones). Returns the object,
 * or NULL on allocation failure. num_variables == n, num_constraints == m. */
fiv_lp_rescaling *fiv_create_lp_rescaling(size_t num_variables, size_t num_constraints);

/* Free the rescaling object and its two factor vectors; *rescaling set to NULL. */
fiv_ret fiv_release_lp_rescaling(fiv_lp_rescaling **rescaling);


/* Ruiz L-infinity equilibration, `ruiz_iterations` passes. Mutates K, c, l, u,
 * q in place and accumulates the factors. `eps_zero` guards divisions/sqrts
 * (use 1e-12, matching pdlp.py). */
fiv_ret fiv_lp_rescale_ruiz(fiv_lp_rescaling *rescaling, fiv_lp_mat *K,
                            fiv_vec *c, fiv_vec *l, fiv_vec *u, fiv_vec *q,
                            int ruiz_iterations, ivf64 eps_zero);

/* Pock-Chambolle operator-norm rescaling (alpha > 0 only; alpha <= 0 is a
 * no-op returning FIV_RET_OK). Mutates in place and accumulates factors.
 *   col_rescale = sqrt( sum_i |K_ij|^(2 - alpha) )
 *   row_rescale = sqrt( sum_j |K_ij|^alpha ) */
fiv_ret fiv_lp_rescale_pock_chambolle(fiv_lp_rescaling *rescaling, fiv_lp_mat *K,
                                      fiv_vec *c, fiv_vec *l, fiv_vec *u, fiv_vec *q,
                                      ivf64 alpha, ivf64 eps_zero);

/* Combined entry used by solve(): Ruiz (ruiz_iterations) then Pock-Chambolle
 * (alpha). Mutates in place and accumulates factors. */
fiv_ret fiv_lp_rescale_solve(fiv_lp_rescaling *rescaling, fiv_lp_mat *K,
                             fiv_vec *c, fiv_vec *l, fiv_vec *u, fiv_vec *q,
                             int ruiz_iterations, ivf64 pock_chambolle_alpha, ivf64 eps_zero);


/* Unscale the primal solution: x_unscaled = x / var_rescale. In-place allowed
 * (x_unscaled may alias x). Length must equal #variables. Mirrors pdlp.py
 * `x / variable_rescaling`. */
fiv_ret fiv_lp_unscale_primal(fiv_vec *x_unscaled, const fiv_lp_rescaling *rescaling,
                              const fiv_vec *x);

/* Unscale the dual solution: y_unscaled = y / con_rescale. In-place allowed.
 * Length must equal #constraints. Mirrors pdlp.py `y / constraint_rescaling`. */
fiv_ret fiv_lp_unscale_dual(fiv_vec *y_unscaled, const fiv_lp_rescaling *rescaling,
                            const fiv_vec *y);


#ifdef __cplusplus
}
#endif

#endif  /* _FIV_LP_RESCALE_H_ */
