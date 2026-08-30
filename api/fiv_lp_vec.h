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

#ifndef _FIV_LP_VEC_H_
#define _FIV_LP_VEC_H_

#include "fiv_ctensor.h"   /* fiv_vec, fiv_scalar, fiv_ret, ivf64 */
#include "fiv_matrix.h"    /* fiv_vec_norm / fiv_vec_axpy / fiv_vec_scale / fiv_vec_dot */

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Vector operators for the PDLP linear-programming solver.
 *
 * Two primitives are SELF-WRITTEN here because FastIV has no equivalent:
 *   - fiv_vec_clamp                : per-element box projection  y = clamp(x, lo, hi)
 *   - fiv_vec_sum_finite_products  : sum a[i]*b[i] over finite a[i] (KKT cert)
 *
 * Everything else is DELEGATED to FastIV -- call it directly, do NOT re-implement:
 *   - L1 / L2 / Inf norm -> fiv_vec_norm(&s, v, FIV_L1_NORM | FIV_L2_NORM | FIV_INF_NORM)
 *   - dot product        -> fiv_vec_dot(&s, a, b)
 *   - y = a*x + y        -> fiv_vec_axpy(y, FIV_SCALAR_FP64(a), x)
 *   - y = scale*x        -> fiv_vec_scale(y, x, FIV_SCALAR_FP64(scale))
 *   - elementwise add / sub / mul / div -> fiv_tensor_add / fiv_tensor_sub / fiv_tensor_mul / fiv_tensor_div
 *                            (void* tensors; requires equal dtype/shape, no broadcast)
 * ========================================================================= */


/* Box projection: dst[i] = max(lower[i], min(upper[i], x[i])).
 * lower / upper may carry +-INF for unbounded sides (matches torch.clamp with
 * infinite bounds). All operands FIV_64F1 and equal length; dst may alias x. */
fiv_ret fiv_vec_clamp(fiv_vec *dst, const fiv_vec *x,
                      const fiv_vec *lower, const fiv_vec *upper);


/* Sum a[i]*multipliers[i] over indices where a[i] is finite (isfinite).
 * Mirrors PDLP's sum_finite_products(values, multipliers): only `values` is
 * checked for finiteness (the box bounds l/u may be +-INF); `multipliers` is
 * assumed finite. *accumulated is set to the result. Used by the KKT /
 * optimality check for the bound terms l_term / u_term. */
fiv_ret fiv_vec_sum_finite_products(const fiv_vec *values, const fiv_vec *multipliers,
                                    ivf64 *accumulated);


#ifdef __cplusplus
}
#endif

#endif  /* _FIV_LP_VEC_H_ */
