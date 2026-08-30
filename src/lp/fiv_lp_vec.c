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

#include "fiv_lp_vec.h"
#include "fiv_matrix.h"   /* fiv_vec_* live here; kept for header completeness */

#include <math.h>          /* isfinite, fabs, isinf */


/* Box projection: dst[i] = max(lower[i], min(upper[i], x[i])).
 * Bounds may be +-INF (unbounded side), matching torch.clamp semantics. */
fiv_ret fiv_vec_clamp(fiv_vec *dst, const fiv_vec *x,
                      const fiv_vec *lower, const fiv_vec *upper)
{
    if (dst == NULL || x == NULL || lower == NULL || upper == NULL)
        return FIV_RET_ERR_PARA;
    if (dst->dtype != FIV_64F1 || x->dtype != FIV_64F1 ||
        lower->dtype != FIV_64F1 || upper->dtype != FIV_64F1)
        return FIV_RET_ERR_PARA;
    if (dst->length != x->length || x->length != lower->length ||
        lower->length != upper->length)
        return FIV_RET_ERR_PARA;

    const ivf64 *input_data = x->data.db;
    const ivf64 *lower_data = lower->data.db;
    const ivf64 *upper_data = upper->data.db;
    ivf64 *output_data = dst->data.db;

    for (size_t index = 0; index < dst->length; index++) {
        ivf64 clamped_value = input_data[index];
        if (clamped_value < lower_data[index]) clamped_value = lower_data[index];
        if (clamped_value > upper_data[index]) clamped_value = upper_data[index];
        output_data[index] = clamped_value;
    }
    return FIV_RET_OK;
}


/* Sum values[i]*multipliers[i] over i where values[i] is finite.
 * Only `values` is tested for finiteness (l/u bounds may be +-INF); the
 * multiplier vector is assumed finite. */
fiv_ret fiv_vec_sum_finite_products(const fiv_vec *values, const fiv_vec *multipliers,
                                    ivf64 *accumulated)
{
    if (values == NULL || multipliers == NULL || accumulated == NULL)
        return FIV_RET_ERR_PARA;
    if (values->dtype != FIV_64F1 || multipliers->dtype != FIV_64F1)
        return FIV_RET_ERR_PARA;
    if (values->length != multipliers->length)
        return FIV_RET_ERR_PARA;

    const ivf64 *value_data = values->data.db;
    const ivf64 *multiplier_data = multipliers->data.db;
    ivf64 total = 0.0;

    for (size_t index = 0; index < values->length; index++) {
        if (isfinite(value_data[index]))
            total += value_data[index] * multiplier_data[index];
    }
    *accumulated = total;
    return FIV_RET_OK;
}
