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

#include "fiv_math.h"
#include "fiv_math_sigmoid.h"
#include "fiv_math_softmax.h"
#include "fiv_math_swiglu.h"


/* ==================== Public API (dispatch only) ====================
   The fiv_math.c file only validates the arguments and forwards to the dtype /
   SIMD-specific backends. The actual kernel implementations live in the
   sibling files:
     - fiv_math_sigmoid.c (scalar real32/real64 + AVX2)
     - fiv_math_softmax.c (scalar real32/real64 + AVX2)
     - fiv_math_swiglu.c  (scalar real32/real64 + AVX2)
   The shared AVX2 vector primitives (exp / reductions / rcp) are factored into
   fiv_math_kernels.h. */

/* Sigmoid is pointwise: every element of the vector is mapped independently,
   so the whole buffer is walked once, with no row/column distinction. dst may
   alias src (in-place). */
fiv_ret fiv_math_sigmoid(fiv_vec* dst, const fiv_vec* src)
{
    if (dst == NULL || src == NULL) return FIV_RET_ERR_PARA;
    if (dst->data.ptr == NULL || src->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (dst->data_continue == 0 || src->data_continue == 0) return FIV_RET_ERR_PARA;
    if (dst->dtype != src->dtype) return FIV_RET_ERR_PARA;
    if (src->dtype != FIV_32F1 && src->dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (dst->shapes[0] != src->shapes[0]) return FIV_RET_ERR_PARA;

    const size_t element_count = src->shapes[0];

    if (src->dtype == FIV_64F1) {
        fiv_math_sigmoid_real64((ivf64*)dst->data.db, (const ivf64*)src->data.db, element_count);
        return FIV_RET_OK;
    }

#if defined(FIV_USE_AVX2)
    fiv_math_sigmoid_avx2_ps((ivf32*)dst->data.fl, (const ivf32*)src->data.fl, element_count);
#else
    fiv_math_sigmoid_real32((ivf32*)dst->data.fl, (const ivf32*)src->data.fl, element_count);
#endif

    return FIV_RET_OK;
}

/* Softmax reduces along the ROW direction only: each row is normalized
   independently (dim == 0). dst may alias src (in-place). FIV_32F1 goes through
   the AVX2+FMA kernel when available; FIV_64F1 stays on the scalar path. */
fiv_ret fiv_math_softmax(fiv_mat* dst, const fiv_mat* src)
{
    if (dst == NULL || src == NULL) return FIV_RET_ERR_PARA;
    if (dst->data.ptr == NULL || src->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (dst->data_continue == 0 || src->data_continue == 0) return FIV_RET_ERR_PARA;
    if (dst->dtype != src->dtype) return FIV_RET_ERR_PARA;
    if (src->dtype != FIV_32F1 && src->dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (dst->rows != src->rows || dst->cols != src->cols) return FIV_RET_ERR_PARA;

    const size_t rows = src->rows;
    const size_t cols = src->cols;

    if (src->dtype == FIV_64F1) {
        fiv_math_softmax_real64((ivf64*)dst->data.db, (const ivf64*)src->data.db, rows, cols);
        return FIV_RET_OK;
    }

#if defined(FIV_USE_AVX2)
    fiv_math_softmax_avx2_ps((ivf32*)dst->data.fl, (const ivf32*)src->data.fl, rows, cols);
#else
    fiv_math_softmax_real32((ivf32*)dst->data.fl, (const ivf32*)src->data.fl, rows, cols);
#endif

    return FIV_RET_OK;
}

/* SwiGLU gated activation, pointwise: dst[i] = gate[i] * sigmoid(gate[i]) *
   up[i] over the whole vector. dst may alias gate or up (in-place). */
fiv_ret fiv_math_swiglu(fiv_vec* dst, const fiv_vec* gate, const fiv_vec* up)
{
    if (dst == NULL || gate == NULL || up == NULL) return FIV_RET_ERR_PARA;
    if (dst->data.ptr == NULL || gate->data.ptr == NULL || up->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (dst->data_continue == 0 || gate->data_continue == 0 || up->data_continue == 0) return FIV_RET_ERR_PARA;
    if (dst->dtype != gate->dtype || gate->dtype != up->dtype) return FIV_RET_ERR_PARA;
    if (gate->dtype != FIV_32F1 && gate->dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (dst->shapes[0] != gate->shapes[0] || gate->shapes[0] != up->shapes[0]) return FIV_RET_ERR_PARA;

    const size_t element_count = gate->shapes[0];

    if (gate->dtype == FIV_64F1) {
        fiv_math_swiglu_real64((ivf64*)dst->data.db, (const ivf64*)gate->data.db,
                               (const ivf64*)up->data.db, element_count);
        return FIV_RET_OK;
    }

#if defined(FIV_USE_AVX2)
    fiv_math_swiglu_avx2_ps((ivf32*)dst->data.fl, (const ivf32*)gate->data.fl,
                            (const ivf32*)up->data.fl, element_count);
#else
    fiv_math_swiglu_real32((ivf32*)dst->data.fl, (const ivf32*)gate->data.fl,
                           (const ivf32*)up->data.fl, element_count);
#endif

    return FIV_RET_OK;
}
