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

#ifndef _FIV_MATH_H_
#define _FIV_MATH_H_

#include "fiv_ctensor.h"

#ifdef __cplusplus
extern "C" {
#endif


/* ============================ Sigmoid (element-wise) ============================ */
/* dst[i] = 1 / (1 + exp(-src[i])) for every element. Sigmoid is a pointwise
   op, so it has no reduction direction: every element of the input vector is
   mapped independently. Float dtype (FIV_32F1 / FIV_64F1), contiguous 1D
   vectors of equal length; dst may alias src (in-place).
   To apply the op to a non-vec tensor, reshape it to a vec first with
   fiv_tensor_reshape (the 1D view shares memory, so the pointwise result maps
   back onto the original tensor). */
fiv_ret fiv_math_sigmoid(fiv_vec* dst, const fiv_vec* src);


/* ============================ SwiGLU (element-wise) ============================ */
/* SwiGLU gated activation (the LLM FFN variant, ported from the reference
   fiv_llm_silu_mul_f32):
     dst[i] = gate[i] / (1 + exp(-gate[i])) * up[i]
            = silu(gate[i]) * up[i]
   Pointwise, so it has no reduction direction: every element of gate is mapped
   independently and multiplied by the matching up element. Float dtype
   (FIV_32F1 / FIV_64F1), contiguous 1D vectors of equal length; dst may alias
   gate or up (in-place).
   To apply the op to non-vec tensors, reshape each of them to a vec first with
   fiv_tensor_reshape (the 1D views share memory, so the pointwise result maps
   back onto the original tensors). */
fiv_ret fiv_math_swiglu(fiv_vec* dst, const fiv_vec* gate, const fiv_vec* up);


/* ============================ Softmax (row-wise) ============================ */
/* Row-wise softmax: each row of src is normalized independently,
   dst[i,j] = exp(src[i,j] - max_j) / sum_j exp(src[i,j] - max_j), where the
   per-row max subtraction keeps the exponentials numerically stable. Only the
   row direction is supported (dim == 0, one softmax per row); a column-wise
   reduction is NOT provided. Float dtype (FIV_32F1 / FIV_64F1), contiguous;
   dst may alias src (in-place). */
fiv_ret fiv_math_softmax(fiv_mat* dst, const fiv_mat* src);


#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MATH_H_ */
