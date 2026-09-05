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

#ifndef _FIV_NN_CONV2D_H_
#define _FIV_NN_CONV2D_H_

#include "fiv_ctensor.h"
#include "fiv_nn.h"
#include "fiv_nn_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Shared AVX2 stride-2 deinterleave primitives, used by the 3x3 / 5x5 / 2x2
   stride-2 plane kernels. even() picks the even columns of a 16-wide window
   [o..o+16), odd() the odd columns; each stride-2 horizontal tap is such a pick
   of a window shifted by 0/1/2 columns. Defined once here (single source). */
#if defined(FIV_USE_AVX2)
static inline __m256 fiv_s2_tap_even(__m256 lo, __m256 hi)
{
    __m256 t0 = _mm256_permute2f128_ps(lo, hi, 0x20);  /* [lo.low | hi.low ] */
    __m256 t1 = _mm256_permute2f128_ps(lo, hi, 0x31);  /* [lo.high| hi.high] */
    return _mm256_shuffle_ps(t0, t1, 0x88);            /* even-column pick   */
}
static __m256 fiv_s2_tap_odd(__m256 lo, __m256 hi)
{
    __m256 t0 = _mm256_permute2f128_ps(lo, hi, 0x20);
    __m256 t1 = _mm256_permute2f128_ps(lo, hi, 0x31);
    return _mm256_shuffle_ps(t0, t1, 0xDD);            /* odd-column pick    */
}
#endif

/* conv2d_method values (see fiv_conv2d_params.conv2d_method). */
enum {
    FIV_CONV2D_STD = 0,        /* dense: out[oc] = sum_ic conv(src[ic], w[oc][ic]) */
    FIV_CONV2D_DEPTHWISE = 1,  /* per-channel: out[c] = conv(src[c], w[c]), no mixing */
    FIV_CONV2D_POINTWISE = 2,  /* reserved */
    FIV_CONV2D_SEPARABLE = 3,  /* reserved */
};

/* CNN-style 2D convolution, any kernel size / stride, explicit pads.
   dst/src/kernel: generic tensors (any 1D~5D tensor pointer, see fiv_tensor_hdr):
   src:    (..., C_in, H, W)
   kernel: (C_out, C_in, kh, kw) for STD/POINTWISE; (C_out, 1, 3, 3) for DEPTHWISE (C_out == C_in)
   dst:    (..., C_out, ceil(H/stride), ceil(W/stride))
   params: fiv_conv2d_params (conv2d_method / kernel_size / stride / padding_method /
           input_channels / output_channels / bias / pad_top-bottom-left-right).
           padding_method 0 = zero fill, 1 = replicate edge element; the pads are
           the explicit start-pads (p0) that enter the index math (see same_pad).
           A legacy 3x3 stride-1 call with all pads 0 keeps the historical
           same-padding (p0 = p1 = 1) and its SIMD fast path. */
fiv_ret fiv_tensor_conv2d(void* dst, void* src, void* kernel, fiv_conv2d_params* params);

/* ---- CONV network node (STD / DEPTHWISE / POINTWISE; fiv_nn_conv2d.c) ---- */

/* Conv op holding its own weights (and optional per-channel bias). weight is
   (C_out, C_in, ky, kx); fiv_nn_op_base must be the first member (see fiv_nn_op.h). */
typedef struct {
    fiv_nn_op_base     base;
    fiv_conv2d_params  params;
    fiv_tensor4d*      weight;        /* (C_out, C_in, ky, kx) */
    fiv_vec*           bias;          /* C_out; NULL when params.bias == 0 */
    fiv_tensor4d*      grad_weight;   /* same shape as weight, engine zeroes per step */
    fiv_vec*           grad_bias;
} fiv_conv2d_node;

void*   fiv_conv2d_node_create(void* params);
void    fiv_conv2d_node_release(void* op_state);
fiv_ret fiv_conv2d_node_forward(void* op_state, void* output, void* input);
fiv_ret fiv_conv2d_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input);
fiv_ret fiv_conv2d_node_inference(void* op_state, void* output, void* input);
void*   fiv_conv2d_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_NN_CONV2D_H_ */
