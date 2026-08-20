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

/* conv2d_method values (see fiv_conv2d_params.conv2d_method). */
enum {
    FIV_CONV2D_STD = 0,        /* dense: out[oc] = sum_ic conv(src[ic], w[oc][ic]) */
    FIV_CONV2D_DEPTHWISE = 1,  /* per-channel: out[c] = conv(src[c], w[c]), no mixing */
    FIV_CONV2D_POINTWISE = 2,  /* reserved */
    FIV_CONV2D_SEPARABLE = 3,  /* reserved */
};

/* CNN-style 2D convolution, kernel 3x3, stride 1, output same spatial size.
   dst/src/kernel: generic tensors (any 1D~5D tensor pointer, see fiv_tensor_hdr):
   src:    (..., C_in, H, W)
   kernel: (C_out, C_in, 3, 3) for STD; (C_out, 1, 3, 3) for DEPTHWISE (C_out == C_in)
   dst:    (..., C_out, H, W)
   params: fiv_conv2d_params (conv2d_method / kernel_size / stride / padding_method /
           input_channels / output_channels / bias); padding_method 0 = zero, 1 = replicate edge. */
fiv_ret fiv_tensor_conv2d(void* dst, void* src, void* kernel, fiv_conv2d_params* params);

/* ---- CONV2D_STD network node (forward/backward live in fiv_nn_conv2d.c) ---- */

/* Conv op holding its own weights (and optional per-channel bias). weight is
   (C_out, C_in, 3, 3); fiv_nn_op_base must be the first member (see fiv_nn_op.h). */
typedef struct {
    fiv_nn_op_base     base;
    fiv_conv2d_params  params;
    fiv_tensor4d*      weight;        /* (C_out, C_in, 3, 3) */
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
