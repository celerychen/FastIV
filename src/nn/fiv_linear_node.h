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

#ifndef _FIV_LINEAR_NODE_H_
#define _FIV_LINEAR_NODE_H_

#include "fiv_nn_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/* LINEAR op. weight [out_features, in_features] and bias [out_features]
   are internal state, not part of the public API. grad_weight/grad_bias
   accumulate the per-batch gradients; cached_input keeps the input of the
   last forward so backward can compute dW/db. */
typedef struct {
    fiv_nn_op_base base;
    int in_features;
    int out_features;
    fiv_mat* weight;
    fiv_vec* bias;
    fiv_mat* grad_weight;
    fiv_vec* grad_bias;
    const fiv_mat* cached_input;
} fiv_linear_node;

void*   fiv_linear_node_create(void* params);   /* params: fiv_linear_node_params* */
void    fiv_linear_node_release(void* op_state);
fiv_ret fiv_linear_node_forward(void* op_state, void* output, void* input);
fiv_ret fiv_linear_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input);
fiv_ret fiv_linear_node_inference(void* op_state, void* output, void* input);
void*   fiv_linear_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_LINEAR_NODE_H_ */
