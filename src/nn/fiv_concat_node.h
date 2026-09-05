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

#ifndef _FIV_CONCAT_NODE_H_
#define _FIV_CONCAT_NODE_H_

#include "fiv_ctensor.h"
#include "fiv_nn.h"
#include "fiv_nn_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/* CONCAT node (multi-input, NCHW): concatenates several tensors that share the
   same batch/HxW along the CHANNEL axis (axis==1). The builder must pass
   params->output_channels = sum of all inputs' channel counts so the output can
   be sized. Uses the three *_multi_fn vtable slots (single-input slots are
   NULL/never invoked). fiv_nn_op_base must be the first member. */
typedef struct {
    fiv_nn_op_base base;
    int axis;             /* concat axis (NCHW), only 1 supported now */
    int output_channels;  /* accumulated channel count of all inputs */
} fiv_concat_node;

void*   fiv_concat_node_create(void* params);
void    fiv_concat_node_release(void* op_state);
fiv_ret fiv_concat_node_forward(void* op_state, void* output, void* input);
fiv_ret fiv_concat_node_forward_multi(void* op_state, void* output, void* const* inputs, int num_inputs);
fiv_ret fiv_concat_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input);
fiv_ret fiv_concat_node_backward_multi(void* op_state, void* const* grad_inputs, const void* grad_output, void* const* inputs, int num_inputs);
fiv_ret fiv_concat_node_inference(void* op_state, void* output, void* input);
fiv_ret fiv_concat_node_inference_multi(void* op_state, void* output, void* const* inputs, int num_inputs);
void*   fiv_concat_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret);

#ifdef __cplusplus
}
#endif
#endif  /* _FIV_CONCAT_NODE_H_ */