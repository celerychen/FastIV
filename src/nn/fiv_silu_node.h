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

#ifndef _FIV_SILU_NODE_H_
#define _FIV_SILU_NODE_H_

#include "fiv_nn_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Element-wise SiLU activation: y = x * sigmoid(x). Dimension-agnostic (flat
   over the whole buffer). No parameters. Non-in-place (YOLO26 fans one input
   out to many consumers, so aliasing the input would corrupt the siblings).
   fiv_nn_op_base must be the first member (see fiv_nn_op.h). */
typedef struct {
    fiv_nn_op_base base;
} fiv_silu_node;

void*   fiv_silu_node_create(void* params);
void    fiv_silu_node_release(void* op_state);
fiv_ret fiv_silu_node_forward(void* op_state, void* output, void* input);
fiv_ret fiv_silu_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input);
fiv_ret fiv_silu_node_inference(void* op_state, void* output, void* input);
void*   fiv_silu_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_SILU_NODE_H_ */
