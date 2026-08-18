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

#ifndef _FIV_ACTIVATE_FN_H_
#define _FIV_ACTIVATE_FN_H_

#include "fiv_nn_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ReLU / ReLU6 elementwise op. For RELU: out[i] = max(0, in[i]); for RELU6:
   out[i] = clamp(in[i], 0, 6). The branch is selected by op_state->node_type.
   in/out must be contiguous float32 tensors with equal element count; out
   may alias in (in-place). */
fiv_ret fiv_relu(void* op_state, void* input, void* output);

/* ReLU / ReLU6 op. ReLU and ReLU6 share the same create/release; the active
   node-type id is stored here (set by the caller at create time from the
   requested variant) and consulted only at compute time.
   fiv_nn_op_base must be the first member (see fiv_nn_op.h) so the engine can
   upcast op_state to fiv_nn_op_base* for dispatch. */
typedef struct {
    fiv_nn_op_base base;
    int node_type;
} fiv_relu_node;

void*   fiv_relu_node_create(void* params);
void    fiv_relu_node_release(void* op_state);
fiv_ret fiv_relu_node_forward(void* op_state, void* output, void* input);
fiv_ret fiv_relu_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input);
fiv_ret fiv_relu_node_inference(void* op_state, void* output, void* input);
void*   fiv_relu_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_ACTIVATE_FN_H_ */
