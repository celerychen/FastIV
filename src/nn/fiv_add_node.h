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

#ifndef _FIV_ADD_NODE_H_
#define _FIV_ADD_NODE_H_

#include "fiv_ctensor.h"
#include "fiv_nn.h"
#include "fiv_nn_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Element-wise ADD of N same-shaped inputs (residual: main path + shortcut).
   All inputs must share the same element count; the output has that shape.
   backward distributes grad_output to every input (d(out)/d(in_i) = 1).
   This is the engine's multi-input node: built via
   fiv_neural_network_add_node_multi, which sets the *_multi vtable slots;
   the single-input *_fn slots are never invoked by the engine. */
typedef struct {
    fiv_nn_op_base base;
} fiv_add_node;

void*   fiv_add_node_create(void* params);
void    fiv_add_node_release(void* op_state);
fiv_ret fiv_add_node_forward(void* op_state, void* output, void* input);
fiv_ret fiv_add_node_forward_multi(void* op_state, void* output, void* const* inputs, int num_inputs);
fiv_ret fiv_add_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input);
fiv_ret fiv_add_node_backward_multi(void* op_state, void* const* grad_inputs, const void* grad_output, void* const* inputs, int num_inputs);
fiv_ret fiv_add_node_inference(void* op_state, void* output, void* input);
fiv_ret fiv_add_node_inference_multi(void* op_state, void* output, void* const* inputs, int num_inputs);
void*   fiv_add_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_ADD_NODE_H_ */
