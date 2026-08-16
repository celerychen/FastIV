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

/* ReLU primitive: out[i] = max(0, in[i]). in/out must be contiguous float32
   tensors with the same total element count; out may alias in (in-place). */
fiv_ret fiv_relu(void* input, void* output);

/* ReLU op. Stateless: forward equals inference, backward masks with x > 0. */
typedef struct {
    fiv_nn_op_base base;
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
