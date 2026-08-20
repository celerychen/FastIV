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

#ifndef _FIV_NN_FLATTEN_H_
#define _FIV_NN_FLATTEN_H_

#include "fiv_ctensor.h"
#include "fiv_nn.h"
#include "fiv_nn_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Flatten node: keeps the first dim as rows, merges the rest into cols
   ((N, C, H, W) -> (N, C*H*W); a 2D input passes through unchanged). The
   output is a zero-copy reshape sharing the input buffer (reference=0, no
   allocation); backward accumulates grad back into the input-shaped buffer
   (same element count). */
typedef struct {
    fiv_nn_op_base base;
} fiv_flatten_node;

void*   fiv_flatten_node_create(void* params);
void    fiv_flatten_node_release(void* op_state);
fiv_ret fiv_flatten_node_forward(void* op_state, void* output, void* input);
fiv_ret fiv_flatten_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input);
fiv_ret fiv_flatten_node_inference(void* op_state, void* output, void* input);
void*   fiv_flatten_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_NN_FLATTEN_H_ */
