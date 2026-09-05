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

#ifndef _FIV_PRELU_NODE_H_
#define _FIV_PRELU_NODE_H_

#include "fiv_ctensor.h"
#include "fiv_nn.h"
#include "fiv_nn_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/* PReLU (parametric ReLU) node: out = x>=0 ? x : alpha[c] * x, where alpha is a
   per-channel learned slope (FaceMesh landmark backbone activation).
   alpha is indexed by channel in NCHW flat order (channels are contiguous, so
   element i belongs to channel (i % channels)). Output shape == input shape.
   fiv_nn_op_base must be the first member (see fiv_nn_op.h). */
typedef struct {
    fiv_nn_op_base base;
    int    channels;  /* number of slopes == input channel count */
    ivf32* alpha;     /* owned per-channel slope array */
} fiv_prelu_node;

void*   fiv_prelu_node_create(void* params);
void    fiv_prelu_node_release(void* op_state);
fiv_ret fiv_prelu_node_forward(void* op_state, void* output, void* input);
fiv_ret fiv_prelu_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input);
fiv_ret fiv_prelu_node_inference(void* op_state, void* output, void* input);
void*   fiv_prelu_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret);

#ifdef __cplusplus
}
#endif
#endif  /* _FIV_PRELU_NODE_H_ */