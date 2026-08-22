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

#ifndef _FIV_PAD_NODE_H_
#define _FIV_PAD_NODE_H_

#include "fiv_ctensor.h"
#include "fiv_nn.h"
#include "fiv_nn_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Channel-pad node (NCHW): copies the input's C_in channels and appends
   zero channels at the END so the output has exactly params.output_channels
   (output_channels >= C_in). No learnable parameters. */
typedef struct {
    fiv_nn_op_base base;
    int output_channels;
} fiv_pad_node;

void*   fiv_pad_node_create(void* params);
void    fiv_pad_node_release(void* op_state);
fiv_ret fiv_pad_node_forward(void* op_state, void* output, void* input);
fiv_ret fiv_pad_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input);
fiv_ret fiv_pad_node_inference(void* op_state, void* output, void* input);
void*   fiv_pad_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_PAD_NODE_H_ */
