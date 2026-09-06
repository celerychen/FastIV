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

#ifndef _FIV_MAXPOOL_NODE_H_
#define _FIV_MAXPOOL_NODE_H_

#include "fiv_ctensor.h"
#include "fiv_nn.h"
#include "fiv_nn_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/* General 2D max pooling over the last two (h, w) dims of an NCHW (or NCHW+)
   tensor. Unlike FIV_NN_NODE_MAX2D (hardcoded 2x2/s2 downsample, used by the
   landmark net) this node is parameterised so the YOLO26 SPPF three-stage
   MaxPool2d(k=5, s=1, p=2) chain can be expressed. Border padding is zero-fill
   (clamped window). Non-in-place. fiv_nn_op_base must be the first member. */
typedef struct {
    fiv_nn_op_base base;
    int kernel_size_x;
    int kernel_size_y;
    int stride;
    int pad_top;
    int pad_bottom;
    int pad_left;
    int pad_right;
    int*   argmax;     /* flat input offset of the max per output element (forward cache) */
    size_t n_out;      /* argmax capacity in elements */
} fiv_maxpool_node;

typedef struct {
    int kernel_size_x;
    int kernel_size_y;
    int stride;
    int pad_top;
    int pad_bottom;
    int pad_left;
    int pad_right;
} fiv_maxpool_node_params;

void*   fiv_maxpool_node_create(void* params);
void    fiv_maxpool_node_release(void* op_state);
fiv_ret fiv_maxpool_node_forward(void* op_state, void* output, void* input);
fiv_ret fiv_maxpool_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input);
fiv_ret fiv_maxpool_node_inference(void* op_state, void* output, void* input);
void*   fiv_maxpool_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MAXPOOL_NODE_H_ */
