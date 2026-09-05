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

#ifndef _FIV_SPATIAL_PAD_NODE_H_
#define _FIV_SPATIAL_PAD_NODE_H_

#include "fiv_ctensor.h"
#include "fiv_nn.h"
#include "fiv_nn_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Spatial pad node (NCHW): pads the HxW plane with a constant `value` (default
   0) on all four sides by explicit margins, extending the spatial extent
   (unlike fiv_pad_node which pads CHANNELS). Output shape == input + margins.
   fiv_nn_op_base must be the first member (see fiv_nn_op.h). */
typedef struct {
    fiv_nn_op_base base;
    int   pad_top;
    int   pad_bottom;
    int   pad_left;
    int   pad_right;
    ivf32 value;
} fiv_spatial_pad_node;

void*   fiv_spatial_pad_node_create(void* params);
void    fiv_spatial_pad_node_release(void* op_state);
fiv_ret fiv_spatial_pad_node_forward(void* op_state, void* output, void* input);
fiv_ret fiv_spatial_pad_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input);
fiv_ret fiv_spatial_pad_node_inference(void* op_state, void* output, void* input);
void*   fiv_spatial_pad_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret);

#ifdef __cplusplus
}
#endif
#endif  /* _FIV_SPATIAL_PAD_NODE_H_ */