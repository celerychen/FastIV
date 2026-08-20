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

#ifndef _FIV_MAX_2D_H_
#define _FIV_MAX_2D_H_

#include "fiv_ctensor.h"
#include "fiv_nn.h"
#include "fiv_nn_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 2x2 stride-2 max pooling node (non-overlapping, downsample by 2).
   fiv_nn_op_base must be the first member (see fiv_nn_op.h). */
typedef struct {
    fiv_nn_op_base base;
    int*   argmax;   /* flat input offset of the max per output element (forward cache) */
    size_t n_out;    /* argmax capacity in elements */
} fiv_max_2d_node;

void*   fiv_max_2d_node_create(void* params);
void    fiv_max_2d_node_release(void* op_state);
fiv_ret fiv_max_2d_node_forward(void* op_state, void* output, void* input);
fiv_ret fiv_max_2d_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input);
fiv_ret fiv_max_2d_node_inference(void* op_state, void* output, void* input);
void*   fiv_max_2d_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MAX_2D_H_ */
