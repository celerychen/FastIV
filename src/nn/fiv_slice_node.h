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

#ifndef _FIV_SLICE_NODE_H_
#define _FIV_SLICE_NODE_H_

#include "fiv_ctensor.h"
#include "fiv_nn.h"
#include "fiv_nn_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Slice: copies the half-open range [start, end) along `axis` of an NCHW (or
   NCHW+) tensor; all other axes are copied wholesale. YOLO26 uses it to split
   a tensor along the channel dim (C2f / C3k / C2PSA) and along the head-proj
   dim (Attention q/k/v). Non-in-place. fiv_nn_op_base must be the first member.
 *
 *   axis mapping (NCHW):  3D 0=c  1=h  2=w
 *                         4D 0=b  1=c  2=h  3=w
 *                         5D 0=b  1=t  2=c  3=h  4=w */
typedef struct {
    fiv_nn_op_base base;
    int axis;   /* which tensor axis to slice along (see mapping above) */
    int start;  /* inclusive start index on that axis */
    int end;    /* exclusive end index on that axis */
} fiv_slice_node;

typedef struct {
    int axis;   /* which tensor axis to slice along (see mapping in node header) */
    int start;  /* inclusive start index on that axis */
    int end;    /* exclusive end index on that axis */
} fiv_slice_node_params;

void*   fiv_slice_node_create(void* params);
void    fiv_slice_node_release(void* op_state);
fiv_ret fiv_slice_node_forward(void* op_state, void* output, void* input);
fiv_ret fiv_slice_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input);
fiv_ret fiv_slice_node_inference(void* op_state, void* output, void* input);
void*   fiv_slice_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_SLICE_NODE_H_ */
