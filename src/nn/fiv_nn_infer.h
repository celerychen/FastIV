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

#ifndef _FIV_NN_INFER_H_
#define _FIV_NN_INFER_H_

#include "fiv_nn.h"
#include "fiv_nn_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-node bookkeeping; node 0 is the implicit INPUT. */
typedef struct {
    int   node_type;
    int   input_src;   /* source node id; -1 for node 0 */
    void* op;          /* op struct carrying the vtable; NULL for node 0 */
    void* input;       /* predecessor node's output (topo order) */
    void* output;      /* this node's output tensor; node 0 aliases caller input */
    int   owns_output; /* 1 if output is a buffer this node must release */
} fiv_nn_node_context;

typedef struct {
    int    node_count;
    int    capacity;
    fiv_nn_node_context* nodes;
    int    output_node;   /* index of the last added node */
    int*   topo_order;
    int    topo_count;
    int    topo_valid;
    int    infer_allocated;  /* 1 once inference pass1 has allocated node outputs */
} fiv_nn_network_context;

/* Training forward pass: wire inputs, allocate outputs and run every node's
   forward_fn in topo order; *final_out receives the output node's tensor. */
fiv_ret fiv_nn_run_forward(fiv_nn_network_context* net, void* input, void** final_out);

/* Inference forward pass: same wiring, runs every node's inference_fn. */
fiv_ret fiv_nn_run_inference(fiv_nn_network_context* net, void* input, void** final_out);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_NN_INFER_H_ */
