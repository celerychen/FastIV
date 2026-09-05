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

/* Toggle the engine-internal per-node-type timing. Both the API declarations
   and the fiv_nn_infer.c definitions/accumulation compile only when this is 1.
   Override with -DFIV_NN_TIMING=0 to strip all timing code from the build. */
#ifndef FIV_NN_TIMING
#define FIV_NN_TIMING 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Per-node bookkeeping; node 0 is the implicit INPUT.
   num_src/src_list: every single-input node has num_src == 1 and src_list
   == NULL (its only source is input_src); a multi-input node (ADD) owns a
   src_list of num_src ids with src_list[0] == input_src. inputs mirrors
   src_list as the wired input tensors and is NULL for single-input nodes. */
typedef struct {
    int   node_type;
    int   input_src;   /* source node id; -1 for node 0 */
    int   num_src;     /* total input count (>= 1); 1 for single-input nodes */
    int*  src_list;    /* owned; all source ids when num_src > 1, else NULL */
    void* op;          /* op struct carrying the vtable; NULL for node 0 */
    void* input;       /* predecessor node's output (topo order); inputs[0] for multi-input */
    void** inputs;     /* owned array of num_src input tensors; NULL for single-input */
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

#if FIV_NN_TIMING
/* Enable per-node-type forward timing for the next run_inference calls on net.
   While enabled, run_inference resets bench_ms each call and pass2 accumulates
   the wall time of every node keyed by its node_type. Read the split with
   fiv_nn_get_bench(). Only compiled when FIV_NN_TIMING is 1. */
void fiv_nn_bench_enable(void* net);

/* Copy the last run_inference's per node-type timing (ms) into out_by_type[0..n-1]. */
void fiv_nn_get_bench(void* net, ivf64* out_by_type, int n);
#endif

/* Training forward pass: wire inputs, allocate outputs and run every node's
   forward_fn in topo order; *final_out receives the output node's tensor. */
fiv_ret fiv_nn_run_forward(fiv_nn_network_context* net, void* input, void** final_out);

/* Inference forward pass: same wiring, runs every node's inference_fn. */
fiv_ret fiv_nn_run_inference(fiv_nn_network_context* net, void* input, void** final_out);

/* Access a node's context by id (node 0 is INPUT); returns NULL on a bad id. */
fiv_nn_node_context* fiv_neural_network_get_node(void* net, int node_id);

/* Return the output tensor of a node produced by the last forward/inference pass
   (NULL before any inference, or on a bad id). This lets a caller read any
   intermediate feature map without adding extra graph structure. */
void* fiv_neural_network_get_node_output(void* net, int node_id);

/* Overwrite a conv node's weight tensor ((C_out, C_in, ky, kx), row-major) or its
   per-output-channel bias vector with externally supplied host data. This lets a
   caller load trained parameters into a conv node after it has been created. */
fiv_ret fiv_neural_network_set_node_weight(void* net, int node_id, const float* weight);
fiv_ret fiv_neural_network_set_node_bias(void* net, int node_id, const float* bias);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_NN_INFER_H_ */
