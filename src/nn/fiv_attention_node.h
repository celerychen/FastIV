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

#ifndef _FIV_ATTENTION_NODE_H_
#define _FIV_ATTENTION_NODE_H_

#include "fiv_data_typedefs.h"
#include "fiv_nn_op.h"
#include "fiv_ctensor.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Composite self-attention node (YOLO26 C2PSA / attn=True C3k2).
 *
 * Reproduces ultralytics.nn.modules.block.Attention end-to-end:
 *   qkv = Conv(x, qkv_w)              # 1x1, no act
 *   q,k,v = channel-split(qkv)        # [dim+2*kd*nh | kd*nh | hd*nh]
 *   S = ((q*scale).T @ k).softmax(-1) # [B, heads, N, N]
 *   O = v @ S.T                       # [B, heads*hd, H, W]
 *   out = Conv(O + pe(v), proj_w)      # pe = depthwise 3x3 on v; proj 1x1
 *
 * The node holds the three conv weight sets (qkv / pe / proj) and applies
 * them internally via fiv_tensor_conv2d; the two batched matmuls reuse the
 * blocked GEMM (fiv_matrix_mul_real32) and the row-softmax
 * (fiv_math_softmax_real32). Weights are injected after creation through
 * fiv_attention_node_set_weights (the engine graph builder uses this same
 * call to load pretrained weights). The op is NOT in-place: it allocates a
 * fresh output tensor, so it is safe under the heavy fan-out of the YOLO26
 * backbone. */

typedef struct {
    int   dim;        /* input/output channels (== head_dim * num_heads) */
    int   num_heads;
    int   key_dim;    /* per-head key dimension (key_dim = int(head_dim*attn_ratio)) */
    int   head_dim;   /* per-head value dimension (== dim / num_heads) */
    ivf32 scale;      /* 1/sqrt(key_dim) */
    int   pe_groups;  /* depthwise groups for positional-encoding conv (== dim) */
} fiv_attention_node_params;

typedef struct {
    fiv_nn_op_base base;
    int   dim, num_heads, key_dim, head_dim, pe_groups;
    ivf32 scale;
    fiv_tensor4d* qkv_w;   /* [dim + 2*key_dim*num_heads, dim, 1, 1] */
    ivf32*        qkv_b;   /* [dim + 2*key_dim*num_heads] */
    fiv_tensor4d* pe_w;    /* [dim, 1, 3, 3]  (depthwise, groups = dim) */
    ivf32*        pe_b;    /* [dim] */
    fiv_tensor4d* proj_w;  /* [dim, dim, 1, 1] */
    ivf32*        proj_b; /* [dim] */
    int           has_weights;
} fiv_attention_node;

void* fiv_attention_node_create(void* params);
void  fiv_attention_node_release(void* op_state);
void* fiv_attention_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret);
fiv_ret fiv_attention_node_forward(void* op_state, void* output, void* input);
fiv_ret fiv_attention_node_inference(void* op_state, void* output, void* input);
fiv_ret fiv_attention_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input);

/* Inject the three conv weight sets. Each weight is copied into the node's own
 * tensor/array; the caller may free its buffers afterwards. All six pointers
 * must be non-NULL. Layout matches PyTorch (and FastIV) conv convention
 * [c_out, c_in, ky, kx]; pe_w is the depthwise layout [dim, 1, 3, 3]. */
fiv_ret fiv_attention_node_set_weights(void* op_state,
                                       const ivf32* qkv_w, const ivf32* qkv_b,
                                       const ivf32* pe_w,  const ivf32* pe_b,
                                       const ivf32* proj_w, const ivf32* proj_b);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_ATTENTION_NODE_H_ */
