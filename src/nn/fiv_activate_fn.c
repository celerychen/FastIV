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

#include <string.h>
#include "fiv_activate_fn.h"
#include "fiv_nn_infer.h"
#include "fiv_common.h"
#include "fiv_ctensor.h"

fiv_ret fiv_relu(void* input, void* output)
{
    if (!input || !output) return FIV_RET_ERR_PARA;

    fiv_tensor_hdr* in  = (fiv_tensor_hdr*)input;
    fiv_tensor_hdr* out = (fiv_tensor_hdr*)output;

    if (in->id < FIV_ID_TENSOR1D || in->id > FIV_ID_TENSOR5D) return FIV_RET_ERR_PARA;
    if (out->id < FIV_ID_TENSOR1D || out->id > FIV_ID_TENSOR5D) return FIV_RET_ERR_PARA;
    if (in->data_continue == 0 || out->data_continue == 0) return FIV_RET_ERR_PARA;
    if (in->dtype != FIV_32F1 || out->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (in->total_bytes != out->total_bytes) return FIV_RET_ERR_PARA;

    ivf32* a = in->data.fl;
    ivf32* b = out->data.fl;
    size_t n = in->total_bytes / sizeof(ivf32);
    for (size_t i = 0; i < n; i++) {
        ivf32 v = a[i];
        b[i] = (v > 6.0f) ? 6.0f : ((v < 0.0f) ? 0.0f : v);
    }
    return FIV_RET_OK;
}

void* fiv_relu_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{
    *out_ret = FIV_RET_OK;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!in) { *out_ret = FIV_RET_ERR_PARA; return NULL; }
    if (in->id < FIV_ID_TENSOR1D || in->id > FIV_ID_TENSOR5D) { *out_ret = FIV_RET_ERR_PARA; return NULL; }
    if (in->dtype != FIV_32F1 || in->data_continue == 0) { *out_ret = FIV_RET_ERR_PARA; return NULL; }
    /* signal in-place: ReLU writes the result into the input buffer. The
       framework aliases the input during inference (zero allocation) and
       allocates a separate buffer during training. */
    return (void*)input;
}

void* fiv_relu_node_create(void* params)
{
    fiv_relu_node* n = (fiv_relu_node*)fiv_malloc(sizeof(fiv_relu_node));
    if (!n) return NULL;
    n->base.create_fn    = fiv_relu_node_create;
    n->base.release_fn   = fiv_relu_node_release;
    n->base.forward_fn   = fiv_relu_node_forward;
    n->base.backward_fn  = fiv_relu_node_backward;
    n->base.inference_fn = fiv_relu_node_inference;
    n->base.alloc_out_fn = fiv_relu_node_alloc_out;
    return n;
}

void fiv_relu_node_release(void* op_state)
{
    fiv_relu_node* n = (fiv_relu_node*)op_state;
    if (n) fiv_free(n);
}

fiv_ret fiv_relu_node_forward(void* op_state, void* output, void* input)
{
    return fiv_relu(input, output);
}

fiv_ret fiv_relu_node_inference(void* op_state, void* output, void* input)
{
    return fiv_relu(input, output);
}

fiv_ret fiv_relu_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    if (!grad_input) return FIV_RET_OK;   /* input gradient not needed */
    const fiv_tensor_hdr* x  = (const fiv_tensor_hdr*)input;
    const fiv_tensor_hdr* go = (const fiv_tensor_hdr*)grad_output;
    fiv_tensor_hdr* gi = (fiv_tensor_hdr*)grad_input;
    if (!x || !go || !gi) return FIV_RET_ERR_PARA;
    if (x->id < FIV_ID_TENSOR1D || x->id > FIV_ID_TENSOR5D) return FIV_RET_ERR_PARA;
    if (x->total_bytes != go->total_bytes || x->total_bytes != gi->total_bytes) return FIV_RET_ERR_PARA;

    const ivf32* a = x->data.fl;
    const ivf32* g = go->data.fl;
    ivf32* h = gi->data.fl;
    size_t n = x->total_bytes / sizeof(ivf32);
    for (size_t i = 0; i < n; i++) h[i] += (a[i] > 0.0f && a[i] < 6.0f) ? g[i] : 0.0f;
    return FIV_RET_OK;
}
