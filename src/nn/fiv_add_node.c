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

#include "fiv_add_node.h"
#include "fiv_common.h"

#include <string.h>

void* fiv_add_node_create(void* params)
{    fiv_add_node* n = (fiv_add_node*)fiv_malloc(sizeof(fiv_add_node));
    if (!n) return NULL;
    memset(n, 0, sizeof(fiv_add_node));
    n->base.create_fn           = fiv_add_node_create;
    n->base.release_fn          = fiv_add_node_release;
    n->base.forward_fn          = fiv_add_node_forward;
    n->base.backward_fn         = fiv_add_node_backward;
    n->base.inference_fn        = fiv_add_node_inference;
    n->base.alloc_out_fn        = fiv_add_node_alloc_out;
    n->base.forward_multi_fn    = fiv_add_node_forward_multi;
    n->base.backward_multi_fn   = fiv_add_node_backward_multi;
    n->base.inference_multi_fn  = fiv_add_node_inference_multi;
    return n;
}

void fiv_add_node_release(void* op_state)
{
    fiv_free(op_state);
}

/* Element-wise sum over N same-shaped inputs, flat float loop (dim-agnostic). */
static fiv_ret fiv_add_compute(void* output, void* const* inputs, int num_inputs)
{
    if (!output || !inputs || num_inputs < 1) return FIV_RET_ERR_PARA;
    const fiv_tensor_hdr* h0 = (const fiv_tensor_hdr*)inputs[0];
    fiv_tensor_hdr* oh = (fiv_tensor_hdr*)output;
    if (h0->dtype != FIV_32F1 || oh->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (h0->data_continue == 0 || oh->data_continue == 0) return FIV_RET_ERR_PARA;
    if (oh->total_bytes != h0->total_bytes) return FIV_RET_ERR_PARA;
    for (int k = 1; k < num_inputs; k++) {
        const fiv_tensor_hdr* h = (const fiv_tensor_hdr*)inputs[k];
        if (h->dtype != FIV_32F1 || h->data_continue == 0) return FIV_RET_ERR_PARA;
        if (h->total_bytes != h0->total_bytes) return FIV_RET_ERR_PARA;
    }

    size_t n = oh->total_bytes / sizeof(ivf32);
    const ivf32* a = h0->data.fl;
    ivf32* d = oh->data.fl;
    for (size_t i = 0; i < n; i++) d[i] = a[i];
    for (int k = 1; k < num_inputs; k++) {
        const ivf32* b = ((const fiv_tensor_hdr*)inputs[k])->data.fl;
        for (size_t i = 0; i < n; i++) d[i] += b[i];
    }
    return FIV_RET_OK;
}

fiv_ret fiv_add_node_forward(void* op_state, void* output, void* input)
{    return fiv_add_compute(output, &input, 1);
}

fiv_ret fiv_add_node_forward_multi(void* op_state, void* output, void* const* inputs, int num_inputs)
{    return fiv_add_compute(output, inputs, num_inputs);
}

fiv_ret fiv_add_node_inference(void* op_state, void* output, void* input)
{
    return fiv_add_node_forward(op_state, output, input);
}

fiv_ret fiv_add_node_inference_multi(void* op_state, void* output, void* const* inputs, int num_inputs)
{
    return fiv_add_node_forward_multi(op_state, output, inputs, num_inputs);
}

/* d(out)/d(in_i) = 1 for every input: grad_input_i += grad_output (accumulate). */
fiv_ret fiv_add_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    if (!grad_input || !grad_output) return FIV_RET_ERR_PARA;
    const fiv_tensor_hdr* go = (const fiv_tensor_hdr*)grad_output;
    fiv_tensor_hdr* gi = (fiv_tensor_hdr*)grad_input;
    if (gi->dtype != FIV_32F1 || go->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (gi->total_bytes != go->total_bytes) return FIV_RET_ERR_PARA;
    size_t n = go->total_bytes / sizeof(ivf32);
    const ivf32* g = go->data.fl;
    ivf32* d = gi->data.fl;
    for (size_t i = 0; i < n; i++) d[i] += g[i];
    return FIV_RET_OK;
}

fiv_ret fiv_add_node_backward_multi(void* op_state, void* const* grad_inputs,
                                    const void* grad_output, void* const* inputs, int num_inputs)
{
    if (!grad_inputs || !grad_output || num_inputs < 1) return FIV_RET_ERR_PARA;
    const fiv_tensor_hdr* go = (const fiv_tensor_hdr*)grad_output;
    if (go->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    size_t n = go->total_bytes / sizeof(ivf32);
    const ivf32* g = go->data.fl;
    for (int k = 0; k < num_inputs; k++) {
        if (!grad_inputs[k]) continue;   /* source is node 0 (external input): no grad buffer */
        fiv_tensor_hdr* gi = (fiv_tensor_hdr*)grad_inputs[k];
        if (gi->dtype != FIV_32F1 || gi->total_bytes != go->total_bytes) return FIV_RET_ERR_PARA;
        ivf32* d = gi->data.fl;
        for (size_t i = 0; i < n; i++) d[i] += g[i];
    }
    return FIV_RET_OK;
}

/* Output shape == input shape (all inputs share it). */
void* fiv_add_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{    *out_ret = FIV_RET_OK;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!in || in->dtype != FIV_32F1 || in->data_continue == 0) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }
    fiv_tensor_hdr* out = (fiv_tensor_hdr*)existing_output;
    if (out && out->id == in->id && out->dtype == FIV_32F1
        && out->data_continue == 1 && out->total_bytes == in->total_bytes) {
        return out;
    }
    if (out) fiv_release_tensor((void**)&out);
    out = (fiv_tensor_hdr*)fiv_create_tensor_like_tensor((void*)in);
    if (!out) { *out_ret = FIV_RET_ERR_MEM; return NULL; }
    return out;
}
