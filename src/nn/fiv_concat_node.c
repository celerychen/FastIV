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

#include "fiv_concat_node.h"
#include "fiv_common.h"

#include <string.h>

/* Read per-channel element count HW = B*H*W and channel count from a tensor. */
static int fiv_concat_dims(const fiv_tensor_hdr* h, size_t* bhw, size_t* C)
{
    if (h->id == FIV_ID_TENSOR3D) {
        const fiv_tensor3d* t = (const fiv_tensor3d*)h;
        *C = t->channels;
        *bhw = (size_t)t->height * t->width;
        return 1;
    } else if (h->id == FIV_ID_TENSOR4D) {
        const fiv_tensor4d* t = (const fiv_tensor4d*)h;
        *C = t->channels;
        *bhw = (size_t)t->batch * t->height * t->width;
        return 1;
    }
    return 0;
}

void* fiv_concat_node_create(void* params)
{
    const fiv_concat_node_params* p = (const fiv_concat_node_params*)params;
    if (!p || p->output_channels <= 0) return NULL;
    fiv_concat_node* n = (fiv_concat_node*)fiv_malloc(sizeof(fiv_concat_node));
    if (!n) return NULL;
    memset(n, 0, sizeof(fiv_concat_node));
    n->base.create_fn          = fiv_concat_node_create;
    n->base.release_fn         = fiv_concat_node_release;
    n->base.forward_multi_fn   = fiv_concat_node_forward_multi;
    n->base.backward_multi_fn  = fiv_concat_node_backward_multi;
    n->base.inference_multi_fn = fiv_concat_node_inference_multi;
    n->axis            = p->axis;
    n->output_channels = p->output_channels;
    return n;
}

void fiv_concat_node_release(void* op_state)
{
    fiv_free(op_state);
}

fiv_ret fiv_concat_node_forward_multi(void* op_state, void* output, void* const* inputs, int num_inputs)
{
    const fiv_concat_node* n = (const fiv_concat_node*)op_state;
    fiv_tensor_hdr* out = (fiv_tensor_hdr*)output;
    if (!n || n->axis != 1 || !out || !inputs || num_inputs < 1) return FIV_RET_ERR_PARA;
    if (out->dtype != FIV_32F1 || !out->data_continue) return FIV_RET_ERR_NOT_SUPPORT;

    size_t bhw0, C0;
    if (!fiv_concat_dims((const fiv_tensor_hdr*)inputs[0], &bhw0, &C0)) return FIV_RET_ERR_PARA;
    size_t total = 0;
    for (int k = 0; k < num_inputs; k++) {
        const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)inputs[k];
        if (in->dtype != FIV_32F1 || !in->data_continue) return FIV_RET_ERR_NOT_SUPPORT;
        size_t bhw, C;
        if (!fiv_concat_dims(in, &bhw, &C) || bhw != bhw0) return FIV_RET_ERR_PARA;
        total += C;
    }
    if (total != (size_t)n->output_channels) return FIV_RET_ERR_PARA;
    if (out->total_bytes / sizeof(ivf32) != bhw0 * total) return FIV_RET_ERR_PARA;

    ivf32* dst = out->data.fl;
    for (int k = 0; k < num_inputs; k++) {
        const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)inputs[k];
        size_t bhw = 0, C = 0;
        fiv_concat_dims(in, &bhw, &C);
        memcpy(dst, in->data.fl, C * bhw * sizeof(ivf32));
        dst += C * bhw;
    }
    return FIV_RET_OK;
}

fiv_ret fiv_concat_node_forward(void* op_state, void* output, void* input)
{
    return fiv_concat_node_forward_multi(op_state, output, &input, 1);
}

fiv_ret fiv_concat_node_inference_multi(void* op_state, void* output, void* const* inputs, int num_inputs)
{
    return fiv_concat_node_forward_multi(op_state, output, inputs, num_inputs);
}

fiv_ret fiv_concat_node_inference(void* op_state, void* output, void* input)
{
    return fiv_concat_node_forward_multi(op_state, output, &input, 1);
}

fiv_ret fiv_concat_node_backward_multi(void* op_state, void* const* grad_inputs,
                                       const void* grad_output, void* const* inputs, int num_inputs)
{
    const fiv_concat_node* n = (const fiv_concat_node*)op_state;
    if (!n || n->axis != 1) return FIV_RET_ERR_NOT_SUPPORT;
    const fiv_tensor_hdr* go = (const fiv_tensor_hdr*)grad_output;
    if (!go) return FIV_RET_ERR_PARA;
    const ivf32* g = go->data.fl;
    for (int k = 0; k < num_inputs; k++) {
        if (!inputs[k] || !grad_inputs[k]) continue;
        fiv_tensor_hdr* gi = (fiv_tensor_hdr*)grad_inputs[k];
        size_t bhw = 0, C = 0;
        fiv_concat_dims((const fiv_tensor_hdr*)inputs[k], &bhw, &C);
        ivf32* d = gi->data.fl;
        for (size_t i = 0; i < C * bhw; i++) d[i] += g[i];
        g += C * bhw;
    }
    return FIV_RET_OK;
}

fiv_ret fiv_concat_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    /* Multi-input op: the engine calls the *_multi slot directly; this is a
       convenience single-input path only (never used with >1 source). */
    void* gis[1] = { grad_input };
    void* iis[1] = { (void*)input };
    return fiv_concat_node_backward_multi(op_state, gis, grad_output, iis, 1);
}

/* Output sized from the primary input's B/H/W with output_channels. */
void* fiv_concat_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{
    const fiv_concat_node* n = (const fiv_concat_node*)op_state;
    *out_ret = FIV_RET_OK;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!n || !in || in->dtype != FIV_32F1 || !in->data_continue) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }
    fiv_tensor_hdr* out = (fiv_tensor_hdr*)existing_output;
    if (out && out->id == in->id && out->dtype == FIV_32F1 && out->data_continue == 1) {
        size_t bhw, C;
        if (fiv_concat_dims(out, &bhw, &C) && C == (size_t)n->output_channels) return out;
    }
    if (out) fiv_release_tensor((void**)&out);
    switch (in->id) {
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)in;
        size_t sh[4] = { t->batch, (size_t)n->output_channels, t->height, t->width };
        out = (fiv_tensor_hdr*)fiv_create_tensor4d(sh, FIV_32F1);
        break;
    }
    case FIV_ID_TENSOR3D: {
        const fiv_tensor3d* t = (const fiv_tensor3d*)in;
        size_t sh[3] = { (size_t)n->output_channels, t->height, t->width };
        out = (fiv_tensor_hdr*)fiv_create_tensor3d(sh, FIV_32F1);
        break;
    }
    default:
        out = (fiv_tensor_hdr*)fiv_create_tensor_like_tensor((void*)in);
        break;
    }
    if (!out) { *out_ret = FIV_RET_ERR_MEM; return NULL; }
    return out;
}