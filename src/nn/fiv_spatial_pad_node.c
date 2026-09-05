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

#include "fiv_spatial_pad_node.h"
#include "fiv_common.h"

#include <string.h>

typedef struct { size_t B, C, H, W; } fiv_spad_dims;

static int fiv_spad_extract(const fiv_tensor_hdr* h, fiv_spad_dims* d)
{
    if (h->id == FIV_ID_TENSOR4D) {
        const fiv_tensor4d* t = (const fiv_tensor4d*)h;
        d->B = t->batch; d->C = t->channels; d->H = t->height; d->W = t->width;
        return 1;
    } else if (h->id == FIV_ID_TENSOR3D) {
        const fiv_tensor3d* t = (const fiv_tensor3d*)h;
        d->B = 1; d->C = t->channels; d->H = t->height; d->W = t->width;
        return 1;
    }
    return 0;
}

void* fiv_spatial_pad_node_create(void* params)
{
    const fiv_spatial_pad_node_params* p = (const fiv_spatial_pad_node_params*)params;
    if (!p) return NULL;
    if (p->pad_top < 0 || p->pad_bottom < 0 || p->pad_left < 0 || p->pad_right < 0) return NULL;
    fiv_spatial_pad_node* n = (fiv_spatial_pad_node*)fiv_malloc(sizeof(fiv_spatial_pad_node));
    if (!n) return NULL;
    memset(n, 0, sizeof(fiv_spatial_pad_node));
    n->base.create_fn    = fiv_spatial_pad_node_create;
    n->base.release_fn   = fiv_spatial_pad_node_release;
    n->base.forward_fn   = fiv_spatial_pad_node_forward;
    n->base.backward_fn  = fiv_spatial_pad_node_backward;
    n->base.inference_fn = fiv_spatial_pad_node_inference;
    n->base.alloc_out_fn = fiv_spatial_pad_node_alloc_out;
    n->pad_top = p->pad_top; n->pad_bottom = p->pad_bottom;
    n->pad_left = p->pad_left; n->pad_right = p->pad_right;
    n->value = p->value;
    return n;
}

void fiv_spatial_pad_node_release(void* op_state)
{
    fiv_free(op_state);
}

static fiv_ret fiv_spad_compute(fiv_spatial_pad_node* n, fiv_tensor_hdr* out, const fiv_tensor_hdr* in)
{
    fiv_spad_dims d;
    if (!n || !out || !in || !fiv_spad_extract(in, &d)) return FIV_RET_ERR_PARA;
    if (in->dtype != FIV_32F1 || out->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (!in->data_continue || !out->data_continue) return FIV_RET_ERR_PARA;
    size_t oH = d.H + (size_t)n->pad_top + (size_t)n->pad_bottom;
    size_t oW = d.W + (size_t)n->pad_left + (size_t)n->pad_right;
    if (out->total_bytes / sizeof(ivf32) != d.B * d.C * oH * oW) return FIV_RET_ERR_PARA;

    ivf32* dst = out->data.fl;
    size_t oHW = oH * oW;
    for (size_t b = 0; b < d.B; b++)
        for (size_t c = 0; c < d.C; c++) {
            ivf32* db = dst + (b * d.C + c) * oHW;
            for (size_t i = 0; i < oHW; i++) db[i] = n->value;
            const ivf32* sb = in->data.fl + (b * d.C + c) * (d.H * d.W);
            for (size_t y = 0; y < d.H; y++) {
                size_t dof = (size_t)(y + n->pad_top) * oW + (size_t)n->pad_left;
                memcpy(db + dof, sb + y * d.W, d.W * sizeof(ivf32));
            }
        }
    return FIV_RET_OK;
}

fiv_ret fiv_spatial_pad_node_forward(void* op_state, void* output, void* input)
{
    return fiv_spad_compute((fiv_spatial_pad_node*)op_state, (fiv_tensor_hdr*)output, (const fiv_tensor_hdr*)input);
}

fiv_ret fiv_spatial_pad_node_inference(void* op_state, void* output, void* input)
{
    return fiv_spatial_pad_node_forward(op_state, output, input);
}

fiv_ret fiv_spatial_pad_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    const fiv_spatial_pad_node* n = (const fiv_spatial_pad_node*)op_state;
    fiv_spad_dims d;
    if (!n || !grad_input || !grad_output || !fiv_spad_extract((const fiv_tensor_hdr*)input, &d))
        return FIV_RET_ERR_PARA;
    fiv_tensor_hdr* gi = (fiv_tensor_hdr*)grad_input;
    const fiv_tensor_hdr* go = (const fiv_tensor_hdr*)grad_output;
    if (gi->dtype != FIV_32F1 || go->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    size_t oH = d.H + (size_t)n->pad_top + (size_t)n->pad_bottom;
    size_t oW = d.W + (size_t)n->pad_left + (size_t)n->pad_right;
    size_t oHW = oH * oW;
    for (size_t b = 0; b < d.B; b++)
        for (size_t c = 0; c < d.C; c++) {
            const ivf32* g = go->data.fl + (b * d.C + c) * oHW;
            ivf32* dg = gi->data.fl + (b * d.C + c) * (d.H * d.W);
            for (size_t y = 0; y < d.H; y++) {
                size_t so = (size_t)(y + n->pad_top) * oW + (size_t)n->pad_left;
                for (size_t x = 0; x < d.W; x++) dg[y * d.W + x] += g[so + x];
            }
        }
    return FIV_RET_OK;
}

void* fiv_spatial_pad_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{
    const fiv_spatial_pad_node* n = (const fiv_spatial_pad_node*)op_state;
    *out_ret = FIV_RET_OK;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!n || !in || in->dtype != FIV_32F1 || !in->data_continue) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }
    fiv_spad_dims d;
    if (!fiv_spad_extract(in, &d)) { *out_ret = FIV_RET_ERR_PARA; return NULL; }
    size_t oH = d.H + (size_t)n->pad_top + (size_t)n->pad_bottom;
    size_t oW = d.W + (size_t)n->pad_left + (size_t)n->pad_right;

    fiv_tensor_hdr* out = (fiv_tensor_hdr*)existing_output;
    if (out && out->id == in->id && out->dtype == FIV_32F1 && out->data_continue == 1) {
        fiv_spad_dims od;
        if (fiv_spad_extract(out, &od) && od.B == d.B && od.C == d.C && od.H == oH && od.W == oW) return out;
    }
    if (out) fiv_release_tensor((void**)&out);
    switch (in->id) {
    case FIV_ID_TENSOR4D: {
        size_t sh[4] = { d.B, d.C, oH, oW };
        out = (fiv_tensor_hdr*)fiv_create_tensor4d(sh, FIV_32F1);
        break;
    }
    case FIV_ID_TENSOR3D: {
        size_t sh[3] = { d.C, oH, oW };
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