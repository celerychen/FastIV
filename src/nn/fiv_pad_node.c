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

#include "fiv_pad_node.h"
#include "fiv_common.h"

#include <string.h>

/* Read C_in / H / W / batch from any of 3D~5D via the shared header. */
typedef struct {
    size_t C_in, H, W, B;
} fiv_pad_dims;

static int fiv_pad_get_dims(const fiv_tensor_hdr* h, fiv_pad_dims* d)
{
    if (h->id < FIV_ID_TENSOR3D || h->id > FIV_ID_TENSOR5D) return 0;
    switch (h->id) {
    case FIV_ID_TENSOR3D:
        d->C_in = ((const fiv_tensor3d*)h)->channels;
        d->H = ((const fiv_tensor3d*)h)->height;
        d->W = ((const fiv_tensor3d*)h)->width;
        d->B = 1;
        break;
    case FIV_ID_TENSOR4D:
        d->C_in = ((const fiv_tensor4d*)h)->channels;
        d->H = ((const fiv_tensor4d*)h)->height;
        d->W = ((const fiv_tensor4d*)h)->width;
        d->B = ((const fiv_tensor4d*)h)->batch;
        break;
    default:
        d->C_in = ((const fiv_tensor5d*)h)->channels;
        d->H = ((const fiv_tensor5d*)h)->height;
        d->W = ((const fiv_tensor5d*)h)->width;
        d->B = ((const fiv_tensor5d*)h)->batch * ((const fiv_tensor5d*)h)->times;
        break;
    }
    return 1;
}

void* fiv_pad_node_create(void* params)
{
    const fiv_pad_node_params* p = (const fiv_pad_node_params*)params;
    if (!p || p->output_channels <= 0) return NULL;
    fiv_pad_node* n = (fiv_pad_node*)fiv_malloc(sizeof(fiv_pad_node));
    if (!n) return NULL;
    memset(n, 0, sizeof(fiv_pad_node));
    n->base.create_fn    = fiv_pad_node_create;
    n->base.release_fn   = fiv_pad_node_release;
    n->base.forward_fn   = fiv_pad_node_forward;
    n->base.backward_fn  = fiv_pad_node_backward;
    n->base.inference_fn = fiv_pad_node_inference;
    n->base.alloc_out_fn = fiv_pad_node_alloc_out;
    n->output_channels = p->output_channels;
    return n;
}

void fiv_pad_node_release(void* op_state)
{
    fiv_free(op_state);
}

/* out = [in's C_in channels] followed by zero channels to output_channels. */
static fiv_ret fiv_pad_compute(fiv_pad_node* n, fiv_tensor_hdr* out, const fiv_tensor_hdr* in)
{
    fiv_pad_dims d;
    if (!n || !out || !in || !fiv_pad_get_dims(in, &d)) return FIV_RET_ERR_PARA;
    if (in->dtype != FIV_32F1 || out->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (!in->data_continue || !out->data_continue) return FIV_RET_ERR_PARA;
    if ((size_t)n->output_channels < d.C_in) return FIV_RET_ERR_PARA;

    size_t HW = d.H * d.W;
    size_t in_chan = d.C_in * HW;
    size_t out_chan = (size_t)n->output_channels * HW;
    if (out->total_bytes / sizeof(ivf32) != d.B * out_chan) return FIV_RET_ERR_PARA;

    const ivf32* s = in->data.fl;
    ivf32* dst = out->data.fl;
    for (size_t b = 0; b < d.B; b++) {
        const ivf32* sb = s + b * in_chan;
        ivf32* db = dst + b * out_chan;
        memcpy(db, sb, in_chan * sizeof(ivf32));              /* copy C_in channels */
        memset(db + in_chan, 0, (out_chan - in_chan) * sizeof(ivf32));  /* zero the rest */
    }
    return FIV_RET_OK;
}

fiv_ret fiv_pad_node_forward(void* op_state, void* output, void* input)
{
    return fiv_pad_compute((fiv_pad_node*)op_state, (fiv_tensor_hdr*)output, (const fiv_tensor_hdr*)input);
}

fiv_ret fiv_pad_node_inference(void* op_state, void* output, void* input)
{
    return fiv_pad_node_forward(op_state, output, input);
}

/* Only the first C_in channels carry gradient; the padded channels are dead. */
fiv_ret fiv_pad_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    fiv_pad_node* n = (fiv_pad_node*)op_state;
    fiv_pad_dims d;
    if (!n || !grad_input || !grad_output || !fiv_pad_get_dims((const fiv_tensor_hdr*)input, &d))
        return FIV_RET_ERR_PARA;
    fiv_tensor_hdr* gi = (fiv_tensor_hdr*)grad_input;
    const fiv_tensor_hdr* go = (const fiv_tensor_hdr*)grad_output;
    if (gi->dtype != FIV_32F1 || go->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    size_t HW = d.H * d.W;
    size_t in_chan = d.C_in * HW;
    size_t out_chan = (size_t)n->output_channels * HW;
    const ivf32* g = go->data.fl;
    ivf32* dg = gi->data.fl;
    for (size_t b = 0; b < d.B; b++) {
        const ivf32* gb = g + b * out_chan;
        ivf32* db = dg + b * in_chan;
        for (size_t k = 0; k < in_chan; k++) db[k] += gb[k];
    }
    return FIV_RET_OK;
}

void* fiv_pad_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{
    const fiv_pad_node* n = (const fiv_pad_node*)op_state;
    *out_ret = FIV_RET_OK;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!n || !in || in->dtype != FIV_32F1 || in->data_continue == 0) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }
    fiv_pad_dims d;
    if (!fiv_pad_get_dims(in, &d)) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }

    fiv_tensor_hdr* out = (fiv_tensor_hdr*)existing_output;
    if (out && out->id == in->id && out->dtype == FIV_32F1 && out->data_continue == 1) {
        size_t oC = 0, oH = 0, oW = 0, oB = 1;
        switch (out->id) {
        case FIV_ID_TENSOR3D:
            oC = ((fiv_tensor3d*)out)->channels;
            oH = ((fiv_tensor3d*)out)->height;
            oW = ((fiv_tensor3d*)out)->width;
            break;
        case FIV_ID_TENSOR4D:
            oB = ((fiv_tensor4d*)out)->batch;
            oC = ((fiv_tensor4d*)out)->channels;
            oH = ((fiv_tensor4d*)out)->height;
            oW = ((fiv_tensor4d*)out)->width;
            break;
        default:
            oB = ((fiv_tensor5d*)out)->batch * ((fiv_tensor5d*)out)->times;
            oC = ((fiv_tensor5d*)out)->channels;
            oH = ((fiv_tensor5d*)out)->height;
            oW = ((fiv_tensor5d*)out)->width;
            break;
        }
        if (oB == d.B && oC == (size_t)n->output_channels && oH == d.H && oW == d.W) return out;
    }
    if (out) fiv_release_tensor((void**)&out);

    switch (in->id) {
    case FIV_ID_TENSOR3D: {
        size_t sh[3] = { (size_t)n->output_channels, d.H, d.W };
        out = (fiv_tensor_hdr*)fiv_create_tensor3d(sh, FIV_32F1);
        break;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)in;
        size_t sh[4] = { t->batch, (size_t)n->output_channels, d.H, d.W };
        out = (fiv_tensor_hdr*)fiv_create_tensor4d(sh, FIV_32F1);
        break;
    }
    default: {
        const fiv_tensor5d* t = (const fiv_tensor5d*)in;
        size_t sh[5] = { t->batch, t->times, (size_t)n->output_channels, d.H, d.W };
        out = (fiv_tensor_hdr*)fiv_create_tensor5d(sh, FIV_32F1);
        break;
    }
    }
    if (!out) { *out_ret = FIV_RET_ERR_MEM; return NULL; }
    return out;
}
