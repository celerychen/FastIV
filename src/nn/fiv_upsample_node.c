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

#include "fiv_upsample_node.h"
#include "fiv_common.h"

#include <string.h>

/* Read batch / channels / h / w from any of 3D~5D via the shared header. */
typedef struct {
    size_t channels, height, width, batch;
} fiv_upsample_dims;

static int fiv_upsample_get_dims(const fiv_tensor_hdr* h, fiv_upsample_dims* d)
{
    if (h->id < FIV_ID_TENSOR3D || h->id > FIV_ID_TENSOR5D) return 0;
    switch (h->id) {
    case FIV_ID_TENSOR3D:
        d->channels = ((const fiv_tensor3d*)h)->channels;
        d->height = ((const fiv_tensor3d*)h)->height;
        d->width = ((const fiv_tensor3d*)h)->width;
        d->batch = 1;
        break;
    case FIV_ID_TENSOR4D:
        d->channels = ((const fiv_tensor4d*)h)->channels;
        d->height = ((const fiv_tensor4d*)h)->height;
        d->width = ((const fiv_tensor4d*)h)->width;
        d->batch = ((const fiv_tensor4d*)h)->batch;
        break;
    default:
        d->channels = ((const fiv_tensor5d*)h)->channels;
        d->height = ((const fiv_tensor5d*)h)->height;
        d->width = ((const fiv_tensor5d*)h)->width;
        d->batch = ((const fiv_tensor5d*)h)->batch * ((const fiv_tensor5d*)h)->times;
        break;
    }
    return 1;
}

/* out[h, w] = in[h >> 1, w >> 1]: each 2x2 output block copies one input pixel. */
static fiv_ret fiv_upsample_compute(fiv_tensor_hdr* out, const fiv_tensor_hdr* in)
{
    fiv_upsample_dims dims;
    if (!out || !in || !fiv_upsample_get_dims(in, &dims)) return FIV_RET_ERR_PARA;
    if (in->dtype != FIV_32F1 || out->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (!in->data_continue || !out->data_continue) return FIV_RET_ERR_PARA;

    size_t height = dims.height;
    size_t width = dims.width;
    size_t in_plane = height * width;
    size_t out_plane = in_plane << 2;                  /* 4x: (2H)*(2W) */
    if (out->total_bytes / sizeof(ivf32) != dims.batch * dims.channels * out_plane)
        return FIV_RET_ERR_PARA;

    const ivf32* src = in->data.fl;
    ivf32* dst = out->data.fl;
    for (size_t b = 0; b < dims.batch; b++) {
        for (size_t c = 0; c < dims.channels; c++) {
            const ivf32* src_plane = src + b * dims.channels * in_plane + c * in_plane;
            ivf32* dst_plane = dst + b * dims.channels * out_plane + c * out_plane;
            for (size_t oh = 0; oh < (height << 1); oh++) {
                const ivf32* src_row = src_plane + (oh >> 1) * width;
                ivf32* dst_row = dst_plane + oh * (width << 1);
                for (size_t ow = 0; ow < (width << 1); ow++) {
                    dst_row[ow] = src_row[ow >> 1];
                }
            }
        }
    }
    return FIV_RET_OK;
}

void* fiv_upsample_node_create(void* params)
{
    fiv_upsample2x_node* node = (fiv_upsample2x_node*)fiv_malloc(sizeof(fiv_upsample2x_node));
    if (!node) return NULL;
    memset(node, 0, sizeof(fiv_upsample2x_node));
    node->base.create_fn    = fiv_upsample_node_create;
    node->base.release_fn   = fiv_upsample_node_release;
    node->base.forward_fn   = fiv_upsample_node_forward;
    node->base.backward_fn  = fiv_upsample_node_backward;
    node->base.inference_fn = fiv_upsample_node_inference;
    node->base.alloc_out_fn = fiv_upsample_node_alloc_out;
    return node;
}

void fiv_upsample_node_release(void* op_state)
{
    fiv_free(op_state);
}

fiv_ret fiv_upsample_node_forward(void* op_state, void* output, void* input)
{
    (void)op_state;
    return fiv_upsample_compute((fiv_tensor_hdr*)output, (const fiv_tensor_hdr*)input);
}

fiv_ret fiv_upsample_node_inference(void* op_state, void* output, void* input)
{
    return fiv_upsample_node_forward(op_state, output, input);
}

/* Nearest-neighbor upsample backward: grad of each (h, w) input element is the
   sum of the grad of its four output descendants at (2h, 2w), (2h+1, 2w), etc. */
fiv_ret fiv_upsample_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    (void)op_state;
    fiv_upsample_dims dims;
    if (!grad_input || !grad_output || !fiv_upsample_get_dims((const fiv_tensor_hdr*)input, &dims))
        return FIV_RET_ERR_PARA;
    fiv_tensor_hdr* grad_i = (fiv_tensor_hdr*)grad_input;
    const fiv_tensor_hdr* grad_o = (const fiv_tensor_hdr*)grad_output;
    if (grad_i->dtype != FIV_32F1 || grad_o->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;

    size_t height = dims.height;
    size_t width = dims.width;
    size_t in_plane = height * width;
    size_t out_plane = in_plane << 2;

    const ivf32* grad_src = grad_o->data.fl;
    ivf32* grad_dst = grad_i->data.fl;
    for (size_t b = 0; b < dims.batch; b++) {
        for (size_t c = 0; c < dims.channels; c++) {
            const ivf32* go_plane = grad_src + b * dims.channels * out_plane + c * out_plane;
            ivf32* gi_plane = grad_dst + b * dims.channels * in_plane + c * in_plane;
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    ivf32 value = 0.0f;
                    for (size_t oh = 0; oh < 2; oh++) {
                        for (size_t ow = 0; ow < 2; ow++) {
                            size_t out_idx = ((h << 1) + oh) * (width << 1) + (w << 1) + ow;
                            value += go_plane[out_idx];
                        }
                    }
                    gi_plane[h * width + w] += value;
                }
            }
        }
    }
    return FIV_RET_OK;
}

/* Output shape is (batch, channels, 2H, 2W); reuse existing_output when it
   already matches to avoid re-allocation each inference. */
void* fiv_upsample_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{
    *out_ret = FIV_RET_OK;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!in || in->dtype != FIV_32F1 || in->data_continue == 0) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }
    fiv_upsample_dims dims;
    if (!fiv_upsample_get_dims(in, &dims)) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }

    fiv_tensor_hdr* out = (fiv_tensor_hdr*)existing_output;
    if (out && out->id == in->id && out->dtype == FIV_32F1 && out->data_continue == 1) {
        const fiv_tensor4d* t4 = (const fiv_tensor4d*)out;
        if (in->id == FIV_ID_TENSOR4D &&
            t4->batch == dims.batch && t4->channels == dims.channels &&
            t4->height == (dims.height << 1) && t4->width == (dims.width << 1))
            return out;
        /* 3D/5D reuse loop is not needed for the YuNet path (all 4D); re-allocate below. */
    }
    if (out) fiv_release_tensor((void**)&out);

    switch (in->id) {
    case FIV_ID_TENSOR3D: {
        size_t shape[3] = { dims.channels, dims.height << 1, dims.width << 1 };
        out = (fiv_tensor_hdr*)fiv_create_tensor3d(shape, FIV_32F1);
        break;
    }
    case FIV_ID_TENSOR4D: {
        size_t shape[4] = { dims.batch, dims.channels, dims.height << 1, dims.width << 1 };
        out = (fiv_tensor_hdr*)fiv_create_tensor4d(shape, FIV_32F1);
        break;
    }
    default: {
        const fiv_tensor5d* t5 = (const fiv_tensor5d*)in;
        size_t shape[5] = { t5->batch, t5->times, dims.channels, dims.height << 1, dims.width << 1 };
        out = (fiv_tensor_hdr*)fiv_create_tensor5d(shape, FIV_32F1);
        break;
    }
    }
    if (!out) { *out_ret = FIV_RET_ERR_MEM; return NULL; }
    return out;
}