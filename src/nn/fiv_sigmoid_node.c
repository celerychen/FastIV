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

#include "fiv_sigmoid_node.h"
#include "fiv_common.h"

#include <math.h>
#include <string.h>

/* Sigmoid with input clamping to stay in the numerically safe range of expf
   (aligns with the reference libfacedetection bound ~ +-88). */
static ivf32 fiv_sigmoid_apply(ivf32 value)
{
    if (value > 88.0f) value = 88.0f;
    else if (value < -88.0f) value = -88.0f;
    return 1.0f / (1.0f + expf(-value));
}

static fiv_ret fiv_sigmoid_compute(fiv_tensor_hdr* out, const fiv_tensor_hdr* in)
{
    if (!out || !in) return FIV_RET_ERR_PARA;
    if (in->dtype != FIV_32F1 || out->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (!in->data_continue || !out->data_continue) return FIV_RET_ERR_PARA;
    if (out->total_bytes != in->total_bytes) return FIV_RET_ERR_PARA;

    size_t samples = in->total_bytes / sizeof(ivf32);
    const ivf32* src = in->data.fl;
    ivf32* dst = out->data.fl;
    for (size_t index = 0; index < samples; index++) {
        dst[index] = fiv_sigmoid_apply(src[index]);
    }
    return FIV_RET_OK;
}

void* fiv_sigmoid_node_create(void* params)
{
    fiv_sigmoid_node* node = (fiv_sigmoid_node*)fiv_malloc(sizeof(fiv_sigmoid_node));
    if (!node) return NULL;
    memset(node, 0, sizeof(fiv_sigmoid_node));
    node->base.create_fn    = fiv_sigmoid_node_create;
    node->base.release_fn   = fiv_sigmoid_node_release;
    node->base.forward_fn   = fiv_sigmoid_node_forward;
    node->base.backward_fn  = fiv_sigmoid_node_backward;
    node->base.inference_fn = fiv_sigmoid_node_inference;
    node->base.alloc_out_fn = fiv_sigmoid_node_alloc_out;
    return node;
}

void fiv_sigmoid_node_release(void* op_state)
{
    fiv_free(op_state);
}

fiv_ret fiv_sigmoid_node_forward(void* op_state, void* output, void* input)
{
    (void)op_state;
    return fiv_sigmoid_compute((fiv_tensor_hdr*)output, (const fiv_tensor_hdr*)input);
}

fiv_ret fiv_sigmoid_node_inference(void* op_state, void* output, void* input)
{
    return fiv_sigmoid_node_forward(op_state, output, input);
}

/* dL/dx = dL/dy * y * (1 - y); re-derive y from the saved input x. */
fiv_ret fiv_sigmoid_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    (void)op_state;
    fiv_tensor_hdr* grad_i = (fiv_tensor_hdr*)grad_input;
    const fiv_tensor_hdr* grad_o = (const fiv_tensor_hdr*)grad_output;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!grad_i || !grad_o || !in) return FIV_RET_ERR_PARA;
    if (in->dtype != FIV_32F1 || grad_o->dtype != FIV_32F1 || grad_i->dtype != FIV_32F1)
        return FIV_RET_ERR_NOT_SUPPORT;

    size_t samples = in->total_bytes / sizeof(ivf32);
    const ivf32* grad_src = grad_o->data.fl;
    const ivf32* src = in->data.fl;
    ivf32* grad_dst = grad_i->data.fl;
    for (size_t index = 0; index < samples; index++) {
        ivf32 value = fiv_sigmoid_apply(src[index]);
        grad_dst[index] += grad_src[index] * value * (1.0f - value);
    }
    return FIV_RET_OK;
}

/* Output mirrors the input shape (same id / dtype / byte count). */
void* fiv_sigmoid_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{
    *out_ret = FIV_RET_OK;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!in || in->dtype != FIV_32F1 || in->data_continue == 0) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }

    fiv_tensor_hdr* out = (fiv_tensor_hdr*)existing_output;
    if (out && out->id == in->id && out->dtype == FIV_32F1 && out->data_continue == 1 &&
        out->total_bytes == in->total_bytes)
        return out;
    if (out) fiv_release_tensor((void**)&out);

    switch (in->id) {
    case FIV_ID_TENSOR3D: {
        const fiv_tensor3d* t = (const fiv_tensor3d*)in;
        size_t shape[3] = { t->channels, t->height, t->width };
        out = (fiv_tensor_hdr*)fiv_create_tensor3d(shape, FIV_32F1);
        break;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)in;
        size_t shape[4] = { t->batch, t->channels, t->height, t->width };
        out = (fiv_tensor_hdr*)fiv_create_tensor4d(shape, FIV_32F1);
        break;
    }
    default: {
        const fiv_tensor5d* t = (const fiv_tensor5d*)in;
        size_t shape[5] = { t->batch, t->times, t->channels, t->height, t->width };
        out = (fiv_tensor_hdr*)fiv_create_tensor5d(shape, FIV_32F1);
        break;
    }
    }
    if (!out) { *out_ret = FIV_RET_ERR_MEM; return NULL; }
    return out;
}