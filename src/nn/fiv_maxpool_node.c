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

#include "fiv_maxpool_node.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "fiv_common.h"

/* Shared dims extraction: (..., c, h, w) -> batch*times, channels, h, w. */
static void fiv_maxpool_dims(const fiv_tensor_hdr* in, size_t* batch, size_t* channels,
                             size_t* height, size_t* width)
{
    switch (in->id) {
    case FIV_ID_TENSOR3D:
        *batch = 1;
        *channels = ((const fiv_tensor3d*)in)->channels;
        *height = ((const fiv_tensor3d*)in)->height;
        *width = ((const fiv_tensor3d*)in)->width;
        break;
    case FIV_ID_TENSOR4D:
        *batch = ((const fiv_tensor4d*)in)->batch;
        *channels = ((const fiv_tensor4d*)in)->channels;
        *height = ((const fiv_tensor4d*)in)->height;
        *width = ((const fiv_tensor4d*)in)->width;
        break;
    default:
        *batch = ((const fiv_tensor5d*)in)->batch * ((const fiv_tensor5d*)in)->times;
        *channels = ((const fiv_tensor5d*)in)->channels;
        *height = ((const fiv_tensor5d*)in)->height;
        *width = ((const fiv_tensor5d*)in)->width;
        break;
    }
}

void* fiv_maxpool_node_create(void* params)
{
    const fiv_maxpool_node_params* p = (const fiv_maxpool_node_params*)params;
    if (!p || p->stride <= 0 || p->kernel_size_x <= 0 || p->kernel_size_y <= 0) return NULL;
    fiv_maxpool_node* n = (fiv_maxpool_node*)fiv_malloc(sizeof(fiv_maxpool_node));
    if (!n) return NULL;
    memset(n, 0, sizeof(fiv_maxpool_node));
    n->base.create_fn    = fiv_maxpool_node_create;
    n->base.release_fn   = fiv_maxpool_node_release;
    n->base.forward_fn   = fiv_maxpool_node_forward;
    n->base.backward_fn  = fiv_maxpool_node_backward;
    n->base.inference_fn = fiv_maxpool_node_inference;
    n->base.alloc_out_fn = fiv_maxpool_node_alloc_out;
    n->kernel_size_x = p->kernel_size_x;
    n->kernel_size_y = p->kernel_size_y;
    n->stride        = p->stride;
    n->pad_top       = p->pad_top;
    n->pad_bottom    = p->pad_bottom;
    n->pad_left      = p->pad_left;
    n->pad_right     = p->pad_right;
    return n;
}

void fiv_maxpool_node_release(void* op_state)
{
    fiv_maxpool_node* n = (fiv_maxpool_node*)op_state;
    if (!n) return;
    if (n->argmax) fiv_free(n->argmax);
    fiv_free(n);
}

void* fiv_maxpool_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{
    const fiv_maxpool_node* n = (const fiv_maxpool_node*)op_state;
    *out_ret = FIV_RET_OK;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!n || !in) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }
    if (in->id < FIV_ID_TENSOR3D || in->id > FIV_ID_TENSOR5D) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }
    if (in->dtype != FIV_32F1 || in->data_continue == 0) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }

    size_t batch    = 0;
    size_t channels = 0;
    size_t height   = 0;
    size_t width    = 0;
    fiv_maxpool_dims(in, &batch, &channels, &height, &width);
    size_t oh = (height + (size_t)n->pad_top + (size_t)n->pad_bottom - (size_t)n->kernel_size_y) / (size_t)n->stride + 1;
    size_t ow = (width + (size_t)n->pad_left + (size_t)n->pad_right - (size_t)n->kernel_size_x) / (size_t)n->stride + 1;
    if (oh == 0 || ow == 0) {
        *out_ret = FIV_RET_ERR_PARA;   /* input too small for the requested window */
        return NULL;
    }

    fiv_tensor_hdr* out = (fiv_tensor_hdr*)existing_output;
    if (out && out->id == in->id && out->dtype == FIV_32F1 && out->data_continue == 1) {
        size_t o_b = 0, o_c = 0, o_h = 0, o_w = 0;
        fiv_maxpool_dims(out, &o_b, &o_c, &o_h, &o_w);
        if (o_b == batch && o_c == channels && o_h == oh && o_w == ow) return out;
    }
    if (out) fiv_release_tensor((void**)&out);

    switch (in->id) {
    case FIV_ID_TENSOR3D: {
        size_t sh[3] = { channels, oh, ow };
        out = (fiv_tensor_hdr*)fiv_create_tensor3d(sh, FIV_32F1);
        break;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)in;
        size_t sh[4] = { t->batch, channels, oh, ow };
        out = (fiv_tensor_hdr*)fiv_create_tensor4d(sh, FIV_32F1);
        break;
    }
    default: {
        const fiv_tensor5d* t = (const fiv_tensor5d*)in;
        size_t sh[5] = { t->batch, t->times, channels, oh, ow };
        out = (fiv_tensor_hdr*)fiv_create_tensor5d(sh, FIV_32F1);
        break;
    }
    }
    if (!out) { *out_ret = FIV_RET_ERR_MEM; return NULL; }
    return out;
}

/* Max pooling with explicit zero-fill padding. For each output element the
   kernel window is clamped to the input bounds; out-of-range positions are
   skipped (equivalent to -inf), so only in-range values compete. The winning
   flat input offset is cached per output element for backward routing. */
static fiv_ret fiv_maxpool_compute(fiv_maxpool_node* n, fiv_tensor_hdr* out, const fiv_tensor_hdr* in)
{
    size_t batch    = 0;
    size_t channels = 0;
    size_t height   = 0;
    size_t width    = 0;
    fiv_maxpool_dims(in, &batch, &channels, &height, &width);

    size_t kh = (size_t)n->kernel_size_y;
    size_t kw = (size_t)n->kernel_size_x;
    size_t st = (size_t)n->stride;
    size_t pt = (size_t)n->pad_top;
    size_t pl = (size_t)n->pad_left;

    size_t oh = (height + pt + (size_t)n->pad_bottom - kh) / st + 1;
    size_t ow = (width + pl + (size_t)n->pad_right - kw) / st + 1;
    size_t n_chan = batch * channels;
    size_t ihw = height * width;
    size_t ohw = oh * ow;
    size_t need = n_chan * ohw;
    if (need == 0) return FIV_RET_OK;
    if (!n->argmax || n->n_out < need) {
        int* a = (int*)fiv_realloc(n->argmax, need * sizeof(int));
        if (!a) return FIV_RET_ERR_MEM;
        n->argmax = a;
        n->n_out = need;
    }

    const ivf32* ip = in->data.fl;
    ivf32*       op = out->data.fl;
    int*         am = n->argmax;
    for (size_t ch = 0; ch < n_chan; ch++) {
        const ivf32* src = ip + ch * ihw;
        ivf32*       dst = op + ch * ohw;
        int*         ama = am + ch * ohw;
        for (size_t oy = 0; oy < oh; oy++) {
            size_t y0 = (oy * st > pt) ? (oy * st - pt) : 0;   /* first in-range row */
            for (size_t ox = 0; ox < ow; ox++) {
                size_t x0 = (ox * st > pl) ? (ox * st - pl) : 0;
                ivf32 maxval = -INFINITY;
                int  maxoff = 0;
                for (size_t ky = 0; ky < kh; ky++) {
                    size_t iy = y0 + ky;
                    if (iy >= height) break;
                    const ivf32* row = src + iy * width;
                    for (size_t kx = 0; kx < kw; kx++) {
                        size_t ix = x0 + kx;
                        if (ix >= width) break;
                        ivf32 v = row[ix];
                        if (v > maxval) {
                            maxval = v;
                            maxoff = (int)(iy * width + ix);
                        }
                    }
                }
                dst[oy * ow + ox] = maxval;
                ama[oy * ow + ox] = maxoff;
            }
        }
    }
    return FIV_RET_OK;
}

fiv_ret fiv_maxpool_node_forward(void* op_state, void* output, void* input)
{
    return fiv_maxpool_compute((fiv_maxpool_node*)op_state, (fiv_tensor_hdr*)output,
                               (const fiv_tensor_hdr*)input);
}

fiv_ret fiv_maxpool_node_inference(void* op_state, void* output, void* input)
{
    return fiv_maxpool_node_forward(op_state, output, input);
}

/* Route each grad_output element to the single input element selected forward. */
fiv_ret fiv_maxpool_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    (void)input;
    fiv_maxpool_node* n = (fiv_maxpool_node*)op_state;
    const fiv_tensor_hdr* go = (const fiv_tensor_hdr*)grad_output;
    fiv_tensor_hdr* gi = (fiv_tensor_hdr*)grad_input;
    if (!n || !go) return FIV_RET_ERR_PARA;
    if (!gi) return FIV_RET_OK;   /* node-0 input: nothing to accumulate */

    size_t batch    = 0;
    size_t channels = 0;
    size_t height   = 0;
    size_t width    = 0;
    fiv_maxpool_dims(gi, &batch, &channels, &height, &width);
    size_t oh = 0, ow = 0;
    fiv_maxpool_dims(go, &batch, &channels, &oh, &ow);

    size_t n_chan = batch * channels;
    size_t ihw = height * width;
    size_t ohw = oh * ow;
    const ivf32* gp = go->data.fl;
    ivf32*       gip = gi->data.fl;
    for (size_t ch = 0; ch < n_chan; ch++) {
        const ivf32* goc = gp + ch * ohw;
        ivf32*       gic = gip + ch * ihw;
        const int*  ama = n->argmax + ch * ohw;
        for (size_t k = 0; k < ohw; k++) gic[(size_t)ama[k]] += goc[k];
    }
    return FIV_RET_OK;
}
