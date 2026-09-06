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

#include "fiv_slice_node.h"

#include <stdlib.h>
#include <string.h>
#include "fiv_common.h"

/* Merged (n_total, c, h, w) view of any 3D~5D tensor, plus the merged-axis
   index that the user's `axis` maps onto. */
typedef struct { int n_total, c, h, w; } fiv_slice_view;

static int fiv_slice_compute_view(const fiv_tensor_hdr* in, fiv_slice_view* v, int user_axis, int* merged_axis)
{
    switch (in->id) {
    case FIV_ID_TENSOR3D: {
        const fiv_tensor3d* t = (const fiv_tensor3d*)in;
        v->n_total = 1; v->c = (int)t->channels; v->h = (int)t->height; v->w = (int)t->width;
        int map[] = { 1, 2, 3 };          /* c, h, w */
        if (user_axis < 0 || user_axis > 2) return 0;
        *merged_axis = map[user_axis];
        return 1;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)in;
        v->n_total = (int)t->batch; v->c = (int)t->channels; v->h = (int)t->height; v->w = (int)t->width;
        int map[] = { 0, 1, 2, 3 };       /* b, c, h, w */
        if (user_axis < 0 || user_axis > 3) return 0;
        *merged_axis = map[user_axis];
        return 1;
    }
    default: {
        const fiv_tensor5d* t = (const fiv_tensor5d*)in;
        v->n_total = (int)t->batch * (int)t->times;
        v->c = (int)t->channels; v->h = (int)t->height; v->w = (int)t->width;
        int map[] = { 0, 0, 1, 2, 3 };    /* b, t -> merged0; c, h, w */
        if (user_axis < 0 || user_axis > 4) return 0;
        *merged_axis = map[user_axis];
        return 1;
    }
    }
}

/* dim size and the contiguous run length (product of dims after this axis). */
static void fiv_slice_axis_info(const fiv_slice_view* v, int merged_axis, int* dim_size, int* stride_axis)
{
    if (merged_axis == 0)      { *dim_size = v->n_total; *stride_axis = v->c * v->h * v->w; }
    else if (merged_axis == 1) { *dim_size = v->c;       *stride_axis = v->h * v->w; }
    else if (merged_axis == 2) { *dim_size = v->h;       *stride_axis = v->w; }
    else                       { *dim_size = v->w;       *stride_axis = 1; }
}

void* fiv_slice_node_create(void* params)
{
    const fiv_slice_node_params* p = (const fiv_slice_node_params*)params;
    if (!p || p->start < 0 || p->end <= p->start) return NULL;
    fiv_slice_node* n = (fiv_slice_node*)fiv_malloc(sizeof(fiv_slice_node));
    if (!n) return NULL;
    memset(n, 0, sizeof(fiv_slice_node));
    n->base.create_fn    = fiv_slice_node_create;
    n->base.release_fn   = fiv_slice_node_release;
    n->base.forward_fn   = fiv_slice_node_forward;
    n->base.backward_fn  = fiv_slice_node_backward;
    n->base.inference_fn = fiv_slice_node_inference;
    n->base.alloc_out_fn = fiv_slice_node_alloc_out;
    n->axis  = p->axis;
    n->start = p->start;
    n->end   = p->end;
    return n;
}

void fiv_slice_node_release(void* op_state)
{
    fiv_free(op_state);
}

void* fiv_slice_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{
    const fiv_slice_node* n = (const fiv_slice_node*)op_state;
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

    fiv_slice_view v;
    int merged_axis = 0;
    if (!fiv_slice_compute_view(in, &v, n->axis, &merged_axis)) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }
    int dim_size = 0, stride_axis = 0;
    fiv_slice_axis_info(&v, merged_axis, &dim_size, &stride_axis);
    if (n->start >= dim_size || n->end > dim_size) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }
    int out_len = n->end - n->start;

    int on = v.n_total, oc = v.c, oh = v.h, ow = v.w;
    if (merged_axis == 0)      on = out_len;
    else if (merged_axis == 1) oc = out_len;
    else if (merged_axis == 2) oh = out_len;
    else                       ow = out_len;

    fiv_tensor_hdr* out = (fiv_tensor_hdr*)existing_output;
    if (out && out->id == in->id && out->dtype == FIV_32F1 && out->data_continue == 1) {
        int eb = 0, ec = 0, eh = 0, ew = 0, et = 0;
        fiv_slice_view ov; int ma = 0;
        if (fiv_slice_compute_view(out, &ov, n->axis, &ma)) {
            eb = ov.n_total; ec = ov.c; eh = ov.h; ew = ov.w;
            int ok = 0;
            if (in->id == FIV_ID_TENSOR3D)      ok = (ec == oc && eh == oh && ew == ow);
            else if (in->id == FIV_ID_TENSOR4D) ok = (eb == on && ec == oc && eh == oh && ew == ow);
            else { const fiv_tensor5d* t = (const fiv_tensor5d*)in; et = (int)t->times;
                   ok = (eb == (int)t->batch * out_len && ec == oc && eh == oh && ew == ow); }
            (void)et;
            if (ok) return out;
        }
    }
    if (out) fiv_release_tensor((void**)&out);

    switch (in->id) {
    case FIV_ID_TENSOR3D: {
        size_t sh[3] = { (size_t)oc, (size_t)oh, (size_t)ow };
        out = (fiv_tensor_hdr*)fiv_create_tensor3d(sh, FIV_32F1);
        break;
    }
    case FIV_ID_TENSOR4D: {
        size_t sh[4] = { (size_t)on, (size_t)oc, (size_t)oh, (size_t)ow };
        out = (fiv_tensor_hdr*)fiv_create_tensor4d(sh, FIV_32F1);
        break;
    }
    default: {
        const fiv_tensor5d* t = (const fiv_tensor5d*)in;
        size_t sh[5] = { (size_t)t->batch, (size_t)t->times, (size_t)oc, (size_t)oh, (size_t)ow };
        out = (fiv_tensor_hdr*)fiv_create_tensor5d(sh, FIV_32F1);
        break;
    }
    }
    if (!out) { *out_ret = FIV_RET_ERR_MEM; return NULL; }
    return out;
}

/* Copy [start, end) along the sliced axis. Every plane (all other axes fixed)
   is a contiguous run of `out_len * stride_axis` elements at offset
   `start * stride_axis` inside the plane block of `dim_size * stride_axis`. */
static size_t fiv_slice_flat(const int* shape, const int* idx)
{
    size_t f = 0;
    for (int d = 0; d < 4; d++) f = f * (size_t)shape[d] + (size_t)idx[d];
    return f;
}

static fiv_ret fiv_slice_copy(const fiv_slice_node* n, ivf32* dst, const ivf32* src,
                           const fiv_tensor_hdr* in)
{
    fiv_slice_view v;
    int merged_axis = 0;
    if (!fiv_slice_compute_view(in, &v, n->axis, &merged_axis))
        return FIV_RET_ERR_PARA;

    int dim_size = 0, stride_axis = 0;
    fiv_slice_axis_info(&v, merged_axis, &dim_size, &stride_axis);
    if (n->start < 0 || n->start >= dim_size || n->end > dim_size || n->end <= n->start)
        return FIV_RET_ERR_PARA;

    int ish[4] = { v.n_total, v.c, v.h, v.w };
    int osh[4];
    for (int d = 0; d < 4; d++) osh[d] = ish[d];
    osh[merged_axis] = n->end - n->start;

    size_t total_out = (size_t)osh[0] * (size_t)osh[1] * (size_t)osh[2] * (size_t)osh[3];
    int idx[4];
    for (size_t f = 0; f < total_out; f++) {
        size_t r = f;
        for (int d = 3; d >= 0; d--) {
            idx[d] = (int)(r % (size_t)osh[d]);
            r /= (size_t)osh[d];
        }
        int in_idx[4];
        for (int d = 0; d < 4; d++) in_idx[d] = idx[d];
        in_idx[merged_axis] = n->start + idx[merged_axis];
        dst[f] = src[fiv_slice_flat(ish, in_idx)];
    }
    return FIV_RET_OK;
}

fiv_ret fiv_slice_node_forward(void* op_state, void* output, void* input)
{
    fiv_slice_node* n = (fiv_slice_node*)op_state;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    fiv_tensor_hdr* out = (fiv_tensor_hdr*)output;
    if (!n || !in || !out) return FIV_RET_ERR_PARA;
    if (in->dtype != FIV_32F1 || out->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (!in->data_continue || !out->data_continue) return FIV_RET_ERR_PARA;
    return fiv_slice_copy(n, out->data.fl, in->data.fl, in);
}

fiv_ret fiv_slice_node_inference(void* op_state, void* output, void* input)
{
    return fiv_slice_node_forward(op_state, output, input);
}

/* Accumulate grad_output into the [start, end) range of grad_input along the
   sliced axis; other positions are left for their other consumers. */
fiv_ret fiv_slice_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    fiv_slice_node* n = (fiv_slice_node*)op_state;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    const fiv_tensor_hdr* go = (const fiv_tensor_hdr*)grad_output;
    fiv_tensor_hdr* gi = (fiv_tensor_hdr*)grad_input;
    if (!n || !in || !go || !gi) return FIV_RET_ERR_PARA;
    if (in->dtype != FIV_32F1 || go->dtype != FIV_32F1 || gi->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (!in->data_continue || !go->data_continue || !gi->data_continue) return FIV_RET_ERR_PARA;

    fiv_slice_view v;
    int merged_axis = 0;
    if (!fiv_slice_compute_view(in, &v, n->axis, &merged_axis)) return FIV_RET_ERR_PARA;
    int dim_size = 0, stride_axis = 0;
    fiv_slice_axis_info(&v, merged_axis, &dim_size, &stride_axis);
    if (n->start < 0 || n->start >= dim_size || n->end > dim_size || n->end <= n->start)
        return FIV_RET_ERR_PARA;

    size_t stride = (size_t)stride_axis;
    size_t block_in = (size_t)dim_size * stride;
    size_t copy_elems = (size_t)(n->end - n->start) * stride;
    size_t total = (size_t)v.n_total * (size_t)v.c * (size_t)v.h * (size_t)v.w;
    size_t n_planes = total / (size_t)dim_size;
    size_t start_off = (size_t)n->start * stride;
    const ivf32* gs = go->data.fl;
    ivf32*       gd = gi->data.fl;
    for (size_t p = 0; p < n_planes; p++) {
        const ivf32* sp = gs + p * copy_elems;
        ivf32*       dp = gd + p * block_in + start_off;
        for (size_t k = 0; k < copy_elems; k++) dp[k] += sp[k];
    }
    return FIV_RET_OK;
}
