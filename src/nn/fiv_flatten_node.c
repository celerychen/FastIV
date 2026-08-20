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

#include "fiv_flatten_node.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "fiv_common.h"

void* fiv_flatten_node_create(void* params)
{
    (void)params;
    fiv_flatten_node* n = (fiv_flatten_node*)fiv_malloc(sizeof(fiv_flatten_node));
    if (!n) return NULL;
    memset(n, 0, sizeof(fiv_flatten_node));
    n->base.create_fn    = fiv_flatten_node_create;
    n->base.release_fn   = fiv_flatten_node_release;
    n->base.forward_fn   = fiv_flatten_node_forward;
    n->base.backward_fn  = fiv_flatten_node_backward;
    n->base.inference_fn = fiv_flatten_node_inference;
    n->base.alloc_out_fn = fiv_flatten_node_alloc_out;
    return n;
}

void fiv_flatten_node_release(void* op_state)
{
    fiv_free(op_state);
}

/* rows = dims[0], cols = product of the rest (total / rows). */
static fiv_ret fiv_flatten_dims(const fiv_tensor_hdr* in, size_t* rows, size_t* cols)
{
    if (!in) return FIV_RET_ERR_PARA;
    if (in->id < FIV_ID_TENSOR1D || in->id > FIV_ID_TENSOR5D) return FIV_RET_ERR_PARA;
    if (in->dtype != FIV_32F1 || in->data_continue == 0) return FIV_RET_ERR_PARA;

    switch (in->id) {
    case FIV_ID_TENSOR1D:
        *rows = 1;
        *cols = ((const fiv_tensor1d*)in)->length;
        break;
    case FIV_ID_TENSOR2D:
        *rows = ((const fiv_tensor2d*)in)->rows;
        *cols = ((const fiv_tensor2d*)in)->cols;
        break;
    case FIV_ID_TENSOR3D:
        *rows = ((const fiv_tensor3d*)in)->channels;
        *cols = ((const fiv_tensor3d*)in)->height * ((const fiv_tensor3d*)in)->width;
        break;
    case FIV_ID_TENSOR4D:
        *rows = ((const fiv_tensor4d*)in)->batch;
        *cols = ((const fiv_tensor4d*)in)->channels
              * ((const fiv_tensor4d*)in)->height
              * ((const fiv_tensor4d*)in)->width;
        break;
    default:
        *rows = ((const fiv_tensor5d*)in)->batch * ((const fiv_tensor5d*)in)->times;
        *cols = ((const fiv_tensor5d*)in)->channels
              * ((const fiv_tensor5d*)in)->height
              * ((const fiv_tensor5d*)in)->width;
        break;
    }
    return FIV_RET_OK;
}

/* Output: 2D (rows, cols) header only; data is bound by fiv_tensor_reshape in
   forward/inference (zero-copy, shares the input buffer, reference = 0). */
void* fiv_flatten_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{
    (void)op_state;
    *out_ret = FIV_RET_OK;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!in) { *out_ret = FIV_RET_ERR_PARA; return NULL; }

    size_t rows = 0, cols = 0;
    *out_ret = fiv_flatten_dims(in, &rows, &cols);
    if (*out_ret != FIV_RET_OK) return NULL;

    fiv_mat* out = (fiv_mat*)existing_output;
    if (out && out->id == FIV_ID_TENSOR2D && out->dtype == FIV_32F1 && out->data_continue == 1
        && out->rows == rows && out->cols == cols) {
        return out;
    }
    if (out) fiv_release_tensor((void**)&out);
    size_t sh[2] = { rows, cols };
    out = fiv_create_tensor2d_header(sh, FIV_32F1);
    if (!out) { *out_ret = FIV_RET_ERR_MEM; return NULL; }
    return out;
}

/* Pure reinterpretation: bind out's data to in's buffer under the 2D shape. */
static fiv_ret fiv_flatten_compute(fiv_mat* out, const fiv_tensor_hdr* in)
{
    size_t rows = 0, cols = 0;
    fiv_ret r = fiv_flatten_dims(in, &rows, &cols);
    if (r != FIV_RET_OK) return r;
    if (!out || out->id != FIV_ID_TENSOR2D || out->dtype != FIV_32F1) return FIV_RET_ERR_PARA;
    size_t sh[2] = { rows, cols };
    return fiv_tensor_reshape(out, in, 2, sh);
}

fiv_ret fiv_flatten_node_forward(void* op_state, void* output, void* input)
{
    (void)op_state;
    return fiv_flatten_compute((fiv_mat*)output, (const fiv_tensor_hdr*)input);
}

fiv_ret fiv_flatten_node_inference(void* op_state, void* output, void* input)
{
    (void)op_state;
    return fiv_flatten_compute((fiv_mat*)output, (const fiv_tensor_hdr*)input);
}

/* Same element count both sides; accumulate grad_output into grad_input. */
fiv_ret fiv_flatten_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    (void)op_state;
    (void)input;
    const fiv_tensor_hdr* go = (const fiv_tensor_hdr*)grad_output;
    fiv_tensor_hdr* gi = (fiv_tensor_hdr*)grad_input;
    if (!go) return FIV_RET_ERR_PARA;
    if (!gi) return FIV_RET_OK;
    if (go->dtype != FIV_32F1 || gi->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (go->total_bytes != gi->total_bytes) return FIV_RET_ERR_PARA;
    const ivf32* gp = go->data.fl;
    ivf32* gip = gi->data.fl;
    size_t n = go->total_bytes / sizeof(ivf32);
    for (size_t i = 0; i < n; i++) gip[i] += gp[i];
    return FIV_RET_OK;
}
