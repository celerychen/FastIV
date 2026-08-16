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

#include <math.h>
#include <string.h>
#include "fiv_linear_node.h"
#include "fiv_matrix.h"
#include "fiv_common.h"

/* Deterministic LCG so training is reproducible across runs. */
static unsigned g_seed = 12345u;

static float fiv_nn_rand(void)
{
    g_seed = g_seed * 1103515245u + 12345u;
    return ((float)((g_seed >> 8) & 0xffffff) / 16777216.0f) * 2.0f - 1.0f;   /* [-1, 1] */
}

static fiv_ret fiv_linear_compute(fiv_linear_node* n, fiv_mat* out, const fiv_mat* in)
{
    if (!n || !n->weight || !in || !out) return FIV_RET_ERR_PARA;
    if (n->in_features <= 0 || n->out_features <= 0) return FIV_RET_ERR_PARA;
    if (in->dtype != FIV_32F1 || out->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (in->data_continue == 0 || out->data_continue == 0) return FIV_RET_ERR_PARA;

    /* output = input * weight^T   (weight stored as [out_features, in_features]) */
    fiv_ret r = fiv_matrix_mul(out, in, n->weight, 0, 1, 1.0f, 0.0f);
    if (r != FIV_RET_OK) return r;

    if (n->bias) {
        if (n->bias->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
        const ivf32* b = n->bias->data.fl;
        ivf32* o = out->data.fl;
        size_t rows = out->rows;
        size_t cols = out->cols;
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                o[i * cols + j] += b[j];
    }
    return FIV_RET_OK;
}

void* fiv_linear_node_create(void* params)
{
    const fiv_linear_node_params* p = (const fiv_linear_node_params*)params;
    if (!p || p->in_features <= 0 || p->out_features <= 0) return NULL;

    fiv_linear_node* n = (fiv_linear_node*)fiv_malloc(sizeof(fiv_linear_node));
    if (!n) return NULL;
    memset(n, 0, sizeof(fiv_linear_node));
    n->base.create_fn    = fiv_linear_node_create;
    n->base.release_fn   = fiv_linear_node_release;
    n->base.forward_fn   = fiv_linear_node_forward;
    n->base.backward_fn  = fiv_linear_node_backward;
    n->base.inference_fn = fiv_linear_node_inference;
    n->base.alloc_out_fn = fiv_linear_node_alloc_out;
    n->in_features  = p->in_features;
    n->out_features = p->out_features;

    size_t wsh[2] = { (size_t)p->out_features, (size_t)p->in_features };
    n->weight = fiv_create_tensor2d(wsh, FIV_32F1);
    if (!n->weight) { fiv_free(n); return NULL; }
    /* random init (1/sqrt(in) scale) breaks symmetry so ReLU layers learn */
    {
        ivf32* w = n->weight->data.fl;
        size_t nw = (size_t)(p->out_features * p->in_features);
        float s = 1.0f / sqrtf((float)p->in_features);
        for (size_t k = 0; k < nw; k++) w[k] = fiv_nn_rand() * s;
    }

    size_t bsh = (size_t)p->out_features;
    n->bias = fiv_create_tensor1d(bsh, FIV_32F1);
    if (!n->bias) { fiv_release_tensor2d(&n->weight); fiv_free(n); return NULL; }
    memset(n->bias->data.ptr, 0, n->bias->total_bytes);

    n->grad_weight = fiv_create_tensor2d(wsh, FIV_32F1);
    if (!n->grad_weight) { fiv_release_tensor1d(&n->bias); fiv_release_tensor2d(&n->weight); fiv_free(n); return NULL; }
    memset(n->grad_weight->data.ptr, 0, n->grad_weight->total_bytes);
    n->grad_bias = fiv_create_tensor1d(bsh, FIV_32F1);
    if (!n->grad_bias) {
        fiv_release_tensor2d(&n->grad_weight);
        fiv_release_tensor1d(&n->bias);
        fiv_release_tensor2d(&n->weight);
        fiv_free(n);
        return NULL;
    }
    memset(n->grad_bias->data.ptr, 0, n->grad_bias->total_bytes);

    return n;
}

void fiv_linear_node_release(void* op_state)
{
    fiv_linear_node* n = (fiv_linear_node*)op_state;
    if (!n) return;
    if (n->weight)      fiv_release_tensor2d(&n->weight);
    if (n->bias)        fiv_release_tensor1d(&n->bias);
    if (n->grad_weight) fiv_release_tensor2d(&n->grad_weight);
    if (n->grad_bias)   fiv_release_tensor1d(&n->grad_bias);
    fiv_free(n);
}

void* fiv_linear_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{
    *out_ret = FIV_RET_OK;
    const fiv_linear_node* n = (const fiv_linear_node*)op_state;
    const fiv_mat* in = (const fiv_mat*)input;
    if (!n || !in) { *out_ret = FIV_RET_ERR_PARA; return NULL; }
    if (in->dtype != FIV_32F1 || in->data_continue == 0) { *out_ret = FIV_RET_ERR_PARA; return NULL; }

    fiv_mat* out = (fiv_mat*)existing_output;
    if (out && out->id == FIV_ID_TENSOR2D && out->dtype == FIV_32F1 && out->data_continue == 1
        && out->rows == in->rows && out->cols == (size_t)n->out_features) {
        return out;
    }

    if (out) fiv_release_tensor((void**)&out);
    size_t sh[2] = { in->rows, (size_t)n->out_features };
    out = fiv_create_tensor2d(sh, FIV_32F1);
    if (!out) { *out_ret = FIV_RET_ERR_MEM; return NULL; }
    return out;
}

fiv_ret fiv_linear_node_forward(void* op_state, void* output, void* input)
{
    fiv_linear_node* n = (fiv_linear_node*)op_state;
    fiv_ret r = fiv_linear_compute(n, (fiv_mat*)output, (const fiv_mat*)input);
    if (r == FIV_RET_OK) n->cached_input = (const fiv_mat*)input;
    return r;
}

fiv_ret fiv_linear_node_inference(void* op_state, void* output, void* input)
{
    return fiv_linear_compute((fiv_linear_node*)op_state, (fiv_mat*)output, (const fiv_mat*)input);
}

/* y = x*W^T + b, x [N,in], W [out,in]:
   dW += go^T * x   (go [N,out] -> [out,in])
   db += row sums of go
   grad_input += go * W   ([N,out] * [out,in] -> [N,in]) */
fiv_ret fiv_linear_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    fiv_linear_node* n = (fiv_linear_node*)op_state;
    const fiv_mat* go = (const fiv_mat*)grad_output;
    const fiv_mat* x  = (const fiv_mat*)input;
    fiv_mat* gi = (fiv_mat*)grad_input;
    if (!n || !n->weight || !n->grad_weight || !go || !x) return FIV_RET_ERR_PARA;

    size_t N   = x->rows;
    size_t out = (size_t)n->out_features;

    fiv_ret r = fiv_matrix_mul(n->grad_weight, go, x, 1, 0, 1.0f, 1.0f);
    if (r != FIV_RET_OK) return r;

    const ivf32* g = go->data.fl;
    ivf32* db = n->grad_bias->data.fl;
    for (size_t j = 0; j < out; j++) {
        float s = 0.0f;
        for (size_t i = 0; i < N; i++) s += g[i * out + j];
        db[j] += s;
    }

    if (gi) {
        r = fiv_matrix_mul(gi, go, n->weight, 0, 0, 1.0f, 1.0f);
        if (r != FIV_RET_OK) return r;
    }
    return FIV_RET_OK;
}
