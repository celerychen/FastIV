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

#include "fiv_attention_node.h"

#include <stdlib.h>
#include <string.h>

#include "fiv_common.h"
#include "fiv_data_typedefs.h"
#include "fiv_nn_conv2d.h"
#include "fiv_matrix.h"
#include "fiv_math_softmax.h"

/* Wrap a contiguous row-major buffer as a 2D matrix view for the generic
   fiv_matrix_mul (which reads shapes/data.ptr/data_continue only). */
static fiv_mat fiv_attn_view(ivf32* data, size_t rows, size_t cols)
{
    fiv_mat m;
    memset(&m, 0, sizeof(m));
    m.id = FIV_ID_TENSOR2D;
    m.dtype = FIV_32F1;
    m.data_continue = 1;
    m.element_bytes = sizeof(ivf32);
    m.shapes[0] = rows;
    m.shapes[1] = cols;
    m.data.ptr = data;
    m.total_bytes = rows * cols * sizeof(ivf32);
    return m;
}

/* Adds per-output-channel bias to a [batch, c_out, H, W] conv result (mirrors
 * fiv_conv2d_compute's bias loop, since fiv_tensor_conv2d only does the
 * multiply). */
static void fiv_attention_add_bias(ivf32* d, const ivf32* b, int batch, int c_out, size_t o_hw)
{
    for (int bb = 0; bb < batch; bb++) {
        for (int oc = 0; oc < c_out; oc++) {
            ivf32 bv = b[oc];
            ivf32* p = d + (size_t)(bb * c_out + oc) * o_hw;
            for (size_t k = 0; k < o_hw; k++) p[k] += bv;
        }
    }
}

/* Conv (std / depthwise) + bias, allocating nothing. dst/src/w are 4D tensors;
 * pad is the explicit same-padding amount (zero fill). */
static fiv_ret fiv_attention_conv(fiv_tensor4d* dst, const fiv_tensor4d* src,
                                 const fiv_tensor4d* w, const ivf32* bias,
                                 int method, int kx, int ky, int stride,
                                 int pad, int c_out)
{
    fiv_conv2d_params p;
    memset(&p, 0, sizeof(p));
    p.conv2d_method  = method;
    p.kernel_size_x  = kx;
    p.kernel_size_y  = ky;
    p.stride         = stride;
    p.padding_method = 0;   /* zero fill */
    p.pad_top = p.pad_bottom = p.pad_left = p.pad_right = pad;
    p.input_channels  = (int)src->channels;
    p.output_channels = c_out;
    p.bias = 1;

    fiv_ret r = fiv_tensor_conv2d((void*)dst, (void*)src, (void*)w, &p);
    if (r != FIV_RET_OK) return r;
    if (bias) {
        size_t o_hw = (size_t)dst->height * (size_t)dst->width;
        fiv_attention_add_bias(dst->data.fl, bias, (int)src->batch, c_out, o_hw);
    }
    return FIV_RET_OK;
}

void* fiv_attention_node_create(void* params)
{
    const fiv_attention_node_params* q = (const fiv_attention_node_params*)params;
    if (!q || q->dim <= 0 || q->num_heads <= 0 || q->key_dim <= 0 || q->head_dim <= 0)
        return NULL;
    if (q->head_dim * q->num_heads != q->dim) return NULL;   /* value dim must tile dim */

    fiv_attention_node* n = (fiv_attention_node*)fiv_malloc(sizeof(fiv_attention_node));
    if (!n) return NULL;
    memset(n, 0, sizeof(fiv_attention_node));
    n->base.create_fn    = fiv_attention_node_create;
    n->base.release_fn   = fiv_attention_node_release;
    n->base.forward_fn   = fiv_attention_node_forward;
    n->base.backward_fn  = fiv_attention_node_backward;
    n->base.inference_fn = fiv_attention_node_inference;
    n->base.alloc_out_fn = fiv_attention_node_alloc_out;
    n->dim        = q->dim;
    n->num_heads  = q->num_heads;
    n->key_dim    = q->key_dim;
    n->head_dim   = q->head_dim;
    n->pe_groups  = q->pe_groups;
    n->scale      = q->scale;
    n->has_weights = 0;
    return n;
}

void fiv_attention_node_release(void* op_state)
{
    fiv_attention_node* n = (fiv_attention_node*)op_state;
    if (!n) return;
    if (n->qkv_w)  fiv_release_tensor((void**)&n->qkv_w);
    if (n->pe_w)   fiv_release_tensor((void**)&n->pe_w);
    if (n->proj_w) fiv_release_tensor((void**)&n->proj_w);
    fiv_free(n->qkv_b);
    fiv_free(n->pe_b);
    fiv_free(n->proj_b);
    fiv_free(n);
}

fiv_ret fiv_attention_node_set_weights(void* op_state,
                                       const ivf32* qkv_w, const ivf32* qkv_b,
                                       const ivf32* pe_w,  const ivf32* pe_b,
                                       const ivf32* proj_w, const ivf32* proj_b)
{
    fiv_attention_node* n = (fiv_attention_node*)op_state;
    if (!n || !qkv_w || !qkv_b || !pe_w || !pe_b || !proj_w || !proj_b)
        return FIV_RET_ERR_PARA;

    int key_ch = n->key_dim * n->num_heads;
    int qkv_ch = n->dim + 2 * key_ch;

    /* qkv: [qkv_ch, dim, 1, 1] */
    size_t qkv_sh[4] = { (size_t)qkv_ch, (size_t)n->dim, 1, 1 };
    fiv_tensor4d* wq = fiv_create_tensor4d(qkv_sh, FIV_32F1);
    if (!wq) return FIV_RET_ERR_MEM;
    memcpy(wq->data.fl, qkv_w, (size_t)qkv_ch * n->dim * sizeof(ivf32));
    ivf32* bq = (ivf32*)fiv_malloc((size_t)qkv_ch * sizeof(ivf32));
    if (!bq) { fiv_release_tensor((void**)&wq); return FIV_RET_ERR_MEM; }
    memcpy(bq, qkv_b, (size_t)qkv_ch * sizeof(ivf32));

    /* pe: depthwise [dim, 1, 3, 3] */
    size_t pe_sh[4] = { (size_t)n->dim, 1, 3, 3 };
    fiv_tensor4d* wp = fiv_create_tensor4d(pe_sh, FIV_32F1);
    if (!wp) { fiv_release_tensor((void**)&wq); fiv_free(bq); return FIV_RET_ERR_MEM; }
    memcpy(wp->data.fl, pe_w, (size_t)n->dim * 1 * 3 * 3 * sizeof(ivf32));
    ivf32* bp = (ivf32*)fiv_malloc((size_t)n->dim * sizeof(ivf32));
    if (!bp) { fiv_release_tensor((void**)&wq); fiv_release_tensor((void**)&wp); fiv_free(bq); return FIV_RET_ERR_MEM; }
    memcpy(bp, pe_b, (size_t)n->dim * sizeof(ivf32));

    /* proj: [dim, dim, 1, 1] */
    size_t proj_sh[4] = { (size_t)n->dim, (size_t)n->dim, 1, 1 };
    fiv_tensor4d* wproj = fiv_create_tensor4d(proj_sh, FIV_32F1);
    if (!wproj) { fiv_release_tensor((void**)&wq); fiv_release_tensor((void**)&wp); fiv_free(bq); fiv_free(bp); return FIV_RET_ERR_MEM; }
    memcpy(wproj->data.fl, proj_w, (size_t)n->dim * n->dim * sizeof(ivf32));
    ivf32* bproj = (ivf32*)fiv_malloc((size_t)n->dim * sizeof(ivf32));
    if (!bproj) { fiv_release_tensor((void**)&wq); fiv_release_tensor((void**)&wp); fiv_release_tensor((void**)&wproj); fiv_free(bq); fiv_free(bp); return FIV_RET_ERR_MEM; }
    memcpy(bproj, proj_b, (size_t)n->dim * sizeof(ivf32));

    if (n->qkv_w)  fiv_release_tensor((void**)&n->qkv_w);
    if (n->pe_w)   fiv_release_tensor((void**)&n->pe_w);
    if (n->proj_w) fiv_release_tensor((void**)&n->proj_w);
    fiv_free(n->qkv_b); fiv_free(n->pe_b); fiv_free(n->proj_b);

    n->qkv_w = wq;  n->qkv_b = bq;
    n->pe_w  = wp;  n->pe_b  = bp;
    n->proj_w = wproj; n->proj_b = bproj;
    n->has_weights = 1;
    return FIV_RET_OK;
}

void* fiv_attention_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{
    const fiv_attention_node* n = (const fiv_attention_node*)op_state;
    *out_ret = FIV_RET_OK;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!n || !in) { *out_ret = FIV_RET_ERR_PARA; return NULL; }
    if (in->id != FIV_ID_TENSOR4D || in->dtype != FIV_32F1 || in->data_continue == 0) {
        *out_ret = FIV_RET_ERR_PARA; return NULL;
    }
    const fiv_tensor4d* t = (const fiv_tensor4d*)in;
    if ((int)t->channels != n->dim) { *out_ret = FIV_RET_ERR_PARA; return NULL; }

    /* Reuse existing output if shape matches (engine passes its cached tensor). */
    if (existing_output) {
        fiv_tensor_hdr* e = (fiv_tensor_hdr*)existing_output;
        if (e->id == FIV_ID_TENSOR4D && e->dtype == FIV_32F1 && e->data_continue == 1) {
            const fiv_tensor4d* et = (const fiv_tensor4d*)e;
            if ((int)et->batch == (int)t->batch && (int)et->channels == n->dim &&
                (int)et->height == (int)t->height && (int)et->width == (int)t->width)
                return e;
        }
        fiv_release_tensor(&existing_output);
    }
    size_t sh[4] = { (size_t)t->batch, (size_t)n->dim, (size_t)t->height, (size_t)t->width };
    fiv_tensor4d* out = fiv_create_tensor4d(sh, FIV_32F1);
    if (!out) { *out_ret = FIV_RET_ERR_MEM; return NULL; }
    return out;
}

fiv_ret fiv_attention_node_forward(void* op_state, void* output, void* input)
{
    fiv_attention_node* n = (fiv_attention_node*)op_state;
    const fiv_tensor4d* in = (const fiv_tensor4d*)input;
    fiv_tensor4d* out = (fiv_tensor4d*)output;
    if (!n || !in || !out) return FIV_RET_ERR_PARA;
    if (!n->has_weights) return FIV_RET_ERR_DATA_UNINITED;
    if (in->id != FIV_ID_TENSOR4D || out->id != FIV_ID_TENSOR4D) return FIV_RET_ERR_PARA;
    if (in->dtype != FIV_32F1 || out->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (!in->data_continue || !out->data_continue) return FIV_RET_ERR_PARA;
    if ((int)in->channels != n->dim || (int)out->channels != n->dim) return FIV_RET_ERR_PARA;

    const int batch  = (int)in->batch;
    const int H      = (int)in->height;
    const int W      = (int)in->width;
    const int N      = H * W;
    const int nh     = n->num_heads;
    const int key_ch = n->key_dim * nh;
    const int val_ch = n->head_dim * nh;            /* == dim */
    const int qkv_ch = n->dim + 2 * key_ch;
    /* qkv output is HEAD-MAJOR: head h owns a contiguous chunk of
       (2*key_dim + head_dim) channels laid out [q(key_dim), k(key_dim),
       v(head_dim)]. qkv_ch == nh * head_ch. */
    const int head_ch = 2 * n->key_dim + n->head_dim;
    const size_t plane = (size_t)N;                  /* one (H,W) channel plane */
    const size_t qkv_plane = (size_t)qkv_ch * N;

    /* temp tensors */
    size_t qkv_sh[4] = { (size_t)batch, (size_t)qkv_ch, (size_t)H, (size_t)W };
    size_t val_sh[4] = { (size_t)batch, (size_t)val_ch, (size_t)H, (size_t)W };
    size_t dim_sh[4] = { (size_t)batch, (size_t)n->dim, (size_t)H, (size_t)W };
    fiv_tensor4d* qkv = fiv_create_tensor4d(qkv_sh, FIV_32F1);
    fiv_tensor4d* O   = fiv_create_tensor4d(val_sh, FIV_32F1);
    fiv_tensor4d* pe  = fiv_create_tensor4d(dim_sh, FIV_32F1);
    fiv_tensor4d* ao  = fiv_create_tensor4d(dim_sh, FIV_32F1);
    if (!qkv || !O || !pe || !ao) {
        if (qkv) fiv_release_tensor((void**)&qkv);
        if (O)   fiv_release_tensor((void**)&O);
        if (pe)  fiv_release_tensor((void**)&pe);
        if (ao)  fiv_release_tensor((void**)&ao);
        return FIV_RET_ERR_MEM;
    }
    ivf32* S = (ivf32*)fiv_malloc((size_t)batch * nh * N * N * sizeof(ivf32));
    if (!S) {
        fiv_release_tensor((void**)&qkv); fiv_release_tensor((void**)&O);
        fiv_release_tensor((void**)&pe);  fiv_release_tensor((void**)&ao);
        return FIV_RET_ERR_MEM;
    }

    fiv_ret r = FIV_RET_OK;

    /* 1) qkv = Conv(x) ; 2) split into q/k/v (views into qkv) */
    r = fiv_attention_conv(qkv, in, n->qkv_w, n->qkv_b, FIV_CONV2D_STD, 1, 1, 1, 0, qkv_ch);
    if (r != FIV_RET_OK) goto cleanup;

    /* 3) per-head attention: S = (q*scale)^T @ k ; A = softmax(S) ; O = v @ A^T */
    for (int bb = 0; bb < batch; bb++) {
        const ivf32* qkv_bb = qkv->data.fl + (size_t)bb * qkv_plane;
        ivf32* O_bb   = O->data.fl + (size_t)bb * (size_t)val_ch * plane;
        ivf32* S_bb   = S + (size_t)bb * nh * N * N;
        for (int h = 0; h < nh; h++) {
            const ivf32* q_h = qkv_bb + (size_t)(h * head_ch) * plane;
            const ivf32* k_h = q_h + (size_t)n->key_dim * plane;
            const ivf32* v_h = q_h + (size_t)(2 * n->key_dim) * plane;
            ivf32* S_h = S_bb + (size_t)h * N * N;
            ivf32* O_h = O_bb + (size_t)(h * n->head_dim) * plane;

            /* S_h = scale * (q_h)^T @ k_h: q_h/k_h are stored [key_dim, N]
               (key_dim rows of N contiguous positions); generic fiv_matrix_mul
               adaptively picks the small/large path by byte size. */
            fiv_mat sA = fiv_attn_view((ivf32*)q_h, (size_t)n->key_dim, (size_t)N);
            fiv_mat sB = fiv_attn_view((ivf32*)k_h, (size_t)n->key_dim, (size_t)N);
            fiv_mat sC = fiv_attn_view(S_h, (size_t)N, (size_t)N);
            r = fiv_matrix_mul(&sC, &sA, &sB, 1, 0,
                               FIV_SCALAR_FP32(n->scale), FIV_SCALAR_FP32(0.0f));
            if (r != FIV_RET_OK) goto cleanup;
            /* A = softmax(S_h) over last dim (rows of length N) */
            fiv_math_softmax_real32(S_h, N, S_h, N, (size_t)N, (size_t)N);
            /* O_h = v_h @ (S_h)^T: v_h stored [head_dim, N], S_h stored [N, N] */
            fiv_mat oA = fiv_attn_view((ivf32*)v_h, (size_t)n->head_dim, (size_t)N);
            fiv_mat oB = fiv_attn_view(S_h, (size_t)N, (size_t)N);
            fiv_mat oC = fiv_attn_view(O_h, (size_t)n->head_dim, (size_t)N);
            r = fiv_matrix_mul(&oC, &oA, &oB, 0, 1,
                               FIV_SCALAR_FP32(1.0f), FIV_SCALAR_FP32(0.0f));
            if (r != FIV_RET_OK) goto cleanup;
        }
    }

    /* 4) pe = depthwise Conv3x3 on v ; attn_out = O + pe
       torch feeds pe with value.reshape(batch, dim, H, W): the split value is
       [B, num_heads, head_dim, N], so its dim-channel order is HEAD-MAJOR,
       channel = h*head_dim + vd. Gather each head's v chunk (qkv offset
       h*head_ch + 2*key_dim) into that contiguous layout first. */
    for (int bb = 0; bb < batch; bb++) {
        const ivf32* qkv_bb = qkv->data.fl + (size_t)bb * qkv_plane;

        fiv_tensor4d vfull;
        memset(&vfull, 0, sizeof(vfull));
        size_t vsh[4] = { 1, (size_t)n->dim, (size_t)H, (size_t)W };
        fiv_tensor4d* vfull_p = fiv_create_tensor4d(vsh, FIV_32F1);
        if (!vfull_p) { r = FIV_RET_ERR_MEM; goto cleanup; }
        for (int h = 0; h < nh; h++) {
            const ivf32* vseg = qkv_bb + (size_t)(h * head_ch + 2 * n->key_dim) * plane;
            ivf32* vdst = vfull_p->data.fl + (size_t)(h * n->head_dim) * plane;
            memcpy(vdst, vseg, (size_t)n->head_dim * plane * sizeof(ivf32));
        }

        fiv_tensor4d pe_bb;
        memset(&pe_bb, 0, sizeof(pe_bb));
        pe_bb.id = FIV_ID_TENSOR4D; pe_bb.dtype = FIV_32F1; pe_bb.data_continue = 1;
        pe_bb.batch = 1; pe_bb.channels = (size_t)n->dim; pe_bb.height = (size_t)H; pe_bb.width = (size_t)W;
        pe_bb.data.fl = pe->data.fl + (size_t)bb * (size_t)n->dim * plane;
        pe_bb.total_bytes = (size_t)n->dim * plane * sizeof(ivf32);

        r = fiv_attention_conv(&pe_bb, vfull_p, n->pe_w, n->pe_b, FIV_CONV2D_DEPTHWISE, 3, 3, 1, 1, n->dim);
        fiv_release_tensor((void**)&vfull_p);
        if (r != FIV_RET_OK) goto cleanup;

        /* attn_out[bb] = O[bb] + pe[bb] */
        ivf32* ao_bb = ao->data.fl + (size_t)bb * (size_t)n->dim * plane;
        const ivf32* pe_bb_d = pe->data.fl + (size_t)bb * (size_t)n->dim * plane;
        const ivf32* O_bb_d  = O->data.fl + (size_t)bb * (size_t)val_ch * plane;
        for (size_t k = 0; k < (size_t)n->dim * plane; k++) ao_bb[k] = O_bb_d[k] + pe_bb_d[k];
    }

    /* 5) out = Conv(attn_out) */
    r = fiv_attention_conv(out, ao, n->proj_w, n->proj_b, FIV_CONV2D_STD, 1, 1, 1, 0, n->dim);

cleanup:
    fiv_release_tensor((void**)&qkv);
    fiv_release_tensor((void**)&O);
    fiv_release_tensor((void**)&pe);
    fiv_release_tensor((void**)&ao);
    fiv_free(S);
    return r;
}

fiv_ret fiv_attention_node_inference(void* op_state, void* output, void* input)
{
    return fiv_attention_node_forward(op_state, output, input);
}

fiv_ret fiv_attention_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    (void)op_state; (void)grad_input; (void)grad_output; (void)input;
    return FIV_RET_ERR_NOT_SUPPORT;   /* YOLO26 port is inference-only */
}
