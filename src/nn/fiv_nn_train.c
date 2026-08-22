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

#include "fiv_nn_train.h"
#include "fiv_nn_topo.h"
#include "fiv_linear_node.h"
#include "fiv_nn_conv2d.h"
#include "fiv_matrix.h"
#include "fiv_common.h"

#include <math.h>
#include <string.h>

/* All tensor dims share this prefix: id, dtype, meta, data pointer, total_bytes
   and the leading shape entries. Reading shape by fiv_data_id (shapes[0..1])
   keeps the training framework free of any fixed matrix/view assumption. */
typedef struct {
    fiv_data_id   id;
    fiv_data_type dtype;
    iv8u          meta_info[4];   /* reference, data_continue, element_bytes, color_space_type */
    void*         data_ptr;
    size_t        total_bytes;
    size_t        shapes[5];
} fiv_nn_thdr;

/* Gradient buffer for node i: same shape as nodes[i].output, reused. */
static void* fiv_nn_grad_get(fiv_nn_network_context* net, void** grads, int i)
{
    void* out = net->nodes[i].output;
    if (!out) return NULL;
    const fiv_nn_thdr* oh = (const fiv_nn_thdr*)out;
    void* g = grads[i];
    if (g) {
        const fiv_nn_thdr* gh = (const fiv_nn_thdr*)g;
        if (gh->id == oh->id && gh->total_bytes == oh->total_bytes
            && gh->dtype == oh->dtype) {
            return g;
        }
        fiv_release_tensor(&grads[i]);
    }
    grads[i] = fiv_create_tensor_like_tensor(out);
    if (grads[i]) {
        fiv_nn_thdr* ng = (fiv_nn_thdr*)grads[i];
        memset(ng->data_ptr, 0, ng->total_bytes);
    }
    return grads[i];
}

static void fiv_nn_grads_zero(fiv_nn_network_context* net, void** grads)
{
    for (int i = 0; i < net->node_count; i++) {
        if (grads[i]) {
            fiv_nn_thdr* g = (fiv_nn_thdr*)grads[i];
            memset(g->data_ptr, 0, g->total_bytes);
        }
    }
}

/* MSE loss: grad_out = 2*(o-l)/N. out/lab/grad are generic tensors read by
   their shared header; out and lab must hold the same element count. */
static fiv_ret fiv_nn_loss_mse(const void* out, const void* lab, void* grad)
{
    const fiv_nn_thdr* o = (const fiv_nn_thdr*)out;
    const fiv_nn_thdr* l = (const fiv_nn_thdr*)lab;
    if (o->dtype != FIV_32F1 || l->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (o->total_bytes != l->total_bytes) return FIV_RET_ERR_PARA;
    size_t n = o->total_bytes / sizeof(ivf32);
    const ivf32* op = (const ivf32*)o->data_ptr;
    const ivf32* lp = (const ivf32*)l->data_ptr;
    ivf32* g = (ivf32*)((fiv_nn_thdr*)grad)->data_ptr;
    for (size_t i = 0; i < n; i++) {
        float d = op[i] - lp[i];
        g[i] = 2.0f * d / (float)n;
    }
    return FIV_RET_OK;
}

/* CE loss with row softmax: out is [N, C] (2D); labels are per-row class
   indices in a 1D tensor. grad_out = (softmax(out) - onehot(label)) / N. */
static fiv_ret fiv_nn_loss_ce(const void* out, const void* lab, void* grad)
{
    const fiv_nn_thdr* o = (const fiv_nn_thdr*)out;
    const fiv_nn_thdr* l = (const fiv_nn_thdr*)lab;
    if (o->dtype != FIV_32F1 || l->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (o->id != FIV_ID_TENSOR2D || l->id != FIV_ID_TENSOR1D) return FIV_RET_ERR_PARA;
    size_t N = o->shapes[0];
    size_t C = o->shapes[1];
    if (N == 0 || C == 0) return FIV_RET_ERR_PARA;
    if (l->total_bytes / sizeof(ivf32) != N) return FIV_RET_ERR_PARA;
    const ivf32* op = (const ivf32*)o->data_ptr;
    const ivf32* lp = (const ivf32*)l->data_ptr;
    ivf32* g = (ivf32*)((fiv_nn_thdr*)grad)->data_ptr;
    for (size_t i = 0; i < N; i++) {
        const ivf32* row = op + i * C;
        float m = row[0];
        for (size_t j = 1; j < C; j++) if (row[j] > m) m = row[j];
        float s = 0.0f;
        for (size_t j = 0; j < C; j++) s += expf(row[j] - m);
        int cls = (int)lp[i];
        if (cls < 0 || (size_t)cls >= C) return FIV_RET_ERR_PARA;
        for (size_t j = 0; j < C; j++) g[i * C + j] = expf(row[j] - m) / s;
        g[i * C + (size_t)cls] -= 1.0f;
    }
    for (size_t i = 0; i < N * C; i++) g[i] /= (float)N;
    return FIV_RET_OK;
}

/* SGD: W -= lr * dW, b -= lr * db, then zero the accumulated gradients. */
static void fiv_nn_sgd_update(fiv_nn_network_context* net, float lr)
{
    for (int i = 1; i < net->node_count; i++) {
        fiv_nn_node_context* nd = &net->nodes[i];
        if (nd->node_type == FIV_NN_NODE_LINEAR) {
            fiv_linear_node* ln = (fiv_linear_node*)nd->op;

            size_t nw = (size_t)(ln->in_features * ln->out_features);
            ivf32* W = ln->weight->data.fl;
            ivf32* dW = ln->grad_weight->data.fl;
            for (size_t k = 0; k < nw; k++) W[k] -= lr * dW[k];
            memset(dW, 0, ln->grad_weight->total_bytes);

            size_t nb = (size_t)ln->out_features;
            ivf32* b = ln->bias->data.fl;
            ivf32* db = ln->grad_bias->data.fl;
            for (size_t k = 0; k < nb; k++) b[k] -= lr * db[k];
            memset(db, 0, ln->grad_bias->total_bytes);
        } else if (nd->node_type == FIV_NN_NODE_CONV2D_STD ||
                   nd->node_type == FIV_NN_NODE_CONV2D_DEPTHWISE ||
                   nd->node_type == FIV_NN_NODE_CONV2D_POINTWISE) {
            fiv_conv2d_node* cn = (fiv_conv2d_node*)nd->op;

            int kcin = (cn->params.conv2d_method == FIV_CONV2D_DEPTHWISE) ? 1
                                                                          : cn->params.input_channels;
            size_t nw = (size_t)cn->params.output_channels * (size_t)kcin *
                        (size_t)cn->params.kernel_size_y * (size_t)cn->params.kernel_size_x;
            ivf32* W = cn->weight->data.fl;
            ivf32* dW = cn->grad_weight->data.fl;
            for (size_t k = 0; k < nw; k++) W[k] -= lr * dW[k];
            memset(dW, 0, cn->grad_weight->total_bytes);

            if (cn->bias && cn->grad_bias) {
                size_t nb = (size_t)cn->params.output_channels;
                ivf32* b = cn->bias->data.fl;
                ivf32* db = cn->grad_bias->data.fl;
                for (size_t k = 0; k < nb; k++) b[k] -= lr * db[k];
                memset(db, 0, cn->grad_bias->total_bytes);
            }
        }
    }
}

fiv_ret fiv_neural_network_train(void* nn_context, fiv_nn_train_params* p, fiv_load_dataset_fn load_dataset)
{
    fiv_nn_network_context* net = (fiv_nn_network_context*)nn_context;
    if (!net || !p || !load_dataset) return FIV_RET_ERR_PARA;
    if (p->epoch_num <= 0 || p->bach_size <= 0 || p->learning_rate <= 0.0f) return FIV_RET_ERR_PARA;
    if (p->loss_fn_type != 0 && p->loss_fn_type != 1) return FIV_RET_ERR_PARA;

    void** grads = (void**)fiv_calloc((size_t)net->node_count, sizeof(void*));
    if (!grads) return FIV_RET_ERR_MEM;
    fiv_ret r = FIV_RET_OK;

    for (int e = 0; e < p->epoch_num; e++) {
        int b = 0;
        for (;;) {
            /* One callback call yields one batch. The callback fills
               info.current_batch_inputs / current_batch_outputs in place;
               the framework reads them directly here. */
            fiv_dadaset_info info;
            memset(&info, 0, sizeof(info));
            info.current_epoch_index = e;   /* current epoch, visible to the callback */
            info.current_batch_index = b;   /* batch index within the epoch */
            r = load_dataset(&info);
            if (r == FIV_RET_ERR_END_OF_FILE) break;   /* current epoch done */
            if (r != FIV_RET_OK) goto fail;
            b++;

            /* The framework makes no assumption about the batch tensor format.
               Input is handed to the network's input node and the label to the
               loss function; both consumers know their own expected data type
               and convert internally. */
            void* b_in  = info.current_batch_inputs;
            void* b_out = info.current_batch_outputs;
            if (!b_in || !b_out) {
                r = FIV_RET_ERR_PARA;
                goto fail;
            }

            /* forward (training mode, may cache per-node state) */
            void* final = NULL;
            r = fiv_nn_run_forward(net, b_in, &final);
            if (r != FIV_RET_OK) goto fail;

            fiv_nn_grads_zero(net, grads);

            /* loss and its gradient w.r.t. the output node */
            void* grad_final = fiv_nn_grad_get(net, grads, net->output_node);
            if (!grad_final) { r = FIV_RET_ERR_MEM; goto fail; }
            r = (p->loss_fn_type == 0)
                  ? fiv_nn_loss_ce(final, b_out, grad_final)
                  : fiv_nn_loss_mse(final, b_out, grad_final);
            if (r != FIV_RET_OK) goto fail;

            /* backward in reverse topo order; each node accumulates its
               input gradient onto its source node(s) */
            for (int k = net->topo_count - 1; k >= 0; k--) {
                int i = net->topo_order[k];
                if (i == 0) continue;
                fiv_nn_node_context* nd = &net->nodes[i];
                fiv_nn_op_base* o = (fiv_nn_op_base*)nd->op;
                if (!o) { r = FIV_RET_ERR_DATA_UNINITED; goto fail; }
                void* grad_out = grads[i];
                if (!grad_out) continue;   /* no downstream contribution */
                if (nd->num_src > 1 && o->backward_multi_fn) {
                    /* multi-input node (ADD): one grad buffer per source;
                       sources that are node 0 (external input) get NULL */
                    void** gi = (void**)fiv_malloc(sizeof(void*) * (size_t)nd->num_src);
                    if (!gi) { r = FIV_RET_ERR_MEM; goto fail; }
                    for (int s = 0; s < nd->num_src; s++) {
                        int src = (s == 0) ? nd->input_src : nd->src_list[s - 1];
                        gi[s] = NULL;
                        if (src != 0) {
                            gi[s] = fiv_nn_grad_get(net, grads, src);
                            if (!gi[s]) { fiv_free(gi); r = FIV_RET_ERR_MEM; goto fail; }
                        }
                    }
                    r = o->backward_multi_fn(nd->op, gi, grad_out, nd->inputs, nd->num_src);
                    fiv_free(gi);
                } else {
                    if (!o->backward_fn) { r = FIV_RET_ERR_DATA_UNINITED; goto fail; }
                    void* grad_in = NULL;
                    if (nd->input_src != 0) {
                        grad_in = fiv_nn_grad_get(net, grads, nd->input_src);
                        if (!grad_in) { r = FIV_RET_ERR_MEM; goto fail; }
                    }
                    r = o->backward_fn(nd->op, grad_in, grad_out, nd->input);
                }
                if (r != FIV_RET_OK) goto fail;
            }

            fiv_nn_sgd_update(net, p->learning_rate);
        }
    }

    r = FIV_RET_OK;
fail:
    for (int i = 0; i < net->node_count; i++) {
        if (grads[i]) fiv_release_tensor(&grads[i]);
    }
    fiv_free(grads);
    return r;
}
