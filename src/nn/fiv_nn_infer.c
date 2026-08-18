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

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "fiv_nn.h"
#include "fiv_nn_infer.h"
#include "fiv_nn_topo.h"
#include "fiv_linear_node.h"
#include "fiv_activate_fn.h"
#include "fiv_matrix.h"
#include "fiv_common.h"

void* fiv_create_neural_network()
{
    fiv_nn_network_context* net = (fiv_nn_network_context*)fiv_malloc(sizeof(fiv_nn_network_context));
    if (!net) return NULL;

    net->capacity = 16;
    net->nodes = (fiv_nn_node_context*)fiv_calloc((size_t)net->capacity, sizeof(fiv_nn_node_context));
    if (!net->nodes) { fiv_free(net); return NULL; }
    net->topo_order = (int*)fiv_malloc(sizeof(int) * (size_t)net->capacity);
    if (!net->topo_order) { fiv_free(net->nodes); fiv_free(net); return NULL; }

    net->node_count         = 1;
    net->nodes[0].node_type = FIV_NN_NODE_INPUT;
    net->nodes[0].input_src = -1;
    net->nodes[0].op        = NULL;
    net->nodes[0].input     = NULL;
    net->nodes[0].output    = NULL;   /* aliased to caller input at inference */
    net->output_node = 0;
    net->topo_count  = 0;
    net->topo_valid  = 0;
    net->infer_allocated = 0;
    return net;
}

static void* fiv_nn_make_op(int node_type, void* params)
{
    switch (node_type) {
    case FIV_NN_NODE_LINEAR: return fiv_linear_node_create(params);
    case FIV_NN_NODE_RELU:
    case FIV_NN_NODE_RELU6:
        return fiv_relu_node_create((void*)(intptr_t)node_type);
    default:                 return NULL;
    }
}

fiv_ret fiv_neural_network_add_node(void* nn_context, int node_type, int index_start, int index_end, void* params)
{
    fiv_nn_network_context* net = (fiv_nn_network_context*)nn_context;
    if (!net) return FIV_RET_ERR_PARA;
    if (node_type <= 0 || node_type >= FIV_NN_NODE_TYPE_NUM) return FIV_RET_ERR_PARA;
    if (index_end <= 0) return FIV_RET_ERR_PARA;                 /* node 0 is reserved */
    if (index_end != net->node_count) return FIV_RET_ERR_PARA;   /* nodes are added in order */
    if (index_start < 0 || index_start >= index_end) return FIV_RET_ERR_PARA;

    if (index_end >= net->capacity) {
        int newcap = net->capacity * 2;
        fiv_nn_node_context* nn = (fiv_nn_node_context*)fiv_realloc(net->nodes,
                                        sizeof(fiv_nn_node_context) * (size_t)newcap);
        if (!nn) return FIV_RET_ERR_MEM;
        memset(nn + net->capacity, 0,
               sizeof(fiv_nn_node_context) * (size_t)(newcap - net->capacity));
        net->nodes = nn;
        int* to = (int*)fiv_realloc(net->topo_order, sizeof(int) * (size_t)newcap);
        if (!to) return FIV_RET_ERR_MEM;
        net->topo_order = to;
        net->capacity = newcap;
    }

    fiv_nn_node_context* nd = &net->nodes[index_end];
    if (nd->op) {
        ((fiv_nn_op_base*)nd->op)->release_fn(nd->op);
        nd->op = NULL;
    }
    if (nd->output && nd->owns_output) fiv_release_tensor(&nd->output);
    nd->output = NULL;
    nd->owns_output = 0;

    void* op = fiv_nn_make_op(node_type, params);
    if (!op) return FIV_RET_ERR_PARA;

    nd->node_type = node_type;
    nd->input_src = index_start;
    nd->op        = op;
    nd->input     = NULL;

    net->node_count  = index_end + 1;
    net->output_node = index_end;
    net->topo_valid  = 0;
    net->infer_allocated = 0;
    /* graph changed: drop any cached inference buffers from a prior run */
    for (int i = 1; i < net->node_count; i++) {
        if (net->nodes[i].output && net->nodes[i].owns_output)
            fiv_release_tensor(&net->nodes[i].output);
        net->nodes[i].output = NULL;
        net->nodes[i].owns_output = 0;
    }
    return FIV_RET_OK;
}

/* An op that returns its input pointer from alloc_out_fn is declaring it can
   run in place (e.g. ReLU); the framework then decides per execution mode. */
static int fiv_nn_op_inplace(const void* out, const void* input)
{
    return out == input;
}

/* Training forward: in-place-capable ops still get their own buffer so a
   predecessor's output stays intact for backward (training is NEVER in place).
   Wiring (input = predecessor.output) is done in topo order, interleaved with
   allocation so a predecessor's output exists before its successors read it. */
static fiv_ret fiv_nn_pass1_forward(fiv_nn_network_context* net)
{
    for (int k = 0; k < net->topo_count; k++) {
        int i = net->topo_order[k];
        if (i == 0) continue;

        fiv_nn_node_context* nd = &net->nodes[i];
        void* src = net->nodes[nd->input_src].output;
        if (!src) return FIV_RET_ERR_DATA_UNINITED;
        nd->input = src;

        fiv_nn_op_base* o = (fiv_nn_op_base*)nd->op;
        if (!o || !o->alloc_out_fn) return FIV_RET_ERR_DATA_UNINITED;
        fiv_ret r;
        void* out = o->alloc_out_fn(nd->op, nd->input, nd->output, &r);
        if (r != FIV_RET_OK || !out) return r;
        if (fiv_nn_op_inplace(out, nd->input)) {
            void* sep = nd->output;
            if (sep == NULL || sep == nd->input) {
                sep = fiv_create_tensor_like_tensor(nd->input);
                if (!sep) { r = FIV_RET_ERR_MEM; return r; }
                if (nd->output && nd->output != nd->input) fiv_release_tensor(&nd->output);
                nd->output = sep;
            }
            nd->owns_output = 1;
        } else {
            nd->output = out;
            nd->owns_output = 1;
        }
    }
    return FIV_RET_OK;
}

/* Inference: in-place-capable ops reuse the input buffer (zero extra
   allocation); the rest keep the buffer alloc_out_fn produced. Wiring is done
   in topo order, same as the training pass. */
static fiv_ret fiv_nn_pass1_inference(fiv_nn_network_context* net)
{
    for (int k = 0; k < net->topo_count; k++) {
        int i = net->topo_order[k];
        if (i == 0) continue;

        fiv_nn_node_context* nd = &net->nodes[i];
        void* src = net->nodes[nd->input_src].output;
        if (!src) return FIV_RET_ERR_DATA_UNINITED;
        nd->input = src;

        fiv_nn_op_base* o = (fiv_nn_op_base*)nd->op;
        if (!o || !o->alloc_out_fn) return FIV_RET_ERR_DATA_UNINITED;
        fiv_ret r;
        void* out = o->alloc_out_fn(nd->op, nd->input, nd->output, &r);
        if (r != FIV_RET_OK || !out) return r;
        if (fiv_nn_op_inplace(out, nd->input)) {
            if (nd->output && nd->output != nd->input) fiv_release_tensor(&nd->output);
            nd->output = nd->input;   /* in place */
            nd->owns_output = 0;
        } else {
            nd->output = out;
            nd->owns_output = 1;
        }
    }
    return FIV_RET_OK;
}

static fiv_ret fiv_nn_pass2_forward(fiv_nn_network_context* net)
{
    for (int k = 0; k < net->topo_count; k++) {
        int i = net->topo_order[k];
        if (i == 0) continue;

        fiv_nn_node_context* nd = &net->nodes[i];
        fiv_nn_op_base* o = (fiv_nn_op_base*)nd->op;
        if (!o || !o->forward_fn) return FIV_RET_ERR_DATA_UNINITED;
        fiv_ret r = o->forward_fn(nd->op, nd->output, nd->input);
        if (r != FIV_RET_OK) return r;
    }
    return FIV_RET_OK;
}

static fiv_ret fiv_nn_pass2_inference(fiv_nn_network_context* net)
{
    for (int k = 0; k < net->topo_count; k++) {
        int i = net->topo_order[k];
        if (i == 0) continue;

        fiv_nn_node_context* nd = &net->nodes[i];
        fiv_nn_op_base* o = (fiv_nn_op_base*)nd->op;
        if (!o || !o->inference_fn) return FIV_RET_ERR_DATA_UNINITED;
        fiv_ret r = o->inference_fn(nd->op, nd->output, nd->input);
        if (r != FIV_RET_OK) return r;
    }
    return FIV_RET_OK;
}

fiv_ret fiv_nn_run_forward(fiv_nn_network_context* net, void* input, void** final_out)
{
    if (!net->topo_valid) {
        fiv_ret r = fiv_nn_topo_sort(net);
        if (r != FIV_RET_OK) return r;
    }
    net->nodes[0].output = input;   /* node 0 aliases the caller input */
    fiv_ret r = fiv_nn_pass1_forward(net);
    if (r != FIV_RET_OK) return r;
    r = fiv_nn_pass2_forward(net);
    if (r != FIV_RET_OK) return r;
    *final_out = net->nodes[net->output_node].output;
    return FIV_RET_OK;
}

/* After pass1_inference has allocated each node's output once, a subsequent
   inference call only needs to re-point the nodes fed directly by node 0 at the
   new input tensor; every other node's input/output buffers are cached and fixed. */
static void fiv_nn_inference_rewire(fiv_nn_network_context* net)
{
    for (int k = 0; k < net->topo_count; k++) {
        int i = net->topo_order[k];
        if (i == 0) continue;
        fiv_nn_node_context* nd = &net->nodes[i];
        if (nd->input_src == 0)
            nd->input = net->nodes[0].output;
    }
}

fiv_ret fiv_nn_run_inference(fiv_nn_network_context* net, void* input, void** final_out)
{
    if (!net->topo_valid) {
        fiv_ret r = fiv_nn_topo_sort(net);
        if (r != FIV_RET_OK) return r;
    }
    net->nodes[0].output = input;   /* node 0 aliases the caller input */
    fiv_ret r;
    if (!net->infer_allocated) {
        r = fiv_nn_pass1_inference(net);   /* allocate node outputs once */
        if (r != FIV_RET_OK) return r;
        net->infer_allocated = 1;
    } else {
        fiv_nn_inference_rewire(net);      /* only re-point node-0 successors at input */
    }
    r = fiv_nn_pass2_inference(net);
    if (r != FIV_RET_OK) return r;
    *final_out = net->nodes[net->output_node].output;
    return FIV_RET_OK;
}

fiv_ret fiv_neural_network_inference(void* output, void* input, void* nn_context)
{
    fiv_nn_network_context* net = (fiv_nn_network_context*)nn_context;
    if (!net || !input || !output) return FIV_RET_ERR_PARA;

    /* common leading header is shared by all tensor dims */
    fiv_mat* in = (fiv_mat*)input;
    if (in->id < FIV_ID_TENSOR1D || in->id > FIV_ID_TENSOR5D) return FIV_RET_ERR_PARA;
    if (in->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (in->data_continue == 0) return FIV_RET_ERR_PARA;

    void* final = NULL;
    fiv_ret r = fiv_nn_run_inference(net, input, &final);
    if (r != FIV_RET_OK) return r;
    if (!final) return FIV_RET_ERR_DATA_UNINITED;

    fiv_mat* out_hdr = (fiv_mat*)output;
    fiv_mat* fin_hdr = (fiv_mat*)final;
    if (out_hdr->id != fin_hdr->id) return FIV_RET_ERR_PARA;
    if (out_hdr->dtype != FIV_32F1 || out_hdr->data_continue == 0) return FIV_RET_ERR_PARA;
    if (fin_hdr->dtype != FIV_32F1 || fin_hdr->data_continue == 0) return FIV_RET_ERR_PARA;
    if (out_hdr->total_bytes != fin_hdr->total_bytes) return FIV_RET_ERR_PARA;
    memcpy(out_hdr->data.ptr, fin_hdr->data.ptr, fin_hdr->total_bytes);
    return FIV_RET_OK;
}

fiv_ret fiv_release_neural_network(void** nn_context)
{
    if (!nn_context || !*nn_context) return FIV_RET_ERR_PARA;
    fiv_nn_network_context* net = (fiv_nn_network_context*)*nn_context;

    for (int i = 1; i < net->node_count; i++) {
        fiv_nn_node_context* nd = &net->nodes[i];
        if (nd->op) {
            fiv_nn_op_base* o = (fiv_nn_op_base*)nd->op;
            if (o->release_fn) o->release_fn(nd->op);
        }
        if (nd->output && nd->owns_output) fiv_release_tensor(&nd->output);
    }
    fiv_free(net->nodes);
    fiv_free(net->topo_order);
    fiv_free(net);
    *nn_context = NULL;
    return FIV_RET_OK;
}

/* Binary model format v1 (native endianness):
   int32 magic "FIVN", int32 version, int32 node_count, then per node
   (i = 1 .. node_count-1): int32 node_type, int32 index_start,
   int32 in_features, int32 out_features; for LINEAR the float weight
   [out*in] and float bias [out] follow. */
#define FIV_NN_MODEL_MAGIC  0x4649564Eu
#define FIV_NN_MODEL_VERSION 1

fiv_ret fiv_neural_network_save_model(char* model_name, void* nn_context)
{
    fiv_nn_network_context* net = (fiv_nn_network_context*)nn_context;
    if (!net) return FIV_RET_ERR_PARA;

    FILE* fp = fopen(model_name, "wb");
    if (!fp) return FIV_RET_ERR_OPEN_FILE;

    int32_t magic = (int32_t)FIV_NN_MODEL_MAGIC;
    int32_t version = FIV_NN_MODEL_VERSION;
    int32_t node_count = (int32_t)net->node_count;
    fwrite(&magic, 4, 1, fp);
    fwrite(&version, 4, 1, fp);
    fwrite(&node_count, 4, 1, fp);

    for (int i = 1; i < net->node_count; i++) {
        fiv_nn_node_context* nd = &net->nodes[i];
        int32_t type = (int32_t)nd->node_type;
        int32_t src  = (int32_t)nd->input_src;
        int32_t in_f = -1, out_f = -1;
        if (nd->node_type == FIV_NN_NODE_LINEAR) {
            fiv_linear_node* ln = (fiv_linear_node*)nd->op;
            in_f  = (int32_t)ln->in_features;
            out_f = (int32_t)ln->out_features;
        }
        fwrite(&type, 4, 1, fp);
        fwrite(&src, 4, 1, fp);
        fwrite(&in_f, 4, 1, fp);
        fwrite(&out_f, 4, 1, fp);
        if (nd->node_type == FIV_NN_NODE_LINEAR) {
            fiv_linear_node* ln = (fiv_linear_node*)nd->op;
            fwrite(ln->weight->data.fl, sizeof(float), (size_t)(out_f * in_f), fp);
            fwrite(ln->bias->data.fl,   sizeof(float), (size_t)out_f, fp);
        }
    }
    fclose(fp);
    return FIV_RET_OK;
}

void* fiv_create_neural_network_from_model(char* model_name)
{
    FILE* fp = fopen(model_name, "rb");
    if (!fp) return NULL;

    int32_t magic, version, node_count;
    if (fread(&magic, 4, 1, fp) != 1 || magic != (int32_t)FIV_NN_MODEL_MAGIC ||
        fread(&version, 4, 1, fp) != 1 || version != FIV_NN_MODEL_VERSION ||
        fread(&node_count, 4, 1, fp) != 1 || node_count < 1) {
        fclose(fp);
        return NULL;
    }

    fiv_nn_network_context* net = (fiv_nn_network_context*)fiv_create_neural_network();
    if (!net) { fclose(fp); return NULL; }

    for (int i = 1; i < node_count; i++) {
        int32_t type, src, in_f, out_f;
        if (fread(&type, 4, 1, fp) != 1 || fread(&src, 4, 1, fp) != 1 ||
            fread(&in_f, 4, 1, fp) != 1 || fread(&out_f, 4, 1, fp) != 1) {
            goto fail;
        }

        if (type == FIV_NN_NODE_LINEAR) {
            fiv_linear_node_params p = { in_f, out_f };
            fiv_ret r = fiv_neural_network_add_node(net, FIV_NN_NODE_LINEAR, (int)src, i, &p);
            if (r != FIV_RET_OK) goto fail;
            fiv_linear_node* ln = (fiv_linear_node*)net->nodes[i].op;
            size_t nw = (size_t)(out_f * in_f), nb = (size_t)out_f;
            if (fread(ln->weight->data.fl, sizeof(float), nw, fp) != nw ||
                fread(ln->bias->data.fl,   sizeof(float), nb, fp) != nb) {
                goto fail;
            }
        } else if (type == FIV_NN_NODE_RELU || type == FIV_NN_NODE_RELU6) {
            if (fiv_neural_network_add_node(net, (int)type, (int)src, i, NULL) != FIV_RET_OK) {
                goto fail;
            }
        } else {
            goto fail;
        }
    }
    fclose(fp);
    return net;

fail:
    fclose(fp);
    fiv_release_neural_network((void**)&net);
    return NULL;
}
