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

/* YOLO26 (yolo26.yaml, detection, end2end, reg_max=1) full-graph builder on
 * the FastIV NN engine. Every conv is driven from the BN-fold table produced
 * by app/yolo26/test/gen_ref.py (fold.json / fold_w.f32 / fold_b.f32, weights
 * already BN-folded, layout [c_out,c_in,ky,kx] = PyTorch, no permutation).
 * Attention composite-node weights come from the attn0_/attn1_ dumps. */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "fiv_common.h"
#include "fiv_nn.h"
#include "fiv_nn_infer.h"
#include "fiv_maxpool_node.h"
#include "fiv_slice_node.h"
#include "fiv_attention_node.h"
#include "fiv_yolo26.h"

/* Per-build state threaded through every block builder: the engine network,
 * the fold table and its flat weight payloads, the folded attention weights,
 * the next free node id and the per-layer output node ids. */
typedef struct {
    void*                    net;         /* engine network */
    const yolo26_fold_entry* fold;        /* fold table */
    int                      fold_count;
    const ivf32*             fold_w;      /* concatenated folded conv weights */
    const ivf32*             fold_b;      /* concatenated folded conv biases */
    const ivf32*             attn_w[2][6]; /* attn<id>: qkv_w, qkv_b, pe_w, pe_b, proj_w, proj_b */
    int                      next_node;   /* next node id (0 = implicit input) */
    int                      layer_out[24];
} fiv_y26_builder;

/* Entry whose full path equals model.<layer>.<rel_path>. */
static const yolo26_fold_entry* fiv_y26_find_entry(const fiv_y26_builder* builder,
                                                   int layer, const char* rel_path)
{
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "model.%d.%s", layer, rel_path);
    for (int idx = 0; idx < builder->fold_count; idx++) {
        const yolo26_fold_entry* entry = &builder->fold[idx];
        if (entry->layer == layer && strcmp(entry->path, pattern) == 0)
            return entry;
    }
    return NULL;
}

/* Whether any fold entry of this layer sits under model.<layer>.<prefix>. */
static int fiv_y26_has_entries(const fiv_y26_builder* builder, int layer,
                               const char* prefix)
{
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "model.%d.%s", layer, prefix);
    size_t prefix_len = strlen(pattern);
    for (int idx = 0; idx < builder->fold_count; idx++) {
        const yolo26_fold_entry* entry = &builder->fold[idx];
        if (entry->layer == layer && strncmp(entry->path, pattern, prefix_len) == 0)
            return 1;
    }
    return 0;
}

static int fiv_y26_add_node(fiv_y26_builder* builder, int node_type,
                            int src_node, void* params)
{
    int node_id = builder->next_node++;
    return fiv_neural_network_add_node(builder->net, node_type, src_node,
                                       node_id, params) == FIV_RET_OK ? node_id : -1;
}

static int fiv_y26_add_multi(fiv_y26_builder* builder, int node_type,
                             const int* src_nodes, int src_count, void* params)
{
    int node_id = builder->next_node++;
    return fiv_neural_network_add_node_multi(builder->net, node_type, (int*)src_nodes,
                                             src_count, node_id, params) == FIV_RET_OK
               ? node_id : -1;
}

/* Element-wise ADD of two nodes. */
static int fiv_y26_add_pair(fiv_y26_builder* builder, int left_node, int right_node)
{
    int fan_in[2] = { left_node, right_node };
    return fiv_y26_add_multi(builder, FIV_NN_NODE_ADD, fan_in, 2, NULL);
}

static int fiv_y26_add_slice(fiv_y26_builder* builder, int src_node,
                             int ch_start, int ch_end)
{
    fiv_slice_node_params slice_params;
    memset(&slice_params, 0, sizeof(slice_params));
    slice_params.axis  = 1;         /* 4D NCHW: axis 1 = channels */
    slice_params.start = ch_start;
    slice_params.end   = ch_end;
    return fiv_y26_add_node(builder, FIV_NN_NODE_SLICE, src_node, &slice_params);
}

static int fiv_y26_add_concat(fiv_y26_builder* builder, const int* src_nodes,
                              int src_count, int out_channels)
{
    fiv_concat_node_params concat_params;
    memset(&concat_params, 0, sizeof(concat_params));
    concat_params.axis = 1;
    concat_params.output_channels = out_channels;
    return fiv_y26_add_multi(builder, FIV_NN_NODE_CONCAT, src_nodes,
                             src_count, &concat_params);
}

/* One folded conv node (+ SiLU node when the Conv wrapper carries one). */
static int fiv_y26_add_conv(fiv_y26_builder* builder, int src_node,
                            const yolo26_fold_entry* entry)
{
    int conv_method;
    int node_type;
    if (entry->groups == 1 && entry->kernel == 1) {
        conv_method = 2;
        node_type   = FIV_NN_NODE_CONV2D_POINTWISE;
    } else if (entry->groups == entry->input_channels) {
        conv_method = 1;
        node_type   = FIV_NN_NODE_CONV2D_DEPTHWISE;
    } else {
        conv_method = 0;
        node_type   = FIV_NN_NODE_CONV2D_STD;
    }

    fiv_conv2d_params conv_params;
    memset(&conv_params, 0, sizeof(conv_params));
    conv_params.conv2d_method   = conv_method;
    conv_params.kernel_size_x   = entry->kernel;
    conv_params.kernel_size_y   = entry->kernel;
    conv_params.stride          = entry->stride;
    conv_params.padding_method  = 0;
    conv_params.input_channels  = entry->input_channels;
    conv_params.output_channels = entry->output_channels;
    conv_params.bias            = 1;            /* folded bias always present */
    conv_params.pad_top = conv_params.pad_bottom = entry->padding;
    conv_params.pad_left = conv_params.pad_right = entry->padding;

    int node_id = builder->next_node++;
    if (fiv_neural_network_add_node(builder->net, node_type, src_node,
                                    node_id, &conv_params) != FIV_RET_OK)
        return -1;
    fiv_neural_network_set_node_weight(builder->net, node_id,
                                       builder->fold_w + entry->w_off);
    fiv_neural_network_set_node_bias(builder->net, node_id,
                                     builder->fold_b + entry->b_off);
    if (entry->act == 1) {
        int silu_node_id = builder->next_node++;
        if (fiv_neural_network_add_node(builder->net, FIV_NN_NODE_SILU,
                                        node_id, silu_node_id, NULL) != FIV_RET_OK)
            return -1;
        return silu_node_id;
    }
    return node_id;
}

/* Bottleneck (module prefix like "m.0." or "m.0.0."): cv1 -> cv2 -> +src. */
static int fiv_y26_add_bottleneck(fiv_y26_builder* builder, int layer,
                                  const char* module_prefix, int src_node)
{
    char rel_cv1[160], rel_cv2[160];
    snprintf(rel_cv1, sizeof(rel_cv1), "%scv1.conv", module_prefix);
    snprintf(rel_cv2, sizeof(rel_cv2), "%scv2.conv", module_prefix);
    const yolo26_fold_entry* entry_cv1 = fiv_y26_find_entry(builder, layer, rel_cv1);
    const yolo26_fold_entry* entry_cv2 = fiv_y26_find_entry(builder, layer, rel_cv2);
    if (!entry_cv1 || !entry_cv2) return -1;

    int conv_a_node = fiv_y26_add_conv(builder, src_node, entry_cv1);
    if (conv_a_node < 0) return -1;
    int conv_b_node = fiv_y26_add_conv(builder, conv_a_node, entry_cv2);
    if (conv_b_node < 0) return -1;
    return fiv_y26_add_pair(builder, conv_b_node, src_node);
}

/* C3k composite block at module_prefix (cv1/cv2 parallel branches, inner
 * m.<j>. bottleneck chain, then cv3). */
static int fiv_y26_add_c3k_block(fiv_y26_builder* builder, int layer,
                                 const char* module_prefix, int src_node)
{
    char rel_path[160];

    snprintf(rel_path, sizeof(rel_path), "%scv1.conv", module_prefix);
    const yolo26_fold_entry* entry_cv1 = fiv_y26_find_entry(builder, layer, rel_path);
    snprintf(rel_path, sizeof(rel_path), "%scv2.conv", module_prefix);
    const yolo26_fold_entry* entry_cv2 = fiv_y26_find_entry(builder, layer, rel_path);
    snprintf(rel_path, sizeof(rel_path), "%scv3.conv", module_prefix);
    const yolo26_fold_entry* entry_cv3 = fiv_y26_find_entry(builder, layer, rel_path);
    if (!entry_cv1 || !entry_cv2 || !entry_cv3) return -1;

    int branch_a_node = fiv_y26_add_conv(builder, src_node, entry_cv1);
    if (branch_a_node < 0) return -1;
    for (int j = 0; j < 8; j++) {
        char inner_prefix[160];
        snprintf(inner_prefix, sizeof(inner_prefix), "%sm.%d.", module_prefix, j);
        if (!fiv_y26_has_entries(builder, layer, inner_prefix)) break;
        branch_a_node = fiv_y26_add_bottleneck(builder, layer, inner_prefix, branch_a_node);
        if (branch_a_node < 0) return -1;
    }
    int branch_b_node = fiv_y26_add_conv(builder, src_node, entry_cv2);
    if (branch_b_node < 0) return -1;

    int branch_srcs[2] = { branch_a_node, branch_b_node };
    int cat_node = fiv_y26_add_concat(builder, branch_srcs, 2, entry_cv3->input_channels);
    if (cat_node < 0) return -1;
    return fiv_y26_add_conv(builder, cat_node, entry_cv3);
}

/* Attention composite node feeding from src_node; weights loaded from
 * builder->attn_w[attn_index]. */
static int fiv_y26_add_attention(fiv_y26_builder* builder, int src_node,
                                 int attn_index, int dim)
{
    fiv_attention_node_params attn_params;
    memset(&attn_params, 0, sizeof(attn_params));
    attn_params.dim       = dim;
    attn_params.num_heads = 2;
    attn_params.head_dim  = dim / 2;
    attn_params.key_dim   = attn_params.head_dim / 2;
    attn_params.scale     = powf((ivf32)attn_params.key_dim, -0.5f);
    attn_params.pe_groups = dim;

    int node_id = builder->next_node++;
    if (fiv_neural_network_add_node(builder->net, FIV_NN_NODE_ATTENTION,
                                    src_node, node_id, &attn_params) != FIV_RET_OK)
        return -1;

    fiv_nn_node_context* node_ctx = fiv_neural_network_get_node(builder->net, node_id);
    if (!node_ctx || !node_ctx->op) return -1;

    const ivf32** weights = builder->attn_w[attn_index];
    for (int part = 0; part < 6; part++)
        if (!weights[part]) return -1;
    if (fiv_attention_node_set_weights(node_ctx->op,
                                       weights[0], weights[1], weights[2],
                                       weights[3], weights[4], weights[5]) != FIV_RET_OK)
        return -1;
    return node_id;
}

/* PSABlock: out = x + ffn(x + attn(x)); module_prefix locates the ffn convs. */
static int fiv_y26_add_psa_block(fiv_y26_builder* builder, int layer,
                                 const char* module_prefix, int src_node,
                                 int attn_index, int dim)
{
    char rel_path[160];

    snprintf(rel_path, sizeof(rel_path), "%sffn.0.conv", module_prefix);
    const yolo26_fold_entry* entry_ffn0 = fiv_y26_find_entry(builder, layer, rel_path);
    snprintf(rel_path, sizeof(rel_path), "%sffn.1.conv", module_prefix);
    const yolo26_fold_entry* entry_ffn1 = fiv_y26_find_entry(builder, layer, rel_path);
    if (!entry_ffn0 || !entry_ffn1) return -1;

    int attn_out_node = fiv_y26_add_attention(builder, src_node, attn_index, dim);
    if (attn_out_node < 0) return -1;
    int residual_node = fiv_y26_add_pair(builder, attn_out_node, src_node);
    if (residual_node < 0) return -1;
    int ffn0_node = fiv_y26_add_conv(builder, residual_node, entry_ffn0);
    if (ffn0_node < 0) return -1;
    int ffn1_node = fiv_y26_add_conv(builder, ffn0_node, entry_ffn1);
    if (ffn1_node < 0) return -1;
    return fiv_y26_add_pair(builder, ffn1_node, residual_node);
}

/* C3k2 (C2f semantics): cv1 -> slice a|b -> module chain on b -> cat -> cv2.
 * The attn variant (layer 22) is a Sequential(Bottleneck, PSABlock) handled
 * explicitly here; all other layers carry plain Bottleneck or C3k modules. */
static int fiv_y26_add_c3k2(fiv_y26_builder* builder, int layer, int src_node)
{
    const yolo26_fold_entry* entry_cv1 = fiv_y26_find_entry(builder, layer, "cv1.conv");
    const yolo26_fold_entry* entry_cv2 = fiv_y26_find_entry(builder, layer, "cv2.conv");
    if (!entry_cv1 || !entry_cv2) return -1;

    int half_ch = entry_cv1->output_channels / 2;
    int conv1_node = fiv_y26_add_conv(builder, src_node, entry_cv1);
    if (conv1_node < 0) return -1;
    int slice_a_node = fiv_y26_add_slice(builder, conv1_node, 0, half_ch);
    int slice_b_node = fiv_y26_add_slice(builder, conv1_node, half_ch, entry_cv1->output_channels);
    if (slice_a_node < 0 || slice_b_node < 0) return -1;

    int cur_node = slice_b_node;
    int module_chain[16];
    int chain_count = 1;                       /* cat keeps b itself */
    module_chain[0] = slice_b_node;

    for (int k = 0; k < 8; k++) {
        char module_prefix[64];
        snprintf(module_prefix, sizeof(module_prefix), "m.%d.", k);
        if (!fiv_y26_has_entries(builder, layer, module_prefix)) break;
        if (layer == 22 && k == 0) {
            /* attn variant: m.0 = Sequential(Bottleneck m.0.0, PSABlock m.0.1). */
            int inner_node = cur_node;
            for (int j = 0; j < 8; j++) {
                char child_prefix[96], probe[96];
                snprintf(child_prefix, sizeof(child_prefix), "m.%d.%d.", k, j);
                if (!fiv_y26_has_entries(builder, layer, child_prefix)) break;
                snprintf(probe, sizeof(probe), "%scv1", child_prefix);
                if (fiv_y26_has_entries(builder, layer, probe))
                    inner_node = fiv_y26_add_bottleneck(builder, layer, child_prefix, inner_node);
                else
                    inner_node = fiv_y26_add_psa_block(builder, layer, child_prefix,
                                                       inner_node, 1, half_ch);
                if (inner_node < 0) return -1;
            }
            cur_node = inner_node;
        } else {
            /* plain Bottleneck chain or single C3k module */
            char probe_cv3[96];
            snprintf(probe_cv3, sizeof(probe_cv3), "%scv3", module_prefix);
            if (fiv_y26_has_entries(builder, layer, probe_cv3)) {
                cur_node = fiv_y26_add_c3k_block(builder, layer, module_prefix, cur_node);
            } else {
                cur_node = fiv_y26_add_bottleneck(builder, layer, module_prefix, cur_node);
            }
        }
        if (cur_node < 0) return -1;
        module_chain[chain_count++] = cur_node;
    }

    int concat_srcs[16];
    concat_srcs[0] = slice_a_node;
    for (int i = 0; i < chain_count; i++) concat_srcs[1 + i] = module_chain[i];
    int cat_node = fiv_y26_add_concat(builder, concat_srcs, 1 + chain_count, entry_cv2->input_channels);
    if (cat_node < 0) return -1;
    return fiv_y26_add_conv(builder, cat_node, entry_cv2);
}

/* SPPF: cv1(1x1, no act) -> 3x maxpool(k5 s1 p2) chain -> cat 4 -> cv2 -> +x. */
static int fiv_y26_add_sppf(fiv_y26_builder* builder, int layer, int src_node)
{
    const yolo26_fold_entry* entry_cv1 = fiv_y26_find_entry(builder, layer, "cv1.conv");
    const yolo26_fold_entry* entry_cv2 = fiv_y26_find_entry(builder, layer, "cv2.conv");
    if (!entry_cv1 || !entry_cv2) return -1;

    int conv1_node = fiv_y26_add_conv(builder, src_node, entry_cv1);
    if (conv1_node < 0) return -1;

    fiv_maxpool_node_params pool_params;
    memset(&pool_params, 0, sizeof(pool_params));
    pool_params.kernel_size_x = 5;
    pool_params.kernel_size_y = 5;
    pool_params.stride        = 1;
    pool_params.pad_top       = 2;
    pool_params.pad_bottom    = 2;
    pool_params.pad_left      = 2;
    pool_params.pad_right     = 2;

    int pool_nodes[3];
    int prev_node = conv1_node;
    for (int i = 0; i < 3; i++) {
        int pool_node = builder->next_node++;
        if (fiv_neural_network_add_node(builder->net, FIV_NN_NODE_MAXPOOL,
                                        prev_node, pool_node, &pool_params) != FIV_RET_OK)
            return -1;
        pool_nodes[i] = pool_node;
        prev_node     = pool_node;
    }

    int pool_srcs[4] = { conv1_node, pool_nodes[0], pool_nodes[1], pool_nodes[2] };
    int cat_node = fiv_y26_add_concat(builder, pool_srcs, 4, entry_cv2->input_channels);
    if (cat_node < 0) return -1;

    int conv2_node = fiv_y26_add_conv(builder, cat_node, entry_cv2);
    if (conv2_node < 0) return -1;
    if (entry_cv1->input_channels == entry_cv2->output_channels) {   /* shortcut residual */
        int add_node = fiv_y26_add_pair(builder, conv2_node, src_node);
        if (add_node < 0) return -1;
        return add_node;
    }
    return conv2_node;
}

/* C2PSA: cv1 -> slice a|b -> b = m.0 PSABlock(attn0) -> cat([a, psa]) -> cv2. */
static int fiv_y26_add_c2psa(fiv_y26_builder* builder, int layer, int src_node)
{
    const yolo26_fold_entry* entry_cv1 = fiv_y26_find_entry(builder, layer, "cv1.conv");
    const yolo26_fold_entry* entry_cv2 = fiv_y26_find_entry(builder, layer, "cv2.conv");
    if (!entry_cv1 || !entry_cv2) return -1;

    int half_ch = entry_cv1->output_channels / 2;
    int conv1_node = fiv_y26_add_conv(builder, src_node, entry_cv1);
    if (conv1_node < 0) return -1;
    int slice_a_node = fiv_y26_add_slice(builder, conv1_node, 0, half_ch);
    int slice_b_node = fiv_y26_add_slice(builder, conv1_node, half_ch, entry_cv1->output_channels);
    if (slice_a_node < 0 || slice_b_node < 0) return -1;

    int psa_node = fiv_y26_add_psa_block(builder, layer, "m.0.", slice_b_node, 0, half_ch);
    if (psa_node < 0) return -1;

    int cat_srcs[2] = { slice_a_node, psa_node };
    int cat_node = fiv_y26_add_concat(builder, cat_srcs, 2, entry_cv2->input_channels);
    if (cat_node < 0) return -1;
    return fiv_y26_add_conv(builder, cat_node, entry_cv2);
}

/* Top-level plain Conv layer: the single folded conv at model.<layer>.conv. */
static int fiv_y26_add_top_conv(fiv_y26_builder* builder, int layer, int src_node)
{
    const yolo26_fold_entry* entry = fiv_y26_find_entry(builder, layer, "conv");
    if (!entry) return -1;
    return fiv_y26_add_conv(builder, src_node, entry);
}

/* Detect one2one head convs on a feature node, one chain per level. */
static int fiv_y26_add_head_chain(fiv_y26_builder* builder, int feature_node,
                                  const char* branch_name, int level)
{
    /* Match "one2one_cv2.<level>" as a bare substring: the chain tail is a
     * plain nn.Conv2d whose fold path has NO ".conv" suffix, so a key ending
     * in "." would silently skip it. The prefix is unique per level. */
    char key[48];
    snprintf(key, sizeof(key), "%s.%d", branch_name, level);

    int cur_node   = feature_node;
    int conv_count = 0;
    for (int idx = 0; idx < builder->fold_count; idx++) {
        const yolo26_fold_entry* entry = &builder->fold[idx];
        if (entry->layer != 23) continue;
        if (strstr(entry->path, key) == NULL) continue;
        cur_node = fiv_y26_add_conv(builder, cur_node, entry);
        if (cur_node < 0) return -1;
        conv_count++;
    }
    return conv_count > 0 ? cur_node : -1;
}

/* Pose26 kpt branch: one2one_cv4.<level> (two 3x3 -> 85ch convs, SiLU) then
 * one2one_cv4_kpts.<level> (1x1 85->51, no act). The kpts 1x1 conv consumes the
 * cv4 output (NOT the feature map), so cv4 must be chained first. */
static int fiv_y26_add_pose_kpt_chain(fiv_y26_builder* builder, int feature_node,
                                      int level)
{
    char pattern[96];
    int  cur_node = feature_node;
    int  conv_count = 0;

    snprintf(pattern, sizeof(pattern), "one2one_cv4.%d.", level);
    for (int idx = 0; idx < builder->fold_count; idx++) {
        const yolo26_fold_entry* entry = &builder->fold[idx];
        if (entry->layer != 23) continue;
        if (strstr(entry->path, pattern) == NULL) continue;
        if (strstr(entry->path, "one2one_cv4_kpts") != NULL) continue; /* kpts/sigma excluded */
        if (strstr(entry->path, "cv4_sigma") != NULL) continue;
        cur_node = fiv_y26_add_conv(builder, cur_node, entry);
        if (cur_node < 0) return -1;
        conv_count++;
    }
    if (conv_count == 0) return -1;

    /* The kpts head is a bare nn.Conv2d (fold path has no ".conv" suffix), so
     * the key must be a bare substring without a trailing dot. */
    snprintf(pattern, sizeof(pattern), "one2one_cv4_kpts.%d", level);
    conv_count = 0;
    for (int idx = 0; idx < builder->fold_count; idx++) {
        const yolo26_fold_entry* entry = &builder->fold[idx];
        if (entry->layer != 23) continue;
        if (strstr(entry->path, pattern) == NULL) continue;
        if (strstr(entry->path, "cv4_sigma") != NULL) continue;
        cur_node = fiv_y26_add_conv(builder, cur_node, entry);
        if (cur_node < 0) return -1;
        conv_count++;
    }
    return conv_count > 0 ? cur_node : -1;
}

fiv_yolo26_graph* fiv_yolo26_build(const yolo26_fold_entry* fold, int fold_count,
                                   const ivf32* fold_w, const ivf32* fold_b,
                                   const ivf32* attn[2][6])
{
    if (!fold || !fold_w || !fold_b || !attn) return NULL;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 6; j++)
            if (!attn[i][j]) return NULL;

    fiv_yolo26_graph* graph = (fiv_yolo26_graph*)fiv_calloc(1, sizeof(fiv_yolo26_graph));
    if (!graph) return NULL;

    void* net = fiv_create_neural_network();
    if (!net) {
        fiv_free(graph);
        return NULL;
    }

    fiv_y26_builder builder;
    memset(&builder, 0, sizeof(builder));
    builder.net        = net;
    builder.fold       = fold;
    builder.fold_count = fold_count;
    builder.fold_w     = fold_w;
    builder.fold_b     = fold_b;
    memcpy(builder.attn_w, attn, sizeof(builder.attn_w));
    builder.next_node = 1;

    for (int i = 0; i < 24; i++) builder.layer_out[i] = -1;
    for (int i = 0; i < 3; i++) {
        graph->mask_node[i] = -1;
        graph->kpt_node[i]  = -1;
    }

    /* ---- backbone + neck: layers 0..22 ---- */
    for (int layer = 0; layer < 23; layer++) {
        int out_node = -1;

        if (layer == 12 || layer == 15 || layer == 18 || layer == 21) {
            /* Concat: fan-in from two earlier layer outputs, consumed by the
             * following C3k2 (whose cv1 input channels fix the concat size). */
            int src_a_layer = (layer == 12) ? 11 : (layer == 15) ? 14 : (layer == 18) ? 17 : 20;
            int src_b_layer = (layer == 12) ? 6  : (layer == 15) ? 4  : (layer == 18) ? 13 : 10;
            if (builder.layer_out[src_a_layer] < 0 || builder.layer_out[src_b_layer] < 0)
                goto fail;
            int fan_in[2] = { builder.layer_out[src_a_layer], builder.layer_out[src_b_layer] };
            const yolo26_fold_entry* next_entry = fiv_y26_find_entry(&builder, layer + 1, "cv1.conv");
            if (!next_entry) goto fail;
            out_node = fiv_y26_add_concat(&builder, fan_in, 2, next_entry->input_channels);
            if (out_node < 0) goto fail;
            builder.layer_out[layer] = out_node;
            continue;
        }

        if (layer != 0 && builder.layer_out[layer - 1] < 0)
            goto fail;
        int src_node = (layer == 0) ? 0 : builder.layer_out[layer - 1];

        switch (layer) {
        case 0: case 1: case 3: case 5: case 7: case 17: case 20:
            out_node = fiv_y26_add_top_conv(&builder, layer, src_node);
            break;
        case 2: case 4: case 6: case 8: case 13: case 16: case 19: case 22:
            out_node = fiv_y26_add_c3k2(&builder, layer, src_node);
            break;
        case 9:
            out_node = fiv_y26_add_sppf(&builder, layer, src_node);
            break;
        case 10:
            out_node = fiv_y26_add_c2psa(&builder, layer, src_node);
            break;
        case 11: case 14:
            out_node = fiv_y26_add_node(&builder, FIV_NN_NODE_UPSAMPLE2X, src_node, NULL);
            break;
        default:
            goto fail;
        }
        if (out_node < 0) goto fail;
        builder.layer_out[layer] = out_node;
    }

    /* ---- Detect one2one heads on features [L16, L19, L22] ---- */
    {
        int feature_nodes[3] = { builder.layer_out[16], builder.layer_out[19], builder.layer_out[22] };
        for (int level = 0; level < 3; level++) {
            int box_node = fiv_y26_add_head_chain(&builder, feature_nodes[level],
                                                  "one2one_cv2", level);
            int cls_node = fiv_y26_add_head_chain(&builder, feature_nodes[level],
                                                  "one2one_cv3", level);
            if (box_node < 0 || cls_node < 0) goto fail;
            graph->head_node[level]     = box_node;
            graph->head_node[3 + level] = cls_node;
        }

        /* Segment26 one2one_cv4 (mask coef) branch - only in seg fold tables. */
        for (int level = 0; level < 3; level++) {
            int mask_node = fiv_y26_add_head_chain(&builder, feature_nodes[level],
                                                   "one2one_cv4", level);
            if (mask_node < 0) break;             /* absent in detection: leave -1 */
            graph->mask_node[level] = mask_node;
        }

        /* Pose26 one2one_cv4_kpts branch - only in pose fold tables. */
        for (int level = 0; level < 3; level++) {
            int kpt_node = fiv_y26_add_pose_kpt_chain(&builder, feature_nodes[level], level);
            if (kpt_node < 0) break;              /* absent in detection/seg: leave -1 */
            graph->kpt_node[level] = kpt_node;
        }
    }

    graph->net = net;
    memcpy(graph->layer_node, builder.layer_out, sizeof(builder.layer_out));
    return graph;

fail:
    fiv_release_neural_network(&net);
    fiv_free(graph);
    return NULL;
}

void fiv_yolo26_release(fiv_yolo26_graph* graph)
{
    if (!graph) return;
    if (graph->net) fiv_release_neural_network(&graph->net);
    fiv_free(graph);
}

/* Add a folded 1x1 pointwise conv fed by raw weight/bias arrays (used for the
 * Classify head conv which is not part of the fold table). */
static int fiv_y26_add_head_conv(fiv_y26_builder* builder, int src_node,
                                 const ivf32* head_w, const ivf32* head_b,
                                 int input_channels, int output_channels)
{
    fiv_conv2d_params params;
    memset(&params, 0, sizeof(params));
    params.conv2d_method   = 2;   /* FIV_CONV2D_POINTWISE */
    params.kernel_size_x   = 1;
    params.kernel_size_y   = 1;
    params.stride          = 1;
    params.padding_method  = 0;
    params.input_channels  = input_channels;
    params.output_channels = output_channels;
    params.bias            = 1;

    int conv_node = builder->next_node++;
    if (fiv_neural_network_add_node(builder->net, FIV_NN_NODE_CONV2D_POINTWISE,
                                    src_node, conv_node, &params) != FIV_RET_OK)
        return -1;
    fiv_neural_network_set_node_weight(builder->net, conv_node, head_w);
    fiv_neural_network_set_node_bias(builder->net, conv_node, head_b);

    int silu_node = builder->next_node++;
    if (fiv_neural_network_add_node(builder->net, FIV_NN_NODE_SILU,
                                    conv_node, silu_node, NULL) != FIV_RET_OK)
        return -1;
    return silu_node;
}

fiv_yolo26_graph* fiv_yolo26_build_classifier(const yolo26_fold_entry* fold,
                                              int fold_count,
                                              const ivf32* fold_w,
                                              const ivf32* fold_b,
                                              const ivf32* attn[2][6],
                                              const ivf32* head_conv_w,
                                              const ivf32* head_conv_b,
                                              int head_output_channels)
{
    if (!fold || !fold_w || !fold_b || !attn || !head_conv_w || !head_conv_b)
        return NULL;
    /* yolo26-cls has a single C2PSA (layer 9) which uses only Attention 0. */
    for (int j = 0; j < 6; j++)
        if (!attn[0][j]) return NULL;

    fiv_yolo26_graph* graph = (fiv_yolo26_graph*)fiv_calloc(1, sizeof(fiv_yolo26_graph));
    if (!graph) return NULL;

    void* net = fiv_create_neural_network();
    if (!net) {
        fiv_free(graph);
        return NULL;
    }

    fiv_y26_builder builder;
    memset(&builder, 0, sizeof(builder));
    builder.net        = net;
    builder.fold       = fold;
    builder.fold_count = fold_count;
    builder.fold_w     = fold_w;
    builder.fold_b     = fold_b;
    memcpy(builder.attn_w, attn, sizeof(builder.attn_w));
    builder.next_node = 1;

    for (int i = 0; i < 24; i++) builder.layer_out[i] = -1;
    for (int i = 0; i < 3; i++) {
        graph->mask_node[i] = -1;
        graph->kpt_node[i]  = -1;
    }

    /* yolo26-cls: 0..8 backbone (Conv at 0/1/3/5/7, C3k2 at 2/4/6/8),
     * 9 = C2PSA, 10 = Classify head conv (raw weights, not in fold). */
    int output_channels = 0;
    for (int layer = 0; layer < 9; layer++) {
        if (layer != 0 && builder.layer_out[layer - 1] < 0)
            goto fail;
        int src_node = (layer == 0) ? 0 : builder.layer_out[layer - 1];
        int out_node = -1;
        switch (layer) {
        case 0: case 1: case 3: case 5: case 7:
            out_node = fiv_y26_add_top_conv(&builder, layer, src_node);
            break;
        case 2: case 4: case 6: case 8:
            out_node = fiv_y26_add_c3k2(&builder, layer, src_node);
            break;
        default:
            goto fail;
        }
        if (out_node < 0) goto fail;
        builder.layer_out[layer] = out_node;
        if (layer == 8) {
            const yolo26_fold_entry* entry = fiv_y26_find_entry(&builder, 8, "cv2.conv");
            if (!entry) goto fail;
            output_channels = entry->output_channels;
        }
    }

    /* layer 9 = C2PSA on the layer-8 output. */
    int c2psa_node = fiv_y26_add_c2psa(&builder, 9, builder.layer_out[8]);
    if (c2psa_node < 0) goto fail;
    builder.layer_out[9] = c2psa_node;

    /* layer 10 = Classify head conv 1x1: in = C2PSA out channels. */
    const yolo26_fold_entry* c2psa_entry = fiv_y26_find_entry(&builder, 9, "cv2.conv");
    if (!c2psa_entry) goto fail;
    int input_channels = c2psa_entry->output_channels;

    int head_node = fiv_y26_add_head_conv(&builder, c2psa_node, head_conv_w,
                                          head_conv_b, input_channels,
                                          head_output_channels);
    if (head_node < 0) goto fail;
    builder.layer_out[10] = head_node;
    (void)output_channels;

    graph->net = net;
    memcpy(graph->layer_node, builder.layer_out, sizeof(builder.layer_out));
    return graph;

fail:
    fiv_release_neural_network(&net);
    fiv_free(graph);
    return NULL;
}

