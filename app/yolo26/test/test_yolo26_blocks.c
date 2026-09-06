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

/* P3 block test for the YOLO26 port (stage 1: plain Conv layers). Every
 * top-level Conv module is a single BN-folded conv (+ SiLU). For each such
 * layer we build a one-node engine graph driven by fold.json, inject the
 * folded weights, run inference on the layer's own torch input dump and
 * compare against the torch output dump. This locks the conv method/pad/
 * stride mapping and the BN-folded weight injection against the engine.
 *
 * Run from build/: ./test_yolo26_blocks
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fiv_ctensor.h"
#include "fiv_nn.h"
#include "fiv_nn_infer.h"
#include "fiv_maxpool_node.h"
#include "fiv_slice_node.h"
#include "fiv_attention_node.h"
#include "yolo26_ref.h"
#include "fiv_common.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK_FAIL(name)                                                  \
    do { printf("  FAIL  %s (missing reference)\n", name); g_fail++; } while (0)

/* NN node type for add_node (api/fiv_nn.h fiv_nn_node_type). */
static int fiv_y26_node_type(const yolo26_fold_entry* e)
{
    if (e->groups == 1 && e->kernel == 1) return FIV_NN_NODE_CONV2D_POINTWISE;
    if (e->groups == e->input_channels && e->groups == e->output_channels) return FIV_NN_NODE_CONV2D_DEPTHWISE;
    return FIV_NN_NODE_CONV2D_STD;
}

/* conv2d_method value for fiv_conv2d_params (FIV_CONV2D_* in fiv_nn_conv2d.h:
 * STD=0, DEPTHWISE=1, POINTWISE=2). Distinct from the NN node enum above. */
static int fiv_y26_conv_method(const yolo26_fold_entry* e)
{
    if (e->groups == 1 && e->kernel == 1) return 2;   /* FIV_CONV2D_POINTWISE */
    if (e->groups == e->input_channels && e->groups == e->output_channels) return 1;  /* FIV_CONV2D_DEPTHWISE */
    return 0;                               /* FIV_CONV2D_STD */
}

/* Build the engine graph for one top-level Conv layer (single conv + optional
 * SiLU), inject the folded weights and run it on `input`. Returns the output
 * node id (>= 1) or -1. The net stays alive; caller reads the node output. */
static int fiv_y26_run_conv_layer(const yolo26_fold_entry* e, const ivf32* in_data,
                                  const size_t in_shape[4], void** net_out)
{
    void* net = fiv_create_neural_network();
    if (!net) return -1;

    fiv_conv2d_params p;
    memset(&p, 0, sizeof(p));
    p.conv2d_method   = fiv_y26_conv_method(e);
    p.kernel_size_x   = e->kernel;
    p.kernel_size_y   = e->kernel;
    p.stride          = e->stride;
    p.padding_method  = 0;                 /* zero fill (torch default padding) */
    p.input_channels  = e->input_channels;
    p.output_channels = e->output_channels;
    p.bias            = 1;                 /* BN fold always yields a bias */
    p.pad_top = p.pad_bottom = p.pad_left = p.pad_right = e->padding;

    int node_type = fiv_y26_node_type(e);
    int conv_id   = 1;                     /* node 0 = implicit INPUT */
    if (fiv_neural_network_add_node(net, node_type, 0, conv_id, &p) != FIV_RET_OK)
        goto fail;

    const ivf32* w = yolo26_ref_load_blob("fold_w.f32", e->w_off, e->w_cnt);
    const ivf32* b = yolo26_ref_load_blob("fold_b.f32", e->b_off, e->b_cnt);
    if (!w || !b) { fiv_free((void*)w); fiv_free((void*)b); goto fail; }
    fiv_neural_network_set_node_weight(net, conv_id, w);
    fiv_neural_network_set_node_bias(net, conv_id, b);
    fiv_free((void*)w); fiv_free((void*)b);

    int out_id = conv_id;
    if (e->act == 1) {
        int silu_id = 2;
        if (fiv_neural_network_add_node(net, FIV_NN_NODE_SILU, conv_id, silu_id, NULL) != FIV_RET_OK)
            goto fail;
        out_id = silu_id;
    }

    fiv_tensor4d* input = fiv_create_tensor4d((size_t*)in_shape, FIV_32F1);
    if (!input) goto fail;
    size_t in_n = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3];
    memcpy(((fiv_tensor_hdr*)input)->data.fl, in_data, in_n * sizeof(ivf32));

    void* final_out = NULL;
    fiv_ret r = fiv_nn_run_inference(net, input, &final_out);
    fiv_release_tensor((void**)&input);
    if (r != FIV_RET_OK) goto fail;

    *net_out = net;
    return out_id;
fail:
    fiv_release_neural_network(&net);
    return -1;
}

static void test_conv_layer(const yolo26_fold_entry* e)
{
    char in_name[64], out_name[64];
    snprintf(in_name, sizeof(in_name), "layer%02d_Conv_in", e->layer);
    snprintf(out_name, sizeof(out_name), "layer%02d_Conv", e->layer);

    int ndim;
    size_t ishape[8];
    const ivf32* in_f = ref_load(in_name, &ndim, ishape);
    if (!in_f) { CHECK_FAIL(in_name); return; }
    if (ndim != 4) { fiv_free((void*)in_f); printf("  FAIL  %s ndim=%d\n", in_name, ndim); g_fail++; return; }

    void* net = NULL;
    int out_id = fiv_y26_run_conv_layer(e, in_f, ishape, &net);
    fiv_free((void*)in_f);
    if (out_id < 0) { printf("  FAIL  layer%02d_Conv graph build/run\n", e->layer); g_fail++; return; }

    fiv_tensor4d* out = (fiv_tensor4d*)fiv_neural_network_get_node_output(net, out_id);
    if (!out) { printf("  FAIL  layer%02d_Conv read output\n", e->layer); g_fail++; fiv_release_neural_network(&net); return; }

    ivf32 max_err;
    int rc = ref_cmp(out_name, ((fiv_tensor_hdr*)out)->data.fl, 1e-4f, &max_err);
    /* ref_cmp already printed the [ ok ]/[FAIL] line; just count here. */
    if (rc == 0) g_pass++; else g_fail++;
    fiv_release_neural_network(&net);
}


/* ---------------- composite-block helpers (SPPF / C3k2 / C2PSA) ---------------- */

static const yolo26_fold_entry* fold_find(const yolo26_fold_entry* es, int n,
                                          int layer, const char* path_tail)
{
    for (int i = 0; i < n; i++)
        if (es[i].layer == layer && strstr(es[i].path, path_tail) != NULL)
            return &es[i];
    return NULL;
}

/* Add a conv node (BN-folded weights already in fold_w/b) whose params derive
   from a fold entry; returns its node id or -1. `src` is the feeding node id. */
static int y26_add_conv(void* net, int next_id, int src, const yolo26_fold_entry* e)
{
    int method;
    if (e->groups == 1 && e->kernel == 1)       method = 2;   /* FIV_CONV2D_POINTWISE */
    else if (e->groups == e->input_channels)         method = 1;   /* FIV_CONV2D_DEPTHWISE */
    else                              method = 0;   /* FIV_CONV2D_STD */
    int node_type;
    if (e->groups == 1 && e->kernel == 1)       node_type = FIV_NN_NODE_CONV2D_POINTWISE;
    else if (e->groups == e->input_channels)         node_type = FIV_NN_NODE_CONV2D_DEPTHWISE;
    else                              node_type = FIV_NN_NODE_CONV2D_STD;

    fiv_conv2d_params p;
    memset(&p, 0, sizeof(p));
    p.conv2d_method   = method;
    p.kernel_size_x   = e->kernel;
    p.kernel_size_y   = e->kernel;
    p.stride          = e->stride;
    p.padding_method  = 0;
    p.input_channels  = e->input_channels;
    p.output_channels = e->output_channels;
    p.bias            = 1;
    p.pad_top = p.pad_bottom = p.pad_left = p.pad_right = e->padding;
    if (fiv_neural_network_add_node(net, node_type, src, next_id, &p) != FIV_RET_OK)
        return -1;
    const ivf32* w = yolo26_ref_load_blob("fold_w.f32", e->w_off, e->w_cnt);
    const ivf32* b = yolo26_ref_load_blob("fold_b.f32", e->b_off, e->b_cnt);
    if (!w || !b) { fiv_free((void*)w); fiv_free((void*)b); return -1; }
    fiv_neural_network_set_node_weight(net, next_id, w);
    fiv_neural_network_set_node_bias(net, next_id, b);
    fiv_free((void*)w); fiv_free((void*)b);
    return next_id;
}

/* Add a SILU node consuming node `src`, returning its id or -1. */
static int y26_add_silu(void* net, int next_id, int src)
{
    if (fiv_neural_network_add_node(net, FIV_NN_NODE_SILU, src, next_id, NULL) != FIV_RET_OK)
        return -1;
    return next_id;
}

/* SPPF(layer): y = cv2(cat(cv1(x), m(cv1), m^2(cv1), m^3(cv1))) + x. From the
   v8.4 SPPF source: c_ = c1//2; cv1(1x1, act=False) c1->c_; n=3 maxpool k5 s1 p2;
   cv2(1x1, act) c_*(n+1) -> c2; add = (shortcut && c1==c2). */
static int y26_run_sppf(const yolo26_fold_entry* entries, int n_entries,
                        const ivf32* in_data, const size_t in_shape[4], void** net_out)
{
    const yolo26_fold_entry* cv1 = fold_find(entries, n_entries, 9, ".cv1.conv");
    const yolo26_fold_entry* cv2 = fold_find(entries, n_entries, 9, ".cv2.conv");
    if (!cv1 || !cv2) return -1;
    int c_ = cv1->output_channels;            /* cv1 out = c_ (c1//2) */
    int cat_ch = c_ * (cv1->kernel ? 4 : 4);   /* n=3 -> 4 cat inputs */
    cat_ch = cv2->input_channels;             /* truth from the fold table */

    void* net = fiv_create_neural_network();
    if (!net) return -1;
    fprintf(stderr, "  DBG sppf: cv1=%s cv2=%s c_=%d cat_ch=%d\n",
            cv1 ? cv1->path : "?", cv2 ? cv2->path : "?", c_, cat_ch);
    int id = 1;
    int cv1_id = y26_add_conv(net, id++, 0, cv1);
    if (cv1_id < 0) { fprintf(stderr, "  DBG sppf cv1 add failed\n"); goto fail; }

    fiv_maxpool_node_params mp;
    memset(&mp, 0, sizeof(mp));
    mp.kernel_size_x = 5; mp.kernel_size_y = 5; mp.stride = 1;
    mp.pad_top = mp.pad_bottom = mp.pad_left = mp.pad_right = 2;
    int mp_ids[3];
    int prev = cv1_id;
    for (int i = 0; i < 3; i++) {
        int mid = id++;
        if (fiv_neural_network_add_node(net, FIV_NN_NODE_MAXPOOL, prev, mid, &mp) != FIV_RET_OK) {
            fprintf(stderr, "  DBG sppf maxpool %d add failed\n", i);
            goto fail;
        }
        mp_ids[i] = mid;
        prev = mid;
    }
    int cat_id = id++;
    {
        int srcs[4] = { cv1_id, mp_ids[0], mp_ids[1], mp_ids[2] };
        fiv_concat_node_params cp;
        memset(&cp, 0, sizeof(cp));
        cp.axis = 1;
        cp.output_channels = cat_ch;
        if (fiv_neural_network_add_node_multi(net, FIV_NN_NODE_CONCAT, srcs, 4, cat_id, &cp) != FIV_RET_OK) {
            fprintf(stderr, "  DBG sppf concat add failed\n");
            goto fail;
        }
    }
    int cv2_id = y26_add_conv(net, id++, cat_id, cv2);
    if (cv2_id < 0) { fprintf(stderr, "  DBG sppf cv2 add failed\n"); goto fail; }
    int silu_id = cv2_id;
    if (cv2->act == 1) { silu_id = y26_add_silu(net, id++, cv2_id); if (silu_id < 0) { fprintf(stderr, "  DBG sppf silu add failed\n"); goto fail; } }
    int add_id = id++;
    {
        int srcs[2] = { silu_id, 0 };   /* residual: output + module input (node 0) */
        if (fiv_neural_network_add_node_multi(net, FIV_NN_NODE_ADD, srcs, 2, add_id, NULL) != FIV_RET_OK) {
            fprintf(stderr, "  DBG sppf add(residual) failed\n");
            goto fail;
        }
    }

    fiv_tensor4d* input = fiv_create_tensor4d((size_t*)in_shape, FIV_32F1);
    if (!input) { fprintf(stderr, "  DBG sppf input alloc failed\n"); goto fail; }
    size_t in_n = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3];
    memcpy(((fiv_tensor_hdr*)input)->data.fl, in_data, in_n * sizeof(ivf32));
    void* final_out = NULL;
    fiv_ret r = fiv_nn_run_inference(net, input, &final_out);
    fiv_release_tensor((void**)&input);
    if (r != FIV_RET_OK) { fprintf(stderr, "  DBG sppf run_inference r=%d\n", r); goto fail; }
    *net_out = net;
    return add_id;
fail:
    fiv_release_neural_network(&net);
    return -1;
}

static void test_sppf(const yolo26_fold_entry* entries, int n_entries)
{
    const char* in_name = "layer09_SPPF_in";
    const char* out_name = "layer09_SPPF";
    int ndim;
    size_t ishape[8];
    const ivf32* in_f = ref_load(in_name, &ndim, ishape);
    if (!in_f) { CHECK_FAIL(in_name); return; }
    void* net = NULL;
    int out_id = y26_run_sppf(entries, n_entries, in_f, ishape, &net);
    fiv_free((void*)in_f);
    if (out_id < 0) { printf("  FAIL  SPPF graph build/run\n"); g_fail++; return; }
    fiv_tensor4d* out = (fiv_tensor4d*)fiv_neural_network_get_node_output(net, out_id);
    if (!out) { printf("  FAIL  SPPF read output\n"); g_fail++; fiv_release_neural_network(&net); return; }
    ivf32 max_err;
    int rc = ref_cmp(out_name, ((fiv_tensor_hdr*)out)->data.fl, 1e-4f, &max_err);
    if (rc == 0) g_pass++; else g_fail++;
    fiv_release_neural_network(&net);
}




/* ---------------- C3k2 (C2f base) composite-block builders ----------------
   Topology (v8.4 source): cv1(1x1->2c, SiLU) -> chunk(a,b) each c channels;
   chain of n modules over b; cat([a, b, m0(b), m1(m0(b)), ...]); cv2(1x1, SiLU).
   Module variant (detected from fold paths of this layer):
     plain : m.<k> is a Bottleneck (conv + conv + residual add)
     c3k   : m.0 is a C3k   = cv3(cat(m(cv1(x)), cv2(x))) with 2 inner Bottlenecks
   Everything (n, channels, act) is read from the fold table; nothing hardcoded. */

typedef struct {
    void*                 net;
    int                   id;     /* next free node id (0 = implicit INPUT) */
    const yolo26_fold_entry* es;
    int                   n;
} y26gb;

/* entry whose full path == model.<lay>.<rel>.conv */
static const yolo26_fold_entry* y26_rel(const yolo26_fold_entry* es, int n,
                                        int lay, const char* rel)
{
    char pat[256];
    snprintf(pat, sizeof(pat), "model.%d.%s.conv", lay, rel);
    for (int i = 0; i < n; i++)
        if (es[i].layer == lay && strcmp(es[i].path, pat) == 0)
            return &es[i];
    return NULL;
}

/* True when any fold entry of this layer sits under `modpref` (a module
   prefix ending in '.', e.groups. "m.0."). Paths are exact strings like
   "model.<lay>.m.0.cv1.conv", so module-existence is a prefix test - an
   exact full-string match against "model.<lay>.m.0..conv" never hits. */
static int y26_has_mod(const yolo26_fold_entry* es, int n, int lay, const char* modpref)
{
    char pref[256];
    snprintf(pref, sizeof(pref), "model.%d.%s", lay, modpref);
    size_t plen = strlen(pref);
    for (int i = 0; i < n; i++)
        if (es[i].layer == lay && strncmp(es[i].path, pref, plen) == 0)
            return 1;
    return 0;
}

/* add one conv node (+ trailing SiLU when its Conv wrapper has act) */
static int y26_gconv(y26gb* g, int src, const yolo26_fold_entry* e)
{
    if (!e) return -1;
    int method, node_type;
    if (e->groups == 1 && e->kernel == 1)      { method = 2; node_type = FIV_NN_NODE_CONV2D_POINTWISE; }
    else if (e->groups == e->input_channels)        { method = 1; node_type = FIV_NN_NODE_CONV2D_DEPTHWISE; }
    else                             { method = 0; node_type = FIV_NN_NODE_CONV2D_STD; }

    fiv_conv2d_params p;
    memset(&p, 0, sizeof(p));
    p.conv2d_method   = method;
    p.kernel_size_x   = e->kernel;
    p.kernel_size_y   = e->kernel;
    p.stride          = e->stride;
    p.padding_method  = 0;
    p.input_channels  = e->input_channels;
    p.output_channels = e->output_channels;
    p.bias            = 1;
    p.pad_top = p.pad_bottom = p.pad_left = p.pad_right = e->padding;

    int nid = g->id++;
    if (fiv_neural_network_add_node(g->net, node_type, src, nid, &p) != FIV_RET_OK)
        return -1;
    const ivf32* w = yolo26_ref_load_blob("fold_w.f32", e->w_off, e->w_cnt);
    const ivf32* b = yolo26_ref_load_blob("fold_b.f32", e->b_off, e->b_cnt);
    if (!w || !b) { fiv_free((void*)w); fiv_free((void*)b); return -1; }
    fiv_neural_network_set_node_weight(g->net, nid, w);
    fiv_neural_network_set_node_bias(g->net, nid, b);
    fiv_free((void*)w); fiv_free((void*)b);
    if (e->act == 1) {
        int sid = g->id++;
        if (fiv_neural_network_add_node(g->net, FIV_NN_NODE_SILU, nid, sid, NULL) != FIV_RET_OK)
            return -1;
        return sid;
    }
    return nid;
}

static int y26_gslice(y26gb* g, int src, int c0, int c1)
{
    fiv_slice_node_params sp;
    memset(&sp, 0, sizeof(sp));
    sp.axis = 1; sp.start = c0; sp.end = c1;
    int nid = g->id++;
    if (fiv_neural_network_add_node(g->net, FIV_NN_NODE_SLICE, src, nid, &sp) != FIV_RET_OK)
        return -1;
    return nid;
}

static int y26_gcat(y26gb* g, const int* srcs, int cnt, int out_ch)
{
    fiv_concat_node_params cp;
    memset(&cp, 0, sizeof(cp));
    cp.axis = 1;
    cp.output_channels = out_ch;
    int nid = g->id++;
    if (fiv_neural_network_add_node_multi(g->net, FIV_NN_NODE_CONCAT,
                                          (int*)srcs, cnt, nid, &cp) != FIV_RET_OK)
        return -1;
    return nid;
}

static int y26_gadd2(y26gb* g, int a, int b)
{
    int srcs[2] = { a, b };
    int nid = g->id++;
    if (fiv_neural_network_add_node_multi(g->net, FIV_NN_NODE_ADD,
                                          srcs, 2, nid, NULL) != FIV_RET_OK)
        return -1;
    return nid;
}

/* Bottleneck: x -> cv1 -> cv2 -> (+ x). mpref e.groups. "m.0." or "m.0.m.1." */
static int y26_bottleneck(y26gb* g, int lay, const char* mpref, int src)
{
    char r1[128], r2[128];
    snprintf(r1, sizeof(r1), "%scv1", mpref);
    snprintf(r2, sizeof(r2), "%scv2", mpref);
    const yolo26_fold_entry* c1 = y26_rel(g->es, g->n, lay, r1);
    const yolo26_fold_entry* c2 = y26_rel(g->es, g->n, lay, r2);
    if (!c1 || !c2) return -1;
    int a1 = y26_gconv(g, src, c1);
    if (a1 < 0) return -1;
    int a2 = y26_gconv(g, a1, c2);
    if (a2 < 0) return -1;
    return y26_gadd2(g, a2, src);
}

/* C3k block at m.0: cv3(cat(m(cv1(x)), cv2(x))); m = m.0.m.<k> bottlenecks */
static int y26_c3k_module(y26gb* g, int lay, int src)
{
    const yolo26_fold_entry* c1 = y26_rel(g->es, g->n, lay, "m.0.cv1");
    const yolo26_fold_entry* c2 = y26_rel(g->es, g->n, lay, "m.0.cv2");
    const yolo26_fold_entry* c3 = y26_rel(g->es, g->n, lay, "m.0.cv3");
    if (!c1 || !c2 || !c3) return -1;
    int b1 = y26_gconv(g, src, c1);          /* m(cv1(x)) branch stem */
    if (b1 < 0) return -1;
    for (int k = 0; k < 8; k++) {            /* inner bottleneck chain */
        char pref[128];
        snprintf(pref, sizeof(pref), "m.0.m.%d.", k);
        if (!y26_has_mod(g->es, g->n, lay, pref)) break;
        b1 = y26_bottleneck(g, lay, pref, b1);
        if (b1 < 0) return -1;
    }
    int b2 = y26_gconv(g, src, c2);          /* cv2(x) parallel branch */
    if (b2 < 0) return -1;
    int srcs[2] = { b1, b2 };
    int cat = y26_gcat(g, srcs, 2, c3->input_channels);
    if (cat < 0) return -1;
    return y26_gconv(g, cat, c3);
}

/* whole C3k2 layer: returns final node id or -1 */
static int y26_c3k2(y26gb* g, int lay)
{
    const yolo26_fold_entry* cv1 = y26_rel(g->es, g->n, lay, "cv1");
    const yolo26_fold_entry* cv2 = y26_rel(g->es, g->n, lay, "cv2");
    if (!cv1 || !cv2) return -1;
    int half = cv1->output_channels / 2;               /* chunk channel c */

    int v1 = y26_gconv(g, 0, cv1);
    if (v1 < 0) return -1;
    int sa = y26_gslice(g, v1, 0, half);
    int sb = y26_gslice(g, v1, half, 2 * half);
    if (sa < 0 || sb < 0) return -1;

    int c3k = (y26_rel(g->es, g->n, lay, "m.0.cv3") != NULL);
    int cur = sb;
    int chain[10], cnt = 0;
    chain[cnt++] = sb;                       /* cat keeps b itself */
    if (c3k) {
        cur = y26_c3k_module(g, lay, sb);
        if (cur < 0) return -1;
        chain[cnt++] = cur;
    } else {
        for (int k = 0; k < 8; k++) {
            char pref[64];
            snprintf(pref, sizeof(pref), "m.%d.", k);
            if (!y26_has_mod(g->es, g->n, lay, pref)) break;
            cur = y26_bottleneck(g, lay, pref, cur);
            if (cur < 0) return -1;
            chain[cnt++] = cur;
        }
    }
    int srcs[12];
    srcs[0] = sa;
    for (int i = 0; i < cnt; i++) srcs[1 + i] = chain[i];
    int cat = y26_gcat(g, srcs, 1 + cnt, cv2->input_channels);
    if (cat < 0) return -1;
    return y26_gconv(g, cat, cv2);
}

static void test_c3k2(const yolo26_fold_entry* entries, int n_entries, int lay)
{
    char tname[64];
    snprintf(tname, sizeof(tname), "layer%02d_C3k2", lay);
    char in_name[64], out_name[64];
    snprintf(in_name, sizeof(in_name), "%s_in", tname);
    snprintf(out_name, sizeof(out_name), "%s", tname);

    int ndim;
    size_t ishape[8];
    const ivf32* in_f = ref_load(in_name, &ndim, ishape);
    if (!in_f) { CHECK_FAIL(in_name); return; }

    void* net = fiv_create_neural_network();
    if (!net) { fiv_free((void*)in_f); g_fail++; return; }
    y26gb g;
    g.net = net; g.id = 1; g.es = entries; g.n = n_entries;
    int out_id = y26_c3k2(&g, lay);
    if (out_id < 0) { fiv_free((void*)in_f); fiv_release_neural_network(&net);
                      printf("  FAIL  %s graph build\\n", tname); g_fail++; return; }

    fiv_tensor4d* input = fiv_create_tensor4d((size_t*)ishape, FIV_32F1);
    if (!input) { fiv_free((void*)in_f); fiv_release_neural_network(&net); g_fail++; return; }
    size_t in_n = ishape[0] * ishape[1] * ishape[2] * ishape[3];
    memcpy(((fiv_tensor_hdr*)input)->data.fl, in_f, in_n * sizeof(ivf32));
    fiv_free((void*)in_f);
    void* final_out = NULL;
    fiv_ret r = fiv_nn_run_inference(net, input, &final_out);
    fiv_release_tensor((void**)&input);
    if (r != FIV_RET_OK) { fiv_release_neural_network(&net);
                           printf("  FAIL  %s run r=%d\\n", tname, r); g_fail++; return; }

    fiv_tensor4d* out = (fiv_tensor4d*)fiv_neural_network_get_node_output(net, out_id);
    if (!out) { fiv_release_neural_network(&net);
                printf("  FAIL  %s read out\\n", tname); g_fail++; return; }
    ivf32 max_err;
    int rc = ref_cmp(out_name, ((fiv_tensor_hdr*)out)->data.fl, 1e-4f, &max_err);
    if (rc == 0) g_pass++; else g_fail++;
    fiv_release_neural_network(&net);
}


/* L10 C2PSA / L22 attn-C3k2: PSABlock embeds the ATTENTION composite node. */

static int y26_gattn(y26gb* g, int src, const char* pref, int dim)
{
    fiv_attention_node_params ap;
    memset(&ap, 0, sizeof(ap));
    ap.dim = dim;
    ap.num_heads = 2;                              /* heads = max(dim/64, 1) */
    ap.head_dim  = dim / ap.num_heads;
    ap.key_dim   = 32;                             /* int(head_dim * 0.5) */
    ap.scale     = 0.1767766952966369f;            /* key_dim^-0.5 */
    ap.pe_groups = dim;
    int nid = g->id++;
    if (fiv_neural_network_add_node(g->net, FIV_NN_NODE_ATTENTION, src, nid, &ap) != FIV_RET_OK)
        return -1;
    const char* parts[6] = { "qkv_w", "qkv_b", "pe_w", "pe_b", "proj_w", "proj_b" };
    const ivf32* bb[6];
    memset(bb, 0, sizeof(bb));
    char nm[80];
    int ndim; size_t sh[8];
    for (int i = 0; i < 6; i++) {
        snprintf(nm, sizeof(nm), "%s_%s", pref, parts[i]);
        bb[i] = ref_load(nm, &ndim, sh);
    }
    fiv_ret rs = FIV_RET_ERR_PARA;
    if (bb[0] && bb[1] && bb[2] && bb[3] && bb[4] && bb[5]) {
        fiv_nn_node_context* nc = fiv_neural_network_get_node(g->net, nid);
        if (nc && nc->op)
            rs = fiv_attention_node_set_weights(nc->op, bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]);
    }
    for (int i = 0; i < 6; i++) fiv_free((void*)bb[i]);
    return rs == FIV_RET_OK ? nid : -1;
}

/* PSABlock forward: x -> x + attn(x) -> m = ffn(that) -> out = m + that.
   stem locates the ffn convs (e.groups. "m.0" or "m.0.1"). */
static int y26_psa(y26gb* g, int lay, const char* stem, int src, const char* attn_pref, int dim)
{
    char r0[96], r1[96];
    snprintf(r0, sizeof(r0), "%s.ffn.0", stem);
    snprintf(r1, sizeof(r1), "%s.ffn.1", stem);
    const yolo26_fold_entry* f0 = y26_rel(g->es, g->n, lay, r0);
    const yolo26_fold_entry* f1 = y26_rel(g->es, g->n, lay, r1);
    if (!f0 || !f1) return -1;
    int t = y26_gattn(g, src, attn_pref, dim);
    if (t < 0) return -1;
    int m1 = y26_gadd2(g, t, src);
    if (m1 < 0) return -1;
    int g0 = y26_gconv(g, m1, f0);
    if (g0 < 0) return -1;
    int g1 = y26_gconv(g, g0, f1);
    if (g1 < 0) return -1;
    return y26_gadd2(g, g1, m1);
}

/* L10 C2PSA: cv1 -> a|b slice -> b=m.0 PSABlock(attn0) -> cat([a,psa]) -> cv2 */
static int y26_c2psa(y26gb* g)
{
    const yolo26_fold_entry* cv1 = y26_rel(g->es, g->n, 10, "cv1");
    const yolo26_fold_entry* cv2 = y26_rel(g->es, g->n, 10, "cv2");
    if (!cv1 || !cv2) return -1;
    int half = cv1->output_channels / 2;
    int v1 = y26_gconv(g, 0, cv1);
    if (v1 < 0) return -1;
    int a = y26_gslice(g, v1, 0, half);
    int b = y26_gslice(g, v1, half, 2 * half);
    if (a < 0 || b < 0) return -1;
    int m = y26_psa(g, 10, "m.0", b, "attn0", half);
    if (m < 0) return -1;
    int s2[2] = { a, m };
    int cc = y26_gcat(g, s2, 2, cv2->input_channels);
    if (cc < 0) return -1;
    return y26_gconv(g, cc, cv2);
}

/* L22 attn-C3k2: cv1 -> a|b slice -> m.0.0 Bottleneck -> m.0.1 PSABlock(attn1)
   -> cat([a,b,psa]) -> cv2 */
static int y26_c3k2_attn(y26gb* g)
{
    const yolo26_fold_entry* cv1 = y26_rel(g->es, g->n, 22, "cv1");
    const yolo26_fold_entry* cv2 = y26_rel(g->es, g->n, 22, "cv2");
    if (!cv1 || !cv2) return -1;
    int half = cv1->output_channels / 2;
    int v1 = y26_gconv(g, 0, cv1);
    if (v1 < 0) return -1;
    int a = y26_gslice(g, v1, 0, half);
    int b = y26_gslice(g, v1, half, 2 * half);
    if (a < 0 || b < 0) return -1;
    int bn = y26_bottleneck(g, 22, "m.0.0.", b);
    if (bn < 0) return -1;
    int m = y26_psa(g, 22, "m.0.1", bn, "attn1", half);
    if (m < 0) return -1;
    int s3[3] = { a, b, m };
    int cc = y26_gcat(g, s3, 3, cv2->input_channels);
    if (cc < 0) return -1;
    return y26_gconv(g, cc, cv2);
}

/* Build graph, run on <in>_in dump, compare against <out> dump. */
static void y26_compare(const char* in_name, const char* out_name,
                        int (*graph)(y26gb*), const yolo26_fold_entry* es, int n)
{
    int ndim; size_t ishape[8];
    const ivf32* in_f = ref_load(in_name, &ndim, ishape);
    if (!in_f) { CHECK_FAIL(in_name); return; }
    void* net = fiv_create_neural_network();
    if (!net) { fiv_free((void*)in_f); printf("  FAIL  %s alloc net\n", out_name); g_fail++; return; }
    y26gb g;
    g.net = net; g.id = 1; g.es = es; g.n = n;
    int out_id = graph(&g);
    if (out_id < 0) {
        fiv_free((void*)in_f); fiv_release_neural_network(&net);
        printf("  FAIL  %s graph build\n", out_name); g_fail++; return;
    }
    fiv_tensor4d* input = fiv_create_tensor4d((size_t*)ishape, FIV_32F1);
    if (!input) { fiv_free((void*)in_f); fiv_release_neural_network(&net); g_fail++; return; }
    size_t in_n = ishape[0] * ishape[1] * ishape[2] * ishape[3];
    memcpy(((fiv_tensor_hdr*)input)->data.fl, in_f, in_n * sizeof(ivf32));
    fiv_free((void*)in_f);
    void* final_out = NULL;
    fiv_ret r = fiv_nn_run_inference(net, input, &final_out);
    fiv_release_tensor((void**)&input);
    if (r != FIV_RET_OK) {
        fiv_release_neural_network(&net);
        printf("  FAIL  %s run r=%d\n", out_name, r); g_fail++; return;
    }
    fiv_tensor4d* out = (fiv_tensor4d*)fiv_neural_network_get_node_output(net, out_id);
    if (!out) {
        fiv_release_neural_network(&net);
        printf("  FAIL  %s read out\n", out_name); g_fail++; return;
    }
    ivf32 max_err;
    int rc = ref_cmp(out_name, ((fiv_tensor_hdr*)out)->data.fl, 1e-4f, &max_err);
    if (rc == 0) g_pass++; else g_fail++;
    fiv_release_neural_network(&net);
}


int main(void)
{
    printf("=== YOLO26 P3 block test: top-level Conv layers ===\n");

    yolo26_fold_entry entries[256];
    int n = yolo26_ref_load_fold(entries, 256);
    if (n <= 0) { printf("FAIL: fold table load\n"); return 1; }

    /* a top-level Conv layer = exactly one folded conv whose immediate parent
       is the Conv wrapper ("model.<i>.conv"); its fold entry.layer == <i>. */
    int checked = 0;
    for (int i = 0; i < n; i++) {
        yolo26_fold_entry* e = &entries[i];
        int single = 1;
        for (int j = 0; j < n; j++) {
            if (j != i && entries[j].layer == e->layer && entries[j].input_channels > 0) { single = 0; break; }
        }
        if (!single) continue;
        if (strcmp(e->parent, "Conv") != 0) continue;
        if (e->layer < 0 || e->layer >= 23) continue;   /* Detect (23) is P4 */
        test_conv_layer(e);
        checked++;
    }
    if (checked == 0) { printf("FAIL: no top-level Conv layers found\n"); return 1; }
    printf("--- SPPF block (layer 9) ---\n");
    {
        int has_l9 = 0;
        for (int i = 0; i < n; i++) if (entries[i].layer == 9) has_l9 = 1;
        if (has_l9) test_sppf(entries, n);
        else { printf("  FAIL  layer 9 (SPPF) not in fold table\n"); g_fail++; }
    }
    printf("--- C3k2 blocks ---\n");
    {
        static const int c3k2_layers[] = { 2, 4, 6, 8, 13, 16, 19 };
        int skip = 0;
        for (unsigned int q = 0; q < sizeof(c3k2_layers) / sizeof(c3k2_layers[0]); q++) {
            int lay = c3k2_layers[q];
            int has = 0;
            for (int i = 0; i < n; i++) if (entries[i].layer == lay) { has = 1; break; }
            if (!has) { printf("  skip L%d (not in fold)\n", lay); skip++; continue; }
            if (y26_rel(entries, n, lay, "cv1") == NULL) { printf("  skip L%d (no cv1)\n", lay); skip++; continue; }
            test_c3k2(entries, n, lay);
        }
        if (skip == 0) { /* no-op */ }
    }
    printf("--- C2PSA / attn-C3k2 blocks ---\n");
    y26_compare("layer10_C2PSA_in", "layer10_C2PSA", y26_c2psa, entries, n);
    y26_compare("layer22_C3k2_in", "layer22_C3k2", y26_c3k2_attn, entries, n);
    printf("=== P3 blocks: pass=%d fail=%d ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
