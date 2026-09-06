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
#include <stdlib.h>
#include <string.h>

#include "fiv_nn.h"
#include "fiv_nn_infer.h"
#include "fiv_maxpool_node.h"
#include "fiv_slice_node.h"
#include "fiv_attention_node.h"
#include "fiv_yolo26.h"

typedef struct {
    void*                     net;      /* engine network */
    const yolo26_fold_entry*  fold;     /* fold table */
    int                       n_fold;
    const float*              fold_w;   /* concat payloads */
    const float*              fold_b;
    const float*              attn_w[2][6]; /* attn<i>: qkv_w,b,pe_w,b,proj_w,b */
    int                       next;     /* next node id (0 = implicit input) */
    int                       layer_out[24];
} y26b;

/* entry whose full path == model.<lay>.<rel>  (rel includes ".conv") */
static const yolo26_fold_entry* y26_entry(const y26b* g, int lay, const char* rel)
{
    char pat[256];
    snprintf(pat, sizeof(pat), "model.%d.%s", lay, rel);
    for (int i = 0; i < g->n_fold; i++)
        if (g->fold[i].layer == lay && strcmp(g->fold[i].path, pat) == 0)
            return &g->fold[i];
    return NULL;
}

/* any entry of this layer under prefix "model.<lay>.<pref>" */
static int y26_has(const y26b* g, int lay, const char* pref)
{
    char p[256];
    snprintf(p, sizeof(p), "model.%d.%s", lay, pref);
    size_t pl = strlen(p);
    for (int i = 0; i < g->n_fold; i++)
        if (g->fold[i].layer == lay && strncmp(g->fold[i].path, p, pl) == 0)
            return 1;
    return 0;
}

static int y26_add_node(y26b* g, int type, int src, void* params)
{
    int id = g->next++;
    return fiv_neural_network_add_node(g->net, type, src, id, params) == FIV_RET_OK ? id : -1;
}

static int y26_add_multi(y26b* g, int type, const int* srcs, int cnt, void* params)
{
    int id = g->next++;
    return fiv_neural_network_add_node_multi(g->net, type, (int*)srcs, cnt, id, params) == FIV_RET_OK ? id : -1;
}

static int y26_add2(y26b* g, int a, int b)
{
    int srcs[2] = { a, b };
    return y26_add_multi(g, FIV_NN_NODE_ADD, srcs, 2, NULL);
}

static int y26_slice(y26b* g, int src, int c0, int c1)
{
    fiv_slice_node_params sp;
    memset(&sp, 0, sizeof(sp));
    sp.axis = 1;              /* 4D NCHW: axis 1 = channels */
    sp.start = c0;
    sp.end = c1;
    return y26_add_node(g, FIV_NN_NODE_SLICE, src, &sp);
}

static int y26_cat(y26b* g, const int* srcs, int cnt, int out_ch)
{
    fiv_concat_node_params cp;
    memset(&cp, 0, sizeof(cp));
    cp.axis = 1;
    cp.output_channels = out_ch;
    return y26_add_multi(g, FIV_NN_NODE_CONCAT, srcs, cnt, &cp);
}

/* one folded conv node (+ SiLU when the Conv wrapper has one) */
static int y26_conv(y26b* g, int src, const yolo26_fold_entry* e)
{
    int method, ntype;
    if (e->g == 1 && e->k == 1)       { method = 2; ntype = FIV_NN_NODE_CONV2D_POINTWISE; }
    else if (e->g == e->in_c)         { method = 1; ntype = FIV_NN_NODE_CONV2D_DEPTHWISE; }
    else                              { method = 0; ntype = FIV_NN_NODE_CONV2D_STD; }

    fiv_conv2d_params p;
    memset(&p, 0, sizeof(p));
    p.conv2d_method   = method;
    p.kernel_size_x   = e->k;
    p.kernel_size_y   = e->k;
    p.stride          = e->s;
    p.padding_method  = 0;
    p.input_channels  = e->in_c;
    p.output_channels = e->out_c;
    p.bias            = 1;            /* folded bias always present */
    p.pad_top = p.pad_bottom = p.pad_left = p.pad_right = e->p;

    int id = g->next++;
    if (fiv_neural_network_add_node(g->net, ntype, src, id, &p) != FIV_RET_OK)
        return -1;
    fiv_neural_network_set_node_weight(g->net, id, g->fold_w + e->w_off);
    fiv_neural_network_set_node_bias(g->net, id, g->fold_b + e->b_off);
    if (e->act == 1) {
        int sid = g->next++;
        if (fiv_neural_network_add_node(g->net, FIV_NN_NODE_SILU, id, sid, NULL) != FIV_RET_OK)
            return -1;
        return sid;
    }
    return id;
}

/* Bottleneck (module prefix like "m.0." or "m.0.0."): cv1 -> cv2 -> +src */
static int y26_bottleneck(y26b* g, int lay, const char* mpref, int src)
{
    char r1[160], r2[160];
    snprintf(r1, sizeof(r1), "%scv1.conv", mpref);
    snprintf(r2, sizeof(r2), "%scv2.conv", mpref);
    const yolo26_fold_entry* e1 = y26_entry(g, lay, r1);
    const yolo26_fold_entry* e2 = y26_entry(g, lay, r2);
    if (!e1 || !e2) return -1;
    int a = y26_conv(g, src, e1);
    if (a < 0) return -1;
    int b = y26_conv(g, a, e2);
    if (b < 0) return -1;
    return y26_add2(g, b, src);
}

/* C3k composite at mpref "m.0." (cv1/cv2 parallel, inner m.0.m.<j>. chain, cv3) */
static int y26_c3k_block(y26b* g, int lay, const char* mpref, int src)
{
    char r[160];
    snprintf(r, sizeof(r), "%scv1.conv", mpref);
    const yolo26_fold_entry* cv1 = y26_entry(g, lay, r);
    snprintf(r, sizeof(r), "%scv2.conv", mpref);
    const yolo26_fold_entry* cv2 = y26_entry(g, lay, r);
    snprintf(r, sizeof(r), "%scv3.conv", mpref);
    const yolo26_fold_entry* cv3 = y26_entry(g, lay, r);
    if (!cv1 || !cv2 || !cv3) return -1;

    int b1 = y26_conv(g, src, cv1);                 /* m(cv1(x)) branch */
    if (b1 < 0) return -1;
    for (int j = 0; j < 8; j++) {
        char inner[160];
        snprintf(inner, sizeof(inner), "%sm.%d.", mpref, j);
        if (!y26_has(g, lay, inner)) break;
        b1 = y26_bottleneck(g, lay, inner, b1);
        if (b1 < 0) return -1;
    }
    int b2 = y26_conv(g, src, cv2);                 /* cv2(x) parallel */
    if (b2 < 0) return -1;
    int s2[2] = { b1, b2 };
    int cc = y26_cat(g, s2, 2, cv3->in_c);
    if (cc < 0) return -1;
    return y26_conv(g, cc, cv3);
}

/* Attention composite node feeding from `src`; loads weights from attn_w[aidx] */
static int y26_attn(y26b* g, int src, int aidx, int dim)
{
    fiv_attention_node_params ap;
    memset(&ap, 0, sizeof(ap));
    ap.dim = dim;
    ap.num_heads = 2;
    ap.head_dim  = dim / 2;
    ap.key_dim   = ap.head_dim / 2;
    ap.scale     = powf((float)ap.key_dim, -0.5f);
    ap.pe_groups = dim;
    int id = g->next++;
    if (fiv_neural_network_add_node(g->net, FIV_NN_NODE_ATTENTION, src, id, &ap) != FIV_RET_OK)
        return -1;
    fiv_nn_node_context* nc = fiv_neural_network_get_node(g->net, id);
    if (!nc || !nc->op) return -1;
    const float** w = g->attn_w[aidx];
    if (!w[0] || !w[1] || !w[2] || !w[3] || !w[4] || !w[5]) return -1;
    if (fiv_attention_node_set_weights(nc->op, w[0], w[1], w[2], w[3], w[4], w[5]) != FIV_RET_OK)
        return -1;
    return id;
}

/* PSABlock at stem "m.0." / "m.0.1.": x -> x + attn(x) -> ffn -> + that */
static int y26_psa(y26b* g, int lay, const char* stem, int src, int aidx, int dim)
{
    char r0[160], r1[160];
    snprintf(r0, sizeof(r0), "%sffn.0.conv", stem);
    snprintf(r1, sizeof(r1), "%sffn.1.conv", stem);
    const yolo26_fold_entry* f0 = y26_entry(g, lay, r0);
    const yolo26_fold_entry* f1 = y26_entry(g, lay, r1);
    if (!f0 || !f1) return -1;
    int ax = y26_attn(g, src, aidx, dim);
    if (ax < 0) return -1;
    int t = y26_add2(g, ax, src);
    if (t < 0) return -1;
    int h0 = y26_conv(g, t, f0);
    if (h0 < 0) return -1;
    int h1 = y26_conv(g, h0, f1);
    if (h1 < 0) return -1;
    return y26_add2(g, h1, t);
}

/* Sequential module (attn variant, L22): m.<k>.0 bottleneck then m.<k>.1 psa */
static int y26_module(y26b* g, int lay, const char* mpref, int src)
{
    char probe[160];
    snprintf(probe, sizeof(probe), "%scv3", mpref);
    if (y26_has(g, lay, probe))
        return y26_c3k_block(g, lay, mpref, src);          /* C3k composite */
    snprintf(probe, sizeof(probe), "%s0.", mpref);
    if (y26_has(g, lay, probe)) {                          /* Sequential children */
        int cur = src;
        for (int j = 0; j < 8; j++) {
            char child[160], p2[160];
            snprintf(child, sizeof(child), "%s%d.", mpref, j);
            if (!y26_has(g, lay, child)) break;
            snprintf(p2, sizeof(p2), "%scv1", child);
            if (y26_has(g, lay, p2)) {
                cur = y26_bottleneck(g, lay, child, cur);
            } else {
                snprintf(p2, sizeof(p2), "%sffn.0", child);
                if (y26_has(g, lay, p2)) {
                    int half = 0;                          /* dim passed by caller via closure below */
                    (void)half;
                    cur = y26_psa(g, lay, child, cur, 1, 0);
                }
            }
            if (cur < 0) return -1;
        }
        return cur;
    }
    return y26_bottleneck(g, lay, mpref, src);             /* plain Bottleneck */
}

/* C3k2 (C2f semantics): cv1 -> slice a|b -> module chain on b -> cat -> cv2 */
static int y26_c3k2(y26b* g, int lay, int src)
{
    const yolo26_fold_entry* cv1 = y26_entry(g, lay, "cv1.conv");
    const yolo26_fold_entry* cv2 = y26_entry(g, lay, "cv2.conv");
    if (!cv1 || !cv2) return -1;
    int half = cv1->out_c / 2;
    int dim  = half;
    int v1 = y26_conv(g, src, cv1);
    if (v1 < 0) return -1;
    int a = y26_slice(g, v1, 0, half);
    int b = y26_slice(g, v1, half, cv1->out_c);
    if (a < 0 || b < 0) return -1;

    /* dispatch any inner PSA on dim: L22 psa is at m.0.1 */
    static int    psa_done = 0;   /* placeholder guard, unused */
    (void)psa_done;
    int cur = b;
    int chain[16], cnt = 0;
    chain[cnt++] = b;
    for (int k = 0; k < 8; k++) {
        char mpref[64];
        snprintf(mpref, sizeof(mpref), "m.%d.", k);
        if (!y26_has(g, lay, mpref)) break;
        /* per-layer attention dim for an inner PSABlock (only L22 has one) */
        if (lay == 22 && k == 0) {
            /* rebuild manually: bottleneck m.0.0 then psa m.0.1 */
            int inner = cur;
            for (int j = 0; j < 8; j++) {
                char child[96], probe[96];
                snprintf(child, sizeof(child), "m.%d.%d.", k, j);
                if (!y26_has(g, lay, child)) break;
                snprintf(probe, sizeof(probe), "%scv1", child);
                if (y26_has(g, lay, probe))
                    inner = y26_bottleneck(g, lay, child, inner);
                else
                    inner = y26_psa(g, lay, child, inner, 1, dim);
                if (inner < 0) return -1;
            }
            cur = inner;
        } else {
            cur = y26_module(g, lay, mpref, cur);
        }
        if (cur < 0) return -1;
        chain[cnt++] = cur;
    }
    int srcs[16];
    srcs[0] = a;
    for (int i = 0; i < cnt; i++) srcs[1 + i] = chain[i];
    int cc = y26_cat(g, srcs, 1 + cnt, cv2->in_c);
    if (cc < 0) return -1;
    return y26_conv(g, cc, cv2);
}

/* SPPF: cv1(1x1,act0) -> 3x maxpool(k5 s1 p2) chain -> cat 4 -> cv2(1x1) -> (+x) */
static int y26_sppf(y26b* g, int lay, int src)
{
    const yolo26_fold_entry* cv1 = y26_entry(g, lay, "cv1.conv");
    const yolo26_fold_entry* cv2 = y26_entry(g, lay, "cv2.conv");
    if (!cv1 || !cv2) return -1;
    int v1 = y26_conv(g, src, cv1);
    if (v1 < 0) return -1;

    fiv_maxpool_node_params mp;
    memset(&mp, 0, sizeof(mp));
    mp.kernel_size_x = 5; mp.kernel_size_y = 5; mp.stride = 1;
    mp.pad_top = mp.pad_bottom = mp.pad_left = mp.pad_right = 2;
    int mids[3], prev = v1;
    for (int i = 0; i < 3; i++) {
        int mid = g->next++;
        if (fiv_neural_network_add_node(g->net, FIV_NN_NODE_MAXPOOL, prev, mid, &mp) != FIV_RET_OK)
            return -1;
        mids[i] = mid;
        prev = mid;
    }
    int s4[4] = { v1, mids[0], mids[1], mids[2] };
    int cc = y26_cat(g, s4, 4, cv2->in_c);
    if (cc < 0) return -1;
    int v2 = y26_conv(g, cc, cv2);
    if (v2 < 0) return -1;
    if (cv1->in_c == cv2->out_c) {              /* add residual */
        int ad = y26_add2(g, v2, src);
        if (ad < 0) return -1;
        return ad;
    }
    return v2;
}

/* C2PSA: cv1 -> slice a|b -> b=m.0 PSABlock(attn0) -> cat([a,psa]) -> cv2 */
static int y26_c2psa(y26b* g, int lay, int src)
{
    const yolo26_fold_entry* cv1 = y26_entry(g, lay, "cv1.conv");
    const yolo26_fold_entry* cv2 = y26_entry(g, lay, "cv2.conv");
    if (!cv1 || !cv2) return -1;
    int half = cv1->out_c / 2;
    int v1 = y26_conv(g, src, cv1);
    if (v1 < 0) return -1;
    int a = y26_slice(g, v1, 0, half);
    int b = y26_slice(g, v1, half, cv1->out_c);
    if (a < 0 || b < 0) return -1;
    int m = y26_psa(g, lay, "m.0.", b, 0, half);
    if (m < 0) return -1;
    int s2[2] = { a, m };
    int cc = y26_cat(g, s2, 2, cv2->in_c);
    if (cc < 0) return -1;
    return y26_conv(g, cc, cv2);
}

/* top-level plain Conv layer: single folded conv at model.<lay>.conv */
static int y26_topconv(y26b* g, int lay, int src)
{
    const yolo26_fold_entry* e = y26_entry(g, lay, "conv");
    if (!e) return -1;
    return y26_conv(g, src, e);
}

/* Detect one2one head convs on feature node `feat`, one per level */
static int y26_head_chain(y26b* g, int feat, const char* branch, int lvl)
{
    /* match "one2one_cv2.<lvl>" as a bare substring: the chain's tail is a
       plain nn.Conv2d whose fold path has NO ".conv" suffix, so a key ending
       in "." would silently skip it. Prefix "one2one_cv2.<lvl>" is unique per
       level (".0" / ".1" / ".2"), so no ambiguity. */
    char key[48];
    snprintf(key, sizeof(key), "%s.%d", branch, lvl);
    int cur = feat;
    int built = 0;
    for (int i = 0; i < g->n_fold; i++) {
        const yolo26_fold_entry* e = &g->fold[i];
        if (e->layer != 23) continue;
        if (strstr(e->path, key) == NULL) continue;
        cur = y26_conv(g, cur, e);
        if (cur < 0) return -1;
        built++;
    }
    return built > 0 ? cur : -1;
}

fiv_yolo26_graph* fiv_yolo26_build(const yolo26_fold_entry* fold, int fold_count,
                                   const float* fold_w, const float* fold_b,
                                   const float* attn[2][6])
{
    if (!fold || !fold_w || !fold_b || !attn) return NULL;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 6; j++)
            if (!attn[i][j]) return NULL;

    fiv_yolo26_graph* graph = (fiv_yolo26_graph*)calloc(1, sizeof(fiv_yolo26_graph));
    if (!graph) return NULL;
    void* net = fiv_create_neural_network();
    if (!net) { free(graph); return NULL; }

    y26b g;
    memset(&g, 0, sizeof(g));
    g.net = net;
    g.fold = fold;
    g.n_fold = fold_count;
    g.fold_w = fold_w;
    g.fold_b = fold_b;
    memcpy(g.attn_w, attn, sizeof(g.attn_w));
    g.next = 1;

    for (int i = 0; i < 24; i++) g.layer_out[i] = -1;

    int rc = 0;
    for (int i = 0; i < 23; i++) {
        int out_id = -1;
        if (i == 12 || i == 15 || i == 18 || i == 21) {
            /* Concat: fan-in from two prior layer outputs; consumed by next C3k2. */
            int a = (i == 12) ? 11 : (i == 15) ? 14 : (i == 18) ? 17 : 20;
            int b = (i == 12) ? 6  : (i == 15) ? 4  : (i == 18) ? 13 : 10;
            if (g.layer_out[a] < 0 || g.layer_out[b] < 0) { rc = -1; break; }
            int s2[2] = { g.layer_out[a], g.layer_out[b] };
            const yolo26_fold_entry* nx = y26_entry(&g, i + 1, "cv1.conv");
            if (!nx) { rc = -1; break; }
            out_id = y26_cat(&g, s2, 2, nx->in_c);
            if (out_id < 0) { rc = -1; break; }
            g.layer_out[i] = out_id;
            continue;
        }
        if (i != 0) {
            if (g.layer_out[i - 1] < 0) { rc = -1; break; }
        }
        int src = (i == 0) ? 0 : g.layer_out[i - 1];
        switch (i) {
        case 0: case 1: case 3: case 5: case 7: case 17: case 20:
            out_id = y26_topconv(&g, i, src); break;
        case 2: case 4: case 6: case 8: case 13: case 16: case 19:
            out_id = y26_c3k2(&g, i, src); break;
        case 22:
            out_id = y26_c3k2(&g, i, src); break;
        case 9:
            out_id = y26_sppf(&g, i, src); break;
        case 10:
            out_id = y26_c2psa(&g, i, src); break;
        case 11: case 14:
            out_id = y26_add_node(&g, FIV_NN_NODE_UPSAMPLE2X, src, NULL); break;
        default:
            rc = -1; break;
        }
        if (rc != 0 || out_id < 0) { rc = -1; break; }
        g.layer_out[i] = out_id;
    }
    if (rc != 0) { fiv_release_neural_network(&net); free(graph); return NULL; }

    /* Detect one2one heads on features [L16, L19, L22] */
    int feat[3] = { g.layer_out[16], g.layer_out[19], g.layer_out[22] };
    for (int lvl = 0; lvl < 3; lvl++) {
        int bb = y26_head_chain(&g, feat[lvl], "one2one_cv2", lvl);
        int cc = y26_head_chain(&g, feat[lvl], "one2one_cv3", lvl);
        if (bb < 0 || cc < 0) { rc = -1; break; }
        graph->head_node[lvl]     = bb;
        graph->head_node[3 + lvl] = cc;
    }
    if (rc != 0) { fiv_release_neural_network(&net); free(graph); return NULL; }

    graph->net = net;
    memcpy(graph->layer_node, g.layer_out, sizeof(g.layer_out));
    return graph;
}

void fiv_yolo26_release(fiv_yolo26_graph* graph)
{
    if (!graph) return;
    if (graph->net) fiv_release_neural_network(&graph->net);
    free(graph);
}
