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

/* yolo26-cls (Classify) full network: layers 0..8 identical to the detection
 * backbone (Conv/C3k2 variants), layer 9 = C2PSA (the SAME module as detection
 * layer 10 - the config just drops SPPF and the neck), layer 10 = Classify head
 * = Conv(1x1 256->1280, SiLU) then the classifier tail is a plain C post-pass:
 *   feature = conv(x)                       [1,1280,H,W]
 *   pooled  = mean(feature, H, W)           [1,1280]  (AdaptiveAvgPool2d(1))
 *   logits  = linear_w . pooled + linear_b  [1,1000]
 *   probs   = softmax(logits)
 * Layer-9 C2PSA contains one Attention composite node (attn0_* weights, same
 * hyperparams as detection).
 *
 * The graph is fold-driven (all convs BN-folded in fold_w/fold_b), with the
 * Classify head conv (model.10, NOT in the fold table) loaded from the
 * head_conv_w/head_conv_b dumps. Layer io is compared against the reference,
 * then pooled/logits/probs against the reference post-head tensors.
 *
 * Run from build/: YOLO26_REF_DIR=../app/yolo26/test/real/ref_cls ./test_yolo26_cls
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fiv_ctensor.h"
#include "fiv_nn.h"
#include "fiv_nn_infer.h"
#include "fiv_slice_node.h"
#include "fiv_attention_node.h"
#include "yolo26_ref.h"
#include "fiv_common.h"

static int g_pass = 0;
static int g_fail = 0;

typedef struct {
    void*                    net;
    const yolo26_fold_entry* es;
    int                      n;
    int                      next;
} cgb;

static const yolo26_fold_entry* ce(const cgb* g, int lay, const char* rel)
{
    char pat[160];
    snprintf(pat, sizeof(pat), "model.%d.%s", lay, rel);
    for (int i = 0; i < g->n; i++)
        if (g->es[i].layer == lay && strcmp(g->es[i].path, pat) == 0)
            return &g->es[i];
    return NULL;
}
static int ch(const cgb* g, int lay, const char* pre)
{
    char pat[160];
    snprintf(pat, sizeof(pat), "model.%d.%s", lay, pre);
    size_t pl = strlen(pat);
    for (int i = 0; i < g->n; i++)
        if (g->es[i].layer == lay && strncmp(g->es[i].path, pat, pl) == 0)
            return 1;
    return 0;
}

static int cadd(cgb* g, int type, int src, void* params)
{
    int id = g->next++;
    return fiv_neural_network_add_node(g->net, type, src, id, params) == FIV_RET_OK ? id : -1;
}
static int cmulti(cgb* g, int type, const int* srcs, int cnt, void* params)
{
    int id = g->next++;
    return fiv_neural_network_add_node_multi(g->net, type, (int*)srcs, cnt, id, params) == FIV_RET_OK ? id : -1;
}
static int cadd2(cgb* g, int a, int b)
{
    int s[2] = { a, b };
    return cmulti(g, FIV_NN_NODE_ADD, s, 2, NULL);
}
static int cslice(cgb* g, int src, int c0, int c1)
{
    fiv_slice_node_params sp; memset(&sp, 0, sizeof(sp));
    sp.axis = 1; sp.start = c0; sp.end = c1;
    return cadd(g, FIV_NN_NODE_SLICE, src, &sp);
}
static int ccat(cgb* g, const int* srcs, int cnt, int out_ch)
{
    fiv_concat_node_params cp; memset(&cp, 0, sizeof(cp));
    cp.axis = 1; cp.output_channels = out_ch;
    return cmulti(g, FIV_NN_NODE_CONCAT, srcs, cnt, &cp);
}
static int cconv(cgb* g, int src, const yolo26_fold_entry* e)
{
    int method, nt;
    if (e->groups == 1 && e->kernel == 1) { method = 2; nt = FIV_NN_NODE_CONV2D_POINTWISE; }
    else if (e->groups == e->input_channels)   { method = 1; nt = FIV_NN_NODE_CONV2D_DEPTHWISE; }
    else                        { method = 0; nt = FIV_NN_NODE_CONV2D_STD; }
    fiv_conv2d_params p; memset(&p, 0, sizeof(p));
    p.conv2d_method = method;
    p.kernel_size_x = e->kernel; p.kernel_size_y = e->kernel;
    p.stride = e->stride; p.padding_method = 0;
    p.input_channels = e->input_channels; p.output_channels = e->output_channels;
    p.bias = 1;
    p.pad_top = p.pad_bottom = p.pad_left = p.pad_right = e->padding;
    int id = g->next++;
    if (fiv_neural_network_add_node(g->net, nt, src, id, &p) != FIV_RET_OK) return -1;
    const ivf32* w = yolo26_ref_load_blob("fold_w.f32", e->w_off, e->w_cnt);
    const ivf32* b = yolo26_ref_load_blob("fold_b.f32", e->b_off, e->b_cnt);
    if (!w || !b) { fiv_free((void*)w); fiv_free((void*)b); return -1; }
    fiv_neural_network_set_node_weight(g->net, id, w);
    fiv_neural_network_set_node_bias(g->net, id, b);
    fiv_free((void*)w); fiv_free((void*)b);
    if (e->act == 1) {
        int sid = g->next++;
        if (fiv_neural_network_add_node(g->net, FIV_NN_NODE_SILU, id, sid, NULL) != FIV_RET_OK) return -1;
        return sid;
    }
    return id;
}
static int cbottle(cgb* g, int lay, const char* mp, int src)
{
    char r1[128], r2[128];
    snprintf(r1, sizeof(r1), "%scv1.conv", mp);
    snprintf(r2, sizeof(r2), "%scv2.conv", mp);
    const yolo26_fold_entry* e1 = ce(g, lay, r1);
    const yolo26_fold_entry* e2 = ce(g, lay, r2);
    if (!e1 || !e2) return -1;
    int a = cconv(g, src, e1); if (a < 0) return -1;
    int b = cconv(g, a, e2);   if (b < 0) return -1;
    return cadd2(g, b, src);
}
static int cattn(cgb* g, int src, int dim)
{
    fiv_attention_node_params ap; memset(&ap, 0, sizeof(ap));
    ap.dim = dim; ap.num_heads = 2; ap.head_dim = dim / 2;
    ap.key_dim = 32; ap.scale = 0.1767766952966369f; ap.pe_groups = dim;
    int id = g->next++;
    if (fiv_neural_network_add_node(g->net, FIV_NN_NODE_ATTENTION, src, id, &ap) != FIV_RET_OK) return -1;
    const char* parts[6] = { "qkv_w","qkv_b","pe_w","pe_b","proj_w","proj_b" };
    const ivf32* bb[6]; memset(bb, 0, sizeof(bb));
    char nm[64]; int nd; size_t sh[8];
    for (int i = 0; i < 6; i++) {
        snprintf(nm, sizeof(nm), "attn0_%s", parts[i]);
        bb[i] = ref_load(nm, &nd, sh);
        if (!bb[i]) { for (int q = 0; q < 6; q++) fiv_free((void*)bb[q]); return -1; }
    }
    fiv_nn_node_context* nc = fiv_neural_network_get_node(g->net, id);
    fiv_ret rr = (nc && nc->op) ? fiv_attention_node_set_weights(nc->op, bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]) : FIV_RET_ERR_PARA;
    for (int i = 0; i < 6; i++) fiv_free((void*)bb[i]);
    return rr == FIV_RET_OK ? id : -1;
}
static int cpsa(cgb* g, int lay, int src, int dim)
{
    const yolo26_fold_entry* f0 = ce(g, lay, "m.0.ffn.0.conv");
    const yolo26_fold_entry* f1 = ce(g, lay, "m.0.ffn.1.conv");
    if (!f0 || !f1) return -1;
    int ax = cattn(g, src, dim); if (ax < 0) return -1;
    int t  = cadd2(g, ax, src);  if (t < 0) return -1;
    int h0 = cconv(g, t, f0);    if (h0 < 0) return -1;
    int h1 = cconv(g, h0, f1);   if (h1 < 0) return -1;
    return cadd2(g, h1, t);
}
static int cc2psa(cgb* g, int lay, int src)
{
    const yolo26_fold_entry* cv1 = ce(g, lay, "cv1.conv");
    const yolo26_fold_entry* cv2 = ce(g, lay, "cv2.conv");
    if (!cv1 || !cv2) return -1;
    int half = cv1->output_channels / 2;
    int v1 = cconv(g, src, cv1); if (v1 < 0) return -1;
    int a = cslice(g, v1, 0, half); if (a < 0) return -1;
    int b = cslice(g, v1, half, cv1->output_channels); if (b < 0) return -1;
    int p = cpsa(g, lay, b, half); if (p < 0) return -1;
    int s[2] = { a, p };
    int cc = ccat(g, s, 2, cv2->input_channels); if (cc < 0) return -1;
    return cconv(g, cc, cv2);
}
static int cc3k2(cgb* g, int lay, int src)
{
    const yolo26_fold_entry* cv1 = ce(g, lay, "cv1.conv");
    const yolo26_fold_entry* cv2 = ce(g, lay, "cv2.conv");
    if (!cv1 || !cv2) return -1;
    int half = cv1->output_channels / 2;
    int v1 = cconv(g, src, cv1); if (v1 < 0) return -1;
    int a = cslice(g, v1, 0, half); if (a < 0) return -1;
    int b = cslice(g, v1, half, cv1->output_channels); if (b < 0) return -1;
    int cur = b;
    int chain[8], cnt = 1;
    chain[0] = b;
    int is_c3k = (ce(g, lay, "m.0.cv3.conv") != NULL);
    if (is_c3k) {
        const yolo26_fold_entry* c1 = ce(g, lay, "m.0.cv1.conv");
        const yolo26_fold_entry* c2 = ce(g, lay, "m.0.cv2.conv");
        const yolo26_fold_entry* c3 = ce(g, lay, "m.0.cv3.conv");
        if (!c1 || !c2 || !c3) return -1;
        int bb1 = cconv(g, cur, c1); if (bb1 < 0) return -1;
        for (int k = 0; k < 8; k++) {
            char mp[128]; snprintf(mp, sizeof(mp), "m.0.m.%d.", k);
            if (!ch(g, lay, mp)) break;
            bb1 = cbottle(g, lay, mp, bb1);
            if (bb1 < 0) return -1;
        }
        int bb2 = cconv(g, cur, c2); if (bb2 < 0) return -1;
        int s2[2] = { bb1, bb2 };
        int cc = ccat(g, s2, 2, c3->input_channels); if (cc < 0) return -1;
        cur = cconv(g, cc, c3); if (cur < 0) return -1;
    } else {
        for (int k = 0; k < 8; k++) {
            char mp[128]; snprintf(mp, sizeof(mp), "m.%d.", k);
            if (!ch(g, lay, mp)) break;
            cur = cbottle(g, lay, mp, cur);
            if (cur < 0) return -1;
        }
    }
    chain[cnt++] = cur;
    int srcs[8]; srcs[0] = a;
    for (int i = 0; i < cnt; i++) srcs[1 + i] = chain[i];
    int cc = ccat(g, srcs, 1 + cnt, cv2->input_channels); if (cc < 0) return -1;
    return cconv(g, cc, cv2);
}

/* model.10 head conv: 1x1 256->1280 SiLU from head_conv_w/b (BN-folded) */
static int chead(cgb* g, int src)
{
    int nd; size_t sh[8];
    const ivf32* w = ref_load("head_conv_w", &nd, sh);
    if (!w) return -1;
    int oc = (int)sh[0], ic = (int)sh[1];
    const ivf32* b = ref_load("head_conv_b", &nd, sh);
    if (!b) { fiv_free((void*)w); return -1; }
    fiv_conv2d_params p; memset(&p, 0, sizeof(p));
    p.conv2d_method = 2;              /* POINTWISE 1x1 */
    p.kernel_size_x = 1; p.kernel_size_y = 1; p.stride = 1;
    p.padding_method = 0; p.input_channels = ic; p.output_channels = oc;
    p.bias = 1;
    int id = g->next++;
    if (fiv_neural_network_add_node(g->net, FIV_NN_NODE_CONV2D_POINTWISE, src, id, &p) != FIV_RET_OK) return -1;
    fiv_neural_network_set_node_weight(g->net, id, w);
    fiv_neural_network_set_node_bias(g->net, id, b);
    int sid = g->next++;
    if (fiv_neural_network_add_node(g->net, FIV_NN_NODE_SILU, id, sid, NULL) != FIV_RET_OK) return -1;
    fiv_free((void*)w); fiv_free((void*)b);
    return sid;
}

static const char* const kL[11] = { "Conv","Conv","C3k2","Conv","C3k2","Conv","C3k2","Conv","C3k2","C2PSA","Classify" };

int main(void)
{
    printf("=== YOLO26-cls full-network test ===\n");
    yolo26_fold_entry fold[300];
    int n = yolo26_ref_load_fold(fold, 300);
    if (n <= 0) { printf("FAIL fold load\n"); return 1; }
    size_t w_tot = 0, b_tot = 0;
    for (int i = 0; i < n; i++) {
        size_t we = fold[i].w_off + fold[i].w_cnt, be = fold[i].b_off + fold[i].b_cnt;
        if (we > w_tot) w_tot = we;
        if (be > b_tot) b_tot = be;
    }
    const ivf32* fw = yolo26_ref_load_blob("fold_w.f32", 0, w_tot);
    const ivf32* fb = yolo26_ref_load_blob("fold_b.f32", 0, b_tot);
    if (!fw || !fb) { printf("FAIL blobs\n"); return 1; }

    void* net = fiv_create_neural_network();
    if (!net) { printf("FAIL net\n"); return 1; }
    cgb g = { net, fold, n, 1 };
    int layer_node[11];
    int rc = 0, src = 0;
    for (int i = 0; i < 10; i++) {
        int out = -1;
        switch (i) {
        case 0: case 1: case 3: case 5: case 7: {
            const yolo26_fold_entry* e = ce(&g, i, "conv");
            out = e ? cconv(&g, src, e) : -1;
            break; }
        case 2: case 4: case 6: case 8:
            out = cc3k2(&g, i, src); break;
        case 9:
            out = cc2psa(&g, i, src); break;
        }
        if (out < 0) { rc = -1; break; }
        layer_node[i] = out;
        src = out;
    }
    if (rc == 0) {
        int hc = chead(&g, src);        /* model.10 Classify conv (post-head handled in C) */
        if (hc < 0) rc = -1;
        else layer_node[10] = hc;
    }
    if (rc != 0) { printf("FAIL graph build rc=%d\n", rc); return 1; }
    printf("graph built; nodes=%d\n", g.next);

    int nd; size_t sh[8];
    const ivf32* in_f = ref_load("input", &nd, sh);
    if (!in_f || nd != 4) { printf("FAIL input\n"); return 1; }
    fiv_tensor4d* input = fiv_create_tensor4d((size_t*)sh, FIV_32F1);
    size_t in_n = sh[0]*sh[1]*sh[2]*sh[3];
    memcpy(((fiv_tensor_hdr*)input)->data.fl, in_f, in_n*sizeof(ivf32));
    fiv_free((void*)in_f);
    void* final_out = NULL;
    fiv_ret r = fiv_nn_run_inference(net, input, &final_out);
    fiv_release_tensor((void**)&input);
    if (r != FIV_RET_OK) { printf("FAIL run r=%d\n", r); return 1; }

    /* per-layer outputs 0..9 (layer10 conv is verified via pooled/logits) */
    for (int i = 0; i < 10; i++) {
        char on[48];
        snprintf(on, sizeof(on), "layer%02d_%s", i, kL[i]);
        fiv_tensor4d* out = (fiv_tensor4d*)fiv_neural_network_get_node_output(net, layer_node[i]);
        if (!out) { g_fail++; printf("  [FAIL] %s read\n", on); continue; }
        ivf32 me; int cr = ref_cmp(on, ((fiv_tensor_hdr*)out)->data.fl, 1e-4f, &me);
        if (cr == 0) g_pass++; else { g_fail++; printf("  [FAIL] %s err=%.2e\n", on, (double)me); }
    }

    /* classifier tail in C: pooled = mean over H,W of the head-conv output, then
       logits = linear_w . pooled + linear_b ; probs = softmax(logits) */
    {
        fiv_tensor4d* headout = (fiv_tensor4d*)fiv_neural_network_get_node_output(net, layer_node[10]);
        int H = (int)headout->height, W = (int)headout->width, C = (int)headout->channels;
        const ivf32* hf = ((fiv_tensor_hdr*)headout)->data.fl;
        size_t hw = (size_t)H * W;
        /* pooled ref */
        const ivf32* pre = ref_load("head_pooled", &nd, sh);
        ivf32* pool = (ivf32*)fiv_malloc((size_t)C * sizeof(ivf32));
        ivf32 pmax = 0.0f;
        for (int c = 0; c < C; c++) {
            double s = 0.0;
            const ivf32* ch_ = hf + (size_t)c * hw;
            for (size_t k = 0; k < hw; k++) s += ch_[k];
            pool[c] = (ivf32)(s / (double)hw);
            if (pre) { ivf32 d = fabsf(pool[c] - pre[c]); if (d > pmax) pmax = d; }
        }
        if (pre) {
            if (pmax < 1e-3f) { g_pass++; printf("  PASS  head_pooled (mean H,W) max_err=%.2e\n", (double)pmax); }
            else { g_fail++; printf("  FAIL  head_pooled max_err=%.2e\n", (double)pmax); }
            fiv_free((void*)pre);
        }
        /* logits = linear_w[1000,1280] . pool[1280] + linear_b */
        const ivf32* lw = ref_load("linear_w", &nd, sh); int nc = (int)sh[0];
        const ivf32* lb = ref_load("linear_b", &nd, sh);
        const ivf32* lr = ref_load("logits", &nd, sh);
        ivf32* logits = (ivf32*)fiv_malloc((size_t)nc * sizeof(ivf32));
        ivf32 lmax = 0.0f;
        for (int c = 0; c < nc; c++) {
            const ivf32* wrow = lw + (size_t)c * C;
            double s = 0.0;
            for (int k = 0; k < C; k++) s += (double)wrow[k] * pool[k];
            logits[c] = (ivf32)s + lb[c];
            if (lr) { ivf32 d = fabsf(logits[c] - lr[c]); if (d > lmax) lmax = d; }
        }
        if (lr) {
            if (lmax < 1e-2f) { g_pass++; printf("  PASS  logits max_err=%.2e\n", (double)lmax); }
            else { g_fail++; printf("  FAIL  logits max_err=%.2e\n", (double)lmax); }
            fiv_free((void*)lr);
        }
        /* top1/top5 class agreement with ref probs */
        const ivf32* pr = ref_load("probs", &nd, sh);
        if (pr) {
            int top1 = 0;
            for (int c = 1; c < nc; c++) if (logits[c] > logits[top1]) top1 = c;
            int rtop1 = 0;
            for (int c = 1; c < nc; c++) if (pr[c] > pr[rtop1]) rtop1 = c;
            if (top1 == rtop1) { g_pass++; printf("  PASS  top-1 class id=%d (match)\n", top1); }
            else { g_fail++; printf("  FAIL  top-1 C=%d ref=%d\n", top1, rtop1); }
            fiv_free((void*)pr);
        }
        fiv_free(logits); fiv_free(pool);
        fiv_free((void*)lw); fiv_free((void*)lb);
    }

    fiv_release_neural_network(&net);
    fiv_free((void*)fw); fiv_free((void*)fb);
    printf("=== P6 cls: pass=%d fail=%d ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
