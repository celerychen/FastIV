/*
 * FastIV - Fast image and vision
 * Copyright (C) 2026 Celery Chen
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * See LICENSE file in project root for full license text.
 *
 * End-to-end BlazeFace check: build the short-range face-detection network with
 * fiv_nn (CONV2D_STD/DEPTHWISE/POINTWISE + RELU + MAX2D + ADD + PAD) and verify
 * the forward output against the reference C port (detect.c / cnn_ops.c):
 *   - every intermediate feature map (input, stem, block_00..block_15) matches
 *     the reference dump within tolerance,
 *   - the flattened regressors/scores match the reference dump,
 *   - the final detections (decode + sigmoid + weighted NMS + project, same
 *     postprocess code as the reference) match blazeface_detect() output.
 *
 * The reference objects (weights.c / geom.c / cnn_ops.c / detect.c) are linked
 * in and used ONLY as the golden source; the forward itself runs on fiv_nn.
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "detect.h"        /* reference: Detection, blazeface_detect(_dump) */
#include "weights.h"       /* reference: Weights, weights_load/get/free */
#include "geom.h"          /* reference: warp for the ImageToTensor ROI */

#include "fiv_nn.h"
#include "fiv_nn_infer.h"  /* internal: read per-node outputs after inference */
#include "fiv_nn_conv2d.h"
#include "fiv_ctensor.h"
#include "fiv_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* reference postprocess (copied verbatim from detect.c so the fiv_nn  */
/* detections go through the identical decode/NMS/project pipeline)    */
/* ------------------------------------------------------------------ */
static const float X_SCALE = 128.0f, Y_SCALE = 128.0f;
static const float W_SCALE = 128.0f, H_SCALE = 128.0f;

static double calc_scale(double min_s, double max_s, int idx, int n) {
    if (n == 1) return (min_s + max_s) * 0.5;
    return min_s + (max_s - min_s) * idx / (n - 1.0);
}

static void generate_anchors(float A[NUM_BOXES][4]) {
    int input_size = TENSOR_SIZE, num_layers = 4;
    int strides[4] = {8, 16, 16, 16};
    double min_scale = 0.1484375, max_scale = 0.75, offset = 0.5;
    int layer_id = 0, idx = 0;
    while (layer_id < num_layers) {
        int last = layer_id;
        while (last < num_layers && strides[last] == strides[layer_id]) last++;
        double ar[16], sc[16]; int na = 0;
        for (int ls = layer_id; ls < last; ls++) {
            double s = calc_scale(min_scale, max_scale, ls, num_layers);
            ar[na] = 1.0; sc[na] = s; na++;
            double snext = (ls == num_layers - 1) ? 1.0
                                                : calc_scale(min_scale, max_scale, ls + 1, num_layers);
            sc[na] = sqrt(s * snext); ar[na] = 1.0; na++;
        }
        int fm = (int)ceil((double)input_size / strides[layer_id]);
        for (int y = 0; y < fm; y++)
            for (int x = 0; x < fm; x++)
                for (int aid = 0; aid < na; aid++) {
                    A[idx][0] = (float)((x + offset) / fm);
                    A[idx][1] = (float)((y + offset) / fm);
                    A[idx][2] = 1.0f; A[idx][3] = 1.0f;
                    idx++;
                }
        layer_id = last;
    }
}

static float sigmoidf(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float e = expf(x);
    return e / (1.0f + e);
}

typedef struct { float score; float bbox[4]; float kps[NUM_KEYPOINTS][2]; } RawDet;

static float iou(const float a[4], const float b[4]) {
    float ix1 = a[0] > b[0] ? a[0] : b[0];
    float iy1 = a[1] > b[1] ? a[1] : b[1];
    float ix2 = a[2] < b[2] ? a[2] : b[2];
    float iy2 = a[3] < b[3] ? a[3] : b[3];
    float iw = ix2 > ix1 ? ix2 - ix1 : 0.0f;
    float ih = iy2 > iy1 ? iy2 - iy1 : 0.0f;
    float inter = iw * ih;
    float area_a = (a[2] > a[0] ? a[2] - a[0] : 0.0f) * (a[3] > a[1] ? a[3] - a[1] : 0.0f);
    float area_b = (b[2] > b[0] ? b[2] - b[0] : 0.0f) * (b[3] > b[1] ? b[3] - b[1] : 0.0f);
    float union_ = area_a + area_b - inter;
    return union_ > 0.0f ? inter / union_ : 0.0f;
}

static int weighted_nms(const RawDet *dets, int n, Detection *out, int max_out) {
    int *order = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (dets[order[j]].score > dets[order[i]].score) {
                int t = order[i]; order[i] = order[j]; order[j] = t;
            }
    char *suppressed = (char *)calloc(n, 1);
    int nout = 0;
    for (int i = 0; i < n && nout < max_out; i++) {
        int ai = order[i];
        if (suppressed[ai]) continue;
        int cand[1024]; int nc = 0;
        for (int j = 0; j < n; j++) {
            if (suppressed[order[j]]) continue;
            if (iou(dets[order[j]].bbox, dets[ai].bbox) > NMS_THRESH) cand[nc++] = order[j];
        }
        float total = 0.0f;
        float wx1 = 0, wy1 = 0, wx2 = 0, wy2 = 0;
        float wkx[NUM_KEYPOINTS], wky[NUM_KEYPOINTS];
        for (int k = 0; k < NUM_KEYPOINTS; k++) { wkx[k] = 0; wky[k] = 0; }
        for (int c = 0; c < nc; c++) {
            const RawDet *d = &dets[cand[c]];
            float s = d->score; total += s;
            wx1 += d->bbox[0] * s; wy1 += d->bbox[1] * s;
            wx2 += d->bbox[2] * s; wy2 += d->bbox[3] * s;
            for (int k = 0; k < NUM_KEYPOINTS; k++) {
                wkx[k] += d->kps[k][0] * s; wky[k] += d->kps[k][1] * s;
            }
        }
        Detection *o = &out[nout++];
        o->score = dets[ai].score;
        o->x = wx1 / total; o->y = wy1 / total;
        o->w = wx2 / total - o->x; o->h = wy2 / total - o->y;
        for (int k = 0; k < NUM_KEYPOINTS; k++) {
            o->kps[k][0] = wkx[k] / total; o->kps[k][1] = wky[k] / total;
        }
        for (int c = 0; c < nc; c++) suppressed[cand[c]] = 1;
    }
    free(order); free(suppressed);
    return nout;
}

static int postprocess(const float reg[NUM_BOXES][NUM_COORDS], const float *scores,
                       int img_w, int img_h, float min_score,
                       Detection *dets, int max_dets) {
    float A[NUM_BOXES][4];
    generate_anchors(A);
    RawDet raw[1024]; int nr = 0;
    for (int i = 0; i < NUM_BOXES; i++) {
        float ax = A[i][0], ay = A[i][1], aw = A[i][2], ah = A[i][3];
        float dx = reg[i][0] / X_SCALE * aw + ax;
        float dy = reg[i][1] / Y_SCALE * ah + ay;
        float dw = (reg[i][2] / W_SCALE) * aw;
        float dh = (reg[i][3] / H_SCALE) * ah;
        float xmin = dx - dw * 0.5f, ymin = dy - dh * 0.5f;
        float xmax = dx + dw * 0.5f, ymax = dy + dh * 0.5f;
        float s = sigmoidf(scores[i]);
        if (s < min_score) continue;
        if ((xmax - xmin) < 0.0f || (ymax - ymin) < 0.0f) continue;
        if (nr >= 1024) break;
        RawDet *d = &raw[nr++];
        d->score = s;
        d->bbox[0] = xmin; d->bbox[1] = ymin; d->bbox[2] = xmax; d->bbox[3] = ymax;
        for (int k = 0; k < NUM_KEYPOINTS; k++) {
            d->kps[k][0] = reg[i][4 + 2 * k] / X_SCALE * aw + ax;
            d->kps[k][1] = reg[i][5 + 2 * k] / Y_SCALE * ah + ay;
        }
    }
    Detection *tmp = (Detection *)malloc(sizeof(Detection) * (nr > 0 ? nr : 1));
    int nkept = weighted_nms(raw, nr, tmp, nr > 0 ? nr : 1);

    double scale = (TENSOR_SIZE / (double)img_w < TENSOR_SIZE / (double)img_h)
                       ? TENSOR_SIZE / (double)img_w : TENSOR_SIZE / (double)img_h;
    double tx = (TENSOR_SIZE - scale * img_w) * 0.5;
    double ty = (TENSOR_SIZE - scale * img_h) * 0.5;

    int nout = 0;
    for (int i = 0; i < nkept && nout < max_dets; i++) {
        Detection *src = &tmp[i];
        double xmin = src->x, ymin = src->y;
        double xmax = src->x + src->w, ymax = src->y + src->h;
        Detection *o = &dets[nout++];
        o->score = src->score;
        o->x = (float)((xmin * TENSOR_SIZE - tx) / scale);
        o->y = (float)((ymin * TENSOR_SIZE - ty) / scale);
        o->w = (float)(((xmax * TENSOR_SIZE - tx) / scale) - o->x);
        o->h = (float)(((ymax * TENSOR_SIZE - ty) / scale) - o->y);
        for (int k = 0; k < NUM_KEYPOINTS; k++) {
            double kx = (src->kps[k][0] * TENSOR_SIZE - tx) / scale;
            double ky = (src->kps[k][1] * TENSOR_SIZE - ty) / scale;
            o->kps[k][0] = (float)(kx / img_w);
            o->kps[k][1] = (float)(ky / img_h);
        }
    }
    free(tmp);
    return nout;
}

/* ------------------------------------------------------------------ */
/* golden dump parser (format: name_len u32 | name | n,c,h,w i32 | f32) */
/* ------------------------------------------------------------------ */
typedef struct { char name[64]; int n, c, h, w; float* data; size_t elems; } GoldT;
typedef struct { GoldT* items; int n; } Gold;

static int gold_load(const char* path, Gold* g) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    g->n = 0;
    g->items = NULL;
    int cap = 0;
    for (;;) {
        unsigned int nl;
        if (fread(&nl, 4, 1, f) != 1) break;   /* EOF after the last tensor */
        if (g->n == cap) {
            cap = cap ? cap * 2 : 32;
            GoldT* ni = (GoldT*)realloc(g->items, (size_t)cap * sizeof(GoldT));
            if (!ni) goto fail;
            g->items = ni;
        }
        GoldT* t = &g->items[g->n];
        if (nl >= 64) nl = 63;
        if (fread(t->name, 1, nl, f) != nl) goto fail;
        t->name[nl] = '\0';
        int hdr[4];
        if (fread(hdr, 4, 4, f) != 4) goto fail;
        t->n = hdr[0]; t->c = hdr[1]; t->h = hdr[2]; t->w = hdr[3];
        t->elems = (size_t)t->n * t->c * t->h * t->w;
        t->data = (float*)malloc(t->elems * sizeof(float));
        if (!t->data) goto fail;
        if (fread(t->data, 4, t->elems, f) != t->elems) goto fail;
        g->n++;
    }
    fclose(f);
    return 1;
fail:
    fclose(f);
    return 0;
}

static const GoldT* gold_find(const Gold* g, const char* name) {
    for (int i = 0; i < g->n; i++)
        if (strcmp(g->items[i].name, name) == 0) return &g->items[i];
    return NULL;
}

/* ------------------------------------------------------------------ */
/* fiv_nn graph builder                                                */
/* ------------------------------------------------------------------ */
static const int BLOCK_SPECS[16][4] = {
    {24, 24, 1, 0}, {24, 28, 1, 0}, {28, 32, 2, 1}, {32, 36, 1, 0},
    {36, 42, 1, 0}, {42, 48, 2, 1}, {48, 56, 1, 0}, {56, 64, 1, 0},
    {64, 72, 1, 0}, {72, 80, 1, 0}, {80, 88, 1, 0}, {88, 96, 2, 1},
    {96, 96, 1, 0}, {96, 96, 1, 0}, {96, 96, 1, 0}, {96, 96, 1, 0},
};

/* conv params for one layer; method: FIV_CONV2D_STD/DEPTHWISE/POINTWISE */
static fiv_conv2d_params convp(int method, int kx, int ky, int st,
                               int pt, int pb, int pl, int pr,
                               int cin, int cout, int bias) {
    fiv_conv2d_params p;
    memset(&p, 0, sizeof(p));
    p.conv2d_method = method;
    p.kernel_size_x = kx;
    p.kernel_size_y = ky;
    p.stride = st;
    p.padding_method = 0;   /* zero fill */
    p.input_channels = cin;
    p.output_channels = cout;
    p.bias = bias;
    p.pad_top = pt; p.pad_bottom = pb; p.pad_left = pl; p.pad_right = pr;
    return p;
}

static int add(void* net, int type, int src, void* params) {
    int id = ((fiv_nn_network_context*)net)->node_count;
    fiv_ret r = fiv_neural_network_add_node(net, type, src, id, params);
    return r == FIV_RET_OK ? id : -1;
}

static int add_multi(void* net, int type, const int* srcs, int n, void* params) {
    int id = ((fiv_nn_network_context*)net)->node_count;
    fiv_ret r = fiv_neural_network_add_node_multi(net, type, srcs, n, id, params);
    return r == FIV_RET_OK ? id : -1;
}

/* permute a TFLite [co,kh,kw,ci] blob into the fiv [co,ci,kh,kw] node layout */
static void load_conv(const Weights* W, fiv_conv2d_node* cn, const char* kname, const char* bname) {
    const float* k = weights_get(W, kname, NULL, NULL);
    const float* b = weights_get(W, bname, NULL, NULL);
    if (!k || !b) { fprintf(stderr, "missing weights %s/%s\n", kname, bname); exit(1); }
    int co = cn->params.output_channels;
    int C_in = cn->params.input_channels;
    int kh = cn->params.kernel_size_y, kw = cn->params.kernel_size_x;
    float* w = cn->weight->data.fl;
    if (cn->params.conv2d_method == FIV_CONV2D_DEPTHWISE) {
        /* blob layout [1, kh, kw, ci]: flat = (ky*kw+kx)*C_in + c (no co term) */
        for (int oc = 0; oc < co; oc++)
            for (int ky = 0; ky < kh; ky++)
                for (int kx = 0; kx < kw; kx++)
                    w[(oc * kh + ky) * kw + kx] = k[(ky * kw + kx) * C_in + oc];
    } else {
        for (int oc = 0; oc < co; oc++)
            for (int ic = 0; ic < C_in; ic++)
                for (int ky = 0; ky < kh; ky++)
                    for (int kx = 0; kx < kw; kx++)
                        w[((oc * C_in + ic) * kh + ky) * kw + kx] =
                            k[((oc * kh + ky) * kw + kx) * C_in + ic];
    }
    if (cn->bias) memcpy(cn->bias->data.fl, b, (size_t)co * sizeof(float));
}

/* One graph: stem + blocks 0..15; the heads attach as fan-out branches at
   block 10 (reg8/cls8, act11) and block 15 (reg16/cls16, act16). A single
   inference pass computes every node's output; the four head tensors are
   read back from the recorded node ids. */
typedef struct {
    int reg8_node, cls8_node;
    int reg16_node, cls16_node;
    int block_relu[16];   /* node id of each block's trailing relu */
    int stem_relu;
} NetIds;

static void build_net(void** netp, const Weights* W, NetIds* ids) {
    void* net = fiv_create_neural_network();
    if (!net) { fprintf(stderr, "create net failed\n"); exit(1); }
    *netp = net;
    ids->reg8_node = ids->cls8_node = -1;
    ids->reg16_node = ids->cls16_node = -1;

    /* stem: 5x5 s2 pad(1,2,1,2), 3 -> 24, relu */
    fiv_conv2d_params cp = convp(FIV_CONV2D_STD, 5, 5, 2, 1, 2, 1, 2, 3, 24, 1);
    int n1 = add(net, FIV_NN_NODE_CONV2D_STD, 0, &cp);
    load_conv(W, (fiv_conv2d_node*)((fiv_nn_network_context*)net)->nodes[n1].op,
              "conv2d/Kernel", "conv2d/Bias");
    int n2 = add(net, FIV_NN_NODE_RELU, n1, NULL);
    ids->stem_relu = n2;
    int feats = n2;

    for (int i = 0; i < 16; i++) {
        int in_c = BLOCK_SPECS[i][0], out_c = BLOCK_SPECS[i][1];
        int stride = BLOCK_SPECS[i][2], maxpool = BLOCK_SPECS[i][3];
        char dk[48], db[48], pk[48], pb[48];
        if (i == 0) strcpy(dk, "depthwise_conv2d/Kernel");
        else snprintf(dk, sizeof(dk), "depthwise_conv2d_%d/Kernel", i);
        if (i == 0) strcpy(db, "depthwise_conv2d/Bias");
        else snprintf(db, sizeof(db), "depthwise_conv2d_%d/Bias", i);
        snprintf(pk, sizeof(pk), "conv2d_%d/Kernel", i + 1);
        snprintf(pb, sizeof(pb), "conv2d_%d/Bias", i + 1);

        int pt = (stride == 1) ? 1 : 0;
        int pd = (stride == 1) ? 1 : 1;
        fiv_conv2d_params dcp = convp(FIV_CONV2D_DEPTHWISE, 3, 3, stride, pt, pd, pt, pd, in_c, in_c, 1);
        int nd = add(net, FIV_NN_NODE_CONV2D_DEPTHWISE, feats, &dcp);
        load_conv(W, (fiv_conv2d_node*)((fiv_nn_network_context*)net)->nodes[nd].op, dk, db);

        fiv_conv2d_params pcp = convp(FIV_CONV2D_POINTWISE, 1, 1, 1, 0, 0, 0, 0, in_c, out_c, 1);
        int np_ = add(net, FIV_NN_NODE_CONV2D_POINTWISE, nd, &pcp);
        load_conv(W, (fiv_conv2d_node*)((fiv_nn_network_context*)net)->nodes[np_].op, pk, pb);

        int res_src = feats;
        if (in_c != out_c) {
            if (maxpool) {
                int nm = add(net, FIV_NN_NODE_MAX2D, feats, NULL);
                res_src = nm;
            }
            fiv_pad_node_params pp;
            pp.output_channels = out_c;
            int npd = add(net, FIV_NN_NODE_PAD, res_src, &pp);
            res_src = npd;
        }
        int srcs[2] = { np_, res_src };
        int na_ = add_multi(net, FIV_NN_NODE_ADD, srcs, 2, NULL);
        int nr_ = add(net, FIV_NN_NODE_RELU, na_, NULL);
        ids->block_relu[i] = nr_;
        feats = nr_;

        /* heads: reg8/cls8 on act11 (block 10), reg16/cls16 on act16 (block 15) */
        if (i == 10) {
            fiv_conv2d_params h8 = convp(FIV_CONV2D_POINTWISE, 1, 1, 1, 0, 0, 0, 0, out_c, 32, 1);
            int nrh = add(net, FIV_NN_NODE_CONV2D_POINTWISE, feats, &h8);
            load_conv(W, (fiv_conv2d_node*)((fiv_nn_network_context*)net)->nodes[nrh].op,
                      "regressor_8/Kernel", "regressor_8/Bias");
            ids->reg8_node = nrh;
            fiv_conv2d_params c8 = convp(FIV_CONV2D_POINTWISE, 1, 1, 1, 0, 0, 0, 0, out_c, 2, 1);
            int nch = add(net, FIV_NN_NODE_CONV2D_POINTWISE, feats, &c8);
            load_conv(W, (fiv_conv2d_node*)((fiv_nn_network_context*)net)->nodes[nch].op,
                      "classificator_8/Kernel", "classificator_8/Bias");
            ids->cls8_node = nch;
        } else if (i == 15) {
            fiv_conv2d_params h16 = convp(FIV_CONV2D_POINTWISE, 1, 1, 1, 0, 0, 0, 0, out_c, 96, 1);
            int nrh = add(net, FIV_NN_NODE_CONV2D_POINTWISE, feats, &h16);
            load_conv(W, (fiv_conv2d_node*)((fiv_nn_network_context*)net)->nodes[nrh].op,
                      "regressor_16/Kernel", "regressor_16/Bias");
            ids->reg16_node = nrh;
            fiv_conv2d_params c16 = convp(FIV_CONV2D_POINTWISE, 1, 1, 1, 0, 0, 0, 0, out_c, 6, 1);
            int nch = add(net, FIV_NN_NODE_CONV2D_POINTWISE, feats, &c16);
            load_conv(W, (fiv_conv2d_node*)((fiv_nn_network_context*)net)->nodes[nch].op,
                      "classificator_16/Kernel", "classificator_16/Bias");
            ids->cls16_node = nch;
        }
    }
}

/* ------------------------------------------------------------------ */
static int g_pass = 0, g_fail = 0;
#define CHECK(c, msg)                                                          \
    do {                                                                       \
        if (c) { g_pass++; }                                                   \
        else   { g_fail++; printf("  [FAIL] %s @%d\n", msg, __LINE__); }       \
    } while (0)

static void cmp_tensor(const char* name, const float* a, const float* b, size_t n,
                       float tol, float* max_diff) {
    float md = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > md) md = d;
    }
    if (md > *max_diff) *max_diff = md;
    char msg[128];
    snprintf(msg, sizeof(msg), "%s max|diff|=%.3g <= %.3g", name, md, tol);
    CHECK(md <= tol, msg);
}

/* ------------------------------------------------------------------ */
/* timing helper: a single macro wraps the timed statement and reports  */
/* avg / best over N runs (CLOCK_MONOTONIC, ns resolution).             */
/* ------------------------------------------------------------------ */
#define BENCH(label, N, stmt)                                                 \
    do {                                                                       \
        struct timespec _bt0, _bt1;                                            \
        double _bsum = 0.0, _bbest = 1e9;                                      \
        for (int _br = 0; _br < (N); _br++) {                                  \
            clock_gettime(CLOCK_MONOTONIC, &_bt0);                             \
            stmt;                                                              \
            clock_gettime(CLOCK_MONOTONIC, &_bt1);                             \
            double _bms = (double)(_bt1.tv_sec - _bt0.tv_sec) * 1000.0 +       \
                          (double)(_bt1.tv_nsec - _bt0.tv_nsec) / 1e6;        \
            _bsum += _bms;                                                     \
            if (_bms < _bbest) _bbest = _bms;                                  \
        }                                                                      \
        printf("  [bench] %-26s avg %.4f ms / best %.4f ms (%d runs)\n",       \
               label, _bsum / (N), _bbest, (N));                              \
    } while (0)

static const char* node_name(int t) {
    switch (t) {
    case FIV_NN_NODE_INPUT:            return "input";
    case FIV_NN_NODE_LINEAR:           return "linear";
    case FIV_NN_NODE_RELU:             return "relu";
    case FIV_NN_NODE_RELU6:            return "relu6";
    case FIV_NN_NODE_CONV2D_STD:       return "conv2d_std(5x5)";
    case FIV_NN_NODE_CONV2D_DEPTHWISE: return "conv2d_depthwise(3x3)";
    case FIV_NN_NODE_CONV2D_POINTWISE: return "conv2d_pointwise(1x1)";
    case FIV_NN_NODE_CONV2D_SEPARABLE: return "conv2d_separable";
    case FIV_NN_NODE_FLATTEN:          return "flatten";
    case FIV_NN_NODE_MAX2D:            return "max2d";
    case FIV_NN_NODE_ADD:              return "add";
    case FIV_NN_NODE_PAD:              return "pad";
    default:                           return "unknown";
    }
}

int main(int argc, char** argv) {
    const char* img_path = argc > 1 ? argv[1]
        : "../src/reference/c_face_detect_release/15.png";
    const char* w_path = argc > 2 ? argv[2]
        : "../src/reference/c_face_detect_release/models/blazeface_weights.bin";
    const char* dump_path = "/tmp/fiv_blazeface_gold.bin";
    const float min_score = 0.5f;
    const float TOL = 2e-3f;          /* intermediate tensors */
    const float DET_TOL = 2e-2f;      /* score / normalized kp / 128-space box */

    int w, h, cn;
    unsigned char* img = stbi_load(img_path, &w, &h, &cn, 3);
    CHECK(img != NULL, "load test image");
    if (!img) return 1;

    Weights W;
    CHECK(weights_load(w_path, &W) == 1, "load weights");
    if (W.n == 0) return 1;

    /* golden: reference pipeline + intermediate dump */
    Detection ref_dets[64];
    int n_ref = blazeface_detect_dump(img, h, w, 3, &W, min_score, ref_dets, 64, dump_path);
    Gold gold;
    CHECK(gold_load(dump_path, &gold) == 1, "parse golden dump");
    if (gold.n == 0) return 1;
    printf("reference: image %s (%dx%d) -> %d faces\n", img_path, w, h, n_ref);

    /* preprocess exactly like reference run_detect() */
    double H[9];
    detection_warp_matrix(w, h, TENSOR_SIZE, H);
    unsigned char warped[TENSOR_SIZE * TENSOR_SIZE * 3];
    warp_perspective_u8(img, h, w, 3, H, TENSOR_SIZE, TENSOR_SIZE, warped, 0);
    size_t ish[4] = { 1, 3, TENSOR_SIZE, TENSOR_SIZE };
    fiv_tensor4d* input = fiv_create_tensor4d(ish, FIV_32F1);
    for (int y = 0; y < TENSOR_SIZE; y++)
        for (int x = 0; x < TENSOR_SIZE; x++)
            for (int c = 0; c < 3; c++) {
                unsigned char v = warped[((size_t)y * TENSOR_SIZE + x) * 3 + c];
                input->data.fl[(c * TENSOR_SIZE + y) * TENSOR_SIZE + x] =
                    (float)v * (2.0f / 255.0f) - 1.0f;
            }

    /* fiv_nn forward: one graph (stem + blocks 0..15 + 4 heads), one pass */
    void* net = NULL;
    NetIds ids;
    build_net(&net, &W, &ids);

    void* final = NULL;
    CHECK(fiv_nn_run_inference((fiv_nn_network_context*)net, input, &final) == FIV_RET_OK,
          "single-graph inference");

    /* ---- intermediate tensor comparison ---- */
    float max_diff = 0.0f;
    const GoldT* g_input = gold_find(&gold, "input");
    CHECK(g_input != NULL, "gold has input");
    if (g_input) cmp_tensor("input", input->data.fl, g_input->data, g_input->elems, 1e-6f, &max_diff);

    fiv_nn_network_context* nctx = (fiv_nn_network_context*)net;
    const GoldT* g_stem = gold_find(&gold, "stem");
    if (g_stem) {
        void* stem_out = nctx->nodes[ids.stem_relu].output;
        cmp_tensor("stem", ((fiv_tensor_hdr*)stem_out)->data.fl, g_stem->data,
                   g_stem->elems, TOL, &max_diff);
    }
    for (int i = 0; i < 16; i++) {
        char nm[16];
        snprintf(nm, sizeof(nm), "block_%02d", i);
        const GoldT* gt = gold_find(&gold, nm);
        if (!gt) continue;
        void* out = nctx->nodes[ids.block_relu[i]].output;
        cmp_tensor(nm, ((fiv_tensor_hdr*)out)->data.fl, gt->data, gt->elems, TOL, &max_diff);
    }

    /* ---- flatten to reg[896][16] / scores[896] exactly like the reference ---- */
    float reg[NUM_BOXES][NUM_COORDS];
    float scores[NUM_BOXES];
    {
        const fiv_tensor4d* reg8 = (const fiv_tensor4d*)nctx->nodes[ids.reg8_node].output;
        const fiv_tensor4d* cls8 = (const fiv_tensor4d*)nctx->nodes[ids.cls8_node].output;
        for (int y = 0; y < 16; y++)
            for (int xx = 0; xx < 16; xx++)
                for (int a = 0; a < 2; a++) {
                    int anchor = (y * 16 + xx) * 2 + a;
                    for (int c = 0; c < 16; c++)
                        reg[anchor][c] = reg8->data.fl[((a * 16 + c) * 16 + y) * 16 + xx];
                    scores[anchor] = cls8->data.fl[(a * 16 + y) * 16 + xx];
                }
        const fiv_tensor4d* reg16 = (const fiv_tensor4d*)nctx->nodes[ids.reg16_node].output;
        const fiv_tensor4d* cls16 = (const fiv_tensor4d*)nctx->nodes[ids.cls16_node].output;
        for (int y = 0; y < 8; y++)
            for (int xx = 0; xx < 8; xx++)
                for (int lid = 0; lid < 6; lid++) {
                    int anchor = 512 + (y * 8 + xx) * 6 + lid;
                    for (int c = 0; c < 16; c++)
                        reg[anchor][c] = reg16->data.fl[((lid * 16 + c) * 8 + y) * 8 + xx];
                    scores[anchor] = cls16->data.fl[(lid * 8 + y) * 8 + xx];
                }
    }

    const GoldT* g_reg = gold_find(&gold, "regressors");
    const GoldT* g_scores = gold_find(&gold, "scores");
    CHECK(g_reg && g_scores, "gold has regressors/scores");
    if (g_reg) cmp_tensor("regressors", &reg[0][0], g_reg->data, NUM_BOXES * NUM_COORDS, TOL, &max_diff);
    if (g_scores) cmp_tensor("scores", scores, g_scores->data, NUM_BOXES, TOL, &max_diff);

    /* ---- final detections through the same postprocess ---- */
    Detection fiv_dets[64];
    int n_fiv = postprocess(reg, scores, w, h, min_score, fiv_dets, 64);
    printf("fiv_nn: %d faces\n", n_fiv);

    CHECK(n_fiv == n_ref, "detection count matches reference");
    int nmatch = n_ref < n_fiv ? n_ref : n_fiv;
    for (int i = 0; i < nmatch; i++) {
        char msg[96];
        snprintf(msg, sizeof(msg), "det[%d] score", i);
        CHECK(fabsf(fiv_dets[i].score - ref_dets[i].score) <= 1e-3f, msg);
        snprintf(msg, sizeof(msg), "det[%d] box", i);
        CHECK(fabsf(fiv_dets[i].x - ref_dets[i].x) <= DET_TOL &&
              fabsf(fiv_dets[i].y - ref_dets[i].y) <= DET_TOL &&
              fabsf(fiv_dets[i].w - ref_dets[i].w) <= DET_TOL &&
              fabsf(fiv_dets[i].h - ref_dets[i].h) <= DET_TOL, msg);
        snprintf(msg, sizeof(msg), "det[%d] kps", i);
        int kps_ok = 1;
        for (int k = 0; k < NUM_KEYPOINTS && kps_ok; k++)
            if (fabsf(fiv_dets[i].kps[k][0] - ref_dets[i].kps[k][0]) > 1e-3f ||
                fabsf(fiv_dets[i].kps[k][1] - ref_dets[i].kps[k][1]) > 1e-3f)
                kps_ok = 0;
        CHECK(kps_ok, msg);
    }

    printf("  max intermediate |diff| vs reference: %.3g (tol %.3g)\n", max_diff, TOL);
    printf("  reference det[0]: score=%.6f x=%.4f y=%.4f w=%.4f h=%.4f\n",
           ref_dets[0].score, ref_dets[0].x, ref_dets[0].y, ref_dets[0].w, ref_dets[0].h);
    printf("  fiv_nn   det[0]: score=%.6f x=%.4f y=%.4f w=%.4f h=%.4f\n",
           fiv_dets[0].score, fiv_dets[0].x, fiv_dets[0].y, fiv_dets[0].w, fiv_dets[0].h);

    /* ---- inference timing: warm up, then average N runs ---- */
    {
        const int WARM = 3, N = 20;
        for (int w = 0; w < WARM; w++)
            fiv_nn_run_inference((fiv_nn_network_context*)net, input, &final);

        /* whole forward (engine bench disabled => clean wall time) */
        BENCH("fiv_nn forward", N,
              fiv_nn_run_inference((fiv_nn_network_context*)net, input, &final));
        /* reference full detect (preprocess + forward + decode + NMS) */
        BENCH("reference detect", 5,
              blazeface_detect(img, h, w, 3, &W, min_score, ref_dets, 64));

        /* precise per-module timing: engine instruments every node's forward
           time keyed by node_type, no estimation. Summed over N runs, /N. */
        double mod[FIV_NN_NODE_TYPE_NUM] = {0};
        fiv_nn_bench_enable(net);
        const int MN = 20;
        for (int r = 0; r < MN; r++) {
            fiv_nn_run_inference((fiv_nn_network_context*)net, input, &final);
            double cur[FIV_NN_NODE_TYPE_NUM];
            fiv_nn_get_bench(net, cur, FIV_NN_NODE_TYPE_NUM);
            for (int t = 0; t < FIV_NN_NODE_TYPE_NUM; t++) mod[t] += cur[t];
        }
        double total = 0.0;
        for (int t = 0; t < FIV_NN_NODE_TYPE_NUM; t++) mod[t] /= MN;
        for (int t = 0; t < FIV_NN_NODE_TYPE_NUM; t++) total += mod[t];
        printf("  per-module (avg over %d runs, engine-instrumented):\n", MN);
        for (int t = 0; t < FIV_NN_NODE_TYPE_NUM; t++) {
            if (mod[t] <= 0.0) continue;
            printf("    %-22s %7.4f ms  (%5.1f%%)\n",
                   node_name(t), mod[t], total > 0 ? mod[t] / total * 100.0 : 0.0);
        }
        printf("    %-22s %7.4f ms  (100.0%%)\n", "FORWARD TOTAL", total);
    }

    /* cleanup */
    fiv_release_neural_network((void**)&net);
    fiv_release_tensor((void**)&input);
    for (int i = 0; i < gold.n; i++) free(gold.items[i].data);
    free(gold.items);
    remove(dump_path);
    weights_free(&W);
    stbi_image_free(img);

    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
