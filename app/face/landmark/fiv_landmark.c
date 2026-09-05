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
 * FaceMesh 478 landmark network builder (see fiv_landmark.h).
 *
 * Parses the reference "LMNT" binary produced by export_landmark_net.py
 * (TFLite float32 NHWC graph) and assembles it into a FastIV network. The
 * node list in the blob is already topologically ordered, so nodes are added
 * to the engine in file order (fiv_neural_network_add_node requires this).
 *
 * Constant tensors and DEQUANTIZE:
 *   Weights, biases and PReLU alphas live in the small-index constant tensors
 *   (t1..t~10) and are referenced by conv/depthwise/prelu nodes through a
 *   DEQUANTIZE (op 6) node whose OUTPUT carries a large logical tensor id with
 *   no data. Every such constant input is resolved through the dequantize map
 *   (output -> input) so the data is read from the real source tensor.
 *
 * Weight layout transposition (reference NHWC -> FastIV NCHW):
 *   - dense conv kernel [co, kh, kw, ci] -> [co, ci, kh, kw]
 *   - depthwise kernel [1, kh, kw, c_out]  -> [c_out, 1, kh, kw] (oc-major);
 *     c_out = mult * c_in, the engine derives mult = c_out / c_in itself.
 * Both biases (per-output-channel) and PReLU alphas are copied verbatim.
 *
 * Padding: every conv/depthwise uses TFLite SAME padding, computed by the
 * reference same_pad(): out = ceil(n/s), pt = ((out-1)*s + k - n)/2 with the
 * remainder on the end. For stride-2 kernels (256->128 stem, 2->1 heads) this
 * yields pt = 0; even-size 3x3 s1 layers yield pt = 1. These literal start
 * pads are passed via pad_top/pad_left and the engine's generic kernels apply
 * exactly this (out-of-range taps are zero skipped). 1x1 pointwise layers
 * ignore padding entirely (no spatial neighborhood).
 *
 * Inputs / outputs:
 *   The graph has one input tensor (t0: 256x256x3, NHWC) fed to node 0 and
 *   three recorded outputs: t473 (1434-d landmark logits), t472 (presence
 *   logit, before the final LOGISTIC) and t475 (a reshaped, unused copy).
 *   Only t473 / t472 are recorded on the graph for later reading via
 *   fiv_neural_network_get_node_output().
 */

#include "fiv_landmark.h"

#include "fiv_common.h"
#include "fiv_ctensor.h"
#include "fiv_nn_infer.h"
#include "fiv_nn_conv2d.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- LMNT blob reader (little-endian) ---- */

typedef struct {
    iv8u*          data;
    size_t         size;
    size_t         pos;
} fiv_lm_reader;

static int fiv_lm_rd_u32(fiv_lm_reader* r, iv32u* v)
{
    if (r->pos + 4 > r->size) return 0;
    *v = (iv32u)r->data[r->pos] | ((iv32u)r->data[r->pos + 1] << 8) |
         ((iv32u)r->data[r->pos + 2] << 16) | ((iv32u)r->data[r->pos + 3] << 24);
    r->pos += 4;
    return 1;
}

static int fiv_lm_rd_i32(fiv_lm_reader* r, iv32s* v)
{
    iv32u u;
    if (!fiv_lm_rd_u32(r, &u)) return 0;
    *v = (iv32s)u;
    return 1;
}

/* ---- tensor table (constants live here; activations are runtime-only) ----
   Constants are stored in the api tensor type fiv_tensor4d (api/fiv_ctensor.h).
   A blob tensor holds up to 4 dims; the iterator's rank is tracked alongside in
   a parallel array (fiv_tensor4d does not expose a rank field). Data is stored
   with fiv_create-style allocated buffers, attached via the data union
   (data.fl for float32, data.ptr32s for int32). */

/* ---- node table ---- */

typedef struct {
    int code;    /* TFLite op id (see fiv_landmark.c op map) */
    int n_in;
    int* ins;
    int n_out;
    int* outs;
    int sh, sw, kh, kw;   /* stride / kernel */
    int mult;             /* depthwise channel multiplier */
    int axis;             /* concat axis (unused for this model) */
} fiv_lm_node;

#define FIV_LM_TF_ADD       0
#define FIV_LM_TF_CONCAT    2
#define FIV_LM_TF_CONV2D    3
#define FIV_LM_TF_DEPTHWISE 4
#define FIV_LM_TF_DEQUANTIZE 6
#define FIV_LM_TF_LOGISTIC 14
#define FIV_LM_TF_MAXPOOL  17
#define FIV_LM_TF_RESHAPE  22
#define FIV_LM_TF_PAD      34
#define FIV_LM_TF_PRELU    54

/* ---- helpers ---- */

static iv32u fiv_lm_elems(const fiv_tensor4d* t, iv32u ndim)
{
    iv32u e = 1;
    for (iv32u k = 0; k < ndim; k++) e *= (iv32u)t->shapes[k];
    return e;
}

/* data.fl and data.ptr32s share the same union storage, so a single free covers
   both the float32 and int32 tensor buffers. */
static void fiv_lm_free_tensors(fiv_tensor4d* t, int n)
{
    if (!t) return;
    for (int i = 0; i < n; i++) fiv_free(t[i].data.fl);
    fiv_free(t);
}

/* One-shot env gate: FIV_BENCH_LM_DUMP=1 prints each conv node (+shape) as the
   graph is built, so we can see exactly which CONV2D nodes run in inference. */
static int fiv_lm_dump_nodes(void)
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("FIV_BENCH_LM_DUMP"); v = (e && e[0] == '1') ? 1 : 0; }
    return v;
}

static void fiv_lm_free_nodes(fiv_lm_node* nd, int n)
{
    if (!nd) return;
    for (int i = 0; i < n; i++) { fiv_free(nd[i].ins); fiv_free(nd[i].outs); }
    fiv_free(nd);
}

/* Gather the real (non-identity) nodes in order, remapping tensor indices so
   the returned array's rows refer to the same original tensor ids. Identity
   ops (DEQUANTIZE/RESHAPE) alias their input, so skipping them keeps the
   producer-consumer chain valid (no real node consumes a DEQUANTIZE/RESHAPE
   output in this model; verified against the blob). */
static int fiv_lm_collect_real(const fiv_lm_node* nodes, int n, fiv_lm_node** out_real)
{
    fiv_lm_node* real = (fiv_lm_node*)fiv_malloc(sizeof(fiv_lm_node) * (size_t)n);
    if (!real) return -1;
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        int c = nodes[i].code;
        if (c == FIV_LM_TF_DEQUANTIZE || c == FIV_LM_TF_RESHAPE) continue;
        real[cnt++] = nodes[i];
    }
    *out_real = real;
    return cnt;
}

/* Map a tensor id to the real-node index that produced it (-1 = graph input). */
static int fiv_lm_producer(const fiv_lm_node* real, int cnt, int tensor_id)
{
    for (int i = 0; i < cnt; i++)
        for (int k = 0; k < real[i].n_out; k++)
            if (real[i].outs[k] == tensor_id) return i;
    return -1;
}

/* Follow the DEQUANTIZE chain to the tensor that actually holds data. */
static int fiv_lm_data_tensor(int ti, const int* dq, int nt)
{
    for (int guard = 0; guard < nt; guard++) {
        if (ti < 0 || ti >= nt || dq[ti] < 0) break;
        ti = dq[ti];
    }
    return ti;
}

/* Map an activation tensor to its producer's FastIV node id (node 0 = graph
   INPUT). Real index p owns FastIV node id p + 1 because node 0 is reserved. */
static int fiv_lm_src(const fiv_lm_node* real, int cnt, int tensor_id, int gin)
{
    int p = fiv_lm_producer(real, cnt, tensor_id);
    if (p >= 0) return p + 1;
    if (tensor_id == gin) return 0;   /* graph input feeds node 0 */
    return -1;
}

/* ---- weight transposition ---- */

/* Dense conv: [co, kh, kw, ci] -> [co, ci, kh, kw]. */
static void fiv_lm_conv_transpose(ivf32* dst, const ivf32* src,
                                  int co, int ci, int kh, int kw)
{
    for (int oc = 0; oc < co; oc++)
        for (int ic = 0; ic < ci; ic++)
            for (int ky = 0; ky < kh; ky++)
                for (int kx = 0; kx < kw; kx++) {
                    size_t s = (((size_t)oc * kh + ky) * kw + kx) * ci + ic;
                    size_t d = (((size_t)oc * ci + ic) * kh + ky) * kw + kx;
                    dst[d] = src[s];
                }
}

/* Depthwise: [1, kh, kw, c] -> [c, 1, kh, kw] (oc-major). */
static void fiv_lm_dw_transpose(ivf32* dst, const ivf32* src,
                                int c, int kh, int kw)
{
    for (int oc = 0; oc < c; oc++)
        for (int ky = 0; ky < kh; ky++)
            for (int kx = 0; kx < kw; kx++) {
                size_t s = ((size_t)ky * kw + kx) * c + oc;
                size_t d = ((size_t)oc * kh + ky) * kw + kx;
                dst[d] = src[s];
            }
}

/* ---- graph builder ---- */

/* TFLite SAME padding split (matches reference same_pad(): the total pad is
   (out-1)*s + k - n with out = ceil(n/s); the remainder goes on the end, so
   the start is floor(total/2) and the end is total - start). */
static void fiv_lm_same_pad(int n, int k, int s, int* p0, int* p1)
{
    int out = (n + s - 1) / s;
    int tot = (out - 1) * s + k - n;
    if (tot < 0) tot = 0;
    *p0 = tot / 2;
    *p1 = tot - *p0;
}

/* Spatial size (H, W) of the tensor feeding a conv, from the producer's
   recorded NHWC output shape [1, H, W, C]; falls back to the 256x256 input. */
static void fiv_lm_input_hw(const fiv_lm_node* real, int cnt,
                            const fiv_tensor4d* tensors, const iv32u* trank, int nt,
                            int tensor_id, int* ih, int* iw)
{
    int p = fiv_lm_producer(real, cnt, tensor_id);
    if (p >= 0) {
        int oti = real[p].outs[0];
        if (oti >= 0 && oti < nt && trank[oti] == 4) {
            *ih = (int)tensors[oti].shapes[1];
            *iw = (int)tensors[oti].shapes[2];
        }
    }
}

fiv_landmark_graph* fiv_create_landmark_graph(const char* path)
{
    fiv_landmark_graph* graph = NULL;
    FILE* f = NULL;
    iv8u* buf = NULL;
    long fsz = 0;

    f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "fiv_landmark: cannot open %s\n", path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0 || (fsz = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f); return NULL;
    }
    buf = (iv8u*)fiv_malloc((size_t)fsz);
    if (!buf || fread(buf, 1, (size_t)fsz, f) != (size_t)fsz) { fclose(f); fiv_free(buf); return NULL; }
    fclose(f);
    f = NULL;

    fiv_lm_reader rd = { buf, (size_t)fsz, 0 };
    iv32u magic = 0, nt = 0, nn = 0, n_gin = 0, n_gout = 0;
    if (!fiv_lm_rd_u32(&rd, &magic) || magic != FIV_LM_MAGIC) goto fail;
    if (!fiv_lm_rd_u32(&rd, &nt) || nt == 0) goto fail;
    fiv_tensor4d* tensors = (fiv_tensor4d*)fiv_calloc(nt, sizeof(fiv_tensor4d));
    if (!tensors) goto fail;
    iv32u* trank = (iv32u*)fiv_calloc(nt, sizeof(iv32u));   /* per-tensor rank */
    if (!trank) { fiv_free(tensors); goto fail; }
    for (iv32u i = 0; i < nt; i++) {
        iv32u idx, ndim, dt, ln;
        if (!fiv_lm_rd_u32(&rd, &idx) || !fiv_lm_rd_u32(&rd, &ndim)) goto fail_tt;
        if (idx >= nt || ndim > 4) goto fail_tt;
        fiv_tensor4d* t = &tensors[idx];
        trank[idx] = ndim;
        for (int k = 0; k < 4; k++) {
            iv32u d;
            if (!fiv_lm_rd_u32(&rd, &d)) goto fail_tt;
            t->shapes[k] = d;
        }
        if (!fiv_lm_rd_u32(&rd, &dt) || !fiv_lm_rd_u32(&rd, &ln)) goto fail_tt;
        t->dtype = (dt == 2) ? FIV_32S1 : FIV_32F1;
        if (ln) {
            size_t bytes = (size_t)ln * 4;
            if (rd.pos + bytes > rd.size) goto fail_tt;
            t->data.fl = (ivf32*)fiv_malloc(bytes);
            if (!t->data.fl) goto fail_tt;
            memcpy(t->data.fl, rd.data + rd.pos, bytes);
            rd.pos += bytes;
        }
    }
    if (!fiv_lm_rd_u32(&rd, &nn) || nn == 0) goto fail_t;

    fiv_lm_node* nodes = (fiv_lm_node*)fiv_calloc(nn, sizeof(fiv_lm_node));
    if (!nodes) goto fail_t;
    for (iv32u i = 0; i < nn; i++) {
        fiv_lm_node* nd = &nodes[i];
        iv32u code, nin, nout;
        if (!fiv_lm_rd_u32(&rd, &code) || !fiv_lm_rd_u32(&rd, &nin)) goto fail_n;
        nd->code = (int)code;
        nd->n_in = (int)nin;
        if (nin) {
            nd->ins = (int*)fiv_malloc(sizeof(int) * nin);
            if (!nd->ins) goto fail_n;
            for (int k = 0; k < (int)nin; k++)
                if (!fiv_lm_rd_i32(&rd, &nd->ins[k])) goto fail_n;
        }
        if (!fiv_lm_rd_u32(&rd, &nout)) goto fail_n;
        nd->n_out = (int)nout;
        if (nout) {
            nd->outs = (int*)fiv_malloc(sizeof(int) * nout);
            if (!nd->outs) goto fail_n;
            for (int k = 0; k < (int)nout; k++)
                if (!fiv_lm_rd_i32(&rd, &nd->outs[k])) goto fail_n;
        }
        if (!fiv_lm_rd_i32(&rd, &nd->sh) || !fiv_lm_rd_i32(&rd, &nd->sw) ||
            !fiv_lm_rd_i32(&rd, &nd->kh) || !fiv_lm_rd_i32(&rd, &nd->kw) ||
            !fiv_lm_rd_i32(&rd, &nd->mult) || !fiv_lm_rd_i32(&rd, &nd->axis)) goto fail_n;
    }
    if (!fiv_lm_rd_u32(&rd, &n_gin)) goto fail_n;
    int* gin = (int*)fiv_malloc(sizeof(int) * (n_gin ? n_gin : 1));
    if (!gin) goto fail_n;
    for (iv32u k = 0; k < n_gin; k++)
        if (!fiv_lm_rd_i32(&rd, &gin[k])) { fiv_free(gin); goto fail_n; }

    if (!fiv_lm_rd_u32(&rd, &n_gout)) goto fail_n;
    int* gout = (int*)fiv_malloc(sizeof(int) * (n_gout ? n_gout : 1));
    if (!gout) goto fail_n;
    for (iv32u k = 0; k < n_gout; k++)
        if (!fiv_lm_rd_i32(&rd, &gout[k])) { fiv_free(gout); goto fail_n; }

    /* ---- optional metadata trailer (LMNT-MTD; 256 model omits it) ---- */
    iv32u n_lm = 478, ts = 256, ih = 256, iw = 256;
    iv32s out_lm_id = 473, out_conf_id = 472;
    {
        iv32u mmagic = 0;
        size_t save = rd.pos;
        if (fiv_lm_rd_u32(&rd, &mmagic) && mmagic == FIV_LM_META_MAGIC) {
            if (!fiv_lm_rd_u32(&rd, &n_lm) || !fiv_lm_rd_u32(&rd, &ts) ||
                !fiv_lm_rd_u32(&rd, &ih)  || !fiv_lm_rd_u32(&rd, &iw) ||
                !fiv_lm_rd_i32(&rd, &out_lm_id) || !fiv_lm_rd_i32(&rd, &out_conf_id)) {
                n_lm = 478; ts = 256; ih = 256; iw = 256;
                out_lm_id = 473; out_conf_id = 472;
            }
        } else {
            rd.pos = save;   /* no trailer; keep defaults */
        }
    }

    /* ---- DEQUANTIZE map: logical constant tensor id -> data source id ---- */
    int* dq = (int*)fiv_malloc(sizeof(int) * (size_t)nt);
    if (!dq) { fiv_free(gout); goto fail_n; }
    for (iv32u i = 0; i < nt; i++) dq[i] = -1;
    for (iv32u i = 0; i < nn; i++) {
        if (nodes[i].code == FIV_LM_TF_DEQUANTIZE && nodes[i].n_in > 0)
            for (int k = 0; k < nodes[i].n_out; k++)
                dq[nodes[i].outs[k]] = nodes[i].ins[0];
    }

    /* ---- collect real nodes (skip DEQUANTIZE / RESHAPE) ---- */
    fiv_lm_node* real = NULL;
    int rcnt = fiv_lm_collect_real(nodes, (int)nn, &real);
    if (rcnt < 0) { fiv_free(dq); fiv_free(gout); goto fail_n; }

    int gin_id = (n_gin > 0) ? gin[0] : 0;

    /* ---- build the FastIV graph ---- */
    graph = (fiv_landmark_graph*)fiv_calloc(1, sizeof(fiv_landmark_graph));
    if (!graph) { fiv_free(dq); fiv_free(gout); fiv_lm_free_nodes(real, rcnt); goto fail_n; }
    graph->net = fiv_create_neural_network();
    if (!graph->net) { fiv_free(dq); fiv_free(gout); fiv_lm_free_nodes(real, rcnt); fiv_free(graph); graph = NULL; goto fail_n; }

    graph->lm_node   = -1;
    graph->conf_node = -1;
    graph->n_landmarks = (int)n_lm;
    graph->tensor_size = (int)ts;
    graph->in_h        = (int)ih;
    graph->in_w        = (int)iw;
    graph->out_lm_id   = out_lm_id;
    graph->out_conf_id = out_conf_id;
    int next = 1;   /* node 0 is the implicit INPUT */

    for (int i = 0; i < rcnt; i++) {
        const fiv_lm_node* nd = &real[i];
        int c = nd->code;
        int node_id = next++;
        fiv_ret r = FIV_RET_OK;
        int src0 = -1;

        switch (c) {
        case FIV_LM_TF_CONV2D: {
            int wti = fiv_lm_data_tensor(nd->ins[1], dq, (int)nt);
            int bti = fiv_lm_data_tensor(nd->ins[2], dq, (int)nt);
            const fiv_tensor4d* wt = &tensors[wti];
            const fiv_tensor4d* bt = &tensors[bti];
            if (!wt->data.fl || !bt->data.fl) { r = FIV_RET_ERR_DATA_UNINITED; break; }
            int ci = (int)wt->shapes[3], co = (int)wt->shapes[0];
            int kh = (int)wt->shapes[1], kw = (int)wt->shapes[2];
            int sh = nd->sh, sw = nd->sw;
            int ih = graph->in_h, iw = graph->in_w;
            fiv_lm_input_hw(real, rcnt, tensors, trank, (int)nt, nd->ins[0], &ih, &iw);
            int pt0, pt1, pl0, pl1;
            fiv_lm_same_pad(ih, kh, sh, &pt0, &pt1);
            fiv_lm_same_pad(iw, kw, sw, &pl0, &pl1);
            fiv_conv2d_params p;
            memset(&p, 0, sizeof(p));
            p.conv2d_method   = FIV_CONV2D_STD;
            p.kernel_size_x   = kw;
            p.kernel_size_y   = kh;
            p.stride          = sh;   /* square stride in this model */
            p.padding_method  = 0;    /* zero pad */
            p.input_channels  = ci;
            p.output_channels = co;
            p.bias            = 1;
            p.pad_top    = pt0;
            p.pad_bottom = pt1;
            p.pad_left   = pl0;
            p.pad_right  = pl1;

            src0 = fiv_lm_src(real, rcnt, nd->ins[0], gin_id);
            if (src0 < 0) { r = FIV_RET_ERR_PARA; break; }
            r = fiv_neural_network_add_node(graph->net, FIV_NN_NODE_CONV2D_STD,
                                            src0, node_id, &p);
            if (fiv_lm_dump_nodes()) {
                int oh = (ih + pt0 + pt1 - kh) / sh + 1;
                int ow = (iw + pl0 + pl1 - kw) / sw + 1;
                fprintf(stderr, "[lm-dump] CONV2D_STD node=%d kernel=%dx%d stride=%d %d->%dch input=[%dx%d] out=[%dx%d]\n",
                        node_id, kh, kw, sh, ci, co, ih, iw, oh, ow);
            }
            if (r == FIV_RET_OK) {
                size_t nw = (size_t)co * ci * kh * kw;
                ivf32* w = (ivf32*)fiv_malloc(nw * sizeof(ivf32));
                if (!w) { r = FIV_RET_ERR_MEM; break; }
                fiv_lm_conv_transpose(w, wt->data.fl, co, ci, kh, kw);
                r = fiv_neural_network_set_node_weight(graph->net, node_id, w);
                fiv_free(w);
                if (r == FIV_RET_OK)
                    r = fiv_neural_network_set_node_bias(graph->net, node_id, bt->data.fl);
            }
            break;
        }
        case FIV_LM_TF_DEPTHWISE: {
            int wti = fiv_lm_data_tensor(nd->ins[1], dq, (int)nt);
            int bti = fiv_lm_data_tensor(nd->ins[2], dq, (int)nt);
            const fiv_tensor4d* wt = &tensors[wti];
            const fiv_tensor4d* bt = &tensors[bti];
            if (!wt->data.fl || !bt->data.fl) { r = FIV_RET_ERR_DATA_UNINITED; break; }
            int mult = nd->mult > 0 ? nd->mult : 1;
            int co = (int)wt->shapes[3];     /* [1, kh, kw, c_out] */
            int ci = co / mult;              /* input channels = c_out / mult */
            int kh = (int)wt->shapes[1], kw = (int)wt->shapes[2];
            int sh = nd->sh, sw = nd->sw;
            int ih = graph->in_h, iw = graph->in_w;
            fiv_lm_input_hw(real, rcnt, tensors, trank, (int)nt, nd->ins[0], &ih, &iw);
            int pt0, pt1, pl0, pl1;
            fiv_lm_same_pad(ih, kh, sh, &pt0, &pt1);
            fiv_lm_same_pad(iw, kw, sw, &pl0, &pl1);
            fiv_conv2d_params p;
            memset(&p, 0, sizeof(p));
            p.conv2d_method   = FIV_CONV2D_DEPTHWISE;
            p.kernel_size_x   = kw;
            p.kernel_size_y   = kh;
            p.stride          = sh;
            p.padding_method  = 0;
            p.input_channels  = ci;
            p.output_channels = co;
            p.bias            = 1;
            p.pad_top    = pt0;
            p.pad_bottom = pt1;
            p.pad_left   = pl0;
            p.pad_right  = pl1;

            src0 = fiv_lm_src(real, rcnt, nd->ins[0], gin_id);
            if (src0 < 0) { r = FIV_RET_ERR_PARA; break; }
            r = fiv_neural_network_add_node(graph->net, FIV_NN_NODE_CONV2D_DEPTHWISE,
                                            src0, node_id, &p);
            if (fiv_lm_dump_nodes()) {
                int oh = (ih + pt0 + pt1 - kh) / sh + 1;
                int ow = (iw + pl0 + pl1 - kw) / sw + 1;
                fprintf(stderr, "[lm-dump] DEPTHWISE node=%d kernel=%dx%d stride=%d %d->%dch input=[%dx%d] out=[%dx%d]\n",
                        node_id, kh, kw, sh, ci, co, ih, iw, oh, ow);
            }
            if (r == FIV_RET_OK) {
                size_t nw = (size_t)co * kh * kw;
                ivf32* w = (ivf32*)fiv_malloc(nw * sizeof(ivf32));
                if (!w) { r = FIV_RET_ERR_MEM; break; }
                fiv_lm_dw_transpose(w, wt->data.fl, co, kh, kw);
                r = fiv_neural_network_set_node_weight(graph->net, node_id, w);
                fiv_free(w);
                if (r == FIV_RET_OK)
                    r = fiv_neural_network_set_node_bias(graph->net, node_id, bt->data.fl);
            }
            break;
        }
        case FIV_LM_TF_PRELU: {
            int ati = fiv_lm_data_tensor(nd->ins[1], dq, (int)nt);
            const fiv_tensor4d* at = &tensors[ati];
            if (!at->data.fl) { r = FIV_RET_ERR_DATA_UNINITED; break; }
            int c = (int)fiv_lm_elems(at, trank[ati]);   /* [1,1,c] / [1,c] / [c] -> c */
            src0 = fiv_lm_src(real, rcnt, nd->ins[0], gin_id);
            if (src0 < 0) { r = FIV_RET_ERR_PARA; break; }
            fiv_prelu_node_params pp;
            memset(&pp, 0, sizeof(pp));
            pp.channels = c;
            pp.alpha    = at->data.fl;
            r = fiv_neural_network_add_node(graph->net, FIV_NN_NODE_PRELU, src0, node_id, &pp);
            break;
        }
        case FIV_LM_TF_ADD: {
            int* srcs = (int*)fiv_malloc(sizeof(int) * (size_t)nd->n_in);
            if (!srcs) { r = FIV_RET_ERR_MEM; break; }
            int ok = 1;
            for (int k = 0; k < nd->n_in; k++) {
                srcs[k] = fiv_lm_src(real, rcnt, nd->ins[k], gin_id);
                if (srcs[k] < 0) { ok = 0; break; }
            }
            if (!ok) { fiv_free(srcs); r = FIV_RET_ERR_PARA; break; }
            r = fiv_neural_network_add_node_multi(graph->net, FIV_NN_NODE_ADD,
                                                  srcs, nd->n_in, node_id, NULL);
            fiv_free(srcs);
            break;
        }
        case FIV_LM_TF_MAXPOOL: {
            /* fiv_max_2d_node is hardwired to a 2x2 stride-2 window (h/2, w/2),
               which is exactly this model's 6 MAXPOOL ops; no params needed. */
            src0 = fiv_lm_src(real, rcnt, nd->ins[0], gin_id);
            if (src0 < 0) { r = FIV_RET_ERR_PARA; break; }
            r = fiv_neural_network_add_node(graph->net, FIV_NN_NODE_MAX2D, src0, node_id, NULL);
            break;
        }
        case FIV_LM_TF_PAD: {
            /* [4,2] int paddings (N,H,W,C): this model pads channels only, so
               the channel-end amount lives at index [3][1] = 16/32/64. */
            int pti = fiv_lm_data_tensor(nd->ins[1], dq, (int)nt);
            const fiv_tensor4d* pt = &tensors[pti];
            if (!pt->data.ptr32s) { r = FIV_RET_ERR_DATA_UNINITED; break; }
            int pc1 = pt->data.ptr32s[7];
            int ic = 3;   /* fallback: graph input has 3 channels */
            int pnode = fiv_lm_producer(real, rcnt, nd->ins[0]);
            if (pnode >= 0) {
                int oti = real[pnode].outs[0];
                if (oti >= 0 && oti < (int)nt && trank[oti] == 4)
                    ic = (int)tensors[oti].shapes[3];
            }
            src0 = fiv_lm_src(real, rcnt, nd->ins[0], gin_id);
            if (src0 < 0) { r = FIV_RET_ERR_PARA; break; }
            fiv_pad_node_params pp;
            memset(&pp, 0, sizeof(pp));
            pp.output_channels = ic + pc1;
            r = fiv_neural_network_add_node(graph->net, FIV_NN_NODE_PAD, src0, node_id, &pp);
            break;
        }
        case FIV_LM_TF_LOGISTIC: {
            src0 = fiv_lm_src(real, rcnt, nd->ins[0], gin_id);
            if (src0 < 0) { r = FIV_RET_ERR_PARA; break; }
            r = fiv_neural_network_add_node(graph->net, FIV_NN_NODE_SIGMOID, src0, node_id, NULL);
            break;
        }
        default:
            fprintf(stderr, "fiv_landmark: unsupported op %d at real node %d\n", c, i);
            fiv_free(dq);
            fiv_free(gout);
            goto fail_graph;
        }

        if (r != FIV_RET_OK) {
            fprintf(stderr, "fiv_landmark: add node %d (op %d) failed: %d\n", i, c, (int)r);
            fiv_free(dq);
            fiv_free(gout);
            goto fail_graph;
        }

        /* record outputs of interest (by model output tensor id) */
        for (int k = 0; k < nd->n_out; k++) {
            if (nd->outs[k] == graph->out_lm_id)   graph->lm_node   = node_id;
            if (nd->outs[k] == graph->out_conf_id) graph->conf_node = node_id;
        }
    }

    fiv_free(dq);
    fiv_free(gout);
    fiv_lm_free_nodes(real, rcnt);
    fiv_lm_free_tensors(tensors, (int)nt);
    fiv_free(trank);
    fiv_free(buf);

    if (graph->lm_node < 0 || graph->conf_node < 0) {
        fprintf(stderr, "fiv_landmark: missing output node(s)\n");
        fiv_release_landmark_graph(graph);
        return NULL;
    }
    return graph;

fail_graph:
    fiv_lm_free_nodes(real, rcnt);
    fiv_lm_free_tensors(tensors, (int)nt);
    fiv_free(trank);
    fiv_free(buf);
    fiv_release_landmark_graph(graph);
    graph = NULL;
    return NULL;
fail_n:
    fiv_lm_free_nodes(nodes, (int)nn);
    fiv_lm_free_tensors(tensors, (int)nt);
    fiv_free(trank);
    fiv_free(buf);
    return NULL;
fail_t:
    fiv_lm_free_tensors(tensors, (int)nt);
    fiv_free(trank);
    fiv_free(buf);
    return NULL;
fail_tt:
    fiv_lm_free_tensors(tensors, (int)nt);
    fiv_free(trank);
    fiv_free(buf);
    return NULL;
fail:
    fiv_free(buf);
    return NULL;
}

void fiv_release_landmark_graph(fiv_landmark_graph* graph)
{
    if (!graph) return;
    if (graph->net) fiv_release_neural_network(&graph->net);
    fiv_free(graph);
}

static ivf32 fiv_lm_sigmoidf(ivf32 x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    ivf32 e = expf(x);
    return e / (1.0f + e);
}

fiv_ret fiv_landmark_graph_run_inference(fiv_landmark_graph* graph,
                                         const fiv_tensor4d* input,
                                         ivf32 out_xyz[FIV_LM_NUM_LANDMARKS][3],
                                         ivf32* out_presence)
{
    if (!graph || !graph->net || !input || !out_xyz || !out_presence)
        return FIV_RET_ERR_PARA;

    const int n_lm    = graph->n_landmarks;
    const int tsz     = graph->tensor_size;
    if (n_lm    <= 0 || n_lm   > FIV_LM_NUM_LANDMARKS) return FIV_RET_ERR_PARA;
    if (tsz     <= 0 || tsz    > FIV_LM_TENSOR_SIZE ) return FIV_RET_ERR_PARA;

    void* final_out = NULL;
    fiv_ret r = fiv_nn_run_inference(graph->net, (void*)input, &final_out);
    if (r != FIV_RET_OK) return r;

    const fiv_tensor_hdr* lm   = (const fiv_tensor_hdr*)
        fiv_neural_network_get_node_output(graph->net, graph->lm_node);
    const fiv_tensor_hdr* conf = (const fiv_tensor_hdr*)
        fiv_neural_network_get_node_output(graph->net, graph->conf_node);
    if (!lm || !conf) return FIV_RET_ERR_UNKNOWN;
    if (lm->total_bytes < (size_t)n_lm * 3 * sizeof(ivf32))
        return FIV_RET_ERR_PARA;

    const ivf32* raw = lm->data.fl;
    for (int ld = 0; ld < n_lm; ld++) {
        out_xyz[ld][0] = raw[3 * ld]     / (ivf32)tsz;
        out_xyz[ld][1] = raw[3 * ld + 1] / (ivf32)tsz;
        out_xyz[ld][2] = raw[3 * ld + 2] / (ivf32)tsz;
    }
    *out_presence = fiv_lm_sigmoidf(conf->data.fl[0]);
    return FIV_RET_OK;
}
