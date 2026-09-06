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

/* YOLO26 NMS-free decode (reg_max=1, end2end). Mirrors the torch reference
 * (gen_ref.py -> ultralytics head.py / tal.py), non-export path, groups=1:
 *   anchors   make_anchors: per level the [2, H*W] grid + 0.5, y-major flatten
 *             (index = y*W + x); row 0 = x, row 1 = y. Levels are concatenated
 *             in stride order (yolo26n strides 8/16/32, levels P3/P4/P5).
 *   dbox      dist2bbox(xyxy): box channels l,t,r,b; x1 = ax - l, y1 = ay - t,
 *             x2 = ax + r, y2 = ay + b, in grid units then scaled by the level
 *             stride, so dbox values are final pixel coordinates.
 *   scores    sigmoid of the raw class logits.
 *   top-k     torch non-export does two EXACT passes:
 *               s1 = scores.amax(nc)                # per-anchor best class
 *               v1, i1 = s1.topk(k)                 # k anchors
 *               t  = scores[:, i1].flatten()        # (k*nc,)
 *               v2, i2 = t.topk(k)
 *               cls = i2 % nc ; anc = i1[i2 // nc]
 *             k = min(max_det, a_count), max_det = 300 (yolo26 default).
 * Output rows [x1, y1, x2, y2, score, class] (+ nm mask coefficients for the
 * Segment26 variant), score descending. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fiv_yolo26_post.h"

typedef struct {
    size_t a_off;    /* first anchor index of this level */
    size_t a_count;  /* H*W of this level */
    int    stride;
    int    width;
    int    height;
} level_info;

/* Shared decode core geometry: box heads give H/W per level (P3/P4/P5, stride
 * order); cls head supplies nc; an optional coef (mask) head set supplies nm. */
static int build_levels(const fiv_tensor4d* box[3], const fiv_tensor4d* cls[3],
                        const fiv_tensor4d* coef[3], const int strides[3],
                        level_info* lv, size_t* a_total, int* nc_out, int* nm_out)
{
    size_t total = 0;
    for (int i = 0; i < 3; i++) {
        if (!box[i] || !cls[i]) return -1;
        if (coef && !coef[i]) return -1;
        lv[i].a_off = total;
        lv[i].width  = (int)box[i]->width;
        lv[i].height = (int)box[i]->height;
        lv[i].stride = strides[i];
        lv[i].a_count = (size_t)box[i]->height * (size_t)box[i]->width;
        total += lv[i].a_count;
    }
    *a_total = total;
    *nc_out = (int)cls[0]->channels;
    *nm_out = coef ? (int)coef[0]->channels : 0;
    return 0;
}

/* softmax-free sigmoid, computed in float. */
static float fiv_sigmoidf(float x)
{
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    float e = expf(x);
    return e / (1.0f + e);
}

/* -------- step 1+2: per-anchor top-k over the best class (EXACT, groups=1) -------- */
static int select_anchors(const float* scores, size_t a_count, int nc, int k,
                          int* anchor_idx, float* best_score)
{
    /* best_score[a] = max over classes */
    for (size_t a = 0; a < a_count; a++) {
        const float* row = scores + a * (size_t)nc;
        float best = row[0];
        for (int c = 1; c < nc; c++) if (row[c] > best) best = row[c];
        best_score[a] = best;
    }
    /* Streaming top-k: start EMPTY and insert every anchor once (a strictly
       increasing scan). When full, replace the current running minimum only
       when the new anchor is strictly better, so the final set holds the k
       largest best-scores and each anchor appears at most once. */
    int filled = 0;
    for (size_t a = 0; a < a_count; a++) {
        if (filled < k) {
            anchor_idx[filled++] = (int)a;
            continue;
        }
        int minpos = 0;
        for (int j = 1; j < filled; j++)
            if (best_score[anchor_idx[j]] < best_score[anchor_idx[minpos]])
                minpos = j;
        if (best_score[a] > best_score[anchor_idx[minpos]])
            anchor_idx[minpos] = (int)a;
    }
    return filled;
}

/* -------- final exact top-k over k*nc flattened candidates -------- */
static int topk_flatten(const float* scores, int nc, const int* anchor_idx, int k,
                        float* out_score,
                        int* out_class, int* out_anchor)
{
    size_t cand = (size_t)k * (size_t)nc;
    float* buf = (float*)malloc(cand * sizeof(float));
    if (!buf) return -1;
    for (int j = 0; j < k; j++) {
        const float* row = scores + (size_t)anchor_idx[j] * (size_t)nc;
        memcpy(buf + (size_t)j * nc, row, (size_t)nc * sizeof(float));
    }
    /* pick k largest with their flat index; simple O(cand*k) with k<=300 */
    int* taken = (int*)calloc(cand, sizeof(int));
    if (!taken) { free(buf); return -1; }
    int kept = 0;
    for (int t = 0; t < k && kept < k; t++) {
        float bestv = -1.0e30f;
        size_t besti = 0;
        int found = 0;
        for (size_t i = 0; i < cand; i++) {
            if (taken[i]) continue;
            if (!found || buf[i] > bestv) { bestv = buf[i]; besti = i; found = 1; }
        }
        if (!found) break;
        taken[besti] = 1;
        int ci = (int)(besti % (size_t)nc);
        int ai = (int)(besti / (size_t)nc);
        out_score[t]  = bestv;
        out_class[t]  = ci;
        out_anchor[t] = anchor_idx[ai];
        kept++;
    }
    free(taken);
    free(buf);
    return kept;
}

/* Shared decode: box[3]/cls[3] are mandatory, coef[3] optional (seg mask
 * coefficient heads; nm = coef[0]->channels, gathered raw per winning anchor).
 * Rows emitted have width 6 + nm: [x1,y1,x2,y2,score,class, coef(0..nm-1)]. */
static int fiv_y26_decode_core(const fiv_tensor4d* box[3], const fiv_tensor4d* cls[3],
                               const fiv_tensor4d* coef[3], const int strides[3],
                               float* out, int max_k)
{
    if (!box || !cls || !strides || !out || max_k <= 0) return -1;
    for (int i = 0; i < 3; i++) if (!box[i] || !cls[i]) return -1;
    if (coef) for (int i = 0; i < 3; i++) if (!coef[i]) return -1;

    level_info lv[3];
    size_t a_total;
    int nc, nm;
    if (build_levels(box, cls, coef, strides, lv, &a_total, &nc, &nm) != 0) return -1;
    if (nc <= 0) return -1;
    int k = (max_k < (int)a_total) ? max_k : (int)a_total;
    const int rw = 6 + nm;

    /* per-anchor flattened buffers, level-major anchor order */
    float* dbox    = (float*)malloc((size_t)4 * a_total * sizeof(float));
    float* scores  = (float*)malloc((size_t)nc * a_total * sizeof(float));
    float* coefs   = nm > 0 ? (float*)malloc((size_t)nm * a_total * sizeof(float)) : NULL;
    float* best    = (float*)malloc(a_total * sizeof(float));
    int*   sel_anc = (int*)malloc((size_t)k * sizeof(int));
    if (!dbox || !scores || !best || !sel_anc || (nm > 0 && !coefs)) {
        free(dbox); free(scores); free(coefs); free(best); free(sel_anc);
        return -1;
    }

    for (int L = 0; L < 3; L++) {
        size_t off = lv[L].a_off;
        int W = lv[L].width, H = lv[L].height;
        float st = (float)lv[L].stride;
        const fiv_tensor4d* hb = box[L];
        const fiv_tensor4d* hc = cls[L];
        const float* bl = ((fiv_tensor_hdr*)hb)->data.fl;
        const float* bt = bl + (size_t)W * H;
        const float* br = bl + (size_t)2 * W * H;
        const float* bb = bl + (size_t)3 * W * H;
        const float* cl = ((fiv_tensor_hdr*)hc)->data.fl;
        const float* cm = coef ? ((fiv_tensor_hdr*)coef[L])->data.fl : NULL;
        size_t hw = (size_t)W * (size_t)H;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                size_t p = (size_t)y * W + (size_t)x;
                size_t a = off + p;
                float ax = (float)x + 0.5f;
                float ay = (float)y + 0.5f;
                dbox[(size_t)4 * a + 0] = (ax - bl[p]) * st;
                dbox[(size_t)4 * a + 1] = (ay - bt[p]) * st;
                dbox[(size_t)4 * a + 2] = (ax + br[p]) * st;
                dbox[(size_t)4 * a + 3] = (ay + bb[p]) * st;
                for (int c = 0; c < nc; c++)
                    scores[a * (size_t)nc + (size_t)c] = fiv_sigmoidf(cl[(size_t)c * hw + p]);
                if (coefs)
                    for (int m = 0; m < nm; m++)
                        coefs[a * (size_t)nm + (size_t)m] = cm[(size_t)m * hw + p];
            }
        }
    }

    /* ---- stage 1: per-anchor class max -> top-k anchors ---- */
    select_anchors(scores, a_total, nc, k, sel_anc, best);

    /* ---- stage 2: flatten k*nc -> final top-k ---- */
    float* s2 = (float*)malloc((size_t)k * sizeof(float));
    int*   c2 = (int*)malloc((size_t)k * sizeof(int));
    int*   a2 = (int*)malloc((size_t)k * sizeof(int));
    if (!s2 || !c2 || !a2) {
        free(dbox); free(scores); free(coefs); free(best); free(sel_anc);
        free(s2); free(c2); free(a2);
        return -1;
    }
    int kept = topk_flatten(scores, nc, sel_anc, k, s2, c2, a2);
    if (kept < 0) {
        free(dbox); free(scores); free(coefs); free(best); free(sel_anc);
        free(s2); free(c2); free(a2);
        return -1;
    }

    /* ---- emit ---- */
    for (int t = 0; t < kept; t++) {
        size_t a = (size_t)a2[t];
        float* r = out + (size_t)rw * t;
        r[0] = dbox[(size_t)4 * a + 0];
        r[1] = dbox[(size_t)4 * a + 1];
        r[2] = dbox[(size_t)4 * a + 2];
        r[3] = dbox[(size_t)4 * a + 3];
        r[4] = s2[t];
        r[5] = (float)c2[t];
        if (coefs)
            memcpy(r + 6, coefs + a * (size_t)nm, (size_t)nm * sizeof(float));
    }

    free(dbox); free(scores); free(coefs); free(best); free(sel_anc);
    free(s2); free(c2); free(a2);
    return kept;
}

/* Detection: 6 heads (box0..2, cls0..2), rows of width 6. */
int fiv_yolo26_postprocess(const fiv_tensor4d* heads[6], const int strides[3],
                           float* out, int max_k)
{
    if (!heads) return -1;
    const fiv_tensor4d* box[3] = { heads[0], heads[1], heads[2] };
    const fiv_tensor4d* cls[3] = { heads[3], heads[4], heads[5] };
    return fiv_y26_decode_core(box, cls, NULL, strides, out, max_k);
}

/* Segment26: heads[0..5] box/cls as detection, heads[6..8] the one2one_cv4 mask
 * coefficient heads; rows have width 6 + nm (nm = heads[6]->channels). */
int fiv_yolo26_postprocess_seg(const fiv_tensor4d* heads[9], const int strides[3],
                               float* out, int max_k)
{
    if (!heads) return -1;
    const fiv_tensor4d* box[3] = { heads[0], heads[1], heads[2] };
    const fiv_tensor4d* cls[3] = { heads[3], heads[4], heads[5] };
    const fiv_tensor4d* coef[3] = { heads[6], heads[7], heads[8] };
    return fiv_y26_decode_core(box, cls, coef, strides, out, max_k);
}
