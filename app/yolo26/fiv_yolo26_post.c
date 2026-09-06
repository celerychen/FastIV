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

/* YOLO26 NMS-free decode (reg_max=1, end2end). Turns the six Detect one2one
 * raw heads (box0..2 then cls0..2, each a contiguous NCHW tensor) into top-k
 * detections mirroring the torch reference (ultralytics head.py + tal.py):
 *   anchors   make_anchors: per level y-major flatten of (x+0.5, y+0.5),
 *             levels concatenated in stride order 8/16/32.
 *   decode    dist2bbox, xyxy: box channels ordered l,t,r,b;
 *             x1 = ax - l, y1 = ay - t, x2 = ax + r, y2 = ay + b
 *             in grid units, then scaled by the level stride (pixels).
 *   scores    sigmoid of the raw class head.
 *   top-k     two exact passes: per-anchor class max keeps k = min(max_k, A)
 *             anchors; the k*nc candidate scores are reduced to the final k.
 * Output rows are [x1, y1, x2, y2, score, class], score-descending. */

#include <math.h>
#include <string.h>

#include "fiv_common.h"
#include "fiv_yolo26_post.h"

/* Per-level decode geometry: anchor start offset, H*W cell count and stride. */
typedef struct {
    size_t anchor_offset;
    size_t anchor_count;
    int    stride;
    int    width;
    int    height;
} y26_level_geometry;

/* Shared decode geometry: box heads give H/W per level (P3/P4/P5, stride
 * order); the class head supplies num_classes; an optional coef (mask/kpt)
 * head set supplies the per-anchor extra column count. */
static int y26_collect_levels(const fiv_tensor4d* box_heads[3],
                              const fiv_tensor4d* cls_heads[3],
                              const fiv_tensor4d* coef_heads[3],
                              const int strides[3],
                              y26_level_geometry* levels,
                              size_t* anchor_total, int* num_classes,
                              int* num_coefs)
{
    size_t total = 0;
    for (int level = 0; level < 3; level++) {
        if (!box_heads[level] || !cls_heads[level]) return -1;
        if (coef_heads && !coef_heads[level]) return -1;
        levels[level].anchor_offset = total;
        levels[level].width         = (int)box_heads[level]->width;
        levels[level].height        = (int)box_heads[level]->height;
        levels[level].stride        = strides[level];
        levels[level].anchor_count  = (size_t)box_heads[level]->height *
                                      (size_t)box_heads[level]->width;
        total += levels[level].anchor_count;
    }
    *anchor_total = total;
    *num_classes  = (int)cls_heads[0]->channels;
    *num_coefs    = coef_heads ? (int)coef_heads[0]->channels : 0;
    return 0;
}

static ivf32 y26_sigmoid(ivf32 value)
{
    if (value >= 0.0f) {
        ivf32 exp_val = expf(-value);
        return 1.0f / (1.0f + exp_val);
    }
    ivf32 exp_val = expf(value);
    return exp_val / (1.0f + exp_val);
}

/* Stage 1: per-anchor best-class score, then exact top-k of those maxima.
 * Streaming insertion keeps the k largest best scores; each anchor enters the
 * set at most once and replaces the current running minimum only when strictly
 * better (deterministic, ties keep earlier anchors). */
static int y26_select_top_anchors(const ivf32* scores, size_t anchor_total,
                                  int num_classes, int topk_count,
                                  int* selected_anchors, ivf32* best_scores)
{
    for (size_t anchor = 0; anchor < anchor_total; anchor++) {
        const ivf32* row = scores + anchor * (size_t)num_classes;
        ivf32 best_val   = row[0];
        for (int class_idx = 1; class_idx < num_classes; class_idx++)
            if (row[class_idx] > best_val) best_val = row[class_idx];
        best_scores[anchor] = best_val;
    }

    int filled = 0;
    for (size_t anchor = 0; anchor < anchor_total; anchor++) {
        if (filled < topk_count) {
            selected_anchors[filled++] = (int)anchor;
            continue;
        }
        int min_pos = 0;
        for (int j = 1; j < filled; j++)
            if (best_scores[selected_anchors[j]] < best_scores[selected_anchors[min_pos]])
                min_pos = j;
        if (best_scores[anchor] > best_scores[selected_anchors[min_pos]])
            selected_anchors[min_pos] = (int)anchor;
    }
    return filled;
}

/* Stage 2: exact top-k over the flattened topk_count * num_classes candidate
 * scores (each selected anchor contributes its full class row). */
static int y26_topk_flatten(const ivf32* scores, int num_classes,
                            const int* selected_anchors, int topk_count,
                            ivf32* out_score, int* out_class, int* out_anchor)
{
    size_t candidate_count = (size_t)topk_count * (size_t)num_classes;
    ivf32* candidate_buf   = (ivf32*)fiv_malloc(candidate_count * sizeof(ivf32));
    if (!candidate_buf) return -1;
    for (int j = 0; j < topk_count; j++) {
        const ivf32* row = scores + (size_t)selected_anchors[j] * (size_t)num_classes;
        memcpy(candidate_buf + (size_t)j * num_classes, row,
               (size_t)num_classes * sizeof(ivf32));
    }

    int* taken = (int*)fiv_calloc(candidate_count, sizeof(int));
    if (!taken) {
        fiv_free(candidate_buf);
        return -1;
    }
    int kept = 0;
    for (int rank = 0; rank < topk_count && kept < topk_count; rank++) {
        ivf32  best_val = -1.0e30f;
        size_t best_pos = 0;
        int    found    = 0;
        for (size_t pos = 0; pos < candidate_count; pos++) {
            if (taken[pos]) continue;
            if (!found || candidate_buf[pos] > best_val) {
                best_val = candidate_buf[pos];
                best_pos = pos;
                found    = 1;
            }
        }
        if (!found) break;
        taken[best_pos] = 1;
        out_score[rank] = best_val;
        out_class[rank] = (int)(best_pos % (size_t)num_classes);
        out_anchor[rank] = selected_anchors[best_pos / (size_t)num_classes];
        kept++;
    }
    fiv_free(taken);
    fiv_free(candidate_buf);
    return kept;
}

/* Shared decode: box/class heads are mandatory, coef_heads optional (seg mask
 * coefficient / pose kpt heads gathered raw per winning anchor). Emitted rows
 * have width 6 + num_coefs: [x1,y1,x2,y2,score,class, coef(0..num_coefs-1)]. */
static int y26_decode_core(const fiv_tensor4d* box_heads[3],
                           const fiv_tensor4d* cls_heads[3],
                           const fiv_tensor4d* coef_heads[3],
                           const int strides[3], ivf32* out, int max_k)
{
    if (!box_heads || !cls_heads || !strides || !out || max_k <= 0) return -1;
    for (int level = 0; level < 3; level++)
        if (!box_heads[level] || !cls_heads[level]) return -1;
    if (coef_heads)
        for (int level = 0; level < 3; level++)
            if (!coef_heads[level]) return -1;

    y26_level_geometry levels[3];
    size_t anchor_total;
    int num_classes, num_coefs;
    if (y26_collect_levels(box_heads, cls_heads, coef_heads, strides,
                           levels, &anchor_total, &num_classes, &num_coefs) != 0)
        return -1;
    if (num_classes <= 0) return -1;
    int topk_count = (max_k < (int)anchor_total) ? max_k : (int)anchor_total;
    const int row_width = 6 + num_coefs;

    /* Per-anchor flattened buffers, level-major anchor order. */
    ivf32* box_pixels = (ivf32*)fiv_malloc((size_t)4 * anchor_total * sizeof(ivf32));
    ivf32* scores     = (ivf32*)fiv_malloc((size_t)num_classes * anchor_total * sizeof(ivf32));
    ivf32* coefs      = num_coefs > 0
                            ? (ivf32*)fiv_malloc((size_t)num_coefs * anchor_total * sizeof(ivf32))
                            : NULL;
    ivf32* best_scores = (ivf32*)fiv_malloc(anchor_total * sizeof(ivf32));
    int*   anchor_sel  = (int*)fiv_malloc((size_t)topk_count * sizeof(int));
    if (!box_pixels || !scores || !best_scores || !anchor_sel ||
        (num_coefs > 0 && !coefs)) {
        fiv_free(box_pixels);
        fiv_free(scores);
        fiv_free(coefs);
        fiv_free(best_scores);
        fiv_free(anchor_sel);
        return -1;
    }

    /* Anchors (grid + 0.5) and per-anchor decode across the three levels. */
    for (int level = 0; level < 3; level++) {
        const y26_level_geometry* geometry = &levels[level];
        int  width  = geometry->width;
        int  height = geometry->height;
        ivf32 stride = (ivf32)geometry->stride;
        size_t plane = (size_t)width * (size_t)height;
        size_t offset = geometry->anchor_offset;

        const fiv_tensor4d* box_head = box_heads[level];
        const fiv_tensor4d* cls_head = cls_heads[level];
        const ivf32* dist_l = ((const fiv_tensor_hdr*)box_head)->data.fl;
        const ivf32* dist_t = dist_l + plane;
        const ivf32* dist_r = dist_l + (size_t)2 * plane;
        const ivf32* dist_b = dist_l + (size_t)3 * plane;
        const ivf32* logits = ((const fiv_tensor_hdr*)cls_head)->data.fl;
        const ivf32* coef_raw = coef_heads
                                    ? ((const fiv_tensor_hdr*)coef_heads[level])->data.fl
                                    : NULL;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                size_t pos  = (size_t)y * (size_t)width + (size_t)x;
                size_t anchor = offset + pos;
                ivf32   ax   = (ivf32)x + 0.5f;
                ivf32   ay   = (ivf32)y + 0.5f;
                box_pixels[(size_t)4 * anchor + 0] = (ax - dist_l[pos]) * stride;
                box_pixels[(size_t)4 * anchor + 1] = (ay - dist_t[pos]) * stride;
                box_pixels[(size_t)4 * anchor + 2] = (ax + dist_r[pos]) * stride;
                box_pixels[(size_t)4 * anchor + 3] = (ay + dist_b[pos]) * stride;
                for (int class_idx = 0; class_idx < num_classes; class_idx++)
                    scores[anchor * (size_t)num_classes + (size_t)class_idx] =
                        y26_sigmoid(logits[(size_t)class_idx * plane + pos]);
                if (coefs)
                    for (int coef = 0; coef < num_coefs; coef++)
                        coefs[anchor * (size_t)num_coefs + (size_t)coef] =
                            coef_raw[(size_t)coef * plane + pos];
            }
        }
    }

    /* Stage 1: per-anchor class max -> top anchors; stage 2: final top-k. */
    y26_select_top_anchors(scores, anchor_total, num_classes,
                           topk_count, anchor_sel, best_scores);
    ivf32* out_score  = (ivf32*)fiv_malloc((size_t)topk_count * sizeof(ivf32));
    int*   out_class  = (int*)fiv_malloc((size_t)topk_count * sizeof(int));
    int*   out_anchor = (int*)fiv_malloc((size_t)topk_count * sizeof(int));
    if (!out_score || !out_class || !out_anchor) {
        fiv_free(box_pixels);
        fiv_free(scores);
        fiv_free(coefs);
        fiv_free(best_scores);
        fiv_free(anchor_sel);
        fiv_free(out_score);
        fiv_free(out_class);
        fiv_free(out_anchor);
        return -1;
    }
    int kept = y26_topk_flatten(scores, num_classes, anchor_sel,
                                topk_count, out_score, out_class, out_anchor);
    if (kept < 0) {
        fiv_free(box_pixels);
        fiv_free(scores);
        fiv_free(coefs);
        fiv_free(best_scores);
        fiv_free(anchor_sel);
        fiv_free(out_score);
        fiv_free(out_class);
        fiv_free(out_anchor);
        return -1;
    }

    /* Emit the final rows. */
    for (int rank = 0; rank < kept; rank++) {
        size_t anchor   = (size_t)out_anchor[rank];
        ivf32* row      = out + (size_t)row_width * (size_t)rank;
        row[0] = box_pixels[(size_t)4 * anchor + 0];
        row[1] = box_pixels[(size_t)4 * anchor + 1];
        row[2] = box_pixels[(size_t)4 * anchor + 2];
        row[3] = box_pixels[(size_t)4 * anchor + 3];
        row[4] = out_score[rank];
        row[5] = (ivf32)out_class[rank];
        if (coefs)
            memcpy(row + 6, coefs + anchor * (size_t)num_coefs,
                   (size_t)num_coefs * sizeof(ivf32));
    }

    fiv_free(box_pixels);
    fiv_free(scores);
    fiv_free(coefs);
    fiv_free(best_scores);
    fiv_free(anchor_sel);
    fiv_free(out_score);
    fiv_free(out_class);
    fiv_free(out_anchor);
    return kept;
}

/* Detection: 6 heads (box0..2, cls0..2), rows of width 6. */
int fiv_yolo26_postprocess(const fiv_tensor4d* heads[6], const int strides[3],
                           ivf32* out, int max_k)
{
    if (!heads) return -1;
    const fiv_tensor4d* box_heads[3] = { heads[0], heads[1], heads[2] };
    const fiv_tensor4d* cls_heads[3] = { heads[3], heads[4], heads[5] };
    return y26_decode_core(box_heads, cls_heads, NULL, strides, out, max_k);
}

/* Segment26: heads[0..5] box/cls as detection, heads[6..8] the one2one_cv4 mask
 * coefficient heads; rows have width 6 + num_coefs (num_coefs = heads[6]->channels). */
int fiv_yolo26_postprocess_seg(const fiv_tensor4d* heads[9], const int strides[3],
                               ivf32* out, int max_k)
{
    if (!heads) return -1;
    const fiv_tensor4d* box_heads[3]  = { heads[0], heads[1], heads[2] };
    const fiv_tensor4d* cls_heads[3]  = { heads[3], heads[4], heads[5] };
    const fiv_tensor4d* coef_heads[3] = { heads[6], heads[7], heads[8] };
    return y26_decode_core(box_heads, cls_heads, coef_heads, strides, out, max_k);
}

/* Pose26 entry: heads[6..8] are the one2one_cv4_kpts raw heads (3*nk = 51 ch,
 * channel order [x,y,vis] triples). Every anchor's kpt block is decoded first
 * (x = (raw_x + anchor_x) * stride, y = (raw_y + anchor_y) * stride, vis =
 * sigmoid(raw_vis)) into synthetic [1,3*nk,H,W] coef-style heads, then the
 * shared core gathers the winning anchors' 51 columns into the final rows. */
int fiv_yolo26_postprocess_pose(const fiv_tensor4d* heads[9], const int strides[3],
                                ivf32* out, int max_k)
{
    if (!heads) return -1;
    for (int i = 0; i < 9; i++)
        if (!heads[i]) return -1;

    int num_coefs = (int)heads[6]->channels;         /* 3*nk = 51 */
    if (num_coefs <= 0 || num_coefs % 3 != 0) return -1;

    const fiv_tensor4d* decoded_heads[3];
    for (int level = 0; level < 3; level++) {
        size_t height = heads[6 + level]->height;
        size_t width  = heads[6 + level]->width;
        size_t plane  = height * width;
        fiv_tensor4d* decoded =
            fiv_create_tensor4d((size_t[]){1, (size_t)num_coefs, height, width}, FIV_32F1);
        if (!decoded) {
            for (int prev = 0; prev < level; prev++)
                fiv_release_tensor((void**)&decoded_heads[prev]);
            return -1;
        }
        decoded_heads[level] = decoded;

        const ivf32* raw_kpts = ((const fiv_tensor_hdr*)heads[6 + level])->data.fl;
        ivf32*       grid     = ((fiv_tensor_hdr*)decoded)->data.fl;
        ivf32 stride          = (ivf32)strides[level];
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                ivf32 ax = (ivf32)x + 0.5f;
                ivf32 ay = (ivf32)y + 0.5f;
                size_t pos = y * width + x;
                for (int coef = 0; coef < num_coefs; coef += 3) {
                    size_t base = (size_t)coef * plane + pos;
                    grid[(size_t)coef * plane + pos]     = (raw_kpts[base] + ax) * stride;
                    grid[(size_t)(coef + 1) * plane + pos] =
                        (raw_kpts[(size_t)(coef + 1) * plane + pos] + ay) * stride;
                    grid[(size_t)(coef + 2) * plane + pos] =
                        y26_sigmoid(raw_kpts[(size_t)(coef + 2) * plane + pos]);
                }
            }
        }
    }

    int result = y26_decode_core(heads, heads + 3, decoded_heads, strides, out, max_k);
    for (int level = 0; level < 3; level++)
        fiv_release_tensor((void**)&decoded_heads[level]);
    return result;
}
