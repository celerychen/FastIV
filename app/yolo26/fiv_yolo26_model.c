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

/* YOLO26 model facades (see fiv_yolo26_model.h). Each facade owns a loaded
 * model directory (fold table + folded weights + attention blobs), builds the
 * engine graph once, and runs single images through letterbox -> inference ->
 * decode. Coordinate spaces: all reported boxes/keypoints live in the model
 * input image (after letterboxing), matching the reference detect_out dumps.
 *
 * Shared plumbing (loader + graph build + forward + decode) lives in
 * fiv_yolo26.c / fiv_yolo26_post.c; this file is only the thin 3-interface
 * surface plus the per-task output assembly.
 */

#include "fiv_yolo26_model.h"
#include "fiv_yolo26.h"
#include "fiv_yolo26_post.h"
#include "fiv_yolo26_proto.h"

#include "fiv_common.h"
#include "fiv_ctensor.h"
#include "fiv_nn_infer.h"
#include "yolo26_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIV_YOLO26_DEFAULT_INPUT_SIZE 320
#define FIV_YOLO26_DEFAULT_CLASSES    80
#define FIV_YOLO26_CLS_HEAD_CHANNELS  1280

typedef enum {
    FIV_YOLO26_TASK_DETECTOR,
    FIV_YOLO26_TASK_SEGMENTER,
    FIV_YOLO26_TASK_POSE,
    FIV_YOLO26_TASK_CLASSIFIER,
} fiv_yolo26_task;

typedef struct {
    fiv_yolo26_task    task;
    fiv_yolo26_graph*  graph;      /* built engine graph (0 when cls) */
    void*              net;        /* engine net (graph->net for det/seg/pose) */
    int                input_size; /* letterbox square side */
    int                classes;    /* det/seg/pose class count */
    /* classifier extras (loaded once) */
    const ivf32*       linear_w;   /* [classes, head_channels] */
    const ivf32*       linear_b;   /* [classes] */
    const ivf32*       head_w;     /* cls head conv weight [1280, in,1,1] */
    const ivf32*       head_b;     /* cls head conv bias [1280] */
    int                head_conv_node;  /* cls head conv node id (with silu) */
    /* seg extras (loaded once): refine0, refine1, fuse, cv1, upsample,
       cv2, cv3 -> w[0..6], b[0..6] */
    const ivf32*       proto_w[7];
    const ivf32*       proto_b[7];
    /* runtime scratch for the classifier head output + logits */
    ivf32*             cls_scratch;
    size_t             cls_scratch_bytes;
    /* segmenter runtime scratch (ctx-owned; valid until the next call) */
    ivf32*             proto_out;       /* [32 * Hp * Wp] */
    ivf32*             mask_scratch;    /* [MAX_DET * Hp * Wp] */
    size_t             proto_plane;     /* Hp * Wp floats per instance/channel */
    /* cumulative stage timing over timing_calls on_image runs (ms) */
    ivf64              timing_pre_ms;   /* letterbox */
    ivf64              timing_infer_ms; /* engine forward */
    ivf64              timing_post_ms;  /* decode + post-processing */
    int                timing_calls;
} fiv_yolo26_model;

/* ------------------------------------------------------------------ */
/* model-directory loader helpers                                      */
/* ------------------------------------------------------------------ */

static int fiv_y26_model_load_blob(const char* dir, const char* name,
                                   ivf32** out_data, size_t* out_count)
{
    char file_name[1024];
    snprintf(file_name, sizeof(file_name), "%s/%s", dir, name);
    FILE* file = fopen(file_name, "rb");
    if (file == NULL) return -1;
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return -1; }
    long byte_count = ftell(file);
    if (byte_count <= 0 || (byte_count % 4) != 0) { fclose(file); return -1; }
    fseek(file, 0, SEEK_SET);
    ivf32* buffer = (ivf32*)fiv_malloc((size_t)byte_count);
    if (buffer == NULL) { fclose(file); return -1; }
    if (fread(buffer, 1, (size_t)byte_count, file) != (size_t)byte_count) {
        fclose(file);
        fiv_free(buffer);
        return -1;
    }
    fclose(file);
    *out_data = buffer;
    if (out_count != NULL) *out_count = (size_t)byte_count / sizeof(ivf32);
    return 0;
}

/* Fold-table load: reuse the loader (it now honors yolo26_ref_set_dir). */
static int fiv_y26_model_load_fold(const char* dir, yolo26_fold_entry* out,
                                   int capacity, int* out_count)
{
    yolo26_ref_set_dir(dir);
    int count = yolo26_ref_load_fold(out, capacity);
    if (count < 0) return -1;
    *out_count = count;
    return 0;
}

/* Copy an attention blob (already in the model dir as attn<id>_<part>.f32)
 * into a malloc'd array. */
static const ivf32* fiv_y26_model_load_attn(const char* dir, int id,
                                            const char* part, size_t* out_count)
{
    char name[128];
    snprintf(name, sizeof(name), "attn%d_%s.f32", id, part);
    ivf32* data = NULL;
    size_t count = 0;
    if (fiv_y26_model_load_blob(dir, name, &data, &count) != 0) return NULL;
    if (out_count) *out_count = count;
    return data;
}

/* ------------------------------------------------------------------ */
/* shared engine forward (feed letterboxed input tensor)               */
/* ------------------------------------------------------------------ */

static fiv_ret fiv_y26_model_forward(fiv_yolo26_model* model,
                                     fiv_tensor4d* input)
{
    void* final_out = NULL;
    return fiv_nn_run_inference(model->net, input, &final_out);
}

/* ------------------------------------------------------------------ */
/* preprocess: letterbox an RGB source into a [1,3,S,S] float tensor   */
/* ------------------------------------------------------------------ */

static fiv_tensor4d* fiv_y26_model_letterbox(fiv_yolo26_model* model,
                                             const fiv_mat* image)
{
    int source_w = (int)image->width;
    int source_h = (int)image->height;
    int target_size = model->input_size;
    const iv8u* source_ptr = image->data.ptr8u;

    /* Letterbox geometry: scale so the full source fits in the square. */
    ivf32 scale = (ivf32)target_size / (ivf32)(source_w > source_h ? source_w : source_h);
    int scaled_w = (int)((ivf32)source_w * scale + 0.5f);
    int scaled_h = (int)((ivf32)source_h * scale + 0.5f);
    if (scaled_w < 1) scaled_w = 1;
    if (scaled_h < 1) scaled_h = 1;
    int offset_x = (target_size - scaled_w) / 2;
    int offset_y = (target_size - scaled_h) / 2;

    size_t size_4d[4] = { 1, 3, (size_t)target_size, (size_t)target_size };
    fiv_tensor4d* tensor = fiv_create_tensor4d(size_4d, FIV_32F1);
    if (tensor == NULL) return NULL;
    ivf32* dst_ptr = ((fiv_tensor_hdr*)tensor)->data.fl;

    /* Bilinear letterbox: content area rescaled from the source (center-aligned
       sample mapping, align_corners = false), padded with gray 114. */
    for (int y = 0; y < target_size; y++) {
        for (int x = 0; x < target_size; x++) {
            int src_x = x - offset_x;
            int src_y = y - offset_y;
            ivf32 px = 114.0f, py = 114.0f, pz = 114.0f;   /* gray fill */
            if (src_x >= 0 && src_x < scaled_w && src_y >= 0 && src_y < scaled_h) {
                ivf32 fx = ((ivf32)src_x + 0.5f) / scale - 0.5f;
                ivf32 fy = ((ivf32)src_y + 0.5f) / scale - 0.5f;
                if (fx < 0.0f) fx = 0.0f;
                if (fy < 0.0f) fy = 0.0f;
                if (fx > (ivf32)source_w - 1.0f) fx = (ivf32)source_w - 1.0f;
                if (fy > (ivf32)source_h - 1.0f) fy = (ivf32)source_h - 1.0f;
                int x0 = (int)fx;
                int y0 = (int)fy;
                int x1 = (x0 + 1 < source_w) ? x0 + 1 : x0;
                int y1 = (y0 + 1 < source_h) ? y0 + 1 : y0;
                ivf32 wx = fx - (ivf32)x0;
                ivf32 wy = fy - (ivf32)y0;
                const iv8u* p00 = source_ptr + ((size_t)y0 * source_w + x0) * 3;
                const iv8u* p10 = source_ptr + ((size_t)y0 * source_w + x1) * 3;
                const iv8u* p01 = source_ptr + ((size_t)y1 * source_w + x0) * 3;
                const iv8u* p11 = source_ptr + ((size_t)y1 * source_w + x1) * 3;
                for (int ch = 0; ch < 3; ch++) {
                    ivf32 top = (1.0f - wx) * p00[ch] + wx * p10[ch];
                    ivf32 bot = (1.0f - wx) * p01[ch] + wx * p11[ch];
                    ivf32 value = (1.0f - wy) * top + wy * bot;
                    dst_ptr[((size_t)ch * target_size + y) * target_size + x] =
                        value / 255.0f;
                }
                continue;
            }
            dst_ptr[(0 * target_size + y) * target_size + x] = px / 255.0f;
            dst_ptr[(1 * target_size + y) * target_size + x] = py / 255.0f;
            dst_ptr[(2 * target_size + y) * target_size + x] = pz / 255.0f;
        }
    }
    return tensor;
}

/* ------------------------------------------------------------------ */
/* classifier tail: pool + linear (+ nothing else).                    */
/* ------------------------------------------------------------------ */

static int fiv_y26_model_classify(fiv_yolo26_model* model, void* result)
{
    fiv_yolo26_cls_result* out = (fiv_yolo26_cls_result*)result;
    const ivf32* head_fl =
        ((fiv_tensor_hdr*)fiv_neural_network_get_node_output(model->net,
                                                             model->head_conv_node))->data.fl;
    const fiv_tensor4d* head_tensor =
        (const fiv_tensor4d*)fiv_neural_network_get_node_output(model->net,
                                                                model->head_conv_node);
    if (head_tensor == NULL || head_fl == NULL) return -1;
    int spatial_h = (int)head_tensor->height;
    int spatial_w = (int)head_tensor->width;
    size_t plane = (size_t)spatial_h * spatial_w;
    int head_channels = (int)head_tensor->channels;

    /* global average pool (H,W) -> [head_channels] */
    ivf32* pooled = (ivf32*)fiv_malloc((size_t)head_channels * sizeof(ivf32));
    if (pooled == NULL) return -1;
    for (int c = 0; c < head_channels; c++) {
        double sum = 0.0;
        const ivf32* channel_ptr = head_fl + (size_t)c * plane;
        for (size_t k = 0; k < plane; k++) sum += (double)channel_ptr[k];
        pooled[c] = (ivf32)(sum / (double)plane);
    }

    int classes = model->classes;
    /* logits = linear_w . pooled + linear_b */
    if (model->cls_scratch_bytes < (size_t)classes * sizeof(ivf32)) {
        fiv_free(model->cls_scratch);
        model->cls_scratch = (ivf32*)fiv_malloc((size_t)classes * sizeof(ivf32));
        if (model->cls_scratch == NULL) { fiv_free(pooled); return -1; }
        model->cls_scratch_bytes = (size_t)classes * sizeof(ivf32);
    }
    ivf32* logits = model->cls_scratch;
    for (int c = 0; c < classes; c++) {
        double sum = 0.0;
        const ivf32* row = model->linear_w + (size_t)c * head_channels;
        for (int k = 0; k < head_channels; k++) sum += (double)row[k] * pooled[k];
        logits[c] = (ivf32)sum + model->linear_b[c];
    }
    fiv_free(pooled);

    /* softmax over logits (single batch row) */
    ivf32 max_logit = logits[0];
    for (int c = 1; c < classes; c++)
        if (logits[c] > max_logit) max_logit = logits[c];
    double prob_sum = 0.0;
    ivf32* probs = (ivf32*)fiv_malloc((size_t)classes * sizeof(ivf32));
    if (probs == NULL) return -1;
    for (int c = 0; c < classes; c++) {
        ivf32 prob = expf(logits[c] - max_logit);
        probs[c] = prob;
        prob_sum += prob;
    }
    for (int c = 0; c < classes; c++) probs[c] = (ivf32)(probs[c] / prob_sum);

    /* top-1 / top-K */
    int top_k = FIV_YOLO26_TOP_K < classes ? FIV_YOLO26_TOP_K : classes;
    for (int t = 0; t < top_k; t++) {
        int best = -1;
        for (int c = 0; c < classes; c++) {
            if (best < 0 || probs[c] > probs[best]) best = c;
        }
        out->top_classes[t] = best;
        out->top_scores[t] = probs[best];
        probs[best] = -1.0f;
    }
    out->class_id = out->top_classes[0];
    out->score = out->top_scores[0];
    out->logits = logits;
    out->num_classes = classes;
    fiv_free(probs);
    return 0;
}

/* ------------------------------------------------------------------ */
/* shared create (loads everything task-generic)                       */
/* ------------------------------------------------------------------ */

static void* fiv_y26_model_create_common(const char* model_dir,
                                         fiv_yolo26_task task)
{
    if (model_dir == NULL) return NULL;
    fiv_yolo26_model* model = (fiv_yolo26_model*)fiv_calloc(1, sizeof(fiv_yolo26_model));
    if (model == NULL) return NULL;
    model->task = task;
    model->input_size = FIV_YOLO26_DEFAULT_INPUT_SIZE;
    model->classes = FIV_YOLO26_DEFAULT_CLASSES;
    model->head_conv_node = -1;

    /* Fold table + payloads. */
    yolo26_fold_entry entries[512];
    int entry_count = 0;
    if (fiv_y26_model_load_fold(model_dir, entries, 512, &entry_count) != 0 ||
        entry_count <= 0) {
        fiv_free(model);
        return NULL;
    }
    size_t weight_bytes = 0, bias_bytes = 0;
    for (int i = 0; i < entry_count; i++) {
        size_t w_end = entries[i].w_off + entries[i].w_cnt;
        size_t b_end = entries[i].b_off + entries[i].b_cnt;
        if (w_end > weight_bytes) weight_bytes = w_end;
        if (b_end > bias_bytes) bias_bytes = b_end;
    }
    ivf32* fold_w = NULL;
    ivf32* fold_b = NULL;
    if (fiv_y26_model_load_blob(model_dir, "fold_w.f32", &fold_w, NULL) != 0 ||
        fiv_y26_model_load_blob(model_dir, "fold_b.f32", &fold_b, NULL) != 0) {
        fiv_free(fold_w);
        fiv_free(fold_b);
        fiv_free(model);
        return NULL;
    }
    (void)weight_bytes;
    (void)bias_bytes;

    /* Attention blobs attn0/attn1: {qkv_w, qkv_b, pe_w, pe_b, proj_w, proj_b} */
    static const char* const parts[6] = {
        "qkv_w", "qkv_b", "pe_w", "pe_b", "proj_w", "proj_b"
    };
    const ivf32* attn[2][6];
    memset(attn, 0, sizeof(attn));
    int attn_ok = 1;
    for (int id = 0; id < 2; id++) {
        for (int p = 0; p < 6; p++) {
            attn[id][p] = fiv_y26_model_load_attn(model_dir, id, parts[p], NULL);
            if (attn[id][p] == NULL) attn_ok = 0;
        }
    }

    /* Build the graph for det/seg/pose. */
    if (task == FIV_YOLO26_TASK_DETECTOR || task == FIV_YOLO26_TASK_SEGMENTER ||
        task == FIV_YOLO26_TASK_POSE) {
        if (!attn_ok) {
            for (int id = 0; id < 2; id++)
                for (int p = 0; p < 6; p++) fiv_free((void*)attn[id][p]);
            fiv_free(fold_w);
            fiv_free(fold_b);
            fiv_free(model);
            return NULL;
        }
        model->graph = fiv_yolo26_build(entries, entry_count, fold_w, fold_b, attn);
        for (int id = 0; id < 2; id++)
            for (int p = 0; p < 6; p++) fiv_free((void*)attn[id][p]);
        if (model->graph == NULL) {
            fiv_free(fold_w);
            fiv_free(fold_b);
            fiv_free(model);
            return NULL;
        }
        model->net = model->graph->net;
        if (task == FIV_YOLO26_TASK_SEGMENTER) {
            static const char* const proto_parts[7] = {
                "feat_refine0", "feat_refine1", "feat_fuse", "cv1",
                "upsample", "cv2", "cv3" };
            int proto_ok = 1;
            for (int i = 0; i < 7; i++) {
                char wname[96], bname[96];
                snprintf(wname, sizeof(wname), "proto_%s_w.f32", proto_parts[i]);
                snprintf(bname, sizeof(bname), "proto_%s_b.f32", proto_parts[i]);
                ivf32* w = NULL;
                ivf32* b = NULL;
                if (fiv_y26_model_load_blob(model_dir, wname, &w, NULL) != 0 ||
                    fiv_y26_model_load_blob(model_dir, bname, &b, NULL) != 0) {
                    fiv_free(w);
                    fiv_free(b);
                    proto_ok = 0;
                    break;
                }
                model->proto_w[i] = w;
                model->proto_b[i] = b;
            }
            if (!proto_ok) {
                for (int i = 0; i < 7; i++) {
                    fiv_free((void*)model->proto_w[i]);
                    fiv_free((void*)model->proto_b[i]);
                    model->proto_w[i] = NULL;
                    model->proto_b[i] = NULL;
                }
                fiv_free(fold_w);
                fiv_free(fold_b);
                fiv_free(model);
                return NULL;
            }
        }
        /* cls-count: from the cls head's channel count is unknown here; keep 80. */
    } else if (task == FIV_YOLO26_TASK_CLASSIFIER) {
        /* head conv + linear weights live as their own blobs. The classifier
           only needs Attention 0 (its single C2PSA). */
        const ivf32* head_w = NULL;
        const ivf32* head_b = NULL;
        size_t linear_b_count = 0;
        if (fiv_y26_model_load_blob(model_dir, "head_conv_w.f32", (ivf32**)&head_w, NULL) != 0 ||
            fiv_y26_model_load_blob(model_dir, "head_conv_b.f32", (ivf32**)&head_b, NULL) != 0 ||
            fiv_y26_model_load_blob(model_dir, "linear_w.f32", (ivf32**)&model->linear_w, NULL) != 0 ||
            fiv_y26_model_load_blob(model_dir, "linear_b.f32", (ivf32**)&model->linear_b,
                                    &linear_b_count) != 0 ||
            attn[0][0] == NULL || attn[0][1] == NULL || attn[0][2] == NULL ||
            attn[0][3] == NULL || attn[0][4] == NULL || attn[0][5] == NULL) {
            fiv_free((void*)head_w);
            fiv_free((void*)head_b);
            fiv_free((void*)model->linear_w);
            fiv_free((void*)model->linear_b);
            fiv_free(fold_w);
            fiv_free(fold_b);
            fiv_free(model);
            return NULL;
        }
        model->head_w = head_w;
        model->head_b = head_b;
        model->classes = (int)linear_b_count;

        model->graph = fiv_yolo26_build_classifier(
            entries, entry_count, fold_w, fold_b, attn, head_w, head_b,
            FIV_YOLO26_CLS_HEAD_CHANNELS);
        for (int id = 0; id < 2; id++)
            for (int part = 0; part < 6; part++) fiv_free((void*)attn[id][part]);
        if (model->graph == NULL) {
            fiv_free((void*)head_w);
            fiv_free((void*)head_b);
            fiv_free((void*)model->linear_w);
            fiv_free((void*)model->linear_b);
            fiv_free(fold_w);
            fiv_free(fold_b);
            fiv_free(model);
            return NULL;
        }
        model->net = model->graph->net;
        model->head_conv_node = model->graph->layer_node[10];
    }

    fiv_free(fold_w);
    fiv_free(fold_b);
    return model;
}

/* ------------------------------------------------------------------ */
/* create / on_image / release                                         */
/* ------------------------------------------------------------------ */

void* fiv_create_yolo26_detector(const char* model_dir)
{
    return fiv_y26_model_create_common(model_dir, FIV_YOLO26_TASK_DETECTOR);
}

void* fiv_create_yolo26_segmenter(const char* model_dir)
{
    return fiv_y26_model_create_common(model_dir, FIV_YOLO26_TASK_SEGMENTER);
}

void* fiv_create_yolo26_pose(const char* model_dir)
{
    return fiv_y26_model_create_common(model_dir, FIV_YOLO26_TASK_POSE);
}

void* fiv_create_yolo26_classifier(const char* model_dir)
{
    return fiv_y26_model_create_common(model_dir, FIV_YOLO26_TASK_CLASSIFIER);
}

static fiv_ret fiv_y26_model_run_det(fiv_yolo26_model* model, void* result,
                                     fiv_mat* image)
{
    fiv_yolo26_det_result* out = (fiv_yolo26_det_result*)result;
    out->count = 0;

    ivf64 _t_pre = fiv_get_current_system_time();
    fiv_tensor4d* input = fiv_y26_model_letterbox(model, image);
    if (input == NULL) return FIV_RET_ERR_MEM;
    model->timing_pre_ms += fiv_get_current_system_time() - _t_pre;
    ivf64 _t_inf = fiv_get_current_system_time();
    fiv_ret run_result = fiv_y26_model_forward(model, input);
    fiv_release_tensor((void**)&input);
    model->timing_infer_ms += fiv_get_current_system_time() - _t_inf;
    ivf64 _t_post = fiv_get_current_system_time();
    if (run_result != FIV_RET_OK) return run_result;

    const fiv_tensor4d* heads[6];
    for (int h = 0; h < 3; h++) {
        heads[h] = (const fiv_tensor4d*)fiv_neural_network_get_node_output(
            model->net, model->graph->head_node[h]);
        heads[3 + h] = (const fiv_tensor4d*)fiv_neural_network_get_node_output(
            model->net, model->graph->head_node[3 + h]);
        if (heads[h] == NULL || heads[3 + h] == NULL) return FIV_RET_ERR_PARA;
    }
    int strides[3];
    for (int level = 0; level < 3; level++)
        strides[level] = model->input_size / (int)heads[level]->width;

    ivf32 decode_rows[FIV_YOLO26_MAX_DETECTIONS * 6];
    int kept = fiv_yolo26_postprocess(heads, strides, decode_rows,
                                      FIV_YOLO26_MAX_DETECTIONS);
    if (kept < 0) return FIV_RET_ERR_PARA;
    out->count = kept;
    for (int i = 0; i < kept && i < FIV_YOLO26_MAX_DETECTIONS; i++) {
        const ivf32* row = decode_rows + (size_t)6 * i;
        fiv_yolo26_box* box = &out->detections[i];
        box->x1 = row[0];
        box->y1 = row[1];
        box->x2 = row[2];
        box->y2 = row[3];
        box->score = row[4];
        box->class_id = (int)row[5];
    }
    model->timing_post_ms += fiv_get_current_system_time() - _t_post;
    model->timing_calls += 1;
    return FIV_RET_OK;
}

static fiv_ret fiv_y26_model_run_pose(fiv_yolo26_model* model, void* result,
                                      fiv_mat* image)
{
    fiv_yolo26_pose_result* out = (fiv_yolo26_pose_result*)result;
    out->count = 0;

    ivf64 _t_pre = fiv_get_current_system_time();
    fiv_tensor4d* input = fiv_y26_model_letterbox(model, image);
    if (input == NULL) return FIV_RET_ERR_MEM;
    model->timing_pre_ms += fiv_get_current_system_time() - _t_pre;
    ivf64 _t_inf = fiv_get_current_system_time();
    fiv_ret run_result = fiv_y26_model_forward(model, input);
    fiv_release_tensor((void**)&input);
    model->timing_infer_ms += fiv_get_current_system_time() - _t_inf;
    ivf64 _t_post = fiv_get_current_system_time();
    if (run_result != FIV_RET_OK) return run_result;

    const fiv_tensor4d* heads[9];
    for (int level = 0; level < 3; level++) {
        heads[level] = (const fiv_tensor4d*)fiv_neural_network_get_node_output(
            model->net, model->graph->head_node[level]);
        heads[3 + level] = (const fiv_tensor4d*)fiv_neural_network_get_node_output(
            model->net, model->graph->head_node[3 + level]);
        heads[6 + level] = (const fiv_tensor4d*)fiv_neural_network_get_node_output(
            model->net, model->graph->kpt_node[level]);
        if (heads[level] == NULL || heads[3 + level] == NULL || heads[6 + level] == NULL)
            return FIV_RET_ERR_PARA;
    }
    int strides[3];
    for (int level = 0; level < 3; level++)
        strides[level] = model->input_size / (int)heads[level]->width;

    const int columns = 6 + FIV_YOLO26_POSE_KEYPOINTS * 3;   /* 57 */
    ivf32* decode_rows = (ivf32*)fiv_malloc(
        (size_t)FIV_YOLO26_MAX_DETECTIONS * (size_t)columns * sizeof(ivf32));
    if (decode_rows == NULL) return FIV_RET_ERR_MEM;
    int kept = fiv_yolo26_postprocess_pose(heads, strides, decode_rows,
                                           FIV_YOLO26_MAX_DETECTIONS);
    if (kept < 0) {
        fiv_free(decode_rows);
        return FIV_RET_ERR_PARA;
    }
    out->count = kept;
    for (int i = 0; i < kept && i < FIV_YOLO26_MAX_DETECTIONS; i++) {
        const ivf32* row = decode_rows + (size_t)columns * i;
        fiv_yolo26_box* box = &out->detections[i];
        box->x1 = row[0];
        box->y1 = row[1];
        box->x2 = row[2];
        box->y2 = row[3];
        box->score = row[4];
        box->class_id = (int)row[5];
        for (int k = 0; k < FIV_YOLO26_POSE_KEYPOINTS; k++) {
            out->keypoints[i][k][0] = row[6 + 3 * k + 0];
            out->keypoints[i][k][1] = row[6 + 3 * k + 1];
            out->keypoints[i][k][2] = row[6 + 3 * k + 2];
        }
    }
    fiv_free(decode_rows);
    model->timing_post_ms += fiv_get_current_system_time() - _t_post;
    model->timing_calls += 1;
    return FIV_RET_OK;
}

fiv_ret fiv_yolo26_detector_on_image(void* result, fiv_mat* image, void* model)
{
    if (result == NULL || image == NULL || model == NULL) return FIV_RET_ERR_PARA;
    fiv_yolo26_model* ctx = (fiv_yolo26_model*)model;
    if (ctx->task != FIV_YOLO26_TASK_DETECTOR) return FIV_RET_ERR_PARA;
    return fiv_y26_model_run_det(ctx, result, image);
}

fiv_ret fiv_yolo26_pose_on_image(void* result, fiv_mat* image, void* model)
{
    if (result == NULL || image == NULL || model == NULL) return FIV_RET_ERR_PARA;
    fiv_yolo26_model* ctx = (fiv_yolo26_model*)model;
    if (ctx->task != FIV_YOLO26_TASK_POSE) return FIV_RET_ERR_PARA;
    return fiv_y26_model_run_pose(ctx, result, image);
}

fiv_ret fiv_yolo26_classifier_on_image(void* result, fiv_mat* image, void* model)
{
    if (result == NULL || image == NULL || model == NULL) return FIV_RET_ERR_PARA;
    fiv_yolo26_model* ctx = (fiv_yolo26_model*)model;
    if (ctx->task != FIV_YOLO26_TASK_CLASSIFIER) return FIV_RET_ERR_PARA;

    ivf64 _t_pre = fiv_get_current_system_time();
    fiv_tensor4d* input = fiv_y26_model_letterbox(ctx, image);
    if (input == NULL) return FIV_RET_ERR_MEM;
    ctx->timing_pre_ms += fiv_get_current_system_time() - _t_pre;
    ivf64 _t_inf = fiv_get_current_system_time();
    fiv_ret run_result = fiv_y26_model_forward(ctx, input);
    fiv_release_tensor((void**)&input);
    ctx->timing_infer_ms += fiv_get_current_system_time() - _t_inf;
    ivf64 _t_post = fiv_get_current_system_time();
    if (run_result != FIV_RET_OK) return run_result;
    if (fiv_y26_model_classify(ctx, result) != 0) return FIV_RET_ERR_MEM;
    ctx->timing_post_ms += fiv_get_current_system_time() - _t_post;
    ctx->timing_calls += 1;
    return FIV_RET_OK;
}

/* ---- segmenter: mask coefficients only (Proto26 integration comes next). -- */
/* ---- segmenter: decode + Proto26 + per-instance mask rebuild ---- */
static fiv_ret fiv_y26_model_run_seg(fiv_yolo26_model* model, void* result,
                                     fiv_mat* image)
{
    fiv_yolo26_seg_result* out = (fiv_yolo26_seg_result*)result;
    out->count = 0;
    out->proto = NULL;
    out->masks = NULL;
    out->proto_grid_w = 0.0f;
    out->proto_grid_h = 0.0f;

    ivf64 _t_pre = fiv_get_current_system_time();
    fiv_tensor4d* input = fiv_y26_model_letterbox(model, image);
    if (input == NULL) return FIV_RET_ERR_MEM;
    model->timing_pre_ms += fiv_get_current_system_time() - _t_pre;
    ivf64 _t_inf = fiv_get_current_system_time();
    fiv_ret run_result = fiv_y26_model_forward(model, input);
    fiv_release_tensor((void**)&input);
    model->timing_infer_ms += fiv_get_current_system_time() - _t_inf;
    ivf64 _t_post = fiv_get_current_system_time();
    if (run_result != FIV_RET_OK) return run_result;

    const fiv_tensor4d* heads[9];
    for (int level = 0; level < 3; level++) {
        heads[level] = (const fiv_tensor4d*)fiv_neural_network_get_node_output(
            model->net, model->graph->head_node[level]);
        heads[3 + level] = (const fiv_tensor4d*)fiv_neural_network_get_node_output(
            model->net, model->graph->head_node[3 + level]);
        heads[6 + level] = (const fiv_tensor4d*)fiv_neural_network_get_node_output(
            model->net, model->graph->mask_node[level]);
        if (heads[level] == NULL || heads[3 + level] == NULL || heads[6 + level] == NULL)
            return FIV_RET_ERR_PARA;
    }
    int strides[3];
    for (int level = 0; level < 3; level++)
        strides[level] = model->input_size / (int)heads[level]->width;

    const int columns = 6 + FIV_YOLO26_MASK_DIMS;   /* 38 */
    ivf32* decode_rows = (ivf32*)fiv_malloc(
        (size_t)FIV_YOLO26_MAX_DETECTIONS * (size_t)columns * sizeof(ivf32));
    if (decode_rows == NULL) return FIV_RET_ERR_MEM;
    int kept = fiv_yolo26_postprocess_seg(heads, strides, decode_rows,
                                          FIV_YOLO26_MAX_DETECTIONS);
    if (kept < 0) {
        fiv_free(decode_rows);
        return FIV_RET_ERR_PARA;
    }
    out->count = kept;
    for (int i = 0; i < kept && i < FIV_YOLO26_MAX_DETECTIONS; i++) {
        const ivf32* row = decode_rows + (size_t)columns * i;
        fiv_yolo26_box* box = &out->detections[i];
        box->x1 = row[0];
        box->y1 = row[1];
        box->x2 = row[2];
        box->y2 = row[3];
        box->score = row[4];
        box->class_id = (int)row[5];
        out->mask_rows[i] = -1;
        for (int m = 0; m < FIV_YOLO26_MASK_DIMS; m++)
            out->coef[i][m] = row[6 + m];
    }
    fiv_free(decode_rows);

    /* ---- Proto26: P3/P4/P5 are the layer-16/19/22 features the heads sat on.
       Reuse those engine-resident tensors; spatial dims equal head level 0. -- */
    if (model->proto_w[0] == NULL) {
        model->timing_post_ms += fiv_get_current_system_time() - _t_post;
        model->timing_calls += 1;
        return FIV_RET_OK;   /* proto weights absent */
    }
    {
        const fiv_tensor4d* p3 = (const fiv_tensor4d*)fiv_neural_network_get_node_output(
            model->net, model->graph->layer_node[16]);
        const fiv_tensor4d* p4 = (const fiv_tensor4d*)fiv_neural_network_get_node_output(
            model->net, model->graph->layer_node[19]);
        const fiv_tensor4d* p5 = (const fiv_tensor4d*)fiv_neural_network_get_node_output(
            model->net, model->graph->layer_node[22]);
        if (p3 == NULL || p4 == NULL || p5 == NULL) return FIV_RET_ERR_PARA;

        int height = (int)heads[0]->height;   /* P3 feature h/w */
        int width  = (int)heads[0]->width;
        int proto_h = height * 2;
        int proto_w = width * 2;
        size_t proto_plane = (size_t)proto_h * (size_t)proto_w;

        /* grow ctx scratch when needed */
        if (model->proto_plane < proto_plane) {
            fiv_free(model->proto_out);
            fiv_free(model->mask_scratch);
            model->proto_out = (ivf32*)fiv_malloc(
                (size_t)FIV_YOLO26_MASK_DIMS * proto_plane * sizeof(ivf32));
            model->mask_scratch = (ivf32*)fiv_malloc(
                (size_t)FIV_YOLO26_MAX_DETECTIONS * proto_plane * sizeof(ivf32));
            if (model->proto_out == NULL || model->mask_scratch == NULL)
                return FIV_RET_ERR_MEM;
            model->proto_plane = proto_plane;
        }

        fiv_yolo26_proto_weights weights;
        memset(&weights, 0, sizeof(weights));
        weights.refine0_w = model->proto_w[0]; weights.refine0_b = model->proto_b[0];
        weights.refine1_w = model->proto_w[1]; weights.refine1_b = model->proto_b[1];
        weights.fuse_w    = model->proto_w[2]; weights.fuse_b    = model->proto_b[2];
        weights.cv1_w     = model->proto_w[3]; weights.cv1_b     = model->proto_b[3];
        weights.upsample_w = model->proto_w[4]; weights.upsample_b = model->proto_b[4];
        weights.cv2_w     = model->proto_w[5]; weights.cv2_b     = model->proto_b[5];
        weights.cv3_w     = model->proto_w[6]; weights.cv3_b     = model->proto_b[6];

        int out_channels = 0, out_height = 0, out_width = 0;
        if (fiv_yolo26_proto_run(
                ((fiv_tensor_hdr*)p3)->data.fl, (int)p3->channels,
                ((fiv_tensor_hdr*)p4)->data.fl, (int)p4->channels,
                ((fiv_tensor_hdr*)p5)->data.fl, (int)p5->channels,
                height, width, &weights, model->proto_out,
                &out_channels, &out_height, &out_width) != 0)
            return FIV_RET_ERR_PARA;

        out->proto = model->proto_out;
        out->proto_grid_w = (ivf32)out_width;
        out->proto_grid_h = (ivf32)out_height;

        /* per-instance mask = sigmoid( sum_m coef[m] * proto[m] ) on the grid.
           Masks are only rebuilt for rows that actually look like detections
           (score >= 0.05); the top-k rows below that are background clutter and
           building 80x80 masks for all 300 of them would dominate the call. */
        ivf32* mask_rows_data = model->mask_scratch;
        for (int i = 0; i < kept && i < FIV_YOLO26_MAX_DETECTIONS; i++) {
            if (out->detections[i].score < 0.05f) continue;
            ivf32* mask = mask_rows_data + (size_t)i * proto_plane;
            for (size_t p = 0; p < proto_plane; p++) {
                ivf32 acc = 0.0f;
                for (int m = 0; m < FIV_YOLO26_MASK_DIMS; m++)
                    acc += out->coef[i][m] *
                           model->proto_out[(size_t)m * proto_plane + p];
                mask[p] = 1.0f / (1.0f + expf(-acc));
            }
            out->mask_rows[i] = i;
        }
        out->masks = mask_rows_data;
    }
    model->timing_post_ms += fiv_get_current_system_time() - _t_post;
    model->timing_calls += 1;
    return FIV_RET_OK;
}

fiv_ret fiv_yolo26_segmenter_on_image(void* result, fiv_mat* image, void* model)
{
    if (result == NULL || image == NULL || model == NULL) return FIV_RET_ERR_PARA;
    fiv_yolo26_model* ctx = (fiv_yolo26_model*)model;
    if (ctx->task != FIV_YOLO26_TASK_SEGMENTER) return FIV_RET_ERR_PARA;
    return fiv_y26_model_run_seg(ctx, result, image);
}

static fiv_ret fiv_y26_model_release(fiv_yolo26_model* model)
{
    if (model == NULL) return FIV_RET_OK;
    if (model->graph != NULL) fiv_yolo26_release(model->graph);
    fiv_free((void*)model->linear_w);
    fiv_free((void*)model->linear_b);
    fiv_free((void*)model->head_w);
    fiv_free((void*)model->head_b);
    fiv_free(model->cls_scratch);
    for (int i = 0; i < 7; i++) {
        fiv_free((void*)model->proto_w[i]);
        fiv_free((void*)model->proto_b[i]);
    }
    fiv_free(model->proto_out);
    fiv_free(model->mask_scratch);
    fiv_free(model);
    return FIV_RET_OK;
}

fiv_ret fiv_release_yolo26_detector(void** model)
{
    if (model == NULL || *model == NULL) return FIV_RET_ERR_PARA;
    fiv_yolo26_model* ctx = (fiv_yolo26_model*)*model;
    if (ctx->task != FIV_YOLO26_TASK_DETECTOR) return FIV_RET_ERR_PARA;
    fiv_ret ret = fiv_y26_model_release(ctx);
    *model = NULL;
    return ret;
}

fiv_ret fiv_release_yolo26_segmenter(void** model)
{
    if (model == NULL || *model == NULL) return FIV_RET_ERR_PARA;
    fiv_yolo26_model* ctx = (fiv_yolo26_model*)*model;
    if (ctx->task != FIV_YOLO26_TASK_SEGMENTER) return FIV_RET_ERR_PARA;
    fiv_ret ret = fiv_y26_model_release(ctx);
    *model = NULL;
    return ret;
}

fiv_ret fiv_release_yolo26_pose(void** model)
{
    if (model == NULL || *model == NULL) return FIV_RET_ERR_PARA;
    fiv_yolo26_model* ctx = (fiv_yolo26_model*)*model;
    if (ctx->task != FIV_YOLO26_TASK_POSE) return FIV_RET_ERR_PARA;
    fiv_ret ret = fiv_y26_model_release(ctx);
    *model = NULL;
    return ret;
}

fiv_ret fiv_release_yolo26_classifier(void** model)
{
    if (model == NULL || *model == NULL) return FIV_RET_ERR_PARA;
    fiv_yolo26_model* ctx = (fiv_yolo26_model*)*model;
    if (ctx->task != FIV_YOLO26_TASK_CLASSIFIER) return FIV_RET_ERR_PARA;
    fiv_ret ret = fiv_y26_model_release(ctx);
    *model = NULL;
    return ret;
}

fiv_ret fiv_yolo26_get_timing(void* model, ivf64 avg_ms[3], int* calls)
{
    fiv_yolo26_model* ctx = (fiv_yolo26_model*)model;
    if (!ctx) return FIV_RET_ERR_PARA;
    int n = ctx->timing_calls;
    if (avg_ms) {
        avg_ms[0] = n > 0 ? ctx->timing_pre_ms / (ivf64)n   : 0.0;
        avg_ms[1] = n > 0 ? ctx->timing_infer_ms / (ivf64)n : 0.0;
        avg_ms[2] = n > 0 ? ctx->timing_post_ms / (ivf64)n  : 0.0;
    }
    if (calls) *calls = n;
    return FIV_RET_OK;
}
