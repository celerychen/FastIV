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
 * BlazeFace short-range face detector: builds the fiv_nn graph (stem + 16
 * blocks + 4 heads), runs inference, and applies decode/sigmoid/weighted-NMS/
 * project postprocess. The weights loader and the ROI warp live in their own
 * modules (fiv_face_weights / fiv_face_warp); this file only wires them.
 */

#include "fiv_face.h"
#include "fiv_face_weights.h"
#include "fiv_face_warp.h"

#include "fiv_nn.h"
#include "fiv_nn_infer.h"
#include "fiv_nn_conv2d.h"
#include "fiv_ctensor.h"
#include "fiv_common.h"
#include "fiv_data_typedefs.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- detector context ---- */
typedef struct {
    void*         net;
    fiv_weights*  weight_blob;
    fiv_tensor4d* input;
    iv8u*         warped;
    ivf32         warp_matrix[9]; /* ROI homography used by the warp impl */
    int           img_w;
    int           img_h;
    int           reg8_node, cls8_node;
    int           reg16_node, cls16_node;
    int           block_relu[16];
    int           stem_relu;
} fiv_face_detector;

/* graph builder (stem + blocks 0..15, heads at block 10 / block 15) */
static const int block_specs[16][4] = {
    {24, 24, 1, 0}, {24, 28, 1, 0}, {28, 32, 2, 1}, {32, 36, 1, 0},
    {36, 42, 1, 0}, {42, 48, 2, 1}, {48, 56, 1, 0}, {56, 64, 1, 0},
    {64, 72, 1, 0}, {72, 80, 1, 0}, {80, 88, 1, 0}, {88, 96, 2, 1},
    {96, 96, 1, 0}, {96, 96, 1, 0}, {96, 96, 1, 0}, {96, 96, 1, 0},
};

static fiv_conv2d_params make_conv_params(int method, int kern_x, int kern_y, int stride,
                                          int pad_top, int pad_bottom, int pad_left, int pad_right,
                                          int in_ch, int out_ch, int use_bias) {
    fiv_conv2d_params params;
    memset(&params, 0, sizeof(params));
    params.conv2d_method  = method;
    params.kernel_size_x  = kern_x;
    params.kernel_size_y  = kern_y;
    params.stride         = stride;
    params.padding_method = 0;
    params.input_channels = in_ch;
    params.output_channels = out_ch;
    params.bias           = use_bias;
    params.pad_top        = pad_top;
    params.pad_bottom     = pad_bottom;
    params.pad_left       = pad_left;
    params.pad_right      = pad_right;
    return params;
}

static int add_node(void* net, int node_type, int src_node, void* node_params) {
    int node_id = ((fiv_nn_network_context*)net)->node_count;
    fiv_ret ret = fiv_neural_network_add_node(net, node_type, src_node, node_id, node_params);
    return ret == FIV_RET_OK ? node_id : -1;
}

static int add_multi(void* net, int node_type, const int* src_nodes, int num_src, void* node_params) {
    int node_id = ((fiv_nn_network_context*)net)->node_count;
    fiv_ret ret = fiv_neural_network_add_node_multi(net, node_type, src_nodes, num_src, node_id, node_params);
    return ret == FIV_RET_OK ? node_id : -1;
}

static void load_conv(const fiv_weights* weight_blob, fiv_conv2d_node* cn,
                      const char* kname, const char* bname) {
    const ivf32* kernel_ptr = fiv_weights_get(weight_blob, kname, NULL, NULL);
    const ivf32* bias_ptr   = fiv_weights_get(weight_blob, bname, NULL, NULL);
    if (!kernel_ptr || !bias_ptr) {
        fprintf(stderr, "missing weights %s/%s\n", kname, bname);
        exit(1);
    }
    int   out_ch = cn->params.output_channels;
    int   in_ch  = cn->params.input_channels;
    int   kern_h = cn->params.kernel_size_y;
    int   kern_w = cn->params.kernel_size_x;
    ivf32* weight_ptr = cn->weight->data.fl;
    if (cn->params.conv2d_method == FIV_CONV2D_DEPTHWISE) {
        for (int oc = 0; oc < out_ch; oc++)
            for (int ky = 0; ky < kern_h; ky++)
                for (int kx = 0; kx < kern_w; kx++)
                    weight_ptr[(oc * kern_h + ky) * kern_w + kx] =
                        kernel_ptr[(ky * kern_w + kx) * in_ch + oc];
    } else {
        for (int oc = 0; oc < out_ch; oc++)
            for (int ic = 0; ic < in_ch; ic++)
                for (int ky = 0; ky < kern_h; ky++)
                    for (int kx = 0; kx < kern_w; kx++)
                        weight_ptr[((oc * in_ch + ic) * kern_h + ky) * kern_w + kx] =
                            kernel_ptr[((oc * kern_h + ky) * kern_w + kx) * in_ch + ic];
    }
    if (cn->bias)
        memcpy(cn->bias->data.fl, bias_ptr, (size_t)out_ch * sizeof(ivf32));
}

static void build_net(void* net, const fiv_weights* weight_blob, fiv_face_detector* det) {
    det->reg8_node  = det->cls8_node  = -1;
    det->reg16_node = det->cls16_node = -1;

    fiv_conv2d_params stem_cp = make_conv_params(FIV_CONV2D_STD, 5, 5, 2, 1, 2, 1, 2, 3, 24, 1);
    int stem_conv_node = add_node(net, FIV_NN_NODE_CONV2D_STD, 0, &stem_cp);
    load_conv(weight_blob, (fiv_conv2d_node*)((fiv_nn_network_context*)net)->nodes[stem_conv_node].op,
              "conv2d/Kernel", "conv2d/Bias");
    int stem_relu_node = add_node(net, FIV_NN_NODE_RELU, stem_conv_node, NULL);
    det->stem_relu = stem_relu_node;
    int feats = stem_relu_node;

    for (int i = 0; i < 16; i++) {
        int in_channels  = block_specs[i][0];
        int out_channels = block_specs[i][1];
        int stride       = block_specs[i][2];
        int use_maxpool  = block_specs[i][3];
        char dw_kname[48], dw_bname[48], pw_kname[48], pw_bname[48];
        if (i == 0) strcpy(dw_kname, "depthwise_conv2d/Kernel");
        else         snprintf(dw_kname, sizeof(dw_kname), "depthwise_conv2d_%d/Kernel", i);
        if (i == 0) strcpy(dw_bname, "depthwise_conv2d/Bias");
        else         snprintf(dw_bname, sizeof(dw_bname), "depthwise_conv2d_%d/Bias", i);
        snprintf(pw_kname, sizeof(pw_kname), "conv2d_%d/Kernel", i + 1);
        snprintf(pw_bname, sizeof(pw_bname), "conv2d_%d/Bias", i + 1);

        int pad_t = (stride == 1) ? 1 : 0;
        int pad_d = (stride == 1) ? 1 : 1;
        fiv_conv2d_params dw_cp = make_conv_params(FIV_CONV2D_DEPTHWISE, 3, 3, stride, pad_t, pad_d, pad_t, pad_d, in_channels, in_channels, 1);
        int dw_node = add_node(net, FIV_NN_NODE_CONV2D_DEPTHWISE, feats, &dw_cp);
        load_conv(weight_blob, (fiv_conv2d_node*)((fiv_nn_network_context*)net)->nodes[dw_node].op, dw_kname, dw_bname);

        fiv_conv2d_params pw_cp = make_conv_params(FIV_CONV2D_POINTWISE, 1, 1, 1, 0, 0, 0, 0, in_channels, out_channels, 1);
        int pw_node = add_node(net, FIV_NN_NODE_CONV2D_POINTWISE, dw_node, &pw_cp);
        load_conv(weight_blob, (fiv_conv2d_node*)((fiv_nn_network_context*)net)->nodes[pw_node].op, pw_kname, pw_bname);

        int res_src = feats;
        if (in_channels != out_channels) {
            if (use_maxpool) {
                int max_node = add_node(net, FIV_NN_NODE_MAX2D, feats, NULL);
                res_src = max_node;
            }
            fiv_pad_node_params pp;
            pp.output_channels = out_channels;
            int pad_node = add_node(net, FIV_NN_NODE_PAD, res_src, &pp);
            res_src = pad_node;
        }
        int add_srcs[2] = { pw_node, res_src };
        int add_node_id = add_multi(net, FIV_NN_NODE_ADD, add_srcs, 2, NULL);
        int relu_node = add_node(net, FIV_NN_NODE_RELU, add_node_id, NULL);
        det->block_relu[i] = relu_node;
        feats = relu_node;

        if (i == 10) {
            fiv_conv2d_params reg8_cp = make_conv_params(FIV_CONV2D_POINTWISE, 1, 1, 1, 0, 0, 0, 0, out_channels, 32, 1);
            int reg_node = add_node(net, FIV_NN_NODE_CONV2D_POINTWISE, feats, &reg8_cp);
            load_conv(weight_blob, (fiv_conv2d_node*)((fiv_nn_network_context*)net)->nodes[reg_node].op,
                      "regressor_8/Kernel", "regressor_8/Bias");
            det->reg8_node = reg_node;
            fiv_conv2d_params cls8_cp = make_conv_params(FIV_CONV2D_POINTWISE, 1, 1, 1, 0, 0, 0, 0, out_channels, 2, 1);
            int cls_node = add_node(net, FIV_NN_NODE_CONV2D_POINTWISE, feats, &cls8_cp);
            load_conv(weight_blob, (fiv_conv2d_node*)((fiv_nn_network_context*)net)->nodes[cls_node].op,
                      "classificator_8/Kernel", "classificator_8/Bias");
            det->cls8_node = cls_node;
        } else if (i == 15) {
            fiv_conv2d_params reg16_cp = make_conv_params(FIV_CONV2D_POINTWISE, 1, 1, 1, 0, 0, 0, 0, out_channels, 96, 1);
            int reg_node = add_node(net, FIV_NN_NODE_CONV2D_POINTWISE, feats, &reg16_cp);
            load_conv(weight_blob, (fiv_conv2d_node*)((fiv_nn_network_context*)net)->nodes[reg_node].op,
                      "regressor_16/Kernel", "regressor_16/Bias");
            det->reg16_node = reg_node;
            fiv_conv2d_params cls16_cp = make_conv_params(FIV_CONV2D_POINTWISE, 1, 1, 1, 0, 0, 0, 0, out_channels, 6, 1);
            int cls_node = add_node(net, FIV_NN_NODE_CONV2D_POINTWISE, feats, &cls16_cp);
            load_conv(weight_blob, (fiv_conv2d_node*)((fiv_nn_network_context*)net)->nodes[cls_node].op,
                      "classificator_16/Kernel", "classificator_16/Bias");
            det->cls16_node = cls_node;
        }
    }
}

/* postprocess: decode + sigmoid + weighted NMS + project */
static const ivf32 x_scale = 128.0f, y_scale = 128.0f;
static const ivf32 w_scale = 128.0f, h_scale = 128.0f;

static ivf32 sigmoidf(ivf32 x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    ivf32 exp_val = expf(x);
    return exp_val / (1.0f + exp_val);
}

static ivf32 calc_scale(ivf32 min_s, ivf32 max_s, int idx, int n) {
    if (n == 1) return (min_s + max_s) * 0.5f;
    return min_s + (max_s - min_s) * (ivf32)idx / (ivf32)(n - 1);
}

static void generate_anchors(ivf32 anchors[FIV_FACE_NUM_BOXES][4]) {
    int  input_size = FIV_FACE_TENSOR_SIZE;
    int  num_layers = 4;
    int  strides[4] = {8, 16, 16, 16};
    ivf32 min_scale = 0.1484375f, max_scale = 0.75f, offset = 0.5f;
    int  layer_id   = 0, idx = 0;
    while (layer_id < num_layers) {
        int last = layer_id;
        while (last < num_layers && strides[last] == strides[layer_id]) last++;
        ivf32 scale_buf[16], aspect_buf[16];
        int  na = 0;
        for (int ls = layer_id; ls < last; ls++) {
            ivf32 scale_val = calc_scale(min_scale, max_scale, ls, num_layers);
            aspect_buf[na] = 1.0f; scale_buf[na] = scale_val; na++;
            ivf32 scale_next = (ls == num_layers - 1) ? 1.0f
                                                         : calc_scale(min_scale, max_scale, ls + 1, num_layers);
            scale_buf[na] = sqrtf(scale_val * scale_next); aspect_buf[na] = 1.0f; na++;
        }
        int fm = (int)ceilf((ivf32)input_size / (ivf32)strides[layer_id]);
        for (int y = 0; y < fm; y++)
            for (int x = 0; x < fm; x++)
                for (int aid = 0; aid < na; aid++) {
                    anchors[idx][0] = (x + offset) / (ivf32)fm;
                    anchors[idx][1] = (y + offset) / (ivf32)fm;
                    anchors[idx][2] = 1.0f;
                    anchors[idx][3] = 1.0f;
                    idx++;
                }
        layer_id = last;
    }
}

typedef struct {
    ivf32 score;
    ivf32 bbox[4];
    ivf32 kps[FIV_FACE_KEYPOINTS][2];
} raw_det;

static ivf32 iou(const ivf32 box_a[4], const ivf32 box_b[4]) {
    ivf32 ix1 = box_a[0] > box_b[0] ? box_a[0] : box_b[0];
    ivf32 iy1 = box_a[1] > box_b[1] ? box_a[1] : box_b[1];
    ivf32 ix2 = box_a[2] < box_b[2] ? box_a[2] : box_b[2];
    ivf32 iy2 = box_a[3] < box_b[3] ? box_a[3] : box_b[3];
    ivf32 iw = ix2 > ix1 ? ix2 - ix1 : 0.0f;
    ivf32 ih = iy2 > iy1 ? iy2 - iy1 : 0.0f;
    ivf32 inter = iw * ih;
    ivf32 area_a = (box_a[2] > box_a[0] ? box_a[2] - box_a[0] : 0.0f) * (box_a[3] > box_a[1] ? box_a[3] - box_a[1] : 0.0f);
    ivf32 area_b = (box_b[2] > box_b[0] ? box_b[2] - box_b[0] : 0.0f) * (box_b[3] > box_b[1] ? box_b[3] - box_b[1] : 0.0f);
    ivf32 union_area = area_a + area_b - inter;
    return union_area > 0.0f ? inter / union_area : 0.0f;
}

static int weighted_nms(const raw_det* raw_dets, int n, fiv_face_detection* out_dets, int max_out) {
    int*   order = (int*)fiv_calloc((size_t)n, sizeof(int));
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (raw_dets[order[j]].score > raw_dets[order[i]].score) {
                int temp_idx = order[i];
                order[i] = order[j];
                order[j] = temp_idx;
            }
    iv8u* suppressed = (iv8u*)fiv_calloc((size_t)n, sizeof(iv8u));
    int   out_count  = 0;
    for (int i = 0; i < n && out_count < max_out; i++) {
        int anchor_idx = order[i];
        if (suppressed[anchor_idx]) continue;
        int cand_idx[1024];
        int cand_count = 0;
        for (int j = 0; j < n; j++) {
            if (suppressed[order[j]]) continue;
            if (iou(raw_dets[order[j]].bbox, raw_dets[anchor_idx].bbox) > FIV_FACE_NMS_THRESH)
                cand_idx[cand_count++] = order[j];
        }
        ivf32 total = 0.0f;
        ivf32 wx1 = 0, wy1 = 0, wx2 = 0, wy2 = 0;
        ivf32 wkx[FIV_FACE_KEYPOINTS], wky[FIV_FACE_KEYPOINTS];
        for (int k = 0; k < FIV_FACE_KEYPOINTS; k++) { wkx[k] = 0; wky[k] = 0; }
        for (int c = 0; c < cand_count; c++) {
            const raw_det* det = &raw_dets[cand_idx[c]];
            ivf32 score_val = det->score;
            total += score_val;
            wx1 += det->bbox[0] * score_val;
            wy1 += det->bbox[1] * score_val;
            wx2 += det->bbox[2] * score_val;
            wy2 += det->bbox[3] * score_val;
            for (int k = 0; k < FIV_FACE_KEYPOINTS; k++) {
                wkx[k] += det->kps[k][0] * score_val;
                wky[k] += det->kps[k][1] * score_val;
            }
        }
        fiv_face_detection* out_det = &out_dets[out_count++];
        out_det->score = raw_dets[anchor_idx].score;
        out_det->x = wx1 / total;
        out_det->y = wy1 / total;
        out_det->w = wx2 / total - out_det->x;
        out_det->h = wy2 / total - out_det->y;
        for (int k = 0; k < FIV_FACE_KEYPOINTS; k++) {
            out_det->kps[k][0] = wkx[k] / total;
            out_det->kps[k][1] = wky[k] / total;
        }
        for (int c = 0; c < cand_count; c++) suppressed[cand_idx[c]] = 1;
    }
    fiv_free(order);
    fiv_free(suppressed);
    return out_count;
}

static int postprocess(const ivf32 reg_pred[FIV_FACE_NUM_BOXES][FIV_FACE_NUM_COORDS], const ivf32 cls_score[],
                       int img_w, int img_h, ivf32 min_score,
                       fiv_face_detection* out_dets, int max_dets) {
    ivf32 anchors[FIV_FACE_NUM_BOXES][4];
    generate_anchors(anchors);
    raw_det raw_dets[1024];
    int     raw_count = 0;
    for (int i = 0; i < FIV_FACE_NUM_BOXES; i++) {
        ivf32 anc_cx = anchors[i][0], anc_cy = anchors[i][1], anc_w = anchors[i][2], anc_h = anchors[i][3];
        ivf32 dx = reg_pred[i][0] / x_scale * anc_w + anc_cx;
        ivf32 dy = reg_pred[i][1] / y_scale * anc_h + anc_cy;
        ivf32 dw = (reg_pred[i][2] / w_scale) * anc_w;
        ivf32 dh = (reg_pred[i][3] / h_scale) * anc_h;
        ivf32 xmin = dx - dw * 0.5f, ymin = dy - dh * 0.5f;
        ivf32 xmax = dx + dw * 0.5f, ymax = dy + dh * 0.5f;
        ivf32 score_val = sigmoidf(cls_score[i]);
        if (score_val < min_score) continue;
        if ((xmax - xmin) < 0.0f || (ymax - ymin) < 0.0f) continue;
        if (raw_count >= 1024) break;
        raw_det* det = &raw_dets[raw_count++];
        det->score = score_val;
        det->bbox[0] = xmin; det->bbox[1] = ymin; det->bbox[2] = xmax; det->bbox[3] = ymax;
        for (int k = 0; k < FIV_FACE_KEYPOINTS; k++) {
            det->kps[k][0] = reg_pred[i][4 + 2 * k] / x_scale * anc_w + anc_cx;
            det->kps[k][1] = reg_pred[i][5 + 2 * k] / y_scale * anc_h + anc_cy;
        }
    }
    fiv_face_detection* nms_dets = (fiv_face_detection*)fiv_calloc((size_t)(raw_count > 0 ? raw_count : 1), sizeof(fiv_face_detection));
    int kept_count = weighted_nms(raw_dets, raw_count, nms_dets, raw_count > 0 ? raw_count : 1);

    ivf32 scale = (FIV_FACE_TENSOR_SIZE / (ivf32)img_w < FIV_FACE_TENSOR_SIZE / (ivf32)img_h)
                      ? FIV_FACE_TENSOR_SIZE / (ivf32)img_w
                      : FIV_FACE_TENSOR_SIZE / (ivf32)img_h;
    ivf32 tx = (FIV_FACE_TENSOR_SIZE - scale * (ivf32)img_w) * 0.5f;
    ivf32 ty = (FIV_FACE_TENSOR_SIZE - scale * (ivf32)img_h) * 0.5f;

    int out_count = 0;
    for (int i = 0; i < kept_count && out_count < max_dets; i++) {
        fiv_face_detection* src    = &nms_dets[i];
        ivf32 xmin = src->x, ymin = src->y;
        ivf32 xmax = src->x + src->w, ymax = src->y + src->h;
        fiv_face_detection* out_det = &out_dets[out_count++];
        out_det->score = src->score;
        out_det->x = (xmin * FIV_FACE_TENSOR_SIZE - tx) / scale;
        out_det->y = (ymin * FIV_FACE_TENSOR_SIZE - ty) / scale;
        out_det->w = ((xmax * FIV_FACE_TENSOR_SIZE - tx) / scale) - out_det->x;
        out_det->h = ((ymax * FIV_FACE_TENSOR_SIZE - ty) / scale) - out_det->y;
        for (int k = 0; k < FIV_FACE_KEYPOINTS; k++) {
            ivf32 kx = (src->kps[k][0] * FIV_FACE_TENSOR_SIZE - tx) / scale;
            ivf32 ky = (src->kps[k][1] * FIV_FACE_TENSOR_SIZE - ty) / scale;
            out_det->kps[k][0] = kx / (ivf32)img_w;
            out_det->kps[k][1] = ky / (ivf32)img_h;
        }
    }
    fiv_free(nms_dets);
    return out_count;
}

/* preprocess: center-square ROI warp (bilinear, zero border) + norm */
static void preprocess(fiv_face_detector* det, const fiv_mat* image) {
    int w = (int)image->width;
    int h = (int)image->height;
    det->img_w = w;
    det->img_h = h;
    fiv_detection_warp_matrix(w, h, FIV_FACE_TENSOR_SIZE, det->warp_matrix);
    fiv_warp_perspective_u8(image->data.ptr8u, h, w, 3, det->warp_matrix,
                        FIV_FACE_TENSOR_SIZE, FIV_FACE_TENSOR_SIZE, det->warped, 0);
    for (int y = 0; y < FIV_FACE_TENSOR_SIZE; y++)
        for (int x = 0; x < FIV_FACE_TENSOR_SIZE; x++)
            for (int c = 0; c < 3; c++) {
                iv8u pixel_val = det->warped[((size_t)y * FIV_FACE_TENSOR_SIZE + x) * 3 + c];
                det->input->data.fl[(c * FIV_FACE_TENSOR_SIZE + y) * FIV_FACE_TENSOR_SIZE + x] =
                    (ivf32)pixel_val * (2.0f / 255.0f) - 1.0f;
            }
}

/* flatten the four head tensors into reg[896][16] / scores[896] */
static void collect_heads(fiv_face_detector* det,
                          ivf32 reg_pred[FIV_FACE_NUM_BOXES][FIV_FACE_NUM_COORDS], ivf32 cls_score[FIV_FACE_NUM_BOXES]) {
    fiv_nn_network_context* nctx = (fiv_nn_network_context*)det->net;
    const fiv_tensor4d* reg8 = (const fiv_tensor4d*)nctx->nodes[det->reg8_node].output;
    const fiv_tensor4d* cls8 = (const fiv_tensor4d*)nctx->nodes[det->cls8_node].output;
    for (int y = 0; y < 16; y++)
        for (int xx = 0; xx < 16; xx++)
            for (int a = 0; a < 2; a++) {
                int anchor = (y * 16 + xx) * 2 + a;
                for (int c = 0; c < 16; c++)
                    reg_pred[anchor][c] = reg8->data.fl[((a * 16 + c) * 16 + y) * 16 + xx];
                cls_score[anchor] = cls8->data.fl[(a * 16 + y) * 16 + xx];
            }
    const fiv_tensor4d* reg16 = (const fiv_tensor4d*)nctx->nodes[det->reg16_node].output;
    const fiv_tensor4d* cls16 = (const fiv_tensor4d*)nctx->nodes[det->cls16_node].output;
    for (int y = 0; y < 8; y++)
        for (int xx = 0; xx < 8; xx++)
            for (int lid = 0; lid < 6; lid++) {
                int anchor = 512 + (y * 8 + xx) * 6 + lid;
                for (int c = 0; c < 16; c++)
                    reg_pred[anchor][c] = reg16->data.fl[((lid * 16 + c) * 8 + y) * 8 + xx];
                cls_score[anchor] = cls16->data.fl[(lid * 8 + y) * 8 + xx];
            }
}

/* public API */
void* fiv_create_face_detetor(char* model_name) {
    fiv_face_detector* det = (fiv_face_detector*)fiv_calloc(1, sizeof(fiv_face_detector));
    if (!det) return NULL;

    det->weight_blob = (fiv_weights*)fiv_calloc(1, sizeof(fiv_weights));
    if (!det->weight_blob) {
        fiv_free(det);
        return NULL;
    }

    if (fiv_weights_load(model_name, det->weight_blob) != 1) {
        fiv_weights_free(det->weight_blob);
        fiv_free(det->weight_blob);
        fiv_free(det);
        return NULL;
    }

    det->net = fiv_create_neural_network();
    if (!det->net) {
        fiv_weights_free(det->weight_blob);
        fiv_free(det->weight_blob);
        fiv_free(det);
        return NULL;
    }
    build_net(det->net, det->weight_blob, det);

    size_t input_shape[4] = { 1, 3, FIV_FACE_TENSOR_SIZE, FIV_FACE_TENSOR_SIZE };
    det->input = fiv_create_tensor4d(input_shape, FIV_32F1);
    det->warped = (iv8u*)fiv_calloc((size_t)FIV_FACE_TENSOR_SIZE * FIV_FACE_TENSOR_SIZE * 3, sizeof(iv8u));
    if (!det->input || !det->warped) {
        fiv_release_tensor((void**)&det->input);
        fiv_free(det->warped);
        fiv_release_neural_network((void**)&det->net);
        fiv_weights_free(det->weight_blob);
        fiv_free(det);
        return NULL;
    }
    return det;
}

fiv_ret fiv_face_detector_on_image(void* face_info, fiv_mat* image, void* detector) {
    if (!face_info || !image || !detector) return FIV_RET_ERR_PARA;
    fiv_face_detector* det    = (fiv_face_detector*)detector;
    fiv_face_result*   result = (fiv_face_result*)face_info;

    preprocess(det, image);

    void* infer_out = NULL;
    fiv_ret ret = fiv_nn_run_inference((fiv_nn_network_context*)det->net, det->input, &infer_out);
    if (ret != FIV_RET_OK) return ret;

    ivf32 reg_pred[FIV_FACE_NUM_BOXES][FIV_FACE_NUM_COORDS];
    ivf32 cls_score[FIV_FACE_NUM_BOXES];
    collect_heads(det, reg_pred, cls_score);

    int det_count = postprocess(reg_pred, cls_score, det->img_w, det->img_h,
                                FIV_FACE_MIN_SCORE, result->detections, FIV_FACE_MAX_DETS);
    result->count = det_count;
    return FIV_RET_OK;
}

fiv_ret fiv_release_face_detector(void** detector) {
    if (!detector || !*detector) return FIV_RET_ERR_PARA;
    fiv_face_detector* det = (fiv_face_detector*)*detector;
    fiv_release_tensor((void**)&det->input);
    fiv_free(det->warped);
    fiv_release_neural_network((void**)&det->net);
    fiv_weights_free(det->weight_blob);
    fiv_free(det->weight_blob);
    fiv_free(det);
    *detector = NULL;
    return FIV_RET_OK;
}
