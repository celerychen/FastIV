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

#include "fiv_yunet.h"
#include "fiv_common.h"
#include "fiv_nn.h"
#include "fiv_nn_infer.h"
#include "fiv_nn_conv2d.h"

#include <math.h>
#include <string.h>

/* YuNet input preprocessing. The network's first real layer is a fused
   stride-2 3x3 direct convolution that reads the raw RGB pixels, so this step
   ONLY packs the image into 3 channel planes and pads the height/width up to
   multiples of 32 (zero-filled) to reproduce the reference align-to-32 grid:
     - in-bounds pixels   -> their R/G/B ivf32 value in channel 0/1/2
     - alignment padding  -> 0 (out-of-range 3x3 taps read zero)
   The old 32-channel im2col expansion is gone: FastIV does direct convolution,
   not GEMM, so no im2col matrix is needed. */
fiv_tensor4d* fiv_yunet_preprocess(const iv8u* src, int width, int height, int width_step)
{
    if (!src || width <= 0 || height <= 0) return NULL;

    int rows = ((height - 1) / 32 + 1) * 32;   /* aligned to multiple of 32 */
    int cols = ((width - 1) / 32 + 1) * 32;
    size_t shape[4] = { 1, 3, (size_t)rows, (size_t)cols };
    fiv_tensor4d* output = fiv_create_tensor4d(shape, FIV_32F1);
    if (!output) return NULL;
    memset(output->data.fl, 0, output->total_bytes);

    size_t hw = (size_t)rows * (size_t)cols;
    ivf32* R = output->data.fl;
    ivf32* G = R + hw;
    ivf32* B = G + hw;
    for (int y = 0; y < height; y++) {
        const iv8u* row = src + (size_t)width_step * y;
        for (int x = 0; x < width; x++) {
            const iv8u* p = row + 3 * x;
            size_t pos = (size_t)y * cols + (size_t)x;
            R[pos] = (ivf32)p[0];
            G[pos] = (ivf32)p[1];
            B[pos] = (ivf32)p[2];
        }
    }
    return output;
}

/* ---- 53-layer CONV topology (mirrors libfacedetection param_pConvInfo) ----
   Ordered exactly as the reference data/graph. depthwise layers use same-pad
   edge-replicate (padding_method 1); pointwise are 1x1 with no extra padding.
   with_relu tells the builder to append a separate ReLU node after this conv. */
typedef struct {
    int input_channels;
    int output_channels;
    int is_depthwise;
    int with_relu;
} fiv_yunet_layer_cfg;

static const fiv_yunet_layer_cfg fiv_yunet_cfg[FIV_YUNET_NUM_CONV] = {
    { 32, 16, 0, 1 }, { 16, 16, 0, 0 }, { 16, 16, 1, 1 }, { 16, 16, 0, 0 },
    { 16, 16, 1, 1 }, { 16, 32, 0, 0 }, { 32, 32, 1, 1 }, { 32, 32, 0, 0 },
    { 32, 32, 1, 1 }, { 32, 64, 0, 0 }, { 64, 64, 1, 1 }, { 64, 64, 0, 0 },
    { 64, 64, 1, 1 }, { 64, 64, 0, 0 }, { 64, 64, 1, 1 }, { 64, 64, 0, 0 },
    { 64, 64, 1, 1 }, { 64, 64, 0, 0 }, { 64, 64, 1, 1 }, { 64, 64, 0, 0 },
    { 64, 64, 1, 1 }, { 64, 64, 0, 0 }, { 64, 64, 1, 1 }, { 64, 64, 0, 0 },
    { 64, 64, 1, 1 }, { 64, 64, 0, 0 }, { 64, 64, 1, 1 }, { 64, 64, 0, 0 },
    { 64, 64, 1, 1 },
    { 64,  1, 0, 0 }, {  1,  1, 1, 0 }, { 64,  1, 0, 0 }, {  1,  1, 1, 0 },
    { 64,  1, 0, 0 }, {  1,  1, 1, 0 },
    { 64,  4, 0, 0 }, {  4,  4, 1, 0 }, { 64,  4, 0, 0 }, {  4,  4, 1, 0 },
    { 64,  4, 0, 0 }, {  4,  4, 1, 0 },
    { 64,  1, 0, 0 }, {  1,  1, 1, 0 }, { 64,  1, 0, 0 }, {  1,  1, 1, 0 },
    { 64,  1, 0, 0 }, {  1,  1, 1, 0 },
    { 64, 10, 0, 0 }, { 10, 10, 1, 0 }, { 64, 10, 0, 0 }, { 10, 10, 1, 0 },
    { 64, 10, 0, 0 }, { 10, 10, 1, 0 },
};

/* Appends one CONV node for layer index `cfg_idx`, records its node id in
   graph->filter_node, appends a ReLU node when the layer has with_relu, and
   returns the id that streams into the next node. Returns -1 on failure. */
static int fiv_yunet_add_conv(void* net, int src, int* next_id,
                              int cfg_idx, fiv_yunet_graph* graph)
{
    const fiv_yunet_layer_cfg* c = &fiv_yunet_cfg[cfg_idx];
    fiv_conv2d_params params;
    memset(&params, 0, sizeof(params));
    params.conv2d_method   = c->is_depthwise ? FIV_CONV2D_DEPTHWISE : FIV_CONV2D_POINTWISE;
    params.kernel_size_x   = c->is_depthwise ? 3 : 1;
    params.kernel_size_y   = c->is_depthwise ? 3 : 1;
    params.stride          = 1;
    params.padding_method  = 0;  /* 0 = zero pad: libfacedetection convolution_3x3depthwise
                                    drops out-of-range 3x3 taps (equivalent to zero padding),
                                    NOT edge-replicate. This matches facedetectcnn.cpp. */
    params.input_channels  = c->input_channels;
    params.output_channels = c->output_channels;
    params.bias            = 1;
    if (c->is_depthwise) {
        params.pad_top = params.pad_bottom = 1;
        params.pad_left = params.pad_right = 1;
    }

    int type = c->is_depthwise ? FIV_NN_NODE_CONV2D_DEPTHWISE : FIV_NN_NODE_CONV2D_POINTWISE;
    int conv_id = (*next_id)++;
    if (fiv_neural_network_add_node(net, type, src, conv_id, &params) != FIV_RET_OK)
        return -1;
    graph->filter_node[cfg_idx] = conv_id;

    int out_id = conv_id;
    if (c->with_relu) {
        int relu_id = (*next_id)++;
        if (fiv_neural_network_add_node(net, FIV_NN_NODE_RELU, conv_id, relu_id, NULL) != FIV_RET_OK)
            return -1;
        out_id = relu_id;
    }
    return out_id;
}

/* conv_head special case: the fused input layer. Instead of the reference's
   pointwise 1x1 over a 32-channel im2col tensor, this is a real stride-2 3x3
   direct convolution 3->16 with zero padding, run by the standard conv2d node
   (FIV_CONV2D_STD, kernel 3, stride 2, SAME pad). Input node 0 now carries the
   3-channel aligned RGB plane tensor produced by fiv_yunet_preprocess. */
static int fiv_yunet_add_convhead(void* net, int src, int* next_id, fiv_yunet_graph* graph)
{
    fiv_conv2d_params params;
    memset(&params, 0, sizeof(params));
    params.conv2d_method   = FIV_CONV2D_STD;
    params.kernel_size_x   = 3;
    params.kernel_size_y   = 3;
    params.stride          = 2;
    params.padding_method  = 0;      /* zero pad, matches reference out-of-range taps */
    params.input_channels  = 3;
    params.output_channels = 16;
    params.bias            = 1;
    params.pad_top = params.pad_bottom = 1;
    params.pad_left = params.pad_right = 1;

    int conv_id = (*next_id)++;
    if (fiv_neural_network_add_node(net, FIV_NN_NODE_CONV2D_STD, src, conv_id, &params) != FIV_RET_OK)
        return -1;
    graph->filter_node[0] = conv_id;

    int out_id = conv_id;
    int relu_id = (*next_id)++;
    if (fiv_neural_network_add_node(net, FIV_NN_NODE_RELU, conv_id, relu_id, NULL) != FIV_RET_OK)
        return -1;
    return relu_id;
}

/* Reindex the reference conv_head weights (pointwise layout [oc][tap*3+ch],
   tap=(ky*3+kx), ch=RGB, raw reference parameter 0) into the fused conv2d node
   layout (C_out, C_in, ky, kx) row-major, then overwrite the node's weight and
   bias. Call on graph->filter_node[0] with the raw reference conv_head buffers. */
fiv_ret fiv_yunet_set_convhead_weight(void* net, int node_id,
                                      const ivf32* ref_w /*[16][32]*/, const ivf32* ref_b /*[16]*/)
{
    if (!net || node_id < 0 || !ref_w || !ref_b) return FIV_RET_ERR_PARA;
    ivf32 fused[16 * 3 * 3 * 3];
    for (int oc = 0; oc < 16; oc++) {
        for (int ic = 0; ic < 3; ic++) {
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    int tap = ky * 3 + kx;
                    size_t src_i = (size_t)oc * 32 + (size_t)tap * 3 + (size_t)ic;
                    size_t dst_i = ((size_t)oc * 3 + (size_t)ic) * 9 + (size_t)ky * 3 + (size_t)kx;
                    fused[dst_i] = ref_w[src_i];
                }
            }
        }
    }
    fiv_ret r = fiv_neural_network_set_node_weight(net, node_id, fused);
    if (r != FIV_RET_OK) return r;
    return fiv_neural_network_set_node_bias(net, node_id, ref_b);
}

static int fiv_yunet_add_pool(void* net, int src, int* next_id)
{
    int id = (*next_id)++;
    if (fiv_neural_network_add_node(net, FIV_NN_NODE_MAX2D, src, id, NULL) != FIV_RET_OK)
        return -1;
    return id;
}

static int fiv_yunet_add_upsample(void* net, int src, int* next_id)
{
    int id = (*next_id)++;
    if (fiv_neural_network_add_node(net, FIV_NN_NODE_UPSAMPLE2X, src, id, NULL) != FIV_RET_OK)
        return -1;
    return id;
}

/* elementAdd(a, b): a two-input ADD node. */
static int fiv_yunet_add_add(void* net, int a, int b, int* next_id)
{
    int id = (*next_id)++;
    int starts[2] = { a, b };
    if (fiv_neural_network_add_node_multi(net, FIV_NN_NODE_ADD, starts, 2, id, NULL) != FIV_RET_OK)
        return -1;
    return id;
}

/* Helpers to keep the per-scale head-id bookkeeping above the add-node calls. */
static int fiv_yunet_add_head(void* net, int feat, int* next_id, fiv_yunet_graph* graph,
                              int pw_idx, int dw_idx)
{
    int t = fiv_yunet_add_conv(net, feat, next_id, pw_idx, graph);  /* P (no relu) */
    if (t < 0) return -1;
    int out = fiv_yunet_add_conv(net, t, next_id, dw_idx, graph);   /* D (no relu) */
    return out;
}

fiv_yunet_graph* fiv_yunet_build_graph(void)
{
    fiv_yunet_graph* graph = (fiv_yunet_graph*)fiv_malloc(sizeof(fiv_yunet_graph));
    if (!graph) return NULL;
    memset(graph, 0, sizeof(*graph));

    void* net = fiv_create_neural_network();
    if (!net) { fiv_free(graph); return NULL; }
    graph->net = net;

    int next = 1;   /* node 0 is the implicit INPUT */
    int cur;

    /* ---- MobileNet backbone ---- */
    cur = fiv_yunet_add_convhead(net, 0, &next, graph);     /* conv_head fused input layer */
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 1, graph);    /* conv0 P     */
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 2, graph);    /* conv0 D     */
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_pool(net, cur, &next);              /* pool0       */
    if (cur < 0) goto fail;

    cur = fiv_yunet_add_conv(net, cur, &next, 3, graph);    /* conv1       */
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 4, graph);
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 5, graph);
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 6, graph);
    if (cur < 0) goto fail;

    cur = fiv_yunet_add_conv(net, cur, &next, 7, graph);    /* conv2       */
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 8, graph);
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 9, graph);
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 10, graph);
    if (cur < 0) goto fail;

    cur = fiv_yunet_add_pool(net, cur, &next);              /* pool3       */
    if (cur < 0) goto fail;

    cur = fiv_yunet_add_conv(net, cur, &next, 11, graph);   /* conv3 -> fb1*/
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 12, graph);
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 13, graph);
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 14, graph);
    if (cur < 0) goto fail;
    int fb1 = cur;

    cur = fiv_yunet_add_pool(net, cur, &next);              /* pool4       */
    if (cur < 0) goto fail;

    cur = fiv_yunet_add_conv(net, cur, &next, 15, graph);   /* conv4 -> fb2*/
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 16, graph);
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 17, graph);
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 18, graph);
    if (cur < 0) goto fail;
    int fb2 = cur;

    cur = fiv_yunet_add_pool(net, cur, &next);              /* pool5       */
    if (cur < 0) goto fail;

    cur = fiv_yunet_add_conv(net, cur, &next, 19, graph);   /* conv5 -> fb3*/
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 20, graph);
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 21, graph);
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 22, graph);
    if (cur < 0) goto fail;
    int fb3 = cur;

    /* ---- top-down FPN: branch5 (fine -> stride 32) ---- */
    cur = fiv_yunet_add_conv(net, fb3, &next, 27, graph);   /* lateral conv 2 */
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 28, graph);
    if (cur < 0) goto fail;
    int fb3_lat = cur;
    graph->cls[2] = fiv_yunet_add_head(net, fb3_lat, &next, graph, 33, 34);
    graph->reg[2] = fiv_yunet_add_head(net, fb3_lat, &next, graph, 39, 40);
    graph->obj[2] = fiv_yunet_add_head(net, fb3_lat, &next, graph, 45, 46);
    graph->kps[2] = fiv_yunet_add_head(net, fb3_lat, &next, graph, 51, 52);
    if (graph->cls[2] < 0 || graph->reg[2] < 0 || graph->obj[2] < 0 || graph->kps[2] < 0) goto fail;

    /* add5: fb2 <- upsample(fb3_lat) + fb2 */
    int up3 = fiv_yunet_add_upsample(net, fb3_lat, &next);
    int fused2 = fiv_yunet_add_add(net, up3, fb2, &next);
    if (up3 < 0 || fused2 < 0) goto fail;

    /* ---- branch4 (mid -> stride 16) ---- */
    cur = fiv_yunet_add_conv(net, fused2, &next, 25, graph); /* lateral conv 1 */
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 26, graph);
    if (cur < 0) goto fail;
    int fb2_lat = cur;
    graph->cls[1] = fiv_yunet_add_head(net, fb2_lat, &next, graph, 31, 32);
    graph->reg[1] = fiv_yunet_add_head(net, fb2_lat, &next, graph, 37, 38);
    graph->obj[1] = fiv_yunet_add_head(net, fb2_lat, &next, graph, 43, 44);
    graph->kps[1] = fiv_yunet_add_head(net, fb2_lat, &next, graph, 49, 50);
    if (graph->cls[1] < 0 || graph->reg[1] < 0 || graph->obj[1] < 0 || graph->kps[1] < 0) goto fail;

    /* add4: fb1 <- upsample(fb2_lat) + fb1 */
    int up2 = fiv_yunet_add_upsample(net, fb2_lat, &next);
    int fused1 = fiv_yunet_add_add(net, up2, fb1, &next);
    if (up2 < 0 || fused1 < 0) goto fail;

    /* ---- branch3 (coarse -> stride 8) ---- */
    cur = fiv_yunet_add_conv(net, fused1, &next, 23, graph); /* lateral conv 0 */
    if (cur < 0) goto fail;
    cur = fiv_yunet_add_conv(net, cur, &next, 24, graph);
    if (cur < 0) goto fail;
    int fb1_lat = cur;
    graph->cls[0] = fiv_yunet_add_head(net, fb1_lat, &next, graph, 29, 30);
    graph->reg[0] = fiv_yunet_add_head(net, fb1_lat, &next, graph, 35, 36);
    graph->obj[0] = fiv_yunet_add_head(net, fb1_lat, &next, graph, 41, 42);
    graph->kps[0] = fiv_yunet_add_head(net, fb1_lat, &next, graph, 47, 48);
    if (graph->cls[0] < 0 || graph->reg[0] < 0 || graph->obj[0] < 0 || graph->kps[0] < 0) goto fail;

    return graph;

fail:
    fiv_yunet_release_graph(graph);
    return NULL;
}

/* Reusable decode scratch block owned by the graph (see helper section before
   fiv_yunet_detect for the exact carving layout). Freed on graph release. */
typedef struct fiv_yunet_scratch {
    void*  mem;
    size_t bytes;
    int    cap;    /* capacity in anchors */
} fiv_yunet_scratch;

void fiv_yunet_release_graph(fiv_yunet_graph* graph)
{
    if (!graph) return;
    if (graph->net) fiv_release_neural_network(&graph->net);
    if (graph->scratch) {
        fiv_free(graph->scratch->mem);
        fiv_free(graph->scratch);
        graph->scratch = NULL;
    }
    fiv_free(graph);
}

/* ---- Detection-head decode + NMS (mirrors libfacedetection) ---- */

/* Read a head tensor (NCHW, channels x H x W) into a per-anchor channel-vector
   buffer laid out row-major like the reference CDataBlob. NCHW index for
   (c,h,w) is (c*H + h)*W + w; the output holds `channels` values per pixel. */
static void fiv_yunet_head_to_vec(const fiv_tensor4d* head, ivf32* vec)
{
    int c = (int)head->channels;
    int h = (int)head->height;
    int w = (int)head->width;
    size_t hw = (size_t)h * (size_t)w;
    const ivf32* src = head->data.fl;
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            ivf32* dst = vec + (size_t)(row * w + col) * c;
            for (int ch = 0; ch < c; ch++)
                dst[ch] = src[(size_t)ch * hw + (size_t)row * w + (size_t)col];
        }
    }
}

/* element-wise sigmoid in place (same clamp as the reference). */
static void fiv_yunet_sigmoid_inplace(ivf32* data, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        ivf32 v = data[i];
        if (v > 88.3762626647949f) v = 88.3762626647949f;
        else if (v < -88.3762626647949f) v = -88.3762626647949f;
        data[i] = 1.0f / (1.0f + expf(-v));
    }
}

/* Append one scale's decoded ties into the per-scale vectors. reg_vec/kps_vec
   hold the raw head maps (NCHW); stride decodes them to padded-image pixels. */
static void fiv_yunet_assemble(const ivf32* cls, const ivf32* reg, const ivf32* kps,
                               const ivf32* obj, int channels, int height, int width,
                               int stride,
                               ivf32* cls_out, ivf32* reg_out, ivf32* kps_out, ivf32* obj_out)
{
    size_t hw = (size_t)height * (size_t)width;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            size_t po = (size_t)row * width + col;                 /* pixel offset */
            size_t ro = po * 4;                                    /* reg offset */
            size_t ko = po * 10;                                   /* kps offset */
            ivf32 priorx = (ivf32)(col * stride);
            ivf32 priory = (ivf32)(row * stride);

            cls_out[po] = cls[po];
            obj_out[po] = obj[po];

            /* bbox_decode: cx = d0*stride + px, w = exp(d2)*stride */
            ivf32 cx = reg[ro + 0] * stride + priorx;
            ivf32 cy = reg[ro + 1] * stride + priory;
            ivf32 bw = expf(reg[ro + 2]) * stride;
            ivf32 bh = expf(reg[ro + 3]) * stride;
            reg_out[ro + 0] = cx - bw / 2.0f;
            reg_out[ro + 1] = cy - bh / 2.0f;
            reg_out[ro + 2] = cx + bw / 2.0f;
            reg_out[ro + 3] = cy + bh / 2.0f;

            /* kps_decode: k = d*stride + prior coordinate */
            for (int n = 0; n < 5; n++) {
                kps_out[ko + 2 * n + 0] = kps[ko + 2 * n + 0] * stride + priorx;
                kps_out[ko + 2 * n + 1] = kps[ko + 2 * n + 1] * stride + priory;
            }
        }
    }
    (void)channels;
}

/* Stable insertion-flavoured selection sort of candidate ids by conf desc. */
static void fiv_yunet_sort_desc(int* idx, const ivf32* conf, int n)
{
    /* simple stable insertion sort (n is small) */
    for (int i = 1; i < n; i++) {
        int key = idx[i];
        ivf32 kc = conf[key];
        int j = i - 1;
        while (j >= 0 && (conf[idx[j]] < kc || (conf[idx[j]] == kc && idx[j] > key))) {
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key;
    }
}

static ivf32 fiv_yunet_jaccard(ivf32 xmin1, ivf32 ymin1, ivf32 xmax1, ivf32 ymax1,
                               ivf32 xmin2, ivf32 ymin2, ivf32 xmax2, ivf32 ymax2)
{
    ivf32 ix = (xmin1 > xmin2 ? xmin1 : xmin2);
    ivf32 iy = (ymin1 > ymin2 ? ymin1 : ymin2);
    ivf32 ax = (xmax1 < xmax2 ? xmax1 : xmax2);
    ivf32 ay = (ymax1 < ymax2 ? ymax1 : ymax2);
    if (ix < ax && iy < ay) {
        ivf32 inter = (ax - ix) * (ay - iy);
        ivf32 a1 = (xmax1 - xmin1) * (ymax1 - ymin1);
        ivf32 a2 = (xmax2 - xmin2) * (ymax2 - ymin2);
        return inter / (a1 + a2 - inter);
    }
    return 0.0f;
}

/* ---- Reusable decode scratch ----
   All per-frame decode buffers live in ONE contiguous block owned by the graph
   (graph->scratch->mem), carved into typed views sized for the current anchor
   count. It is allocated lazily on the first detect and only reallocated (and
   grown) when the anchor count increases, so steady-state frames perform no
   heap allocation here. Layout per anchor: 32 ivf32 words (reg*4 + kps*10 +
   cls + obj + cls_s + reg_s*4 + obj_s + kps_s*10 + cand_score) followed by 4
   int words (cand, order, accept, accept_idx). */
typedef struct {
    ivf32* reg_total;    /* total * 4 */
    ivf32* kps_total;    /* total * 10 */
    ivf32* cls_total;    /* total */
    ivf32* obj_total;    /* total */
    ivf32* cls_s;        /* per-scale, sized total */
    ivf32* reg_s;        /* per-scale * 4 */
    ivf32* obj_s;        /* per-scale */
    ivf32* kps_s;        /* per-scale * 10 */
    ivf32* cand_score;   /* total */
    int*   cand;         /* total */
    int*   order;        /* total */
    int*   accept;       /* total */
    int*   accept_idx;   /* total */
} fiv_yunet_scratch_view;

static fiv_ret fiv_yunet_scratch_grow(fiv_yunet_graph* graph, int anchors)
{
    if (!graph->scratch) {
        graph->scratch = (fiv_yunet_scratch*)fiv_malloc(sizeof(fiv_yunet_scratch));
        if (!graph->scratch) return FIV_RET_ERR_MEM;
        graph->scratch->mem = NULL;
        graph->scratch->bytes = 0;
        graph->scratch->cap = 0;
    }
    if (graph->scratch->cap >= anchors) return FIV_RET_OK;
    size_t words = sizeof(ivf32) * 32 + sizeof(int) * 4; /* bytes per anchor */
    size_t bytes = words * (size_t)anchors;
    void* nb = fiv_malloc(bytes);
    if (!nb) return FIV_RET_ERR_MEM;
    fiv_free(graph->scratch->mem);
    graph->scratch->mem = nb;
    graph->scratch->bytes = bytes;
    graph->scratch->cap = anchors;
    return FIV_RET_OK;
}

static void fiv_yunet_scratch_carve(const void* mem, int anchors, fiv_yunet_scratch_view* v)
{
    size_t t = (size_t)anchors;
    ivf32* f = (ivf32*)mem;
    v->reg_total = f;              f += t * 4;
    v->kps_total = f;              f += t * 10;
    v->cls_total = f;              f += t;
    v->obj_total = f;              f += t;
    v->cls_s     = f;              f += t;
    v->reg_s     = f;              f += t * 4;
    v->obj_s     = f;              f += t;
    v->kps_s     = f;              f += t * 10;
    v->cand_score = f;             f += t;
    int* p = (int*)f;
    v->cand      = p;              p += t;
    v->order     = p;              p += t;
    v->accept    = p;              p += t;
    v->accept_idx = p;             p += t;
}

fiv_ret fiv_yunet_detect(fiv_yunet_graph* graph, fiv_yunet_result* out, int* out_count,
                         const iv8u* rgb, int width, int height, int width_step)
{
    if (!graph || !out || !out_count || !rgb || width <= 0 || height <= 0)
        return FIV_RET_ERR_PARA;

    fiv_tensor4d* input = fiv_yunet_preprocess(rgb, width, height, width_step);
    if (!input) return FIV_RET_ERR_MEM;

    void* final_out = NULL;
    fiv_ret ret = fiv_nn_run_inference(graph->net, input, &final_out);
    fiv_release_tensor((void**)&input);
    if (ret != FIV_RET_OK) return ret;

    int total_anchors = 0;
    int h[3], w[3];
    int stride[3] = { 8, 16, 32 };
    for (int s = 0; s < 3; s++) {
        fiv_tensor4d* t = (fiv_tensor4d*)fiv_neural_network_get_node_output(graph->net, graph->cls[s]);
        h[s] = (int)t->height;
        w[s] = (int)t->width;
        total_anchors += h[s] * w[s];
    }

    /* All decode buffers come from one reusable scratch (no per-frame malloc). */
    fiv_yunet_scratch_view v;
    if (fiv_yunet_scratch_grow(graph, total_anchors) != FIV_RET_OK)
        return FIV_RET_ERR_MEM;
    fiv_yunet_scratch_carve(graph->scratch->mem, total_anchors, &v);
    ivf32* cls_total = v.cls_total;
    ivf32* reg_total = v.reg_total;
    ivf32* kps_total = v.kps_total;
    ivf32* obj_total = v.obj_total;

    size_t base = 0;
    for (int s = 0; s < 3; s++) {
        size_t hs = (size_t)h[s] * (size_t)w[s];
        fiv_tensor4d* cls_t = (fiv_tensor4d*)fiv_neural_network_get_node_output(graph->net, graph->cls[s]);
        fiv_tensor4d* reg_t = (fiv_tensor4d*)fiv_neural_network_get_node_output(graph->net, graph->reg[s]);
        fiv_tensor4d* obj_t = (fiv_tensor4d*)fiv_neural_network_get_node_output(graph->net, graph->obj[s]);
        fiv_tensor4d* kps_t = (fiv_tensor4d*)fiv_neural_network_get_node_output(graph->net, graph->kps[s]);
        ivf32* cls_s = v.cls_s;
        ivf32* reg_s = v.reg_s;
        ivf32* obj_s = v.obj_s;
        ivf32* kps_s = v.kps_s;
        fiv_yunet_head_to_vec(cls_t, cls_s);
        fiv_yunet_head_to_vec(reg_t, reg_s);
        fiv_yunet_head_to_vec(obj_t, obj_s);
        fiv_yunet_head_to_vec(kps_t, kps_s);
        fiv_yunet_assemble(cls_s, reg_s, kps_s, obj_s, (int)cls_t->channels,
                           h[s], w[s], stride[s], cls_total + base, reg_total + base * 4,
                           kps_total + base * 10, obj_total + base);
        base += hs;
    }

    /* sigmoid on cls and obj (matches reference concat then sigmoid). */
    fiv_yunet_sigmoid_inplace(cls_total, (size_t)total_anchors);
    fiv_yunet_sigmoid_inplace(obj_total, (size_t)total_anchors);

    /* Gather candidate ids with conf >= threshold. */
    const ivf32 confidence_threshold = 0.2f;
    int top_k = 1000;
    const ivf32 overlap_threshold = 0.45f;
    const int keep_top_k = 512;

    int* cand = v.cand;
    ivf32* cand_score = v.cand_score;
    int n_cand = 0;
    for (int i = 0; i < total_anchors; i++) {
        ivf32 conf = sqrtf(cls_total[i] * obj_total[i]);
        if (conf >= confidence_threshold) {
            cand[n_cand] = i;
            cand_score[n_cand] = conf;
            n_cand++;
        }
    }
    if (n_cand > top_k) n_cand = top_k;

    /* Sort candidates by score desc (stable). */
    int* order = v.order;
    for (int i = 0; i < n_cand; i++) order[i] = i;
    fiv_yunet_sort_desc(order, cand_score, n_cand);

    /* Greedy NMS against accepted boxes. */
    int* accept = v.accept;
    int* accept_idx = v.accept_idx;
    int n_accept = 0;
    for (int i = 0; i < n_cand; i++) {
        int oi = cand[order[i]];
        const ivf32* b1 = reg_total + (size_t)oi * 4;
        int keep = 1;
        for (int k = 0; k < n_accept; k++) {
            const ivf32* b2 = reg_total + (size_t)accept_idx[k] * 4;
            if (fiv_yunet_jaccard(b1[0], b1[1], b1[2], b1[3], b2[0], b2[1], b2[2], b2[3]) > overlap_threshold) {
                keep = 0;
                break;
            }
        }
        if (keep && n_accept < keep_top_k) {
            accept[n_accept] = i;
            accept_idx[n_accept] = oi;
            n_accept++;
        }
    }

    /* Emit results in accepted (score-desc) order. */
    for (int i = 0; i < n_accept; i++) {
        int oi = cand[order[accept[i]]];
        const ivf32* b0 = reg_total + (size_t)oi * 4;
        const ivf32* k0 = kps_total + (size_t)oi * 10;
        fiv_yunet_result* r = &out[i];
        r->score = sqrtf(cls_total[oi] * obj_total[oi]);
        r->x = (int)b0[0];
        r->y = (int)b0[1];
        r->w = (int)(b0[2] - b0[0]);
        r->h = (int)(b0[3] - b0[1]);
        for (int n = 0; n < 10; n++) r->lm[n] = (int)k0[n];
    }
    *out_count = n_accept;

    return FIV_RET_OK;   /* decode scratch is reused; freed in fiv_yunet_release_graph */
}