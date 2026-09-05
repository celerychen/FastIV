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

#ifndef _FIV_YUNET_H_
#define _FIV_YUNET_H_

#include "fiv_ctensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* libfacedetection YuNet: a 53-layer MobileNet-style face detector ported onto
   the FastIV nn engine. This header exposes the building blocks: RGB input
   packing, graph assembly, weight loading and post-processing. */

/* Input preprocessing: packs the RGB image into 3 channel planes and zero-pads
   height/width up to multiples of 32, producing a tensor of shape (1, 3, rows,
   cols) with rows/cols aligned to 32. The first graph layer (conv_head) is a
   fused stride-2 3x3 direct convolution over this tensor; FastIV uses direct
   convolution, so the reference's 32-channel im2col expansion is not needed.
   The caller releases the returned tensor. */
fiv_tensor4d* fiv_yunet_preprocess(const iv8u* src, int width, int height, int width_step);

/* Fully-assembled YuNet graph plus the node ids needed for weight loading
   (filter_node) and detection-head decode (cls/reg/obj/kps per scale). The
   three pyramid scales are indexed 0=coarse->2=fine, i.e. scale k is the head
   produced on the level with 2^(k+3)-pixel stride over the input image
   (stride 8 / 16 / 32). filter_node[i] is the conv node id of the i-th
   libfacedetection CONV layer (0..52) and is what weight injection addresses. */
#define FIV_YUNET_NUM_CONV 53

struct fiv_yunet_scratch;

typedef struct {
    void* net;        /* fiv_nn_network_context* built by fiv_yunet_build_graph */
    int   filter_node[FIV_YUNET_NUM_CONV]; /* conv node id per CONV layer index */
    int   cls[3];     /* classification head node id per scale */
    int   reg[3];     /* bbox-regression head node id per scale */
    int   obj[3];     /* objectness head node id per scale */
    int   kps[3];     /* landmark head node id per scale */
    /* Reusable decode scratch, allocated lazily on first detect and grown only
       when the per-frame anchor count increases. Freed in fiv_yunet_release_graph
       so steady-state frames do no heap allocation for local decode buffers. */
    struct fiv_yunet_scratch* scratch;
} fiv_yunet_graph;

/* Assemble the full 53-layer YuNet (RGB input -> conv_head fused stride-2 3x3
   -> MobileNet backbone -> FPN/lateral fusion -> per-scale detection heads).
   Node 0 is the implicit input that the caller feeds with fiv_yunet_preprocess
   output. Weights are left at engine default until overwritten via
   fiv_neural_network_set_node_weight / _set_node_bias using filter_node ids;
   conv_head (filter_node[0]) additionally needs fiv_yunet_set_convhead_weight,
   which reindexes the reference's pointwise conv_head parameters into the fused
   stride-2 3x3 node layout. Returns NULL on any node-add failure. */
fiv_yunet_graph* fiv_yunet_build_graph(void);

/* Load the fused conv_head layer. ref_w is the RAW reference conv_head weight
   in pointwise layout [oc][tap*3+ch] (16 x 32) and ref_b its 16 biases; the two
   are reindexed into the stride-2 3x3 node layout and written to node node_id
   (graph->filter_node[0]). Without this the conv_head node keeps its engine
   default weights. */
fiv_ret fiv_yunet_set_convhead_weight(void* net, int node_id,
                                      const ivf32* ref_w /*[16][32]*/, const ivf32* ref_b /*[16]*/);

/* Tear down a graph created by fiv_yunet_build_graph, releasing its network. */
void fiv_yunet_release_graph(fiv_yunet_graph* graph);

/* ---- Detection-head decode + NMS (matches libfacedetection) ---- */

/* A detected face: pixel coordinates in the padded input image, plus five
   landmarks (paired x/y, scaled by the feature stride in input pixels). */
typedef struct {
    ivf32 score;
    int   x;
    int   y;
    int   w;
    int   h;
    int   lm[10];   /* five (x,y) landmark pixels */
} fiv_yunet_result;

#define FIV_YUNET_MAX_FACES 512   /* matches keep_top_k of the reference */

/* Run the full pipeline: preprocess + inference + per-scale decode + NMS, and
   fill *out with up to FIV_YUNET_MAX_FACES detections (same maths as
   facedetect_cnn: conf = sqrt(cls*obj) >= 0.20, IoU NMS @ 0.45). */
fiv_ret fiv_yunet_detect(fiv_yunet_graph* graph, fiv_yunet_result* out, int* out_count,
                         const iv8u* rgb, int width, int height, int width_step);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_YUNET_H_ */