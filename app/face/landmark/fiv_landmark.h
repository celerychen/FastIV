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
 * FaceMesh 478-point 3D landmark model (landmark_net.bin).
 *
 * This loader parses the reference "LMNT" binary (see export_landmark_net.py in
 * the c_face_detect_release reference tree) and assembles the graph into a
 * FastIV neural network (api/fiv_nn.h) at runtime. Every conv/depthwise weight
 * is transposed from the TFLite NHWC layout [co,kh,kw,ci] into FastIV's NCHW
 * layout [co,ci,kh,kw] before injection; depthwise weights are reindexed from
 * tap-major to output-channel-major. PReLU alphas and conv biases are injected
 * directly. The graph is a MobileNetV2-style backbone (input 256x256x3 ->
 * stride-2 stem 3x3 -> 128x128x16, then 8/16/32/64-channel inverted residuals
 * with ADD skip connections), followed by three heads:
 *   - t473: 1434 raw 3D landmark logits (478 landmarks * (x, y, z))
 *   - t472 -> t474: presence confidence logit (sigmoid)
 * All activations are PReLU (per-channel slope alpha).
 */

#ifndef _FIV_LANDMARK_H_
#define _FIV_LANDMARK_H_

#include "fiv_nn.h"
#include "fiv_ctensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Number of 3D landmarks produced by the network. */
#define FIV_LM_NUM_LANDMARKS 478

/* Spatial size of the network input (square, RGB). */
#define FIV_LM_TENSOR_SIZE 256

/* Model file signature ("LMNT", little-endian). */
#define FIV_LM_MAGIC 0x4C4D4E54u

/* LDNT metadata trailer magic ("TMDL", little-endian). An LMNT blob that ends
   with this trailer carries per-model metadata (landmark count, tensor size,
   input spatial size and the two output tensor ids) so a single loader can
   serve both the 478x256 model and the 468x192 model. When the trailer is
   absent the defaults below apply (256x256 / 478 / out 473,472). */
#define FIV_LM_META_MAGIC 0x4C444D54u

/* The graph built from landmark_net.bin. `net` owns every node; the node ids
   of the two readable outputs are recorded so the caller can fetch them with
   fiv_neural_network_get_node_output() after fiv_nn_run_inference(). The
   remaining fields describe the model (defaults in parentheses) and override
   the compile-time constants when loaded from a metadata-carrying blob. */
typedef struct {
    void* net;             /* fiv_nn_network_context* */
    int   lm_node;         /* node id producing the landmark logits */
    int   conf_node;       /* node id producing the presence logit */
    int   n_landmarks;     /* 478 (default) / 468 for the 192 model */
    int   tensor_size;     /* 256 (default) / 192 for the 192 model */
    int   in_h, in_w;      /* graph-input spatial fallback (256,256) */
    int   out_lm_id;       /* output tensor id of the landmark logits (473) */
    int   out_conf_id;     /* output tensor id of the presence logit (472) */
} fiv_landmark_graph;

/* Build the landmark network graph from landmark_net.bin at `path`.
   Returns an allocated graph, or NULL on any parse / build error. */
fiv_landmark_graph* fiv_create_landmark_graph(const char* path);

/* Release a graph previously created by fiv_create_landmark_graph. */
void fiv_release_landmark_graph(fiv_landmark_graph* graph);

/* Run one inference pass on the landmark graph.
   input   : NCHW [1,3,256,256] float32 tensor (channel-first, [0,1]).
   out_xyz : 478*3 normalized landmark outputs (xyz = raw_logit / 256).
   out_presence : sigmoid of the presence logit, [0,1].
   Returns FIV_RET_OK on success. These are the raw network outputs (before the
   higher-level ROI inverse-warp / pixel projection done by fiv_lm_model). */
fiv_ret fiv_landmark_graph_run_inference(fiv_landmark_graph* graph,
                                         const fiv_tensor4d* input,
                                         ivf32 out_xyz[FIV_LM_NUM_LANDMARKS][3],
                                         ivf32* out_presence);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_LANDMARK_H_ */
