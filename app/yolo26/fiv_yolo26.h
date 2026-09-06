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

/* YOLO26 (ultralytics yolo26n, detection, end2end/reg_max=1) full-graph
 * builder on the FastIV NN engine. Drives every conv from the BN-fold table
 * produced by app/yolo26/test/gen_ref.py (fold.json / fold_w.f32 /
 * fold_b.f32); attention composite-node weights come from the attn0_/attn1_
 * dumps. Weight layout is [c_out, c_in, ky, kx] row-major, identical to
 * PyTorch (no permutation).
 *
 * The fold table type is the loader struct declared in
 * app/yolo26/test/yolo26_ref.h (this app header includes it); the caller
 * loads table + weight blobs via the loader and passes them in. */

#ifndef FIV_YOLO26_H
#define FIV_YOLO26_H

#include "fiv_data_typedefs.h"
#include "yolo26_ref.h"

typedef struct {
    void* net;              /* fiv neural network (engine) */
    int   layer_node[24];   /* output node id of model.model[i], i = 0..22 */
    int   head_node[6];     /* Detect one2one raw heads: box0,box1,box2,cls0,cls1,cls2 */
    int   mask_node[3];     /* Segment26 one2one_cv4 (mask coef) heads; -1 when absent */
    int   kpt_node[3];      /* Pose26 one2one_cv4_kpts (kpt reg) heads; -1 when absent */
} fiv_yolo26_graph;

/* attn[i][0..5] = { qkv_w, qkv_b, pe_w, pe_b, proj_w, proj_b } for the i-th
 * Attention instance (i=0 -> layer10 C2PSA, i=1 -> layer22 attn-C3k2).
 * Returns NULL on build failure. */
fiv_yolo26_graph* fiv_yolo26_build(const yolo26_fold_entry* fold, int fold_count,
                                   const ivf32* fold_w, const ivf32* fold_b,
                                   const ivf32* attn[2][6]);

/* Build the yolo26-cls (Classify) network: layers 0..8 are the same backbone
 * blocks as detection (Conv / C3k2), layer 9 is C2PSA (no SPPF/neck), layer 10
 * is the Classify head conv 1x1 (folded head_conv_w/b, SiLU) whose output the
 * caller pools and runs through the separately-loaded Linear head. The head
 * conv node id is returned in layer_node[10]. */
fiv_yolo26_graph* fiv_yolo26_build_classifier(const yolo26_fold_entry* fold,
                                              int fold_count,
                                              const ivf32* fold_w,
                                              const ivf32* fold_b,
                                              const ivf32* attn[2][6],
                                              const ivf32* head_conv_w,
                                              const ivf32* head_conv_b,
                                              int head_output_channels);

void fiv_yolo26_release(fiv_yolo26_graph* graph);

#endif /* FIV_YOLO26_H */
