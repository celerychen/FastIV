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

#include "yolo26_ref.h"

typedef struct {
    void* net;              /* fiv neural network (engine) */
    int   layer_node[24];   /* output node id of model.model[i], i = 0..22 */
    int   head_node[6];     /* Detect one2one raw heads: box0,box1,box2,cls0,cls1,cls2 */
    int   mask_node[3];     /* Segment26 one2one_cv4 (mask coef) heads; -1 when absent */
} fiv_yolo26_graph;

/* attn[i][0..5] = { qkv_w, qkv_b, pe_w, pe_b, proj_w, proj_b } for the i-th
 * Attention instance (i=0 -> layer10 C2PSA, i=1 -> layer22 attn-C3k2).
 * Returns NULL on build failure. */
fiv_yolo26_graph* fiv_yolo26_build(const yolo26_fold_entry* fold, int fold_count,
                                   const float* fold_w, const float* fold_b,
                                   const float* attn[2][6]);

void fiv_yolo26_release(fiv_yolo26_graph* graph);

#endif /* FIV_YOLO26_H */
