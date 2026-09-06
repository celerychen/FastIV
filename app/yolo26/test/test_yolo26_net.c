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

/* P4 end-to-end network test: builds the full YOLO26 detection graph with
 * app/yolo26/fiv_yolo26.c (driven by the BN-fold table + attn weight dumps),
 * runs inference on the torch reference input, and compares every layer
 * output (layerNN_<Type>) plus the six Detect one2one raw heads
 * (head_box0..2 / head_cls0..2) against the reference tensors.
 *
 * Run from build/: ./test_yolo26_net
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fiv_ctensor.h"
#include "fiv_nn_infer.h"
#include "yolo26_ref.h"
#include "fiv_yolo26.h"

static int g_pass = 0;
static int g_fail = 0;

/* yolo26.yaml backbone+neck top-level type per index (Detect is index 23). */
static const char* const kLayerType[24] = {
    "Conv", "Conv", "C3k2", "Conv", "C3k2", "Conv", "C3k2", "Conv",
    "C3k2", "SPPF", "C2PSA", "Upsample", "Concat", "C3k2", "Upsample",
    "Concat", "C3k2", "Conv", "Concat", "C3k2", "Conv", "Concat", "C3k2",
    "-"
};

int main(void)
{
    printf("=== YOLO26 P4 full-network test (vs torch reference) ===\n");

    yolo26_fold_entry fold[300];
    int n_fold = yolo26_ref_load_fold(fold, 300);
    if (n_fold <= 0) { printf("FAIL: fold table load\n"); return 1; }

    /* weight payload spans: [0, max(w_off+w_cnt)) */
    size_t w_tot = 0, b_tot = 0;
    for (int i = 0; i < n_fold; i++) {
        size_t we = fold[i].w_off + fold[i].w_cnt;
        size_t be = fold[i].b_off + fold[i].b_cnt;
        if (we > w_tot) w_tot = we;
        if (be > b_tot) b_tot = be;
    }
    const float* fold_w = yolo26_ref_load_blob("fold_w.f32", 0, w_tot);
    const float* fold_b = yolo26_ref_load_blob("fold_b.f32", 0, b_tot);
    if (!fold_w || !fold_b) { printf("FAIL: fold weight blobs\n"); return 1; }

    /* attention weights attn[2][6] */
    const char* part[6] = { "qkv_w", "qkv_b", "pe_w", "pe_b", "proj_w", "proj_b" };
    const float* attn[2][6];
    memset(attn, 0, sizeof(attn));
    char nm[64];
    int ndim; size_t sh[8];
    for (int a = 0; a < 2; a++) {
        for (int j = 0; j < 6; j++) {
            snprintf(nm, sizeof(nm), "attn%d_%s", a, part[j]);
            attn[a][j] = ref_load(nm, &ndim, sh);
            if (!attn[a][j]) { printf("FAIL: load %s\n", nm); return 1; }
        }
    }

    fiv_yolo26_graph* graph = fiv_yolo26_build(fold, n_fold, fold_w, fold_b, attn);
    if (!graph) { printf("FAIL: full graph build\n"); return 1; }

    /* run on the torch reference input */
    const float* in_f = ref_load("input", &ndim, sh);
    if (!in_f) { printf("FAIL: load input\n"); return 1; }
    if (ndim != 4) { printf("FAIL: input ndim %d\n", ndim); return 1; }
    fiv_tensor4d* input = fiv_create_tensor4d((size_t*)sh, FIV_32F1);
    if (!input) { printf("FAIL: alloc input tensor\n"); return 1; }
    size_t in_n = sh[0] * sh[1] * sh[2] * sh[3];
    memcpy(((fiv_tensor_hdr*)input)->data.fl, in_f, in_n * sizeof(float));
    free((void*)in_f);

    void* final_out = NULL;
    fiv_ret r = fiv_nn_run_inference(graph->net, input, &final_out);
    fiv_release_tensor((void**)&input);
    if (r != FIV_RET_OK) { printf("FAIL: run_inference r=%d\n", r); return 1; }

    /* layer outputs: layerNN_<Type> (only layers that produced a node) */
    for (int i = 0; i < 23; i++) {
        if (graph->layer_node[i] < 0) continue;
        if (i == 12 || i == 15 || i == 18 || i == 21) continue; /* concat covered via next */
        char out_name[40];
        snprintf(out_name, sizeof(out_name), "layer%02d_%s", i, kLayerType[i]);
        fiv_tensor4d* out = (fiv_tensor4d*)fiv_neural_network_get_node_output(graph->net, graph->layer_node[i]);
        if (!out) { g_fail++; printf("  [FAIL] %s read output\n", out_name); continue; }
        float max_err;
        int rc = ref_cmp(out_name, ((fiv_tensor_hdr*)out)->data.fl, 1e-4f, &max_err);
        if (rc == 0) g_pass++; else g_fail++;
    }
    /* also compare the concat outputs themselves */
    for (int i = 0; i < 23; i++) {
        if (graph->layer_node[i] < 0) continue;
        if (i != 12 && i != 15 && i != 18 && i != 21) continue;
        char out_name[40];
        snprintf(out_name, sizeof(out_name), "layer%02d_%s", i, kLayerType[i]);
        fiv_tensor4d* out = (fiv_tensor4d*)fiv_neural_network_get_node_output(graph->net, graph->layer_node[i]);
        if (!out) { g_fail++; printf("  [FAIL] %s read output\n", out_name); continue; }
        float max_err;
        int rc = ref_cmp(out_name, ((fiv_tensor_hdr*)out)->data.fl, 1e-4f, &max_err);
        if (rc == 0) g_pass++; else g_fail++;
    }

    /* Detect one2one raw heads */
    const char* head_name[6] = { "head_box0", "head_box1", "head_box2",
                                 "head_cls0", "head_cls1", "head_cls2" };
    for (int h = 0; h < 6; h++) {
        fiv_tensor4d* out = (fiv_tensor4d*)fiv_neural_network_get_node_output(graph->net, graph->head_node[h]);
        if (!out) { g_fail++; printf("  [FAIL] %s read output\n", head_name[h]); continue; }
        float max_err;
        int rc = ref_cmp(head_name[h], ((fiv_tensor_hdr*)out)->data.fl, 1e-4f, &max_err);
        if (rc == 0) g_pass++; else g_fail++;
    }

    fiv_yolo26_release(graph);
    free((void*)fold_w);
    free((void*)fold_b);
    for (int a = 0; a < 2; a++)
        for (int j = 0; j < 6; j++) free((void*)attn[a][j]);

    printf("=== P4 full net: pass=%d fail=%d ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
