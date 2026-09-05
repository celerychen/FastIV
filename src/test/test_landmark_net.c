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
 * test_landmark_net.c - end-to-end check of the FaceMesh 478 landmark network
 * built by fiv_create_landmark_graph().
 *
 * Usage: test_landmark_net [model.bin] [input.raw] [out_lm.raw]
 *   model.bin   LMNT blob (default ../app/face/models/landmark_net.bin)
 *   input.raw   786432 float32 bytes, NHWC 256x256x3 (dumped from gt_input.npy)
 *   out_lm.raw  written: 1434 landmark logits + 1 presence logit (float32)
 *
 * The input is transposed NHWC -> NCHW because the FastIV engine keeps feature
 * maps channel-major (C, H, W). The written landmark logits are compared
 * against gt_lm_logits.npy by build/cmp_lm.py.
 */

#include "fiv_landmark.h"
#include "fiv_nn_infer.h"
#include "fiv_ctensor.h"
#include "fiv_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIV_LM_INPUT_H 256
#define FIV_LM_INPUT_W 256
#define FIV_LM_INPUT_C 3
#define FIV_LM_OUTPUT_N 1434   /* 478 landmarks * (x, y, z) */

int main(int argc, char** argv)
{
    const char* model_path = (argc > 1) ? argv[1]
        : "../app/face/models/landmark_net.bin";
    const char* input_path = (argc > 2) ? argv[2] : "gt_input_256.raw";
    const char* out_path   = (argc > 3) ? argv[3] : "out_lm.raw";

    /* ---- build the landmark graph first: its metadata carries the input
       size (256 / 192) and landmark count (478 / 468) ---- */
    fiv_landmark_graph* graph = fiv_create_landmark_graph(model_path);
    if (!graph) { fprintf(stderr, "build failed\n"); return 1; }
    const int ts = graph->tensor_size > 0 ? graph->tensor_size : 256;
    const int n_out = graph->n_landmarks * 3;

    /* ---- load NHWC ts x ts x 3 input ---- */
    FILE* f = fopen(input_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", input_path); return 1; }
    size_t n_in = (size_t)ts * ts * FIV_LM_INPUT_C;
    ivf32* nhwc = (ivf32*)fiv_malloc(n_in * sizeof(ivf32));
    if (!nhwc || fread(nhwc, sizeof(ivf32), n_in, f) != n_in) {
        fprintf(stderr, "short read %s\n", input_path);
        fiv_free(nhwc); fclose(f); return 1;
    }
    fclose(f);

    /* ---- wrap as NCHW [1, 3, ts, ts] ---- */
    size_t sh[4] = { 1, FIV_LM_INPUT_C, (size_t)ts, (size_t)ts };
    fiv_tensor4d* in = fiv_create_tensor4d(sh, FIV_32F1);
    if (!in) { fiv_free(nhwc); fiv_release_landmark_graph(graph); return 1; }
    for (int c = 0; c < FIV_LM_INPUT_C; c++)
        for (int h = 0; h < ts; h++)
            for (int w = 0; w < ts; w++)
                in->data.fl[((size_t)c * ts + h) * ts + w]
                    = nhwc[((size_t)h * ts + w) * FIV_LM_INPUT_C + c];
    fiv_free(nhwc);

    /* ---- inference ---- */
    void* final_out = NULL;
    fiv_ret r = fiv_nn_run_inference(graph->net, in, &final_out);
    if (r != FIV_RET_OK) { fprintf(stderr, "inference failed: %d\n", (int)r); goto done; }

    const fiv_tensor_hdr* lm   = (const fiv_tensor_hdr*)fiv_neural_network_get_node_output(graph->net, graph->lm_node);
    const fiv_tensor_hdr* conf = (const fiv_tensor_hdr*)fiv_neural_network_get_node_output(graph->net, graph->conf_node);
    if (!lm || !conf) { fprintf(stderr, "missing outputs\n"); goto done; }

    printf("lm out dims (N,C,H,W) = %zu,%zu,%zu,%zu  head = %.4f %.4f %.4f %.4f %.4f %.4f\n",
           ((const fiv_tensor4d*)lm)->batch, ((const fiv_tensor4d*)lm)->channels,
           ((const fiv_tensor4d*)lm)->height, ((const fiv_tensor4d*)lm)->width,
           lm->data.fl[0], lm->data.fl[1], lm->data.fl[2],
           lm->data.fl[3], lm->data.fl[4], lm->data.fl[5]);
    printf("presence logit = %.6f\n", conf->data.fl[0]);

    /* ---- dump raw outputs for the numpy comparison ---- */
    f = fopen(out_path, "wb");
    if (f) {
        fwrite(lm->data.fl, sizeof(ivf32), n_out, f);
        fwrite(conf->data.fl, sizeof(ivf32), 1, f);
        fclose(f);
        printf("wrote %s\n", out_path);
    }

done:
    fiv_release_landmark_graph(graph);
    fiv_release_tensor((void**)&in);
    return r == FIV_RET_OK ? 0 : 1;
}
