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
 * Implementation of the simple 3-interface YuNet wrapper (see fiv_yunet_model.h).
 * Weight loading replicates the YWT1 blob format that iv_tv_time reads: a
 * "YWT1" magic, a layer count, then per layer a 6-int header plus the raw /w and
 * /b ivf32s. Depthwise weights are stored transposed (tap-major) and are
 * reindexed to channel-major before injection, exactly like load_bin() does.
 */

#include <stdio.h>
#include <string.h>

#include "fiv_yunet_model.h"
#include "fiv_common.h"
#include "fiv_nn.h"
#include "fiv_nn_infer.h"

/* Opaque detector: the assembled graph plus a per-layer weight cache used while
   injecting. Weights are freed once they have been copied into the engine. */
typedef struct {
    fiv_yunet_graph* graph;
    ivf32* W[FIV_YUNET_NUM_CONV];
    ivf32* B[FIV_YUNET_NUM_CONV];
} fiv_yunet_model;

static void fiv_yunet_model_free_weights(fiv_yunet_model* det)
{
    for (int i = 0; i < FIV_YUNET_NUM_CONV; i++) { fiv_free(det->W[i]); fiv_free(det->B[i]); }
}

void* fiv_create_yunet_detector(char* model_name)
{
    fiv_yunet_model* det = (fiv_yunet_model*)fiv_calloc(1, sizeof(fiv_yunet_model));
    if (!det) return NULL;

    FILE* f = fopen(model_name, "rb");
    if (!f) { fiv_free(det); return NULL; }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "YWT1", 4) != 0) { fclose(f); fiv_free(det); return NULL; }
    int32_t count;
    if (fread(&count, 4, 1, f) != 1 || count != FIV_YUNET_NUM_CONV) { fclose(f); fiv_free(det); return NULL; }

    for (int i = 0; i < count; i++) {
        int32_t v[6];
        if (fread(v, 4, 6, f) != 6) { fclose(f); fiv_yunet_model_free_weights(det); fiv_free(det); return NULL; }
        int channels = v[0], dw = v[2], wc = v[4], bc = v[5];
        ivf32* rawW = (ivf32*)fiv_calloc(wc > 0 ? (size_t)wc : 1, sizeof(ivf32));
        ivf32* rawB = (ivf32*)fiv_calloc(bc > 0 ? (size_t)bc : 1, sizeof(ivf32));
        if (!rawW || !rawB) { fiv_free(rawW); fiv_free(rawB); fclose(f); fiv_yunet_model_free_weights(det); fiv_free(det); return NULL; }
        if (fread(rawW, 4, (size_t)wc, f) != (size_t)wc ||
            fread(rawB, 4, (size_t)bc, f) != (size_t)bc) {
            fiv_free(rawW); fiv_free(rawB); fclose(f);
            fiv_yunet_model_free_weights(det); fiv_free(det); return NULL;
        }
        det->W[i] = (ivf32*)fiv_calloc(wc > 0 ? (size_t)wc : 1, sizeof(ivf32));
        det->B[i] = (ivf32*)fiv_calloc(bc > 0 ? (size_t)bc : 1, sizeof(ivf32));
        if (!det->W[i] || !det->B[i]) {
            fiv_free(rawW); fiv_free(rawB); fclose(f);
            fiv_yunet_model_free_weights(det); fiv_free(det); return NULL;
        }
        memcpy(det->B[i], rawB, sizeof(ivf32) * (size_t)bc);
        if (dw) {
            /* depthwise: raw is tap-major [9][ch], engine wants channel-major [ch][9] */
            for (int c = 0; c < channels; c++)
                for (int t = 0; t < 9; t++)
                    det->W[i][c * 9 + t] = rawW[t * channels + c];
        } else {
            memcpy(det->W[i], rawW, sizeof(ivf32) * (size_t)wc);
        }
        fiv_free(rawW); fiv_free(rawB);
    }
    fclose(f);

    det->graph = fiv_yunet_build_graph();
    if (!det->graph) { fiv_yunet_model_free_weights(det); fiv_free(det); return NULL; }

    for (int i = 0; i < FIV_YUNET_NUM_CONV; i++) {
        if (i == 0) {
            /* conv_head is the fused stride-2 3x3 node: reindex via the helper. */
            if (fiv_yunet_set_convhead_weight(det->graph->net, det->graph->filter_node[0], det->W[0], det->B[0]) != FIV_RET_OK) {
                fiv_yunet_model_free_weights(det); fiv_yunet_release_graph(det->graph); fiv_free(det); return NULL;
            }
        } else {
            if (fiv_neural_network_set_node_weight(det->graph->net, det->graph->filter_node[i], det->W[i]) != FIV_RET_OK ||
                fiv_neural_network_set_node_bias(det->graph->net, det->graph->filter_node[i], det->B[i]) != FIV_RET_OK) {
                fiv_yunet_model_free_weights(det); fiv_yunet_release_graph(det->graph); fiv_free(det); return NULL;
            }
        }
    }

    /* weights are now owned by the engine; drop the temporary cache */
    fiv_yunet_model_free_weights(det);
    return det;
}

fiv_ret fiv_yunet_detector_on_image(void* face_info, fiv_mat* image, void* detector)
{
    if (!face_info || !image || !detector) return FIV_RET_ERR_PARA;
    fiv_yunet_model*        det    = (fiv_yunet_model*)detector;
    fiv_yunet_model_result* result = (fiv_yunet_model_result*)face_info;

    int W = (int)image->width, H = (int)image->height;
    int wstep = W * 3;   /* RGB, 3 channels */
    return fiv_yunet_detect(det->graph, result->detections, &result->count,
                            image->data.ptr8u, W, H, wstep);
}

fiv_ret fiv_release_yunet_detector(void** detector)
{
    if (!detector || !*detector) return FIV_RET_ERR_PARA;
    fiv_yunet_model* det = (fiv_yunet_model*)*detector;
    fiv_yunet_release_graph(det->graph);
    fiv_free(det);
    *detector = NULL;
    return FIV_RET_OK;
}