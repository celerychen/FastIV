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
 * FaceMesh 478-landmark model (see fiv_landmark_model.h). Port of the
 * reference face_landmarker.c's landmark stage: wraps the landmark graph
 * (fiv_landmark) plus the geometry chain (fiv_landmark_geom) into a single
 * rect-driven pipeline. Face detection is intentionally NOT part of this
 * model - the caller detects faces separately and feeds the landmark stage
 * normalized rects. All float math mirrors MediaPipe's float32 operators
 * (operand order preserved).
 */

#include "fiv_landmark_model.h"

#include "fiv_common.h"
#include "fiv_ctensor.h"
#include "fiv_nn.h"
#include "fiv_nn_infer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- model context ---- */
typedef struct {
    fiv_landmark_graph* graph;      /* landmark network (owns nodes) */
    fiv_tensor4d*       input;      /* NCHW [1,3,ts,ts] network input */
    iv8u*               warped;     /* ts*ts*3 uint8 warp scratch */
    ivf32*              tensor;     /* ts*ts*3 float32 [0,1] NHWC */
    int                 nl;         /* landmark count (468 or 478) */
    int                 ts;         /* network input size (192 or 256) */

    /* per-stage profiler, gated by env FIV_BENCH_LM=1 (default off) and only
       printed by fiv_release_face_landmark; costs nothing when it is off */
    int   lm_bench;
    ivf64 st_warp_ms, st_conv_ms, st_infer_ms, st_proj_ms;
#if FIV_NN_TIMING
    ivf64 st_type[FIV_NN_NODE_TYPE_NUM];   /* inference time by node type */
#endif
    ivf64 st_nfaces;
} fiv_lm_model;

static ivf32 fiv_lm_sigmoidf(ivf32 x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    ivf32 e = expf(x);
    return e / (1.0f + e);
}

/* Warp the ROI to the 256x256 tensor, run the landmark graph and project the
   478 landmarks back to source-image pixel coordinates. Mirrors
   face_landmarker.c::landmark_from_rect. */
static fiv_ret fiv_lm_run_rect(fiv_lm_model* m, const fiv_mat* image,
                               const fiv_lm_rect* rect, fiv_lm_face* out) {
    int h = (int)image->height;
    int w = (int)image->width;

    ivf64 _t0 = 0.0, _t1, _t2, _t3, _t4;
    if (m->lm_bench) _t0 = fiv_get_current_system_time();

    fiv_lm_roi_to_tensor(image->data.ptr8u, h, w, 3, rect, m->ts, m->warped, m->tensor);
    if (m->lm_bench) _t1 = fiv_get_current_system_time();

    /* NHWC [0,1] tensor -> NCHW network input */
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < m->ts; y++)
            for (int x = 0; x < m->ts; x++)
                m->input->data.fl[((size_t)c * m->ts + y) * m->ts + x] =
                    m->tensor[((size_t)y * m->ts + x) * 3 + c];
    if (m->lm_bench) _t2 = fiv_get_current_system_time();

    void* final_out = NULL;
#if FIV_NN_TIMING
    ivf64 _bt[FIV_NN_NODE_TYPE_NUM];
    if (m->lm_bench) fiv_nn_bench_enable(m->graph->net);
#endif
    fiv_ret r = fiv_nn_run_inference(m->graph->net, m->input, &final_out);
    if (r != FIV_RET_OK) return r;
    if (m->lm_bench) {
        _t3 = fiv_get_current_system_time();
#if FIV_NN_TIMING
        fiv_nn_get_bench(m->graph->net, _bt, FIV_NN_NODE_TYPE_NUM);
#endif
    }

    const fiv_tensor_hdr* lm   = (const fiv_tensor_hdr*)fiv_neural_network_get_node_output(m->graph->net, m->graph->lm_node);
    const fiv_tensor_hdr* conf = (const fiv_tensor_hdr*)fiv_neural_network_get_node_output(m->graph->net, m->graph->conf_node);
    if (!lm || !conf) return FIV_RET_ERR_UNKNOWN;

    const ivf32* raw = lm->data.fl;
    for (int ld = 0; ld < m->nl; ld++) {
        out->points[ld].x = raw[3 * ld] / (ivf32)m->ts;
        out->points[ld].y = raw[3 * ld + 1] / (ivf32)m->ts;
        out->points[ld].z = raw[3 * ld + 2] / (ivf32)m->ts;
    }

    /* keep_aspect_ratio = false -> no letterbox padding (0,0) */
    fiv_lm_letterbox_removal((ivf32(*)[3])out->points, 0.0f, 0.0f);

    /* LandmarkProjection: rebuild the pixel ROI and project with the z scale */
    ivf32 roi[5];
    fiv_lm_get_roi(rect, w, h, roi);
    ivf32 mm[16];
    fiv_lm_sub_rect_to_rect_matrix(roi, w, h, 0, mm);
    ivf32 z_scale = fiv_lm_calc_z_scale(mm);
    for (int ld = 0; ld < m->nl; ld++) {
        ivf32 nx, ny;
        fiv_lm_project_xy(out->points[ld].x, out->points[ld].y, out->points[ld].z, mm, &nx, &ny);
        out->points[ld].x = nx;
        out->points[ld].y = ny;
        out->points[ld].z = z_scale * out->points[ld].z;
    }
    out->presence = fiv_lm_sigmoidf(conf->data.fl[0]);

    if (m->lm_bench) {
        _t4 = fiv_get_current_system_time();
        m->st_warp_ms   += _t1 - _t0;
        m->st_conv_ms   += _t2 - _t1;
        m->st_infer_ms  += _t3 - _t2;
        m->st_proj_ms   += _t4 - _t3;
#if FIV_NN_TIMING
        for (int t = 0; t < FIV_NN_NODE_TYPE_NUM; t++) m->st_type[t] += _bt[t];
#endif
        m->st_nfaces    += 1.0;
    }
    return FIV_RET_OK;
}

/* ---- public API ---- */
void* fiv_create_face_landmark(const char* lm_model) {
    if (!lm_model) return NULL;

    fiv_lm_model* m = (fiv_lm_model*)fiv_calloc(1, sizeof(fiv_lm_model));
    if (!m) return NULL;

    m->graph = fiv_create_landmark_graph(lm_model);
    if (!m->graph) { fiv_free(m); return NULL; }

    const char* e = getenv("FIV_BENCH_LM");
    m->lm_bench = (e && e[0] == '1') ? 1 : 0;

    m->nl = m->graph->n_landmarks;
    m->ts = m->graph->tensor_size;
    if (m->nl <= 0 || m->nl > FIV_LM_NUM_LANDMARKS) m->nl = FIV_LM_NUM_LANDMARKS;
    if (m->ts <= 0 || m->ts > FIV_LM_TENSOR_SIZE) m->ts = FIV_LM_TENSOR_SIZE;

    size_t sh[4] = { 1, 3, (size_t)m->ts, (size_t)m->ts };
    m->input  = fiv_create_tensor4d(sh, FIV_32F1);
    m->warped = (iv8u*)fiv_calloc((size_t)m->ts * m->ts * 3, sizeof(iv8u));
    m->tensor = (ivf32*)fiv_calloc((size_t)m->ts * m->ts * 3, sizeof(ivf32));
    if (!m->input || !m->warped || !m->tensor) {
        fiv_release_tensor((void**)&m->input);
        fiv_free(m->warped);
        fiv_free(m->tensor);
        fiv_release_landmark_graph(m->graph);
        fiv_free(m);
        return NULL;
    }
    return m;
}

fiv_ret fiv_face_landmark_from_faces(void* result, fiv_mat* image,
                                     void* model, const fiv_lm_rect* rects,
                                     iv32s n) {
    if (!result || !image || !model || !rects || n <= 0) return FIV_RET_ERR_PARA;
    fiv_lm_model_result*  out  = (fiv_lm_model_result*)result;
    fiv_lm_model*         m    = (fiv_lm_model*)model;
    int cnt = n;
    if (cnt > FIV_LM_MODEL_MAX_FACES) cnt = FIV_LM_MODEL_MAX_FACES;
    out->count       = cnt;
    out->n_landmarks = m->nl;
    for (int i = 0; i < cnt; i++) {
        fiv_ret r = fiv_lm_run_rect(m, image, &rects[i], &out->faces[i]);
        if (r != FIV_RET_OK) return r;
    }
    return FIV_RET_OK;
}

fiv_ret fiv_release_face_landmark(void** model) {
    if (!model || !*model) return FIV_RET_ERR_PARA;
    fiv_lm_model* m = (fiv_lm_model*)*model;
    if (m->lm_bench && m->st_nfaces > 0.0) {
        ivf64 n = m->st_nfaces;
        fprintf(stderr,
                "[lm-bench] faces=%g\n"
                "  warp   =%8.3f ms/face\n"
                "  convert=%8.3f ms/face\n"
                "  infer  =%8.3f ms/face\n"
                "  project=%8.3f ms/face\n"
                "  total  =%8.3f ms/face\n",
                n, m->st_warp_ms / n, m->st_conv_ms / n,
                m->st_infer_ms / n, m->st_proj_ms / n,
                (m->st_warp_ms + m->st_conv_ms + m->st_infer_ms + m->st_proj_ms) / n);
#if FIV_NN_TIMING
        fprintf(stderr, "  inference by node type (ms/face):\n");
        for (int t = 0; t < FIV_NN_NODE_TYPE_NUM; t++)
            if (m->st_type[t] / n > 0.0005)
                fprintf(stderr, "    node_type=%d  %.4f ms\n", t, m->st_type[t] / n);
#endif
    }
    fiv_release_tensor((void**)&m->input);
    fiv_free(m->warped);
    fiv_free(m->tensor);
    fiv_release_landmark_graph(m->graph);
    fiv_free(m);
    *model = NULL;
    return FIV_RET_OK;
}
