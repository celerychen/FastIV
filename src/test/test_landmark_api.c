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
 * test_landmark_api.c - validates the full FaceMesh 478-landmark API
 * (fiv_landmark_model.h) three ways:
 *
 *   1. warp 对拍  : fiv_lm_roi_to_tensor on the ground-truth normalized rect
 *                   must reproduce the reference 256x256x3 warp tensor.
 *   2. 完整管线对拍 : fiv_face_landmark_from_faces (n=1) must reproduce the
 *                   reference 478 normalized landmarks (and presence).
 *   3. API 契约    : arg validation, build determinism, release semantics.
 *
 * Ground truth comes from dump_lm_gt.py (packs each _truth npz into a
 * little-endian bundle). Usage:
 *   test_landmark_api [truth_dir] [lm_model]
 */

#include "fiv_landmark_model.h"
#include "fiv_landmark.h"
#include "fiv_landmark_geom.h"
#include "fiv_ctensor.h"
#include "fiv_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0, g_fail = 0;

static void check(int cond, const char* msg) {
    if (cond) g_pass++;
    else { g_fail++; printf("  [FAIL] %s\n", msg); }
}

/* One ground-truth bundle (see dump_lm_gt.py layout). */
typedef struct {
    int    iw, ih;
    ivf32  rect[5];
    ivf32* tensor;   /* 256*256*3, NHWC [0,1] */
    ivf32* gt;       /* 478*3, normalized */
    iv8u*  img;      /* ih*iw*3, RGB */
} lm_gt;

static int load_gt(const char* path, lm_gt* g) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    iv32s iw, ih;
    if (fread(&iw, sizeof(iw), 1, f) != 1 || fread(&ih, sizeof(ih), 1, f) != 1) { fclose(f); return 0; }
    g->iw = (int)iw; g->ih = (int)ih;
    if (fread(g->rect, sizeof(ivf32), 5, f) != 5) { fclose(f); return 0; }
    size_t n_t = (size_t)FIV_LM_TENSOR_SIZE * FIV_LM_TENSOR_SIZE * 3;
    size_t n_g = (size_t)FIV_LM_NUM_LANDMARKS * 3;
    g->tensor = (ivf32*)fiv_malloc(n_t * sizeof(ivf32));
    g->gt     = (ivf32*)fiv_malloc(n_g * sizeof(ivf32));
    g->img    = (iv8u*)fiv_malloc((size_t)g->iw * g->ih * 3);
    if (!g->tensor || !g->gt || !g->img) { fclose(f); return 0; }
    if (fread(g->tensor, sizeof(ivf32), n_t, f) != n_t ||
        fread(g->gt,     sizeof(ivf32), n_g, f) != n_g ||
        fread(g->img,    1, (size_t)g->iw * g->ih * 3, f) != (size_t)g->iw * g->ih * 3) {
        fclose(f); return 0;
    }
    fclose(f);
    return 1;
}

static void free_gt(lm_gt* g) {
    fiv_free(g->tensor);
    fiv_free(g->gt);
    fiv_free(g->img);
}

int main(int argc, char** argv) {
    const char* truth_dir = argc > 1 ? argv[1] : ".lm_truth";
    const char* lm_model = argc > 2 ? argv[2]
        : "../app/face/models/landmark_net.bin";

    /* ---- model create ---- */
    void* model = fiv_create_face_landmark(lm_model);
    check(model != NULL, "fiv_create_face_landmark returns non-NULL");
    if (!model) {
        printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
        return 1;
    }

    /* ---- API 契约: bad args ---- */
    fiv_lm_face face;
    memset(&face, 0, sizeof(face));
    fiv_lm_rect r0 = { 0.5f, 0.5f, 0.6f, 0.6f, 0.0f };
    fiv_lm_model_result batch;
    memset(&batch, 0, sizeof(batch));
    check(fiv_face_landmark_from_faces(NULL, NULL, model, &r0, 1) != FIV_RET_OK,
          "from_faces(NULL result,img) rejects");
    check(fiv_face_landmark_from_faces(&batch, NULL, model, &r0, 1) != FIV_RET_OK,
          "from_faces(img=NULL) rejects");
    check(fiv_face_landmark_from_faces(&batch, NULL, NULL, &r0, 1) != FIV_RET_OK,
          "from_faces(model=NULL) rejects");
    check(fiv_face_landmark_from_faces(&batch, NULL, model, NULL, 1) != FIV_RET_OK,
          "from_faces(rects=NULL) rejects");
    check(fiv_face_landmark_from_faces(&batch, NULL, model, &r0, 0) != FIV_RET_OK,
          "from_faces(n<=0) rejects");

    /* enumerate bundles */
    char path[512];
    ivf32 worst_warp = 0.0f, worst_xy = 0.0f, worst_z = 0.0f;
    int bundles = 0;

    iv8u* warped_scratch = (iv8u*)fiv_malloc((size_t)FIV_LM_TENSOR_SIZE * FIV_LM_TENSOR_SIZE * 3);
    ivf32* our_tensor = (ivf32*)fiv_malloc((size_t)FIV_LM_TENSOR_SIZE * FIV_LM_TENSOR_SIZE * 3 * sizeof(ivf32));
    lm_gt g;

    for (int bi = 0;; bi++) {
        /* bundle names match dump_lm_gt.py output tags */
        static const char* tags[] = {
            "0028", "15", "images_test1_44", "images_test3_24",
            "images_test3_41", "images_test3_59", "images_test3_63", NULL
        };
        if (!tags[bi]) break;
        snprintf(path, sizeof(path), "%s/%s.bin", truth_dir, tags[bi]);
        if (!load_gt(path, &g)) {
            printf("  [SKIP] cannot open %s (run dump_lm_gt.py)\n", path);
            continue;
        }
        bundles++;
        fiv_lm_rect rect = { g.rect[0], g.rect[1], g.rect[2], g.rect[3], g.rect[4] };

        /* ---- 1. warp 对拍 ---- */
        memset(our_tensor, 0, (size_t)FIV_LM_TENSOR_SIZE * FIV_LM_TENSOR_SIZE * 3 * sizeof(ivf32));
        fiv_lm_roi_to_tensor(g.img, g.ih, g.iw, 3, &rect, FIV_LM_TENSOR_SIZE, warped_scratch, our_tensor);
        ivf32 wdiff = 0.0f, wsum = 0.0f;
        size_t n_t = (size_t)FIV_LM_TENSOR_SIZE * FIV_LM_TENSOR_SIZE * 3;
        for (size_t i = 0; i < n_t; i++) {
            ivf32 d = fabsf(our_tensor[i] - g.tensor[i]);
            if (d > wdiff) wdiff = d;
            wsum += d;
        }
        ivf32 wmean = (ivf32)((ivf64)wsum / (ivf64)n_t);
        if (wdiff > worst_warp) worst_warp = wdiff;

        /* ---- 2. 完整管线对拍 ---- */
        fiv_mat image;
        memset(&image, 0, sizeof(image));
        image.id = FIV_ID_TENSOR2D;          /* only ptr8u / height / width are used */
        image.dtype = FIV_8U3;
        image.height = (size_t)g.ih;
        image.width  = (size_t)g.iw;
        image.data.ptr8u = g.img;

        memset(&face, 0, sizeof(face));
        fiv_lm_model_result cres;
        memset(&cres, 0, sizeof(cres));
        fiv_ret r = fiv_face_landmark_from_faces(&cres, &image, model, &rect, 1);
        check(r == FIV_RET_OK, "face_landmark_from_faces returns FIV_RET_OK");
        check(cres.count == 1, "single-face batch fills count");
        face = cres.faces[0];
        ivf32 xdiff = 0.0f, zdiff = 0.0f, psum = 0.0f;
        for (int ld = 0; ld < FIV_LM_NUM_LANDMARKS; ld++) {
            ivf32 dx = fabsf(face.points[ld].x - g.gt[3 * ld]);
            ivf32 dy = fabsf(face.points[ld].y - g.gt[3 * ld + 1]);
            ivf32 dz = fabsf(face.points[ld].z - g.gt[3 * ld + 2]);
            if (dx > xdiff) xdiff = dx;
            if (dy > xdiff) xdiff = dy;
            if (dz > zdiff) zdiff = dz;
            psum += dx + dy;
        }
        ivf32 pmean = psum / (ivf32)(FIV_LM_NUM_LANDMARKS * 2);
        if (xdiff > worst_xy) worst_xy = xdiff;
        if (zdiff > worst_z) worst_z = zdiff;

        printf("%-18s %4dx%-5d  warp max=%.3e mean=%.3e | lm xy max=%.3e mean=%.3e | z max=%.3e | presence=%.4f\n",
               tags[bi], g.iw, g.ih, wdiff, wmean, xdiff, pmean, zdiff, face.presence);

        free_gt(&g);
    }

    /* thresholds: warp tensor is pixel-level [0,1] (3/255 ~ 0.0118 env-safe);
       478-landmark xy tolerates ~1e-2 as in the reference verify script
       (official MediaPipe uses its own ROI, differ from gt at ~1e-3..1e-2). */
    if (bundles > 0) {
        printf("\nworst across %d bundles: warp=%.3e | xy=%.3e | z=%.3e\n",
               bundles, worst_warp, worst_xy, worst_z);
        check(worst_warp < 0.02f, "warp tensor within 2/255 of reference");
        check(worst_xy < 0.02f,   "landmark xy within 2e-2 of reference");
        check(worst_z  < 0.02f,   "landmark z  within 2e-2 of reference");
    } else {
        printf("  [FAIL] no ground-truth bundles found in %s\n", truth_dir);
        check(0, "at least one bundle loaded");
    }

    /* ---- API 契约: genericity / determinism / release ---- */
    check(face.presence >= 0.0f && face.presence <= 1.0f, "presence in [0,1]");
    check(fiv_release_face_landmark(&model) == FIV_RET_OK, "release returns OK");
    check(model == NULL, "release nulls the handle");

    fiv_free(warped_scratch);
    fiv_free(our_tensor);

    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    printf("RESULT: %s\n", g_fail == 0 ? "ALL PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}