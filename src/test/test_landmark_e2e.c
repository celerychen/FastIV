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
 * test_landmark_e2e.c - END-TO-END joint test: face detection -> landmark.
 *
 * Unlike test_landmark_api.c, which feeds the landmark stage a pre-computed
 * normalized rect, this test runs the real pipeline on a source image:
 *
 *   1. load 15.png (RGB24)
 *   2. fiv_face_detector_on_image  -> BlazeFace detections (bbox + 6 kps)
 *   3. fiv_lm_expanded_face_rect   -> normalized rect per face
 *   4. fiv_face_landmark_from_faces                    -> 478 landmarks per face
 *
 * This validates the STRICT SEPARATION contract: the landmark model must accept
 * the rects produced from a real detector and produce landmarks consistent with
 * the reference output (bundle "15" from dump_lm_gt.py, containing the
 * reference rect and the reference 478 landmarks).
 *
 * Usage: test_landmark_e2e [image] [lm_model] [blazeface_model] [truth_dir]
 */
#include "fiv_ctensor.h"
#include "fiv_image.h"
#include "fiv_common.h"
#include "fiv_face.h"
#include "fiv_landmark_model.h"
#include "fiv_landmark_geom.h"
#include "face_mesh_topology.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0, g_fail = 0;

static void check(int cond, const char* msg) {
    if (cond) g_pass++;
    else { g_fail++; printf("  [FAIL] %s\n", msg); }
}

/* Draw a filled circle (radius r pixels) for a NORMALIZED landmark (0..1)
   directly into a contiguous RGB8 fiv_mat. */
static void draw_point(fiv_mat* img, ivf32 nx, ivf32 ny, int r, iv8u cr, iv8u cg, iv8u cb) {
    int iw = (int)img->width, ih = (int)img->height;
    int cx = (int)(nx * (ivf32)iw);
    int cy = (int)(ny * (ivf32)ih);
    iv8u* p = img->data.ptr8u;
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int x = cx + dx, y = cy + dy;
            if (x < 0 || x >= iw || y < 0 || y >= ih) continue;
            size_t o = ((size_t)y * (size_t)iw + (size_t)x) * 3;
            p[o + 0] = cr; p[o + 1] = cg; p[o + 2] = cb;
        }
    }
}

/* Project a normalized landmark to the source image, clamp, and draw a
   colored filled circle (radius r pixels). */
static void draw_point_norm(fiv_mat* img, const fiv_lm_point* pt, int r,
                            iv8u cr, iv8u cg, iv8u cb) {
    ivf32 nx = pt->x, ny = pt->y;
    if (nx < 0.0f) nx = 0.0f; else if (nx > 1.0f) nx = 1.0f;
    if (ny < 0.0f) ny = 0.0f; else if (ny > 1.0f) ny = 1.0f;
    draw_point(img, nx, ny, r, cr, cg, cb);
}

/* 3D->2D projection for the mesh, mirroring face_landmarker_mesh.py's
   _draw_connections: the landmark's normalized x,y ARE the projected 2D
   position (the z parallax was already folded into them by the model's
   LandmarkProjection), so the screen point is simply clamp(x,0,1)*W (and y*H). */
static void proj_px(const fiv_lm_point* pt, int iw, int ih, int* px, int* py) {
    ivf32 nx = pt->x, ny = pt->y;
    if (nx < 0.0f) nx = 0.0f; else if (nx > 1.0f) nx = 1.0f;
    if (ny < 0.0f) ny = 0.0f; else if (ny > 1.0f) ny = 1.0f;
    *px = (int)(nx * (ivf32)iw);
    *py = (int)(ny * (ivf32)ih);
}

/* Bresenham line into the contiguous RGB8 fiv_mat, with a small thickness
   brush (thick >= 1). Mirrors cv2.line in the python reference. */
static void draw_seg(fiv_mat* img, int x0, int y0, int x1, int y1,
                     int thick, iv8u cr, iv8u cg, iv8u cb) {
    int iw = (int)img->width, ih = (int)img->height;
    iv8u* p = img->data.ptr8u;
    if (x0 < 0) x0 = 0; if (x0 >= iw) x0 = iw - 1;
    if (x1 < 0) x1 = 0; if (x1 >= iw) x1 = iw - 1;
    if (y0 < 0) y0 = 0; if (y0 >= ih) y0 = ih - 1;
    if (y1 < 0) y1 = 0; if (y1 >= ih) y1 = ih - 1;
    int dx = abs(x1 - x0), dy = -abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (int ox = -thick / 2; ox <= thick / 2; ox++) {
        for (int oy = -thick / 2; oy <= thick / 2; oy++) {
            int x = x0, y = y0, e2 = err;
            for (;;) {
                int xx = x + ox, yy = y + oy;
                if (xx >= 0 && xx < iw && yy >= 0 && yy < ih) {
                    size_t o = ((size_t)yy * (size_t)iw + (size_t)xx) * 3;
                    p[o + 0] = cr; p[o + 1] = cg; p[o + 2] = cb;
                }
                if (x == x1 && y == y1) break;
                e2 = 2 * err;
                if (e2 >= dy) { err += dy; x += sx; }
                if (e2 <= dx) { err += dx; y += sy; }
            }
        }
    }
}

/* Draw all edges of one topology table for one face. */
static void draw_edges(fiv_mat* img, const fiv_lm_face* face,
                       const int edges[][2], int n, int thick,
                       iv8u cr, iv8u cg, iv8u cb) {
    int iw = (int)img->width, ih = (int)img->height;
    for (int i = 0; i < n; i++) {
        int a = edges[i][0], b = edges[i][1];
        if (a < 0 || a >= FIV_LM_NUM_LANDMARKS || b < 0 || b >= FIV_LM_NUM_LANDMARKS)
            continue;
        int x0, y0, x1, y1;
        proj_px(&face->points[a], iw, ih, &x0, &y0);
        proj_px(&face->points[b], iw, ih, &x1, &y1);
        draw_seg(img, x0, y0, x1, y1, thick, cr, cg, cb);
    }
}

/* Draw the FaceMesh 3D wireframe for one face: triangles + face oval + lips
   + iris dots. Colors follow the python reference (translated BGR->RGB). */
static void draw_mesh(fiv_mat* img, const fiv_lm_face* face) {
    /* cyan triangles, thin */
    draw_edges(img, face, FIV_LM_TESSELATION, 2556, 1, 0, 255, 255);
    /* light-gray face oval, thick */
    draw_edges(img, face, FIV_LM_FACE_OVAL, 36, 2, 224, 224, 224);
    /* pink lips, thick */
    draw_edges(img, face, FIV_LM_LIPS, 40, 2, 255, 120, 180);
    /* yellow iris dots */
    for (int i = 0; i < 4; i++) {
        int a = FIV_LM_LEFT_IRIS[i][0];
        int b = FIV_LM_RIGHT_IRIS[i][0];
        draw_point_norm(img, &face->points[a], 2, 255, 255, 0);
        draw_point_norm(img, &face->points[b], 2, 255, 255, 0);
    }
}

/* Reference bundle for the tag "15" (rect + gt only; tensor/img are not needed
   because this test loads the source image from disk). Layout matches
   dump_lm_gt.py (iw, ih, rect[5], tensor[], gt[478*3], img[]). */
typedef struct { int iw, ih; ivf32 rect[5]; ivf32* gt; } e2e_gt;

static int load_gt_rect(const char* path, e2e_gt* g) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    iv32s iw, ih;
    if (fread(&iw, sizeof(iw), 1, f) != 1 || fread(&ih, sizeof(ih), 1, f) != 1) { fclose(f); return 0; }
    g->iw = (int)iw; g->ih = (int)ih;
    if (fread(g->rect, sizeof(ivf32), 5, f) != 5) { fclose(f); return 0; }
    /* skip the warp tensor, then read the 478*3 landmark ground truth */
    size_t n_t = (size_t)FIV_LM_TENSOR_SIZE * FIV_LM_TENSOR_SIZE * 3;
    if (fseek(f, (long)(n_t * sizeof(ivf32)), SEEK_CUR) != 0) { fclose(f); return 0; }
    size_t n_g = (size_t)FIV_LM_NUM_LANDMARKS * 3;
    g->gt = (ivf32*)fiv_malloc(n_g * sizeof(ivf32));
    if (!g->gt) { fclose(f); return 0; }
    if (fread(g->gt, sizeof(ivf32), n_g, f) != n_g) { fiv_free(g->gt); g->gt = NULL; fclose(f); return 0; }
    fclose(f);
    return 1;
}

int main(int argc, char** argv) {
    const char* img_path  = argc > 1 ? argv[1] : "../src/reference/c_face_detect_release/15.png";
    const char* lm_model  = argc > 2 ? argv[2] : "../app/face/models/landmark_net.bin";
    const char* det_model = argc > 3 ? argv[3] : "../app/face/models/blazeface_weights.bin";
    const char* truth_dir = argc > 4 ? argv[4] : ".lm_truth";

    /* ---- create the two strictly-separated stages ---- */
    void* detector = fiv_create_face_detetor((char*)det_model);
    check(detector != NULL, "fiv_create_face_detetor returns non-NULL");
    void* model = fiv_create_face_landmark(lm_model);
    check(model != NULL, "fiv_create_face_landmark returns non-NULL");
    if (!detector || !model) {
        printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
        return 1;
    }

    /* ---- load the source image (the reference ran detection+landmark here) ---- */
    double tl0 = fiv_get_current_system_time();
    fiv_mat* image = fiv_create_image_from_file((char*)img_path, FIV_RGB24_CS);
    double tl1 = fiv_get_current_system_time();
    printf("time_load: %.3f ms\n", tl1 - tl0);
    check(image != NULL, "fiv_create_image_from_file returns non-NULL");
    if (!image) {
        fiv_release_face_detector(&detector);
        fiv_release_face_landmark(&model);
        printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
        return 1;
    }
    int img_w = (int)image->width, img_h = (int)image->height;
    check(img_w > 0 && img_h > 0, "image has positive dimensions");

    /* ---- 1. face detection ---- */
    fiv_face_result fdres;
    memset(&fdres, 0, sizeof(fdres));
    double td0 = fiv_get_current_system_time();
    check(fiv_face_detector_on_image(&fdres, image, detector) == FIV_RET_OK,
          "fiv_face_detector_on_image returns FIV_RET_OK");
    double td1 = fiv_get_current_system_time();
    printf("time_detect: %.3f ms\n", td1 - td0);
    check(fdres.count >= 1, "detector found at least one face");
    printf("detected %d face(s) on %dx%d image\n", fdres.count, img_w, img_h);

    /* pick the highest-scoring detection for the landmark comparison */
    int best = 0;
    for (int i = 1; i < fdres.count; i++)
        if (fdres.detections[i].score > fdres.detections[best].score) best = i;
    if (fdres.count > 0) {
        const fiv_face_detection* d = &fdres.detections[best];
        char m[96];
        snprintf(m, sizeof(m), "best detection score %.3f >= 0.5", d->score);
        check(d->score >= FIV_FACE_MIN_SCORE, m);
        check(d->w > 0.0f && d->h > 0.0f, "best detection has positive size");
    }

    /* ---- 2 + 3. detector bbox/kps -> normalized landmark rect ---- */
    fiv_lm_rect rects[FIV_LM_MODEL_MAX_FACES];
    int n_rects = fdres.count;
    if (n_rects > FIV_LM_MODEL_MAX_FACES) n_rects = FIV_LM_MODEL_MAX_FACES;
    for (int i = 0; i < n_rects; i++) {
        const fiv_face_detection* d = &fdres.detections[i];
        fprintf(stderr,
                "DET[%d] score=%.6f x=%.4f y=%.4f w=%.4f h=%.4f kp0=(%.6f,%.6f) kp1=(%.6f,%.6f) kp2=(%.6f,%.6f) kp3=(%.6f,%.6f) kp4=(%.6f,%.6f) kp5=(%.6f,%.6f)\n",
                i, d->score, d->x, d->y, d->w, d->h,
                d->kps[0][0], d->kps[0][1], d->kps[1][0], d->kps[1][1],
                d->kps[2][0], d->kps[2][1], d->kps[3][0], d->kps[3][1],
                d->kps[4][0], d->kps[4][1], d->kps[5][0], d->kps[5][1]);
        ivf32 bxc = (d->x + (d->w * 0.5f)) / (ivf32)img_w;
        ivf32 byc = (d->y + (d->h * 0.5f)) / (ivf32)img_h;
        ivf32 bw  = (d->w) / (ivf32)img_w;
        ivf32 bh  = (d->h) / (ivf32)img_h;
        rects[i] = fiv_lm_expanded_face_rect(bxc, byc, bw, bh,
                                             d->kps, img_w, img_h);
        fprintf(stderr, "RECT[%d] xc=%.4f yc=%.4f w=%.4f h=%.4f rot=%.4f pxbbox=(%.0f,%.0f,w=%.0f)\n",
                i, rects[i].xc, rects[i].yc, rects[i].w, rects[i].h, rects[i].rot,
                d->x, d->y, d->w);
    }

    /* ---- 4. landmark on every detected face (batch fills all of them) ---- */
    fiv_lm_model_result cres;
    memset(&cres, 0, sizeof(cres));
    check(fiv_face_landmark_from_faces(&cres, image, model, rects, n_rects) == FIV_RET_OK,
          "fiv_face_landmark_from_faces returns FIV_RET_OK");
    check(cres.count == n_rects, "face_landmark_from_faces fills count");

    /* landmark bench: N iterations of the full batch (image still clean) */
    {
        const int bench_n = 50;
        double tsum = 0.0;
        for (int bi = 0; bi < bench_n; bi++) {
            fiv_lm_model_result bres;
            memset(&bres, 0, sizeof(bres));
            double ts = fiv_get_current_system_time();
            fiv_face_landmark_from_faces(&bres, image, model, rects, n_rects);
            tsum += fiv_get_current_system_time() - ts;
        }
        printf("landmark_bench: %d iters  avg=%.3f ms/frame  (%d faces/batch)  %.1f FPS\n",
               bench_n, tsum / bench_n, n_rects, 1000.0 * bench_n / tsum);
    }

    /* debug: dump normalized landmarks to build/our15_face{i}.txt for diff vs MediaPipe */
    {
        FILE* df = fopen("build/our15_faces.txt", "w");
        if (df) {
            fprintf(df, "%d\n", cres.count);
            for (int i = 0; i < cres.count; i++) {
                for (int ld = 0; ld < FIV_LM_NUM_LANDMARKS; ld++)
                    fprintf(df, "%.9f %.9f %.9f\n",
                            cres.faces[i].points[ld].x,
                            cres.faces[i].points[ld].y,
                            cres.faces[i].points[ld].z);
            }
            fclose(df);
        }
    }

    /* best face = landmark of the highest-scoring detection */
    check(best >= 0 && best < cres.count, "best face index in range");
    const fiv_lm_face* face = &cres.faces[best];
    check(face->presence > 0.5f, "best face presence > 0.5");

    /* all points finite and within the image-feature-normalized range */
    int in_range = 1;
    for (int ld = 0; ld < FIV_LM_NUM_LANDMARKS; ld++)
        if (!isfinite(face->points[ld].x) || !isfinite(face->points[ld].y) || !isfinite(face->points[ld].z) ||
            fabsf(face->points[ld].x) > 4.0f || fabsf(face->points[ld].y) > 4.0f)
            in_range = 0;
    check(in_range, "all 478 landmarks finite and bounded");

    /* ---- compare end-to-end output to the reference (bundle "15") ---- */
    char path[512];
    snprintf(path, sizeof(path), "%s/15.bin", truth_dir);
    e2e_gt g;
    if (load_gt_rect(path, &g)) {
        /* reference rect vs detector-derived rect (normalized center/size) */
        ivf32 dcen = fabsf(rects[best].xc - g.rect[0]) + fabsf(rects[best].yc - g.rect[1]);
        ivf32 dsize = fabsf(rects[best].w - g.rect[2]) + fabsf(rects[best].h - g.rect[3]);
        check(dcen < 0.10f,  "detector rect center within 0.10 of reference");
        check(dsize < 0.20f, "detector rect size within 0.20 of reference");

        ivf32 xy_max = 0.0f;
        for (int ld = 0; ld < FIV_LM_NUM_LANDMARKS; ld++) {
            ivf32 dx = fabsf(face->points[ld].x - g.gt[3 * ld]);
            ivf32 dy = fabsf(face->points[ld].y - g.gt[3 * ld + 1]);
            if (dx > xy_max) xy_max = dx;
            if (dy > xy_max) xy_max = dy;
        }
        printf("e2e vs reference: rect_center_delta=%.3e rect_size_delta=%.3e | lm xy max=%.3e | presence=%.4f\n",
               dcen, dsize, xy_max, face->presence);
        check(xy_max < 0.10f, "end-to-end landmark xy within 0.10 of reference");
        fiv_free(g.gt);
    } else {
        printf("  [SKIP] no 15.bin ground truth in %s (run dump_lm_gt.py); structural checks only\n", truth_dir);
    }

    /* ---- draw the 3D mesh wireframe of EVERY detected face, then save ---- */
    if (cres.count > 0 && image->dtype == FIV_8U3) {
        double tg0 = fiv_get_current_system_time();
        for (int fi = 0; fi < cres.count; fi++)
            draw_mesh(image, &cres.faces[fi]);
        double tg1 = fiv_get_current_system_time();
        char out[512];
        snprintf(out, sizeof(out), "e2e_%s_mesh.png", "15");
        fiv_ret wr = fiv_image_write(out, image);
        double tg2 = fiv_get_current_system_time();
        printf("time_draw_mesh: %.3f ms\n", tg1 - tg0);
        printf("time_write_png: %.3f ms\n", tg2 - tg1);
        check(wr == FIV_RET_OK, "fiv_image_write saves landmark overlay");
        if (wr == FIV_RET_OK)
            printf("  saved landmark overlay (%d faces) -> %s\n", cres.count, out);
    }

    /* ---- release ---- */
    check(fiv_release_face_landmark(&model) == FIV_RET_OK, "release model returns OK");
    check(fiv_release_face_detector(&detector) == FIV_RET_OK, "release detector returns OK");
    fiv_release_image(image);

    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    printf("RESULT: %s\n", g_fail == 0 ? "ALL PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}