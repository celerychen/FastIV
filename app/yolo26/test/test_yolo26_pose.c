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

/* P6/pose test: yolo26-pose (Pose26) full network. Backbone 0..22 is identical
 * to detection; layer23 Pose26 = one2one_cv2(box) + one2one_cv3(cls) +
 * one2one_cv4(85ch) -> one2one_cv4_kpts(1x1->51ch). The engine builds all of
 * it (kpt heads through the cv4-fed chain), runs on the reference input and
 * compares per-layer outputs + 9 raw heads (box/cls/kpt x 3). Then the kpts
 * decode is verified: detected rows are [x1,y1,x2,y2,score,class,
 * kpt(51)] with kpt_i = (raw_i + anchor_xy) * stride, vis = sigmoid(raw) -
 * the decode is exercised on the raw heads directly (independent of the engine
 * box/cls top-k) and the full detect_out rows are compared via set-match.
 *
 * Run from build/: YOLO26_REF_DIR=../app/yolo26/test/real/ref_pose ./test_yolo26_pose
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fiv_ctensor.h"
#include "fiv_nn_infer.h"
#include "yolo26_ref.h"
#include "fiv_yolo26.h"
#include "fiv_yolo26_post.h"
#include "fiv_image.h"
#include "fiv_common.h"

static int g_pass = 0;
static int g_fail = 0;

static const char* const kLayerType[24] = {
    "Conv","Conv","C3k2","Conv","C3k2","Conv","C3k2","Conv",
    "C3k2","SPPF","C2PSA","Upsample","Concat","C3k2","Upsample",
    "Concat","C3k2","Conv","Concat","C3k2","Conv","Concat","C3k2","-" };

/* ---- pose overlay rendering (RGB8 HWC image, all coords are PIXELS) ---- */

/* COCO-17 limb pairs (0-based): (l/r)-shoulder->elbow->wrist, hip->knee->ankle,
   torso, shoulders, and the head chain (mirrors the yolo pose skeleton). */
static const int kSkeleton[19][2] = {
    {15,13},{13,11},{16,14},{14,12},{11,12},{5,11},{6,12},{5,6},
    {5,7},{6,8},{7,9},{8,10},{1,2},{0,1},{0,2},{1,3},{2,4},{3,5},{4,6}
};

static void pose_draw_circle(fiv_mat* img, int cx, int cy, int rad,
                             iv8u cr, iv8u cg, iv8u cb)
{
    int iw = (int)img->width, ih = (int)img->height;
    iv8u* p = img->data.ptr8u;
    for (int dy = -rad; dy <= rad; dy++)
        for (int dx = -rad; dx <= rad; dx++) {
            int x = cx + dx, y = cy + dy;
            if (x < 0 || x >= iw || y < 0 || y >= ih) continue;
            size_t o = ((size_t)y * (size_t)iw + (size_t)x) * 3;
            p[o + 0] = cr; p[o + 1] = cg; p[o + 2] = cb;
        }
}

/* Bresenham line with a thickness brush. */
static void pose_draw_line(fiv_mat* img, int x0, int y0, int x1, int y1,
                           int thick, iv8u cr, iv8u cg, iv8u cb)
{
    int iw = (int)img->width, ih = (int)img->height;
    iv8u* p = img->data.ptr8u;
    if (x0 < 0) x0 = 0; if (x0 >= iw) x0 = iw - 1;
    if (x1 < 0) x1 = 0; if (x1 >= iw) x1 = iw - 1;
    if (y0 < 0) y0 = 0; if (y0 >= ih) y0 = ih - 1;
    if (y1 < 0) y1 = 0; if (y1 >= ih) y1 = ih - 1;
    int dx = abs(x1 - x0), dy = -abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (int ox = -thick / 2; ox <= thick / 2; ox++)
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

/* Map a letterbox-grid point (model input coords, 0..in) to source-image
   pixels given letterbox geometry, then clamp to the image. */
static void pose_map(const ivf32* l, int in, int iw, int ih,
                     int nw, int nh, int* px, int* py)
{
    ivf32 rscale = (ivf32)nw / (ivf32)iw;   /* == nh/ih */
    ivf32 dx = ((ivf32)in - (ivf32)nw) * 0.5f;
    ivf32 dy = ((ivf32)in - (ivf32)nh) * 0.5f;
    int x = (int)((l[0] - dx) / rscale + 0.5f);
    int y = (int)((l[1] - dy) / rscale + 0.5f);
    if (x < 0) x = 0; else if (x >= iw) x = iw - 1;
    if (y < 0) y = 0; else if (y >= ih) y = ih - 1;
    *px = x; *py = y;
}

/* Rectangle outline (4 segments, thickness 2). */
static void pose_draw_box(fiv_mat* img, int x0, int y0, int x1, int y1,
                          iv8u cr, iv8u cg, iv8u cb)
{
    pose_draw_line(img, x0, y0, x1, y0, 2, cr, cg, cb);
    pose_draw_line(img, x1, y0, x1, y1, 2, cr, cg, cb);
    pose_draw_line(img, x1, y1, x0, y1, 2, cr, cg, cb);
    pose_draw_line(img, x0, y1, x0, y0, 2, cr, cg, cb);
}

int main(void)
{
    printf("=== YOLO26-pose full-network test ===\n");
    yolo26_fold_entry fold[400];
    int n_fold = yolo26_ref_load_fold(fold, 400);
    if (n_fold <= 0) { printf("FAIL fold load\n"); return 1; }
    printf("fold convs: %d\n", n_fold);

    size_t w_tot = 0, b_tot = 0;
    for (int i = 0; i < n_fold; i++) {
        size_t we = fold[i].w_off + fold[i].w_cnt, be = fold[i].b_off + fold[i].b_cnt;
        if (we > w_tot) w_tot = we;
        if (be > b_tot) b_tot = be;
    }
    const ivf32* fold_w = yolo26_ref_load_blob("fold_w.f32", 0, w_tot);
    const ivf32* fold_b = yolo26_ref_load_blob("fold_b.f32", 0, b_tot);
    if (!fold_w || !fold_b) { printf("FAIL blobs\n"); return 1; }

    const char* part[6] = {"qkv_w","qkv_b","pe_w","pe_b","proj_w","proj_b"};
    const ivf32* attn[2][6];
    memset(attn, 0, sizeof(attn));
    char nm[64]; int ndim; size_t sh[8];
    for (int a = 0; a < 2; a++) for (int j = 0; j < 6; j++) {
        snprintf(nm, sizeof(nm), "attn%d_%s", a, part[j]);
        attn[a][j] = ref_load(nm, &ndim, sh);
        if (!attn[a][j]) { printf("FAIL load %s\n", nm); return 1; }
    }

    fiv_yolo26_graph* g = fiv_yolo26_build(fold, n_fold, fold_w, fold_b, attn);
    if (!g) { printf("FAIL build\n"); return 1; }
    printf("kpt_node: %d %d %d\n", g->kpt_node[0], g->kpt_node[1], g->kpt_node[2]);
    if (!(g->kpt_node[0] >= 0 && g->kpt_node[1] >= 0 && g->kpt_node[2] >= 0)) {
        printf("FAIL kpt branch not built\n"); g_fail++;
    }

    const ivf32* in_f = ref_load("input", &ndim, sh);
    if (!in_f || ndim != 4) { printf("FAIL input\n"); return 1; }
    int in_w = (int)sh[3];
    fiv_tensor4d* input = fiv_create_tensor4d((size_t*)sh, FIV_32F1);
    size_t in_n = sh[0]*sh[1]*sh[2]*sh[3];
    memcpy(((fiv_tensor_hdr*)input)->data.fl, in_f, in_n*sizeof(ivf32));
    fiv_free((void*)in_f);
    void* final_out = NULL;
    fiv_ret r = fiv_nn_run_inference(g->net, input, &final_out);
    fiv_release_tensor((void**)&input);
    if (r != FIV_RET_OK) { printf("FAIL run r=%d\n", r); return 1; }

    /* per-layer outputs 0..22 */
    for (int i = 0; i < 23; i++) {
        if (g->layer_node[i] < 0) continue;
        char on[40];
        snprintf(on, sizeof(on), "layer%02d_%s", i, kLayerType[i]);
        fiv_tensor4d* out = (fiv_tensor4d*)fiv_neural_network_get_node_output(g->net, g->layer_node[i]);
        if (!out) { g_fail++; printf("  [FAIL] %s read\n", on); continue; }
        ivf32 me; int rc = ref_cmp(on, ((fiv_tensor_hdr*)out)->data.fl, 1e-4f, &me);
        if (rc == 0) g_pass++; else { g_fail++; printf("  [FAIL] %s err=%.2e\n", on, (double)me); }
    }

    /* nine raw heads */
    const char* hn[9] = {"head_box0","head_box1","head_box2",
                         "head_cls0","head_cls1","head_cls2",
                         "head_kpt0","head_kpt1","head_kpt2"};
    int node[9];
    for (int i = 0; i < 3; i++) { node[i] = g->head_node[i]; node[3+i] = g->head_node[3+i]; node[6+i] = g->kpt_node[i]; }
    for (int h = 0; h < 9; h++) {
        fiv_tensor4d* out = (fiv_tensor4d*)fiv_neural_network_get_node_output(g->net, node[h]);
        if (!out) { g_fail++; printf("  [FAIL] %s read\n", hn[h]); continue; }
        ivf32 me; int rc = ref_cmp(hn[h], ((fiv_tensor_hdr*)out)->data.fl, 1e-3f, &me);
        if (rc == 0) g_pass++; else { g_fail++; printf("  [FAIL] %s err=%.2e\n", hn[h], (double)me); }
    }

    /* pose decode (57 cols = 6 + 51 kpts) set-match against detect_out */
    {
        const fiv_tensor4d* hh[9];
        for (int i = 0; i < 9; i++) hh[i] = (const fiv_tensor4d*)fiv_neural_network_get_node_output(g->net, node[i]);
        int strides[3];
        for (int L = 0; L < 3; L++) strides[L] = in_w / (int)hh[L]->width;
        ivf32* det = (ivf32*)fiv_malloc((size_t)300 * 57 * sizeof(ivf32));
        int kept = fiv_yolo26_postprocess_pose(hh, strides, det, 300);
        if (kept <= 0) { g_fail++; printf("  [FAIL] pose decode kept=%d\n", kept); }
        else {
            const ivf32* ref = ref_load("detect_out", &ndim, sh);
            int refk = (int)sh[1];
            printf("  pose decode kept=%d refk=%d\n", kept, refk);
            int* used = (int*)fiv_calloc((size_t)refk, sizeof(int));
            int* cmap = (int*)fiv_malloc((size_t)refk * sizeof(int));
            for (int i = 0; i < refk; i++) cmap[i] = -1;
            int unmatched = 0;
            for (int i = 0; i < refk; i++) {
                const ivf32* t = ref + (size_t)57 * i;
                int found = -1;
                for (int j = 0; j < kept; j++) {
                    if (used[j]) continue;
                    const ivf32* cdet = det + (size_t)57 * j;
                    int ok = (int)lrintf(cdet[5]) == (int)lrintf(t[5]) &&
                             fabsf(cdet[4]-t[4]) <= 1e-4f &&
                             fabsf(cdet[0]-t[0]) <= 1e-2f && fabsf(cdet[1]-t[1]) <= 1e-2f &&
                             fabsf(cdet[2]-t[2]) <= 1e-2f && fabsf(cdet[3]-t[3]) <= 1e-2f;
                    if (ok) { found = j; break; }
                }
                if (found < 0) { unmatched++; if (unmatched <= 3) printf("    ref row %d unmatched\n", i); }
                else { used[found] = 1; cmap[i] = found; }
            }
            /* kpt columns (6..56) of matched rows must agree */
            int kpt_bad = 0;
            if (unmatched == 0)
                for (int i = 0; i < refk; i++) {
                    int j = cmap[i];
                    if (j < 0) { kpt_bad++; continue; }
                    const ivf32* t = ref + (size_t)57 * i;
                    const ivf32* cdet = det + (size_t)57 * j;
                    ivf32 worst = 0.0f;
                    for (int c = 6; c < 57; c++) {
                        ivf32 d = fabsf(cdet[c] - t[c]);
                        if (d > worst) worst = d;
                    }
                    if (worst > 0.3f) { kpt_bad++; if (kpt_bad <= 3) printf("    row %d kpt worst=%.3f\n", i, (double)worst); }
                }
            if (kept == refk && unmatched == 0 && kpt_bad == 0) {
                g_pass++; printf("  PASS  pose decode set-matched (%d rows, kpt 6..56 ok)\n", kept);
            } else {
                g_fail++; printf("  FAIL  pose decode unmatched=%d kpt_bad=%d\n", unmatched, kpt_bad);
            }
            fiv_free(used); fiv_free(cmap); fiv_free((void*)ref);
        }
        /* ---- render detected poses (top rows, score>=0.3) onto bus.jpg ---- */
        {
            const int in = in_w;                 /* model input is a square letterbox */
            fiv_mat* img = fiv_create_image_from_file(
                (char*)"../app/yolo26/test/real/bus.jpg", FIV_RGB24_CS);
            if (!img) { g_fail++; printf("  [FAIL] load bus.jpg for pose overlay\n"); }
            else {
                const int iw = (int)img->width, ih = (int)img->height;
                const ivf32 rscale = (ivf32)in / (iw > ih ? (ivf32)iw : (ivf32)ih);
                const int nw = (int)(iw * rscale + 0.5f);
                const int nh = (int)(ih * rscale + 0.5f);
                int accx[64], accy[64];          /* accepted box centers (orig px) */
                int n_acc = 0;
                for (int t = 0; t < kept && n_acc < 64; t++) {
                    const ivf32* row = det + (size_t)57 * t;
                    if (row[4] < 0.3f) break;       /* rows are score-descending */
                    int x1, y1, x2, y2;
                    pose_map(&row[0], in, iw, ih, nw, nh, &x1, &y1);
                    pose_map(&row[2], in, iw, ih, nw, nh, &x2, &y2);
                    int cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
                    int dup = 0;
                    for (int a = 0; a < n_acc; a++)
                        if (abs(cx - accx[a]) + abs(cy - accy[a]) < 10) dup = 1;
                    if (dup) continue;
                    accx[n_acc] = cx;
                    accy[n_acc] = cy;
                    n_acc++;
                    pose_draw_box(img, x1, y1, x2, y2, 40, 220, 40);
                    int px[17], py[17], vis[17];
                    for (int j = 0; j < 17; j++) {
                        const ivf32* k = &row[6 + 3 * j];
                        ivf32 l2[2] = { k[0], k[1] };
                        pose_map(l2, in, iw, ih, nw, nh, &px[j], &py[j]);
                        vis[j] = k[2] >= 0.5f;
                        if (vis[j])
                            pose_draw_circle(img, px[j], py[j], 3, 235, 60, 60);
                    }
                    for (int e = 0; e < 19; e++) {
                        int a = kSkeleton[e][0], b = kSkeleton[e][1];
                        if (vis[a] && vis[b])
                            pose_draw_line(img, px[a], py[a], px[b], py[b], 2, 0, 255, 255);
                    }
                }
                printf("  overlay: %d pose instance(s) drawn (score>=0.3)\n", n_acc);
                fiv_ret wr = fiv_image_write(
                    (char*)"../app/yolo26/test/real/bus_pose_C.png", img);
                if (wr == FIV_RET_OK) { g_pass++; printf("  PASS  pose overlay saved (bus_pose_C.png)\n"); }
                else { g_fail++; printf("  [FAIL] fiv_image_write pose overlay (%d)\n", wr); }
                fiv_release_image(img);
            }
        }
        fiv_free(det);
    }

    fiv_yolo26_release(g);
    fiv_free((void*)fold_w); fiv_free((void*)fold_b);
    for (int a = 0; a < 2; a++) for (int j = 0; j < 6; j++) fiv_free((void*)attn[a][j]);
    printf("=== P6 pose: pass=%d fail=%d ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
