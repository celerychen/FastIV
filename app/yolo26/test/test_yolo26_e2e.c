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

/* P5 end-to-end test: build the full YOLO26 graph (P4), run the six Detect
 * one2one raw heads through the NMS-free decode (fiv_yolo26_post.c) and
 * compare the final detections against the torch reference detect_out.
 *
 *   out rows are [x1, y1, x2, y2, score, class], score-descending;
 *   k = min(max_det=300, a_count). At 64x64 a_count = 84 < 300 so every
 *   anchor survives and detect_out[1,84,6] is compared row-for-row.
 *
 * Run from build/: ./test_yolo26_e2e
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
#include "fiv_common.h"

static int g_pass = 0;
static int g_fail = 0;

int main(void)
{
    printf("=== YOLO26 P5 end-to-end test (decode vs torch detect_out) ===\n");

    yolo26_fold_entry fold[300];
    int n_fold = yolo26_ref_load_fold(fold, 300);
    if (n_fold <= 0) { printf("FAIL: fold table load\n"); return 1; }

    size_t w_tot = 0, b_tot = 0;
    for (int i = 0; i < n_fold; i++) {
        size_t we = fold[i].w_off + fold[i].w_cnt;
        size_t be = fold[i].b_off + fold[i].b_cnt;
        if (we > w_tot) w_tot = we;
        if (be > b_tot) b_tot = be;
    }
    const ivf32* fold_w = yolo26_ref_load_blob("fold_w.f32", 0, w_tot);
    const ivf32* fold_b = yolo26_ref_load_blob("fold_b.f32", 0, b_tot);
    if (!fold_w || !fold_b) { printf("FAIL: fold weight blobs\n"); return 1; }

    const char* part[6] = { "qkv_w", "qkv_b", "pe_w", "pe_b", "proj_w", "proj_b" };
    const ivf32* attn[2][6];
    memset(attn, 0, sizeof(attn));
    char nm[64];
    int ndim; size_t sh[8];
    for (int a = 0; a < 2; a++)
        for (int j = 0; j < 6; j++) {
            snprintf(nm, sizeof(nm), "attn%d_%s", a, part[j]);
            attn[a][j] = ref_load(nm, &ndim, sh);
            if (!attn[a][j]) { printf("FAIL: load %s\n", nm); return 1; }
        }

    fiv_yolo26_graph* graph = fiv_yolo26_build(fold, n_fold, fold_w, fold_b, attn);
    if (!graph) { printf("FAIL: full graph build\n"); return 1; }

    const ivf32* in_f = ref_load("input", &ndim, sh);
    if (!in_f || ndim != 4) { printf("FAIL: input\n"); return 1; }
    int in_w = (int)sh[3];
    fiv_tensor4d* input = fiv_create_tensor4d((size_t*)sh, FIV_32F1);
    size_t in_n = sh[0] * sh[1] * sh[2] * sh[3];
    memcpy(((fiv_tensor_hdr*)input)->data.fl, in_f, in_n * sizeof(ivf32));
    fiv_free((void*)in_f);

    void* final_out = NULL;
    fiv_ret r = fiv_nn_run_inference(graph->net, input, &final_out);
    fiv_release_tensor((void**)&input);
    if (r != FIV_RET_OK) { printf("FAIL: run_inference r=%d\n", r); return 1; }

    /* collect the six raw heads (head_node order: box0,box1,box2,cls0,cls1,cls2) */
    fiv_tensor4d* heads[6];
    for (int h = 0; h < 6; h++) {
        heads[h] = (fiv_tensor4d*)fiv_neural_network_get_node_output(graph->net, graph->head_node[h]);
        if (!heads[h]) { printf("FAIL: head %d output\n", h); return 1; }
    }
    /* level strides = input / feature per level (yolo26n: 8/16/32) */
    int strides[3];
    for (int L = 0; L < 3; L++) strides[L] = in_w / (int)heads[L]->width;

    int k = 300;
    ivf32* det = (ivf32*)fiv_malloc((size_t)k * 6 * sizeof(ivf32));
    int kept = fiv_yolo26_postprocess((const fiv_tensor4d**)heads, strides, det, k);
    if (kept <= 0) { printf("FAIL: postprocess kept=%d\n", kept); return 1; }

    /* reference detect_out [1,k,6] */
    const ivf32* ref = ref_load("detect_out", &ndim, sh);
    if (!ref) { printf("FAIL: detect_out load\n"); return 1; }
    int ref_k = (ndim == 3) ? (int)sh[1] : (int)(sh[0] * sh[1]);
    printf("  kept=%d ref_k=%d\n", kept, ref_k);
    if (kept != ref_k) { printf("FAIL: detection count mismatch\n"); g_fail++; }

    /* At 64x64 every anchor survives top-k (k = min(300, A) = A = 84) and the
     * random-init model yields all-background scores that agree to ~1e-7, so a
     * row-for-row order check is numerically ill-conditioned (torch top-k and
     * this C decode tie-break differently at the last ulp). Verify instead that
     * the C output is EXACTLY the reference SET: every reference row appears
     * once in the C output within tolerances (box 1e-2, score 1e-5, class
     * exact). This proves anchors/decode/sigmoid/class mapping, which is the
     * P5 decode deliverable; the top-k ORDER semantics get a separate planted
     * check below where scores are well separated. */
    /* Real-weights run the full ~120-conv graph in fp32; torch-vs-C head
     * logits drift by ~1e-5..1e-4 absolute at 320x320 (larger activations than
     * the random-init reference). The single observed outlier was a duplicate
     * low-confidence box (s~0.058) whose score differed by 1.7e-5 - same box
     * (dbox 1.3e-4), same class. score_tol 1e-4 matches the net/blocks tests. */
    const ivf32 box_tol = 1e-2f, score_tol = 1e-4f;
    int unmatched = 0;
    if (kept != ref_k) {
        unmatched = 1;
        printf("  FAIL  count mismatch: kept=%d ref_k=%d\n", kept, ref_k);
    } else {
        int* used = (int*)fiv_calloc((size_t)kept, sizeof(int));
        if (!used) { printf("FAIL: calloc\n"); return 1; }
        for (int i = 0; i < ref_k; i++) {
            const ivf32* t = ref + (size_t)6 * i;
            int found = -1;
            for (int j = 0; j < kept; j++) {
                if (used[j]) continue;
                const ivf32* g = det + (size_t)6 * j;
                int ok = (int)lrintf(g[5]) == (int)lrintf(t[5]) &&
                         fabsf(g[4] - t[4]) <= score_tol &&
                         fabsf(g[0] - t[0]) <= box_tol && fabsf(g[1] - t[1]) <= box_tol &&
                         fabsf(g[2] - t[2]) <= box_tol && fabsf(g[3] - t[3]) <= box_tol;
                if (ok) { found = j; break; }
            }
            if (found < 0) {
                unmatched++;
                if (unmatched <= 3)
                    printf("  ref row %d unmatched: [%g,%g,%g,%g, %g, %d]\n",
                           i, t[0], t[1], t[2], t[3], t[4], (int)lrintf(t[5]));
            } else {
                used[found] = 1;
            }
        }
        fiv_free(used);
    }
    if (kept == ref_k && unmatched == 0) { g_pass++; printf("  PASS  detect_out set-matched (%d rows)\n", kept); }
    else { g_fail++; printf("  FAIL  detect_out set-mismatch=%d\n", unmatched); }

    fiv_free(det);
    fiv_free((void*)ref);
    fiv_yolo26_release(graph);
    fiv_free((void*)fold_w); fiv_free((void*)fold_b);
    for (int a = 0; a < 2; a++) for (int j = 0; j < 6; j++) fiv_free((void*)attn[a][j]);

    /* ---- planted top-k ORDER check: well-separated scores, k=2 < A=3. The
     * two-stage top-k lets ONE strong anchor emit BOTH of its top classes
     * (torch semantics: stage2 scans the k*nc flattened candidate scores), so
     * the expected rows are (anchor0,cls0) and (anchor0,cls1) - a config that a
     * one-stage per-anchor-max decode would get wrong (it would emit
     * anchor1's max instead of anchor0's second class). 3 levels of 1x1,
     * nc = 2, box heads all zero (decode = grid +- 0, so boxes are the grid
     * cell corners scaled by stride):
     *   level0 (a0, stride 8):  cls0 logit +9 -> s=0.9998766, cls1 logit +7 -> 0.9990888
     *   level1 (a1, stride 16): cls0 logit +3 -> s=0.9525741, cls1 logit -9
     *   level2 (a2, stride 32): both -9
     * amax -> a0=0.99988, a1=0.95257, a2~1e-4 -> stage1 keeps {a0,a1};
     * stage2 flatten 4 candidates -> top2 = a0cls0 (0.9998766), a0cls1 (0.9990888). */
    {
        fiv_tensor4d* hs[6];
        for (int L = 0; L < 3; L++) {
            size_t bs[4] = {1, 4, 1, 1};
            hs[L] = fiv_create_tensor4d(bs, FIV_32F1);
            size_t cs[4] = {1, 2, 1, 1};
            hs[L + 3] = fiv_create_tensor4d(cs, FIV_32F1);
            memset(((fiv_tensor_hdr*)hs[L])->data.fl, 0, 4 * sizeof(ivf32));
        }
        ivf32* cl0 = ((fiv_tensor_hdr*)hs[3])->data.fl;   /* level0: ch0=cls0 ch1=cls1 */
        ivf32* cl1 = ((fiv_tensor_hdr*)hs[4])->data.fl;   /* level1 */
        ivf32* cl2 = ((fiv_tensor_hdr*)hs[5])->data.fl;   /* level2 */
        cl0[0] = 9.0f;  cl0[1] = 7.0f;      /* anchor0: cls0 0.9998766, cls1 0.9990888 */
        cl1[0] = 3.0f;  cl1[1] = -9.0f;     /* anchor1: cls0 0.9525741 */
        cl2[0] = -9.0f; cl2[1] = -9.0f;     /* anchor2: ~1e-4 */
        int strides[3] = {8, 16, 32};
        ivf32 buf[2 * 6];
        const fiv_tensor4d* hh[6];
        for (int i = 0; i < 6; i++) hh[i] = hs[i];
        int kk = fiv_yolo26_postprocess(hh, strides, buf, 2);
        /* expect row0: anchor0 cls0, box (4,4,4,4), s 0.9998766
           expect row1: anchor0 cls1, box (4,4,4,4), s 0.9990888 */
        int ok_row0 = kk == 2 &&
            (int)lrintf(buf[5]) == 0 && fabsf(buf[4] - 0.9998766f) < 1e-4f &&
            fabsf(buf[0] - 4.0f) < 1e-3f && fabsf(buf[1] - 4.0f) < 1e-3f &&
            fabsf(buf[2] - 4.0f) < 1e-3f && fabsf(buf[3] - 4.0f) < 1e-3f;
        int ok_row1 = ok_row0 && (int)lrintf(buf[11]) == 1 &&
            fabsf(buf[10] - 0.9990888f) < 1e-4f &&
            fabsf(buf[6] - 4.0f) < 1e-3f && fabsf(buf[8] - 4.0f) < 1e-3f;
        if (ok_row1) { g_pass++; printf("  PASS  planted two-stage top-k (one anchor, two classes)\n"); }
        else { g_fail++; printf("  FAIL  planted top-k kk=%d\n  row0=[%g,%g,%g,%g, %g, %d]\n  row1=[%g,%g,%g,%g, %g, %d]\n",
              kk, buf[0],buf[1],buf[2],buf[3],buf[4],(int)lrintf(buf[5]),
              buf[6],buf[7],buf[8],buf[9],buf[10],(int)lrintf(buf[11])); }
        for (int i = 0; i < 6; i++) fiv_release_tensor((void**)&hs[i]);
    }

    printf("=== P5 e2e: pass=%d fail=%d ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
