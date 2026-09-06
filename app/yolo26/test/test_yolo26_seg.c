/*
 * FastIV - Fast image and vision
 * Copyright (C) 2026 Celery Chen
 *
 * P6-3 seg test: build the full YOLO26-seg graph (backbone+neck 0..22 identical
 * to detection + Segment26 one2one_cv2/cv3/cv4 heads) from the seg fold table,
 * run on the reference input and compare every layer output plus the nine raw
 * heads (box/cls/mask x 3) against the reference tensors.
 *
 * Run from build/: YOLO26_REF_DIR=../app/yolo26/test/real/ref_seg ./test_yolo26_seg
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

static const char* const kLayerType[24] = {
    "Conv","Conv","C3k2","Conv","C3k2","Conv","C3k2","Conv",
    "C3k2","SPPF","C2PSA","Upsample","Concat","C3k2","Upsample",
    "Concat","C3k2","Conv","Concat","C3k2","Conv","Concat","C3k2","-" };

int main(void){
    printf("=== YOLO26-seg full-network test (mask branch) ===\n");
    yolo26_fold_entry fold[400];
    int n_fold = yolo26_ref_load_fold(fold, 400);
    if (n_fold <= 0) { printf("FAIL fold load\n"); return 1; }
    printf("fold convs: %d\n", n_fold);

    size_t w_tot=0, b_tot=0;
    for (int i=0;i<n_fold;i++){
        size_t we=fold[i].w_off+fold[i].w_cnt, be=fold[i].b_off+fold[i].b_cnt;
        if (we>w_tot) w_tot=we;
        if (be>b_tot) b_tot=be;
    }
    const ivf32* fold_w = yolo26_ref_load_blob("fold_w.f32",0,w_tot);
    const ivf32* fold_b = yolo26_ref_load_blob("fold_b.f32",0,b_tot);
    if (!fold_w || !fold_b) { printf("FAIL blobs\n"); return 1; }

    const char* part[6]={"qkv_w","qkv_b","pe_w","pe_b","proj_w","proj_b"};
    const ivf32* attn[2][6];
    memset(attn,0,sizeof(attn));
    char nm[64]; int ndim; size_t sh[8];
    for (int a=0;a<2;a++) for (int j=0;j<6;j++){
        snprintf(nm,sizeof(nm),"attn%d_%s",a,part[j]);
        attn[a][j]=ref_load(nm,&ndim,sh);
        if (!attn[a][j]) { printf("FAIL load %s\n",nm); return 1; }
    }

    fiv_yolo26_graph* g = fiv_yolo26_build(fold,n_fold,fold_w,fold_b,attn);
    if (!g) { printf("FAIL build\n"); return 1; }
    printf("mask_node: %d %d %d\n", g->mask_node[0], g->mask_node[1], g->mask_node[2]);
    int have_mask = g->mask_node[0]>=0 && g->mask_node[1]>=0 && g->mask_node[2]>=0;
    if (!have_mask) { printf("FAIL mask branch not built\n"); g_fail++; }

    const ivf32* in_f = ref_load("input",&ndim,sh);
    if (!in_f || ndim!=4) { printf("FAIL input\n"); return 1; }
    fiv_tensor4d* input = fiv_create_tensor4d((size_t*)sh,FIV_32F1);
    size_t in_n = sh[0]*sh[1]*sh[2]*sh[3];
    memcpy(((fiv_tensor_hdr*)input)->data.fl,in_f,in_n*sizeof(ivf32));
    fiv_free((void*)in_f);
    void* final_out = NULL;
    fiv_ret r = fiv_nn_run_inference(g->net,input,&final_out);
    fiv_release_tensor((void**)&input);
    if (r != FIV_RET_OK) { printf("FAIL run r=%d\n",r); return 1; }

    /* per-layer outputs 0..22 (skip concat id 12/15/18/21 which route into next) */
    for (int i=0;i<23;i++){
        if (g->layer_node[i]<0) continue;
        if (i==12||i==15||i==18||i==21) continue;
        char on[40];
        snprintf(on,sizeof(on),"layer%02d_%s",i,kLayerType[i]);
        fiv_tensor4d* out = (fiv_tensor4d*)fiv_neural_network_get_node_output(g->net,g->layer_node[i]);
        if (!out){ g_fail++; printf("  [FAIL] %s read\n",on); continue; }
        ivf32 me;
        int rc = ref_cmp(on,((fiv_tensor_hdr*)out)->data.fl,1e-4f,&me);
        if (rc==0) g_pass++; else g_fail++;
    }
    /* concat outputs */
    for (int i=0;i<23;i++){
        if (g->layer_node[i]<0) continue;
        if (i!=12&&i!=15&&i!=18&&i!=21) continue;
        char on[40];
        snprintf(on,sizeof(on),"layer%02d_%s",i,kLayerType[i]);
        fiv_tensor4d* out=(fiv_tensor4d*)fiv_neural_network_get_node_output(g->net,g->layer_node[i]);
        if (!out){ g_fail++; printf("  [FAIL] %s read\n",on); continue; }
        ivf32 me; int rc=ref_cmp(on,((fiv_tensor_hdr*)out)->data.fl,1e-4f,&me);
        if (rc==0) g_pass++; else g_fail++;
    }
    /* nine raw heads: box0..2 cls0..2 mask0..2 */
    const char* hn[9]={"head_box0","head_box1","head_box2","head_cls0","head_cls1","head_cls2",
                       "head_mask0","head_mask1","head_mask2"};
    int node[9];
    for (int i=0;i<3;i++){
        node[i]=g->head_node[i];
        node[3+i]=g->head_node[3+i];
        node[6+i]=g->mask_node[i];
    }
    for (int h=0;h<9;h++){
        fiv_tensor4d* out=(fiv_tensor4d*)fiv_neural_network_get_node_output(g->net,node[h]);
        if (!out){ g_fail++; printf("  [FAIL] %s read\n",hn[h]); continue; }
        /* Real weights: cls1/cls2 are deepest heads (layer19/22+5 convs), fp32 drift ~3e-4 */
        ivf32 me; int rc=ref_cmp(hn[h],((fiv_tensor_hdr*)out)->data.fl,1e-3f,&me);
        if (rc==0) g_pass++; else g_fail++;
    }

    /* ---- P6-3 seg decode: rows [x1,y1,x2,y2,score,class,coef(32)] ---- */
    {
        int in_w = (int)sh[3];
        const fiv_tensor4d* hh[9];
        for (int i = 0; i < 9; i++) hh[i] = (const fiv_tensor4d*)fiv_neural_network_get_node_output(g->net, node[i]);
        int strides[3];
        for (int L = 0; L < 3; L++) strides[L] = in_w / (int)hh[L]->width;
        ivf32* det = (ivf32*)fiv_malloc((size_t)300 * 38 * sizeof(ivf32));
        int kept = fiv_yolo26_postprocess_seg(hh, strides, det, 300);
        if (kept <= 0) { g_fail++; printf("  [FAIL] seg decode kept=%d\n", kept); }
        else {
            const ivf32* ref = ref_load("detect_out", &ndim, sh);
            int ref_k = (ndim == 3) ? (int)sh[1] : (int)(sh[0]*sh[1]);
            printf("  seg decode kept=%d ref_k=%d\n", kept, ref_k);
            /* set-match: box 1e-2 / score 1e-4 / class exact (fp32 drift on the
               deepest heads was ~3e-4, decode adds nothing beyond sigmoid). */
            int* used = (int*)fiv_calloc((size_t)ref_k, sizeof(int));
            int* cmap = (int*)fiv_malloc((size_t)ref_k * sizeof(int));   /* torch row i -> C row */
            for (int i = 0; i < ref_k; i++) cmap[i] = -1;
            int unmatched = 0;
            for (int i = 0; i < ref_k; i++) {
                const ivf32* t = ref + (size_t)38 * i;
                int found = -1;
                for (int j = 0; j < kept; j++) {
                    if (used[j]) continue;
                    const ivf32* cdet = det + (size_t)38 * j;
                    int ok = (int)lrintf(cdet[5]) == (int)lrintf(t[5]) &&
                             fabsf(cdet[4]-t[4]) <= 1e-4f &&
                             fabsf(cdet[0]-t[0]) <= 1e-2f && fabsf(cdet[1]-t[1]) <= 1e-2f &&
                             fabsf(cdet[2]-t[2]) <= 1e-2f && fabsf(cdet[3]-t[3]) <= 1e-2f;
                    if (ok) { found = j; break; }
                }
                if (found < 0) { unmatched++; if (unmatched <= 3) printf("    ref row %d unmatched\n", i); }
                else { used[found] = 1; cmap[i] = found; }
            }
            if (kept == ref_k && unmatched == 0) { g_pass++; printf("  PASS  seg detect_out set-matched (%d rows)\n", kept); }
            else { g_fail++; printf("  FAIL  seg detect_out set-mismatch=%d\n", unmatched); }

            /* ---- mask rebuild: sigmoid(sum_nm coef[nm] * proto[nm]) vs mask_gt0..7 ---- */
            {
                const ivf32* proto = ref_load("proto", &ndim, sh);
                if (proto && kept >= 8) {
                    /* proto ref is [1,nm,Hp,Wp] contiguous; here sh holds its shape */
                    int nm = (int)sh[1], Hp = (int)sh[2], Wp = (int)sh[3];
                    int all_ok = 1;
                    for (int i = 0; i < 8; i++) {
                        int j = cmap[i];
                        if (j < 0) { all_ok = 0; break; }
                        char gt_name[24];
                        snprintf(gt_name, sizeof(gt_name), "mask_gt%d", i);
                        const ivf32* gt = ref_load(gt_name, &ndim, sh);
                        if (!gt) { all_ok = 0; break; }
                        const ivf32* c = det + (size_t)38 * j + 6;
                        ivf32* mbuf = (ivf32*)fiv_malloc((size_t)Hp * Wp * sizeof(ivf32));
                        for (int y = 0; y < Hp; y++)
                            for (int x = 0; x < Wp; x++) {
                                ivf32 acc = 0.0f;
                                for (int m = 0; m < nm; m++)
                                    acc += c[m] * proto[(size_t)m * Hp * Wp + (size_t)y * Wp + x];
                                mbuf[(size_t)y * Wp + x] = 1.0f / (1.0f + expf(-acc));
                            }
                        ivf32 worst = 0.0f;
                        for (int k = 0; k < Hp*Wp; k++) {
                            ivf32 d = fabsf(mbuf[k] - gt[k]);
                            if (d > worst) worst = d;
                        }
                        fiv_free(mbuf);
                        fiv_free((void*)gt);
                        if (worst > 1e-2f) { all_ok = 0; printf("    mask_gt%d worst=%.3e\n", i, (double)worst); }
                    }
                    if (all_ok) { g_pass++; printf("  PASS  mask rebuild sigmoid(proto@coef) rows 0..7\n"); }
                    else { g_fail++; printf("  FAIL  mask rebuild\n"); }
                    fiv_free((void*)proto);
                } else { g_fail++; printf("  FAIL  mask rebuild (proto/mask missing)\n"); }
            }
            fiv_free(cmap);
            fiv_free(used);
            fiv_free((void*)ref);
        }
        fiv_free(det);
    }

    fiv_yolo26_release(g);
    fiv_free((void*)fold_w); fiv_free((void*)fold_b);
    for (int a=0;a<2;a++) for (int j=0;j<6;j++) fiv_free((void*)attn[a][j]);
    printf("=== P6-3 seg net: pass=%d fail=%d ===\n",g_pass,g_fail);
    return g_fail==0?0:1;
}
