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

/* P2 operator test for the YOLO26 port: verifies the composite ATTENTION node
 * (YOLO26 C2PSA / attn=True C3k2) end-to-end against torch reference tensors
 * dumped by app/yolo26/test/gen_ref.py. Three cases run:
 *   attn0 / attn1  real Attention instances of the 64x64 net (2x2 maps, N=4;
 *                  q/k/v ~0 so these are weak for catching scale/qk/pe bugs)
 *   syn_attn       synthetic Attention (same hyperparams) on a 16x16 input
 *                  (N=256) with O(1) activations - the discriminating case.
 * For each we load the node input, the three conv weight sets (qkv / pe /
 * proj), run the node, and compare its output to <prefix>_out (the exact module
 * output produced by ultralytics Attention.forward).
 *
 * Build: project Makefile wires this as `test_yolo26_attn`; run from build/
 * (loader default YOLO26_REF_DIR = ../app/yolo26/test/ref). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fiv_ctensor.h"
#include "fiv_attention_node.h"
#include "yolo26_ref.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (cond) { g_pass++; printf("  PASS  %s\n", msg); }               \
        else      { g_fail++; printf("  FAIL  %s\n", msg); }               \
    } while (0)

static void test_attention(const char* prefix, int dim, int num_heads, int key_dim, int head_dim, float scale)
{
    printf("[Attention %s] dim=%d heads=%d key_dim=%d head_dim=%d\n", prefix, dim, num_heads, key_dim, head_dim);

    /* node input */
    int ndim; size_t ishape[4];
    char in_name[40];
    snprintf(in_name, sizeof(in_name), "%s_in", prefix);
    const float* in_f = ref_load(in_name, &ndim, ishape);
    if (!in_f) { CHECK(0, "load input"); return; }
    fiv_tensor4d* in = fiv_create_tensor4d((size_t*)ishape, FIV_32F1);
    if (!in) { CHECK(0, "create input tensor"); free((void*)in_f); return; }
    size_t in_n = ishape[0]*ishape[1]*ishape[2]*ishape[3];
    memcpy(((fiv_tensor_hdr*)in)->data.fl, in_f, in_n*sizeof(float));
    free((void*)in_f);

    /* weights */
    char nm[64];
    #define LOADW(suffix) do { snprintf(nm,sizeof(nm),"%s_%s",prefix,#suffix); } while(0)
    LOADW(qkv_w); const float* qkv_w = ref_load(nm, &ndim, ishape);
    LOADW(qkv_b); const float* qkv_b = ref_load(nm, &ndim, ishape);
    LOADW(pe_w);  const float* pe_w  = ref_load(nm, &ndim, ishape);
    LOADW(pe_b);  const float* pe_b  = ref_load(nm, &ndim, ishape);
    LOADW(proj_w);const float* proj_w= ref_load(nm, &ndim, ishape);
    LOADW(proj_b);const float* proj_b= ref_load(nm, &ndim, ishape);

    if (!qkv_w || !qkv_b || !pe_w || !pe_b || !proj_w || !proj_b) {
        CHECK(0, "load all weights");
        fiv_release_tensor((void**)&in);
        return;
    }

    fiv_attention_node_params p;
    memset(&p, 0, sizeof(p));
    p.dim = dim; p.num_heads = num_heads; p.key_dim = key_dim;
    p.head_dim = head_dim; p.scale = scale; p.pe_groups = dim;

    void* op = fiv_attention_node_create(&p);
    if (!op) { CHECK(0, "create node"); fiv_release_tensor((void**)&in); return; }

    fiv_ret rs = fiv_attention_node_set_weights(op, qkv_w, qkv_b, pe_w, pe_b, proj_w, proj_b);
    if (rs != FIV_RET_OK) { CHECK(0, "set_weights"); fiv_attention_node_release(op); fiv_release_tensor((void**)&in); return; }

    fiv_ret ra;
    fiv_tensor_hdr* out = (fiv_tensor_hdr*)fiv_attention_node_alloc_out(op, in, NULL, &ra);
    if (!out || ra != FIV_RET_OK) { CHECK(0, "alloc_out"); fiv_attention_node_release(op); fiv_release_tensor((void**)&in); return; }

    fiv_ret rf = fiv_attention_node_forward(op, out, (fiv_tensor_hdr*)in);
    CHECK(rf == FIV_RET_OK, "forward");

    char out_name[40];
    snprintf(out_name, sizeof(out_name), "%s_out", prefix);
    float max_err;
    int rc = ref_cmp(out_name, out->data.fl, 1e-4f, &max_err);
    CHECK(rc == 0, "matches torch Attention.forward output");
    if (rc != 0) printf("        max_abs_err=%.3e\n", max_err);

    fiv_release_tensor((void**)&out);
    fiv_attention_node_release(op);
    fiv_release_tensor((void**)&in);
}

int main(void)
{
    printf("=== YOLO26 P2 attention node test (vs torch reference) ===\n");
    /* hyperparams match every Attention instance: num_heads=2, head_dim=64 ->
     * dim=128, key_dim=32, scale=1/sqrt(32). attn0/attn1 are the real (2x2,
     * near-degenerate) instances; syn_attn is the 16x16 discriminating case. */
    const float scale = 0.1767766952966369f;
    test_attention("attn0",     128, 2, 32, 64, scale);
    test_attention("attn1",     128, 2, 32, 64, scale);
    test_attention("syn_attn",  128, 2, 32, 64, scale);
    printf("=== P2 attn: pass=%d fail=%d ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
