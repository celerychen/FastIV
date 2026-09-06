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

/* P1 operator test for the YOLO26 port: verifies the three new NN nodes
 *   - FIV_NN_NODE_SILU    (y = x * sigmoid(x))
 *   - FIV_NN_NODE_MAXPOOL (general 2D max-pool, k=5 s=1 p=2 from SPPF)
 *   - FIV_NN_NODE_SLICE   (channel split along axis=1, C2f/C3k style)
 * against the torch reference tensors dumped by app/yolo26/test/gen_ref.py.
 *
 * Build: the project Makefile wires this as `test_yolo26_ops`; run from
 * build/ (the loader default YOLO26_REF_DIR = ../app/yolo26/test/ref).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fiv_ctensor.h"
#include "fiv_silu_node.h"
#include "fiv_maxpool_node.h"
#include "fiv_slice_node.h"
#include "yolo26_ref.h"
#include "fiv_common.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                         \
    do {                                                         \
        if (cond) { g_pass++; printf("  PASS  %s\n", msg); }    \
        else      { g_fail++; printf("  FAIL  %s\n", msg); }    \
    } while (0)

/* Wrap a reference buffer into a contiguous 4D tensor. */
static fiv_tensor4d* make_tensor4d(const ivf32* src, const size_t shape[4], size_t* n_elems)
{
    fiv_tensor4d* t = fiv_create_tensor4d((size_t*)shape, FIV_32F1);
    if (!t) return NULL;
    size_t n = shape[0] * shape[1] * shape[2] * shape[3];
    memcpy(((fiv_tensor_hdr*)t)->data.fl, src, n * sizeof(ivf32));
    if (n_elems) *n_elems = n;
    return t;
}

/* ---------------- SiLU ---------------- */
static void test_silu(void)
{
    printf("[SiLU] y = x * sigmoid(x)\n");
    int ndim; size_t shape[4];
    const ivf32* src = ref_load("syn_silu_src", &ndim, shape);
    if (!src) { CHECK(0, "load syn_silu_src"); return; }
    size_t n; fiv_tensor4d* in = make_tensor4d(src, shape, &n);
    fiv_free((void*)src);
    if (!in) { CHECK(0, "create input tensor"); return; }

    void* op = fiv_silu_node_create(NULL);
    fiv_ret r;
    fiv_tensor_hdr* out = (fiv_tensor_hdr*)fiv_silu_node_alloc_out(op, in, NULL, &r);
    if (!out || r != FIV_RET_OK) { CHECK(0, "alloc_out"); fiv_release_tensor((void**)&in); fiv_silu_node_release(op); return; }

    r = fiv_silu_node_forward(op, out, (fiv_tensor_hdr*)in);
    CHECK(r == FIV_RET_OK, "forward");

    ivf32 max_err;
    int rc = ref_cmp("syn_silu_out", out->data.fl, 1e-5f, &max_err);
    CHECK(rc == 0, "matches torch F.silu reference");
    if (rc != 0) printf("        max_abs_err=%.3e\n", max_err);

    fiv_release_tensor((void**)&out);
    fiv_release_tensor((void**)&in);
    fiv_silu_node_release(op);
}

/* ---------------- MaxPool (SPPF stage) ---------------- */
static void test_maxpool(void)
{
    printf("[MaxPool] k=5 s=1 p=2 (one SPPF stage)\n");
    int ndim; size_t shape[4];
    const ivf32* src = ref_load("layer09_sppf_y0", &ndim, shape);
    if (!src) { CHECK(0, "load layer09_sppf_y0"); return; }
    size_t n; fiv_tensor4d* in = make_tensor4d(src, shape, &n);
    fiv_free((void*)src);
    if (!in) { CHECK(0, "create input tensor"); return; }

    fiv_maxpool_node_params p;
    memset(&p, 0, sizeof(p));
    p.kernel_size_x = 5; p.kernel_size_y = 5; p.stride = 1;
    p.pad_top = 2; p.pad_bottom = 2; p.pad_left = 2; p.pad_right = 2;

    void* op = fiv_maxpool_node_create(&p);
    fiv_ret r;
    fiv_tensor_hdr* out = (fiv_tensor_hdr*)fiv_maxpool_node_alloc_out(op, in, NULL, &r);
    if (!out || r != FIV_RET_OK) { CHECK(0, "alloc_out"); fiv_release_tensor((void**)&in); fiv_maxpool_node_release(op); return; }

    r = fiv_maxpool_node_forward(op, out, (fiv_tensor_hdr*)in);
    CHECK(r == FIV_RET_OK, "forward");

    ivf32 max_err;
    int rc = ref_cmp("layer09_sppf_y1", out->data.fl, 1e-5f, &max_err);
    CHECK(rc == 0, "matches torch MaxPool2d(k=5,s=1,p=2) reference");
    if (rc != 0) printf("        max_abs_err=%.3e\n", max_err);

    fiv_release_tensor((void**)&out);
    fiv_release_tensor((void**)&in);
    fiv_maxpool_node_release(op);
}

/* ---------------- Slice (channel split) ---------------- */
static void test_slice(void)
{
    printf("[Slice] channel split axis=1 (C2f / C3k style)\n");
    int ndim; size_t shape[4];
    const ivf32* src = ref_load("syn_slice_src", &ndim, shape);
    if (!src) { CHECK(0, "load syn_slice_src"); return; }
    size_t n; fiv_tensor4d* in = make_tensor4d(src, shape, &n);
    fiv_free((void*)src);
    if (!in) { CHECK(0, "create input tensor"); return; }

    int c = (int)shape[1];
    int half = c / 2;

    fiv_slice_node_params p0; memset(&p0, 0, sizeof(p0));
    p0.axis = 1; p0.start = 0; p0.end = half;
    void* op0 = fiv_slice_node_create(&p0);
    fiv_ret r0; fiv_tensor_hdr* out0 = (fiv_tensor_hdr*)fiv_slice_node_alloc_out(op0, in, NULL, &r0);
    if (out0 && r0 == FIV_RET_OK) {
        r0 = fiv_slice_node_forward(op0, out0, (fiv_tensor_hdr*)in);
        ivf32 e0; int rc0 = ref_cmp("syn_slice_half0", out0->data.fl, 1e-6f, &e0);
        CHECK(rc0 == 0, "slice [0,half) matches reference");
        if (rc0 != 0) printf("        max_abs_err=%.3e\n", e0);
        fiv_release_tensor((void**)&out0);
    } else { CHECK(0, "slice [0,half) alloc_out"); }
    fiv_slice_node_release(op0);

    fiv_slice_node_params p1; memset(&p1, 0, sizeof(p1));
    p1.axis = 1; p1.start = half; p1.end = c;
    void* op1 = fiv_slice_node_create(&p1);
    fiv_ret r1; fiv_tensor_hdr* out1 = (fiv_tensor_hdr*)fiv_slice_node_alloc_out(op1, in, NULL, &r1);
    if (out1 && r1 == FIV_RET_OK) {
        r1 = fiv_slice_node_forward(op1, out1, (fiv_tensor_hdr*)in);
        ivf32 e1; int rc1 = ref_cmp("syn_slice_half1", out1->data.fl, 1e-6f, &e1);
        CHECK(rc1 == 0, "slice [half,c) matches reference");
        if (rc1 != 0) printf("        max_abs_err=%.3e\n", e1);
        fiv_release_tensor((void**)&out1);
    } else { CHECK(0, "slice [half,c) alloc_out"); }
    fiv_slice_node_release(op1);

    fiv_release_tensor((void**)&in);
}

int main(void)
{
    printf("=== YOLO26 P1 operator test (vs torch reference) ===\n");
    test_silu();
    test_maxpool();
    test_slice();
    printf("=== P1 ops: pass=%d fail=%d ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
