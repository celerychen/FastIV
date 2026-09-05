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

#include "fiv_ctensor.h"
#include "fiv_nn.h"
#include "fiv_nn_op.h"
#include "fiv_prelu_node.h"
#include "fiv_concat_node.h"
#include "fiv_spatial_pad_node.h"
#include "fiv_nn_conv2d.h"
#include <stdio.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  FAIL [%s] (%s:%d)\n", msg, __FILE__, __LINE__); g_fail++; } } while (0)

static void test_prelu(void)
{
    printf("[PReLU]\n");
    ivf32 alpha[2] = { 0.5f, -0.25f };
    fiv_prelu_node_params pp = { 2, alpha };
    void* op = fiv_prelu_node_create(&pp);
    CHECK(op != NULL, "create");

    fiv_tensor3d* in = NULL;
    { size_t sh[3] = { 2, 1, 2 }; in = (fiv_tensor3d*)fiv_create_tensor3d(sh, FIV_32F1); }
    fiv_ret r = FIV_RET_OK;
    fiv_tensor_hdr* out = (fiv_tensor_hdr*)fiv_prelu_node_alloc_out(op, (void*)in, NULL, &r);
    CHECK(out != NULL && r == FIV_RET_OK, "alloc out");

    ivf32 x[4] = { 1.0f, -2.0f, -4.0f, 0.5f };  /* ch0: 1,-2 ; ch1: -4,0.5 */
    memcpy(in->data.fl, x, sizeof(x));
    CHECK(fiv_prelu_node_forward(op, out, (void*)in) == FIV_RET_OK, "forward");
    const ivf32* o = out->data.fl;
    CHECK(o[0] == 1.0f, "ch0 x>=0");
    CHECK(o[1] == -1.0f, "ch0 x<0 ->0.5*-2");
    CHECK(o[2] == 1.0f, "ch1 x<0 ->-0.25*-4");
    CHECK(o[3] == 0.5f, "ch1 x>=0");

    fiv_release_tensor((void**)&out);
    fiv_release_tensor((void**)&in);
    fiv_prelu_node_release(op);
}

static void test_concat(void)
{
    printf("[CONCAT]\n");
    fiv_concat_node_params cp = { 1, 3 };
    void* op = fiv_concat_node_create(&cp);
    CHECK(op != NULL, "create");

    fiv_tensor4d* a = NULL, * b = NULL;
    { size_t sh[4] = { 1, 2, 1, 1 }; a = (fiv_tensor4d*)fiv_create_tensor4d(sh, FIV_32F1); sh[1] = 1; b = (fiv_tensor4d*)fiv_create_tensor4d(sh, FIV_32F1); }
    a->data.fl[0] = 1.0f; a->data.fl[1] = 2.0f;
    b->data.fl[0] = 3.0f;

    fiv_ret r = FIV_RET_OK;
    fiv_tensor_hdr* out = (fiv_tensor_hdr*)fiv_concat_node_alloc_out(op, (void*)a, NULL, &r);
    CHECK(out != NULL && r == FIV_RET_OK, "alloc out");
    void* ins[2] = { a, b };
    CHECK(fiv_concat_node_inference_multi(op, out, ins, 2) == FIV_RET_OK, "inference_multi");
    const ivf32* o = out->data.fl;
    CHECK(o[0] == 1.0f && o[1] == 2.0f && o[2] == 3.0f, "c0|c1|c2");

    fiv_release_tensor((void**)&out);
    fiv_release_tensor((void**)&a);
    fiv_release_tensor((void**)&b);
    fiv_concat_node_release(op);
}

static void test_spatial_pad(void)
{
    printf("[SPATIAL_PAD]\n");
    fiv_spatial_pad_node_params sp = { 1, 1, 0, 0, 0.0f };
    void* op = fiv_spatial_pad_node_create(&sp);
    CHECK(op != NULL, "create");

    fiv_tensor4d* in = NULL;
    { size_t sh[4] = { 1, 1, 2, 2 }; in = (fiv_tensor4d*)fiv_create_tensor4d(sh, FIV_32F1); }
    ivf32 x[4] = { 1, 2, 3, 4 };
    memcpy(in->data.fl, x, sizeof(x));

    fiv_ret r = FIV_RET_OK;
    fiv_tensor_hdr* out = (fiv_tensor_hdr*)fiv_spatial_pad_node_alloc_out(op, (void*)in, NULL, &r);
    CHECK(out != NULL && r == FIV_RET_OK, "alloc out");
    fiv_tensor4d* o = (fiv_tensor4d*)out;
    CHECK(o->height == 4 && o->width == 2, "out H=4 W=2");
    CHECK(fiv_spatial_pad_node_forward(op, out, (void*)in) == FIV_RET_OK, "forward");
    const ivf32* p = o->data.fl;
    /* 4x2 plane flat row-major: [0,0, 1,2, 3,4, 0,0] */
    CHECK(p[0] == 0 && p[1] == 0, "top pad");
    CHECK(p[2] == 1 && p[3] == 2, "row1");
    CHECK(p[4] == 3 && p[5] == 4, "row2");
    CHECK(p[6] == 0 && p[7] == 0, "bottom pad");

    fiv_release_tensor((void**)&out);
    fiv_release_tensor((void**)&in);
    fiv_spatial_pad_node_release(op);
}

/* Depthwise 3x3 stride-2 with channel multiplier 2 (C_out = 2 * C_in).
   out[oc] = conv3x3_s2(in[oc/mult], w[oc]); SAME pad p0=p1=1. */
static void test_depthwise_mult(void)
{
    printf("[DEPTHWISE_MULT]\n");
    fiv_conv2d_params cp;
    memset(&cp, 0, sizeof(cp));
    cp.conv2d_method    = FIV_CONV2D_DEPTHWISE;
    cp.kernel_size_x    = 3;
    cp.kernel_size_y    = 3;
    cp.stride           = 2;
    cp.padding_method   = 0;
    cp.input_channels   = 2;
    cp.output_channels  = 4;   /* depth_mult = 2 */
    cp.bias             = 0;

    /* in: 2 channels, 4x4 */
    fiv_tensor4d* in = NULL;
    { size_t sh[4] = { 1, 2, 4, 4 }; in = (fiv_tensor4d*)fiv_create_tensor4d(sh, FIV_32F1); }
    for (int c = 0; c < 2; c++)
        for (int i = 0; i < 16; i++) in->data.fl[c * 16 + i] = (ivf32)(c + 1) + (ivf32)(i % 4);

    /* weight: (4, 1, 3, 3) */
    fiv_tensor4d* w = NULL;
    { size_t sh[4] = { 4, 1, 3, 3 }; w = (fiv_tensor4d*)fiv_create_tensor4d(sh, FIV_32F1); }
    for (int oc = 0; oc < 4; oc++)
        for (int k = 0; k < 9; k++) w->data.fl[oc * 9 + k] = (ivf32)(oc + 1);

    fiv_tensor4d* out = NULL;
    { size_t sh[4] = { 1, 4, 2, 2 }; out = (fiv_tensor4d*)fiv_create_tensor4d(sh, FIV_32F1); }

    CHECK(fiv_tensor_conv2d(out, in, w, &cp) == FIV_RET_OK, "depthwise_mult conv");

    /* oc0,oc1 read in[0]; oc2,oc3 read in[1]. Weight all ones(1..4). */
    /* in[0] 4x4: row0 1 2 3 4 / row1 1 2 3 4 / ... stride-2 at (1,1) with p0=1: */
    /* out[0,0] taps (0,0),(0,1),(1,0),(1,1) = 1,2,1,2 => sum 6 */
    const ivf32* o = out->data.fl;
    /* out[0][0,0]: weight 1 => 6 */
    CHECK(o[0] == 6.0f, "oc0 center");
    /* out[1][0,0]: same taps weight=2 => 12 */
    CHECK(o[4] == 12.0f, "oc1 center");
    /* out[2][0,0]: in[1] taps 2,3,2,3 = 10 * weight 3 = 30 */
    CHECK(o[8] == 30.0f, "oc2 center");

    fiv_release_tensor((void**)&in);
    fiv_release_tensor((void**)&w);
    fiv_release_tensor((void**)&out);
}

int main(void)
{
    test_prelu();
    test_concat();
    test_spatial_pad();
    test_depthwise_mult();
    if (g_fail) { printf("\ntest_landmark_ops: %d FAILURES\n", g_fail); return 1; }
    printf("\ntest_landmark_ops: all PASS\n");
    return 0;
}