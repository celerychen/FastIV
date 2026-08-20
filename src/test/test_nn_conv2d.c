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

/* fiv_tensor_conv2d correctness: hand-computed fixed cases + a naive reference
   implementation compared pixel-wise over random shapes/methods/paddings, plus
   error-path checks. */

#include "fiv_nn_conv2d.h"
#include "fiv_max_2d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_pass = 0;
static int g_fail = 0;
#define CHECK(c, msg)                                                           \
    do {                                                                        \
        if (!(c)) { printf("  [FAIL] %s @%d\n", (msg), __LINE__); g_fail++; }   \
        else       { g_pass++; }                                                \
    } while (0)

static const ivf32 K_IDENT[9] = {0,0,0, 0,1,0, 0,0,0};  /* picks center pixel */
static const ivf32 K_TL[9]    = {1,0,0, 0,0,0, 0,0,0};  /* picks src[y-1][x-1] */

/* ---- naive reference (independent of the optimized kernel) ---- */

static void ref_conv_plane(ivf32* out, const ivf32* src, size_t H, size_t W,
                           const ivf32* coef, int zero_pad, int accumulate)
{
    for (size_t y = 0; y < H; y++)
        for (size_t x = 0; x < W; x++) {
            ivf32 s = 0.0f;
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    int sy = (int)y + ky - 1;
                    int sx = (int)x + kx - 1;
                    if (sy < 0 || sy >= (int)H || sx < 0 || sx >= (int)W) {
                        if (zero_pad) continue;   /* zero pad: skip */
                        if (sy < 0) sy = 0; if (sy >= (int)H) sy = (int)H - 1;
                        if (sx < 0) sx = 0; if (sx >= (int)W) sx = (int)W - 1;
                    }
                    s += src[(size_t)sy * W + (size_t)sx] * coef[(size_t)ky * 3 + (size_t)kx];
                }
            out[y * W + x] = accumulate ? out[y * W + x] + s : s;
        }
}

static void ref_conv(ivf32* out, const ivf32* in, size_t C_in, size_t C_out,
                     size_t H, size_t W, const ivf32* kern, int method, int zero_pad)
{
    size_t HW = H * W;
    if (method == FIV_CONV2D_STD) {
        for (size_t oc = 0; oc < C_out; oc++) {
            for (size_t ic = 0; ic < C_in; ic++)
                ref_conv_plane(out + oc * HW, in + ic * HW, H, W,
                               kern + (oc * C_in + ic) * 9, zero_pad, ic > 0);
        }
    } else {  /* DEPTHWISE: out[c] = conv(src[c], kern[c]) */
        for (size_t c = 0; c < C_out; c++)
            ref_conv_plane(out + c * HW, in + c * HW, H, W, kern + c * 9, zero_pad, 0);
    }
}

/* ---- helpers ---- */

static fiv_conv2d_params mk_params(int method, int pad, int cin, int cout)
{
    fiv_conv2d_params p;
    memset(&p, 0, sizeof(p));
    p.conv2d_method   = method;
    p.kernel_size_x   = 3;
    p.kernel_size_y   = 3;
    p.stride          = 1;
    p.padding_method  = pad;
    p.input_channels  = cin;
    p.output_channels = cout;
    return p;
}

static void fill_rand(ivf32* p, size_t n, unsigned* seed)
{
    for (size_t i = 0; i < n; i++) {
        *seed = *seed * 1664525u + 1013904223u;
        p[i] = (ivf32)((int)(*seed % 7u) - 3);
    }
}

static int cmp_plane(const ivf32* a, const ivf32* b, size_t n, const char* msg)
{
    for (size_t i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > 1e-4f) {
            printf("  [FAIL] %s: idx %zu got %.6f want %.6f @%d\n", msg, i, a[i], b[i], __LINE__);
            g_fail++;
            return 0;
        }
    }
    g_pass++;
    return 1;
}

/* ---- fixed hand-computed cases ---- */

static void test_fixed(void)
{
    const ivf32 A[9] = {1,2,3, 4,5,6, 7,8,9};
    const ivf32 B[9] = {9,8,7, 6,5,4, 3,2,1};
    size_t sh3[3] = {1, 3, 3};
    fiv_tensor3d* out3;

    /* STD, C_in=1, C_out=1, identity kernel: out == src (both paddings) */
    {
        fiv_tensor3d* src = fiv_create_tensor3d(sh3, FIV_32F1);
        size_t ksh[4] = {1, 1, 3, 3};
        fiv_tensor4d* k = fiv_create_tensor4d(ksh, FIV_32F1);
        out3 = fiv_create_tensor3d(sh3, FIV_32F1);
        memcpy(src->data.fl, A, sizeof(A));
        memcpy(k->data.fl, K_IDENT, sizeof(K_IDENT));
        fiv_conv2d_params p = mk_params(FIV_CONV2D_STD, 1, 1, 1);
        CHECK(fiv_tensor_conv2d(out3, src, k, &p) == FIV_RET_OK, "std ident edge ok");
        CHECK(cmp_plane(out3->data.fl, A, 9, "std identity edge == src"), "");
        p.padding_method = 0;
        CHECK(fiv_tensor_conv2d(out3, src, k, &p) == FIV_RET_OK, "std ident zero ok");
        CHECK(cmp_plane(out3->data.fl, A, 9, "std identity zero == src"), "");
        fiv_release_tensor((void**)&k);
        fiv_release_tensor((void**)&src);
        fiv_release_tensor((void**)&out3);
    }

    /* STD, K_TL kernel: out[y][x] = src[clamp(y-1)][clamp(x-1)] (edge) or 0 (zero) */
    {
        fiv_tensor3d* src = fiv_create_tensor3d(sh3, FIV_32F1);
        size_t ksh[4] = {1, 1, 3, 3};
        fiv_tensor4d* k = fiv_create_tensor4d(ksh, FIV_32F1);
        out3 = fiv_create_tensor3d(sh3, FIV_32F1);
        memcpy(src->data.fl, A, sizeof(A));
        memcpy(k->data.fl, K_TL, sizeof(K_TL));
        fiv_conv2d_params p = mk_params(FIV_CONV2D_STD, 1, 1, 1);
        CHECK(fiv_tensor_conv2d(out3, src, k, &p) == FIV_RET_OK, "std tl edge ok");
        const ivf32 exp_e[9] = {1,1,2, 1,1,2, 4,4,5};
        CHECK(cmp_plane(out3->data.fl, exp_e, 9, "std K_TL edge"), "");
        p.padding_method = 0;
        CHECK(fiv_tensor_conv2d(out3, src, k, &p) == FIV_RET_OK, "std tl zero ok");
        const ivf32 exp_z[9] = {0,0,0, 0,1,2, 0,4,5};
        CHECK(cmp_plane(out3->data.fl, exp_z, 9, "std K_TL zero"), "");
        fiv_release_tensor((void**)&k);
        fiv_release_tensor((void**)&src);
        fiv_release_tensor((void**)&out3);
    }

    /* STD, C_in=2 -> C_out=1: accumulate across channels, kernel 2*identity */
    {
        size_t sh3c[3] = {2, 3, 3};
        fiv_tensor3d* src = fiv_create_tensor3d(sh3c, FIV_32F1);
        size_t ksh[4] = {1, 2, 3, 3};
        fiv_tensor4d* k = fiv_create_tensor4d(ksh, FIV_32F1);
        out3 = fiv_create_tensor3d(sh3, FIV_32F1);
        memcpy(src->data.fl, A, sizeof(A));
        memcpy(src->data.fl + 9, B, sizeof(B));
        ivf32 k2i[9]; for (int i = 0; i < 9; i++) k2i[i] = 2.0f * K_IDENT[i];
        memcpy(k->data.fl, k2i, 9 * sizeof(ivf32));
        memcpy(k->data.fl + 9, k2i, 9 * sizeof(ivf32));
        fiv_conv2d_params p = mk_params(FIV_CONV2D_STD, 1, 2, 1);
        CHECK(fiv_tensor_conv2d(out3, src, k, &p) == FIV_RET_OK, "std 2ch acc ok");
        const ivf32 exp[9] = {20,20,20, 20,20,20, 20,20,20};
        CHECK(cmp_plane(out3->data.fl, exp, 9, "std 2ch accumulate == 2*(A+B)"), "");
        fiv_release_tensor((void**)&k);
        fiv_release_tensor((void**)&src);
        fiv_release_tensor((void**)&out3);
    }

    /* STD, C_in=1 -> C_out=2: two output channels with different kernels */
    {
        fiv_tensor3d* src = fiv_create_tensor3d(sh3, FIV_32F1);
        size_t ksh[4] = {2, 1, 3, 3};
        fiv_tensor4d* k = fiv_create_tensor4d(ksh, FIV_32F1);
        size_t osh[3] = {2, 3, 3};
        fiv_tensor3d* out = fiv_create_tensor3d(osh, FIV_32F1);
        memcpy(src->data.fl, A, sizeof(A));
        memcpy(k->data.fl, K_IDENT, sizeof(K_IDENT));
        memcpy(k->data.fl + 9, K_TL, sizeof(K_TL));
        fiv_conv2d_params p = mk_params(FIV_CONV2D_STD, 1, 1, 2);
        CHECK(fiv_tensor_conv2d(out, src, k, &p) == FIV_RET_OK, "std 2out ok");
        CHECK(cmp_plane(out->data.fl, A, 9, "std out0 == src"), "");
        const ivf32 exp1[9] = {1,1,2, 1,1,2, 4,4,5};
        CHECK(cmp_plane(out->data.fl + 9, exp1, 9, "std out1 == K_TL edge"), "");
        fiv_release_tensor((void**)&k);
        fiv_release_tensor((void**)&src);
        fiv_release_tensor((void**)&out);
    }

    /* 4D batch=2, identity: each batch slice independent */
    {
        size_t sh4[4] = {2, 1, 3, 3};
        fiv_tensor4d* src = fiv_create_tensor4d(sh4, FIV_32F1);
        size_t ksh[4] = {1, 1, 3, 3};
        fiv_tensor4d* k = fiv_create_tensor4d(ksh, FIV_32F1);
        fiv_tensor4d* out = fiv_create_tensor4d(sh4, FIV_32F1);
        memcpy(src->data.fl, A, sizeof(A));
        for (int i = 0; i < 9; i++) src->data.fl[9 + i] = (ivf32)(10 + i);
        memcpy(k->data.fl, K_IDENT, sizeof(K_IDENT));
        fiv_conv2d_params p = mk_params(FIV_CONV2D_STD, 1, 1, 1);
        CHECK(fiv_tensor_conv2d(out, src, k, &p) == FIV_RET_OK, "batch2 ok");
        CHECK(cmp_plane(out->data.fl, A, 9, "batch0 == A"), "");
        const ivf32 expb[9] = {10,11,12, 13,14,15, 16,17,18};
        CHECK(cmp_plane(out->data.fl + 9, expb, 9, "batch1 == 10..18"), "");
        fiv_release_tensor((void**)&k);
        fiv_release_tensor((void**)&src);
        fiv_release_tensor((void**)&out);
    }

    /* DEPTHWISE, C_in=C_out=2: per-channel kernels, no cross-channel mixing */
    {
        size_t sh3c[3] = {2, 3, 3};
        fiv_tensor3d* src = fiv_create_tensor3d(sh3c, FIV_32F1);
        size_t ksh[4] = {2, 1, 3, 3};
        fiv_tensor4d* k = fiv_create_tensor4d(ksh, FIV_32F1);
        fiv_tensor3d* out = fiv_create_tensor3d(sh3c, FIV_32F1);
        memcpy(src->data.fl, A, sizeof(A));
        memcpy(src->data.fl + 9, B, sizeof(B));
        memcpy(k->data.fl, K_IDENT, sizeof(K_IDENT));
        memcpy(k->data.fl + 9, K_TL, sizeof(K_TL));
        fiv_conv2d_params p = mk_params(FIV_CONV2D_DEPTHWISE, 1, 2, 2);
        CHECK(fiv_tensor_conv2d(out, src, k, &p) == FIV_RET_OK, "dw 2ch ok");
        CHECK(cmp_plane(out->data.fl, A, 9, "dw ch0 == A"), "");
        /* ch1 uses B with K_TL: out[y][x] = B[clamp(y-1)][clamp(x-1)] */
        const ivf32 expd[9] = {9,9,8, 9,9,8, 6,6,5};
        CHECK(cmp_plane(out->data.fl + 9, expd, 9, "dw ch1 == B K_TL edge"), "");
        /* zero pad: border becomes 0 */
        p.padding_method = 0;
        CHECK(fiv_tensor_conv2d(out, src, k, &p) == FIV_RET_OK, "dw zero ok");
        const ivf32 expdz[9] = {0,0,0, 0,9,8, 0,6,5};
        CHECK(cmp_plane(out->data.fl + 9, expdz, 9, "dw ch1 zero pad"), "");
        fiv_release_tensor((void**)&k);
        fiv_release_tensor((void**)&src);
        fiv_release_tensor((void**)&out);
    }

    /* 5D {1,1,1,3,3} identity: 5D shape extraction path */
    {
        size_t sh5[5] = {1, 1, 1, 3, 3};
        fiv_tensor5d* src = fiv_create_tensor5d(sh5, FIV_32F1);
        size_t ksh[4] = {1, 1, 3, 3};
        fiv_tensor4d* k = fiv_create_tensor4d(ksh, FIV_32F1);
        fiv_tensor5d* out = fiv_create_tensor5d(sh5, FIV_32F1);
        memcpy(src->data.fl, A, sizeof(A));
        memcpy(k->data.fl, K_IDENT, sizeof(K_IDENT));
        fiv_conv2d_params p = mk_params(FIV_CONV2D_STD, 1, 1, 1);
        CHECK(fiv_tensor_conv2d(out, src, k, &p) == FIV_RET_OK, "5d ok");
        CHECK(cmp_plane(out->data.fl, A, 9, "5d identity == src"), "");
        fiv_release_tensor((void**)&k);
        fiv_release_tensor((void**)&src);
        fiv_release_tensor((void**)&out);
    }
}

/* ---- randomized reference comparison (hits NEON/AVX2 interior, tails, accumulate) ---- */

static void test_random(void)
{
    static const size_t hws[][2] = {
        {3,3}, {4,10}, {9,9}, {10,4}, {5,7}, {11,12}, {12,11}
    };
    unsigned seed = 0x12345678u;
    int trials = 0;

    for (size_t t = 0; t < 60; t++) {
        const size_t* hw = hws[t % (sizeof(hws) / sizeof(hws[0]))];
        size_t H = hw[0], W = hw[1];
        int method  = (t & 1) ? FIV_CONV2D_DEPTHWISE : FIV_CONV2D_STD;
        int pad     = (t & 2) ? 1 : 0;
        int batch   = (t & 4) ? 2 : 1;
        size_t C_in  = 1 + (size_t)(t % 3);
        size_t C_out = (method == FIV_CONV2D_DEPTHWISE) ? C_in : 1 + (size_t)((t / 3) % 3);

        size_t sh4[4] = { (size_t)batch, C_in, H, W };
        fiv_tensor4d* src = fiv_create_tensor4d(sh4, FIV_32F1);
        size_t ksh[4] = { C_out, (method == FIV_CONV2D_DEPTHWISE) ? 1 : C_in, 3, 3 };
        fiv_tensor4d* k = fiv_create_tensor4d(ksh, FIV_32F1);
        size_t osh4[4] = { (size_t)batch, C_out, H, W };
        fiv_tensor4d* out = fiv_create_tensor4d(osh4, FIV_32F1);
        size_t nin  = (size_t)batch * C_in * H * W;
        size_t nout = (size_t)batch * C_out * H * W;
        size_t nk   = C_out * ksh[1] * 9;
        fill_rand(src->data.fl, nin, &seed);
        fill_rand(k->data.fl, nk, &seed);

        fiv_conv2d_params p = mk_params(method, pad, (int)C_in, (int)C_out);
        fiv_ret r = fiv_tensor_conv2d(out, src, k, &p);
        CHECK(r == FIV_RET_OK, "random conv ok");

        /* reference over the same batch */
        ivf32* ref = (ivf32*)malloc(sizeof(ivf32) * nout);
        for (int b = 0; b < batch; b++)
            ref_conv(ref + (size_t)b * C_out * H * W, src->data.fl + (size_t)b * C_in * H * W,
                     C_in, C_out, H, W, k->data.fl, method, pad == 0);
        CHECK(cmp_plane(out->data.fl, ref, nout, "random vs reference"), "");
        free(ref);
        fiv_release_tensor((void**)&k);
        fiv_release_tensor((void**)&src);
        fiv_release_tensor((void**)&out);
        trials++;
    }
    printf("  random trials: %d\n", trials);
}

/* ---- error paths ---- */

static void test_errors(void)
{
    size_t sh3[3] = {1, 3, 3};
    size_t ksh[4] = {1, 1, 3, 3};
    fiv_tensor3d* src = fiv_create_tensor3d(sh3, FIV_32F1);
    fiv_tensor4d* k = fiv_create_tensor4d(ksh, FIV_32F1);
    fiv_tensor3d* out = fiv_create_tensor3d(sh3, FIV_32F1);
    fiv_conv2d_params p = mk_params(FIV_CONV2D_STD, 1, 1, 1);
    memset(src->data.fl, 0, sizeof(ivf32) * 9);
    memset(k->data.fl, 0, sizeof(ivf32) * 9);
    memset(out->data.fl, 0, sizeof(ivf32) * 9);

    CHECK(fiv_tensor_conv2d(NULL, src, k, &p) == FIV_RET_ERR_PARA, "null dst");
    CHECK(fiv_tensor_conv2d(out, NULL, k, &p) == FIV_RET_ERR_PARA, "null src");
    CHECK(fiv_tensor_conv2d(out, src, NULL, &p) == FIV_RET_ERR_PARA, "null kernel");
    CHECK(fiv_tensor_conv2d(out, src, k, NULL) == FIV_RET_ERR_PARA, "null params");

    /* src must be 3D..5D */
    {
        size_t sh2[2] = {3, 3};
        fiv_mat* s2 = fiv_create_tensor2d(sh2, FIV_32F1);
        CHECK(fiv_tensor_conv2d(out, s2, k, &p) == FIV_RET_ERR_PARA, "src 2D rejected");
        fiv_release_tensor((void**)&s2);
    }

    /* kernel must be 4D */
    {
        fiv_tensor3d* k3 = fiv_create_tensor3d(sh3, FIV_32F1);
        CHECK(fiv_tensor_conv2d(out, src, k3, &p) == FIV_RET_ERR_PARA, "kernel 3D rejected");
        fiv_release_tensor((void**)&k3);
    }

    /* dst dim must match src dim */
    {
        size_t sh4[4] = {1, 1, 3, 3};
        fiv_tensor4d* o4 = fiv_create_tensor4d(sh4, FIV_32F1);
        CHECK(fiv_tensor_conv2d(o4, src, k, &p) == FIV_RET_ERR_PARA, "dst dim mismatch");
        fiv_release_tensor((void**)&o4);
    }

    /* unsupported dtype */
    {
        fiv_tensor3d* si = fiv_create_tensor3d(sh3, FIV_32S1);
        CHECK(fiv_tensor_conv2d(out, si, k, &p) == FIV_RET_ERR_NOT_SUPPORT, "int dtype rejected");
        fiv_release_tensor((void**)&si);
    }

    /* non-contiguous src (strided 3D view) */
    {
        size_t shb[3] = {2, 2, 2};
        fiv_tensor3d* base = fiv_create_tensor3d(shb, FIV_32F1);
        fiv_tensor3d view;
        size_t off[3] = {0, 1, 0};
        size_t sz[3]  = {2, 1, 2};
        CHECK(fiv_tensor_view(&view, base, off, sz) == FIV_RET_OK, "view ok");
        CHECK(view.data_continue == 0, "view non-contiguous");
        CHECK(fiv_tensor_conv2d(out, &view, k, &p) == FIV_RET_ERR_PARA, "strided src rejected");
        fiv_release_tensor((void**)&base);
    }

    /* params constraints */
    {
        fiv_conv2d_params q = p;
        q.conv2d_method = FIV_CONV2D_POINTWISE;
        CHECK(fiv_tensor_conv2d(out, src, k, &q) == FIV_RET_ERR_NOT_SUPPORT, "pointwise not implemented");
        q = p; q.conv2d_method = FIV_CONV2D_SEPARABLE;
        CHECK(fiv_tensor_conv2d(out, src, k, &q) == FIV_RET_ERR_NOT_SUPPORT, "separable not implemented");
        q = p; q.conv2d_method = 99;
        CHECK(fiv_tensor_conv2d(out, src, k, &q) == FIV_RET_ERR_NOT_SUPPORT, "bad method value");
        q = p; q.kernel_size_x = 5;
        CHECK(fiv_tensor_conv2d(out, src, k, &q) == FIV_RET_ERR_NOT_SUPPORT, "kernel size != 3");
        q = p; q.stride = 2;
        CHECK(fiv_tensor_conv2d(out, src, k, &q) == FIV_RET_ERR_NOT_SUPPORT, "stride != 1");
        q = p; q.padding_method = 7;
        CHECK(fiv_tensor_conv2d(out, src, k, &q) == FIV_RET_ERR_PARA, "bad padding method");
        q = p; q.input_channels = 2;
        CHECK(fiv_tensor_conv2d(out, src, k, &q) == FIV_RET_ERR_PARA, "input_channels mismatch");
        q = p; q.output_channels = 2;
        CHECK(fiv_tensor_conv2d(out, src, k, &q) == FIV_RET_ERR_PARA, "output_channels mismatch");
    }

    /* kernel shape mismatches */
    {
        size_t kbad1[4] = {1, 2, 3, 3};  /* kCin != C_in for STD */
        fiv_tensor4d* kb = fiv_create_tensor4d(kbad1, FIV_32F1);
        CHECK(fiv_tensor_conv2d(out, src, kb, &p) == FIV_RET_ERR_PARA, "kernel C_in mismatch");
        fiv_release_tensor((void**)&kb);

        size_t kbad2[4] = {1, 1, 5, 3};  /* kernel spatial != 3x3 */
        fiv_tensor4d* kc = fiv_create_tensor4d(kbad2, FIV_32F1);
        CHECK(fiv_tensor_conv2d(out, src, kc, &p) == FIV_RET_ERR_PARA, "kernel not 3x3");
        fiv_release_tensor((void**)&kc);

        /* DEPTHWISE: kCin must be 1 and kCout == C_in */
        fiv_conv2d_params dp = mk_params(FIV_CONV2D_DEPTHWISE, 1, 1, 1);
        size_t kbad3[4] = {2, 1, 3, 3};  /* kCout=2 != C_in=1 */
        fiv_tensor4d* kd = fiv_create_tensor4d(kbad3, FIV_32F1);
        CHECK(fiv_tensor_conv2d(out, src, kd, &dp) == FIV_RET_ERR_PARA, "depthwise kCout != C_in");
        fiv_release_tensor((void**)&kd);
    }

    /* dst wrong channel count / spatial */
    {
        size_t obad1[3] = {2, 3, 3};
        fiv_tensor3d* o1 = fiv_create_tensor3d(obad1, FIV_32F1);
        CHECK(fiv_tensor_conv2d(o1, src, k, &p) == FIV_RET_ERR_PARA, "dst channel mismatch");
        fiv_release_tensor((void**)&o1);
        size_t obad2[3] = {1, 2, 3};
        fiv_tensor3d* o2 = fiv_create_tensor3d(obad2, FIV_32F1);
        CHECK(fiv_tensor_conv2d(o2, src, k, &p) == FIV_RET_ERR_PARA, "dst spatial mismatch");
        fiv_release_tensor((void**)&o2);
    }

    fiv_release_tensor((void**)&k);
    fiv_release_tensor((void**)&src);
    fiv_release_tensor((void**)&out);
}

/* ---- CONV2D_STD node: forward vs reference, backward vs numeric gradient ---- */

static float conv_node_fwd_sum(fiv_conv2d_node* n, fiv_tensor4d* x, fiv_tensor4d* out)
{
    if (n->base.forward_fn(n, out, x) != FIV_RET_OK) return 0.0f;
    float s = 0.0f;
    for (size_t i = 0; i < out->total_bytes / sizeof(ivf32); i++) s += out->data.fl[i];
    return s;
}

static void test_conv_node(void)
{
    fiv_conv2d_params p = mk_params(FIV_CONV2D_STD, 0, 1, 2);
    p.bias = 1;

    CHECK(fiv_conv2d_node_create(NULL) == NULL, "conv node null params");
    {
        fiv_conv2d_params q = p;
        q.conv2d_method = FIV_CONV2D_DEPTHWISE;
        CHECK(fiv_conv2d_node_create(&q) == NULL, "conv node method rejected");
        q = p; q.kernel_size_x = 5;
        CHECK(fiv_conv2d_node_create(&q) == NULL, "conv node kernel size rejected");
        q = p; q.stride = 2;
        CHECK(fiv_conv2d_node_create(&q) == NULL, "conv node stride rejected");
        q = p; q.padding_method = 7;
        CHECK(fiv_conv2d_node_create(&q) == NULL, "conv node padding rejected");
        q = p; q.bias = 5;
        CHECK(fiv_conv2d_node_create(&q) == NULL, "conv node bias rejected");
        q = p; q.input_channels = 0;
        CHECK(fiv_conv2d_node_create(&q) == NULL, "conv node channels rejected");
    }

    void* op = fiv_conv2d_node_create(&p);
    CHECK(op != NULL, "conv node create ok");
    fiv_conv2d_node* n = (fiv_conv2d_node*)op;
    CHECK(n->base.create_fn && n->base.release_fn && n->base.forward_fn
          && n->base.backward_fn && n->base.inference_fn && n->base.alloc_out_fn,
          "conv node vtable complete");

    size_t xsh[4] = {1, 1, 3, 3};
    fiv_tensor4d* x = fiv_create_tensor4d(xsh, FIV_32F1);
    const ivf32 xd[9] = {1,2,3, 4,5,6, 7,8,9};
    memcpy(x->data.fl, xd, sizeof(xd));

    fiv_ret r;
    fiv_tensor4d* out = (fiv_tensor4d*)n->base.alloc_out_fn(op, x, NULL, &r);
    CHECK(r == FIV_RET_OK && out != NULL, "conv node alloc_out");
    CHECK(out->channels == 2 && out->height == 3 && out->width == 3, "conv node out shape");
    void* out2 = n->base.alloc_out_fn(op, x, out, &r);
    CHECK(r == FIV_RET_OK && out2 == (void*)out, "conv node alloc_out reuse");
    CHECK(n->base.forward_fn(op, out, x) == FIV_RET_OK, "conv node forward");

    /* reference: fiv_tensor_conv2d + per-channel bias */
    fiv_tensor4d* ref = fiv_create_tensor4d(out->shapes, FIV_32F1);
    CHECK(fiv_tensor_conv2d(ref, x, n->weight, &p) == FIV_RET_OK, "conv node ref conv");
    {
        ivf32* rp = ref->data.fl;
        const ivf32* b = n->bias->data.fl;
        for (int oc = 0; oc < 2; oc++)
            for (int k = 0; k < 9; k++) rp[oc * 9 + k] += b[oc];
    }
    CHECK(cmp_plane(out->data.fl, ref->data.fl, 18, "conv node forward == ref+bias"), "");
    fiv_release_tensor((void**)&ref);

    /* ---- numeric gradient of L = sum(out) vs backward ---- */
    size_t nw = (size_t)p.output_channels * p.input_channels * 9;
    const float eps = 1e-3f;
    ivf32* W = n->weight->data.fl;
    ivf32* Wsave = (ivf32*)malloc(nw * sizeof(ivf32));
    memcpy(Wsave, W, nw * sizeof(ivf32));
    memset(n->grad_weight->data.fl, 0, n->grad_weight->total_bytes);
    if (n->grad_bias) memset(n->grad_bias->data.fl, 0, n->grad_bias->total_bytes);

    fiv_tensor4d* go = fiv_create_tensor4d(out->shapes, FIV_32F1);
    for (size_t i = 0; i < go->total_bytes / sizeof(ivf32); i++) go->data.fl[i] = 1.0f;
    fiv_tensor4d* gi = fiv_create_tensor4d(xsh, FIV_32F1);
    memset(gi->data.ptr, 0, gi->total_bytes);

    CHECK(n->base.backward_fn(n, gi, go, x) == FIV_RET_OK, "conv node backward");

    int dw_ok = 1;
    for (size_t idx = 0; idx < nw; idx++) {
        memcpy(W, Wsave, nw * sizeof(ivf32));
        W[idx] += eps; float lp = conv_node_fwd_sum(n, x, out);
        memcpy(W, Wsave, nw * sizeof(ivf32));
        W[idx] -= eps; float lm = conv_node_fwd_sum(n, x, out);
        memcpy(W, Wsave, nw * sizeof(ivf32));
        float num = (lp - lm) / (2.0f * eps);
        if (fabsf(num - n->grad_weight->data.fl[idx]) > 1e-3f) {
            printf("  [FAIL] dW[%zu]: backward %.6f numeric %.6f @%d\n", idx,
                   n->grad_weight->data.fl[idx], num, __LINE__);
            dw_ok = 0;
        }
    }
    g_pass += dw_ok;
    if (!dw_ok) g_fail++;

    int din_ok = 1;
    ivf32* X = x->data.fl;
    ivf32* Xsave = (ivf32*)malloc(9 * sizeof(ivf32));
    memcpy(Xsave, X, 9 * sizeof(ivf32));
    for (size_t idx = 0; idx < 9; idx++) {
        memcpy(W, Wsave, nw * sizeof(ivf32));
        memcpy(X, Xsave, 9 * sizeof(ivf32));
        X[idx] += eps; float lp = conv_node_fwd_sum(n, x, out);
        memcpy(X, Xsave, 9 * sizeof(ivf32));
        X[idx] -= eps; float lm = conv_node_fwd_sum(n, x, out);
        memcpy(X, Xsave, 9 * sizeof(ivf32));
        float num = (lp - lm) / (2.0f * eps);
        if (fabsf(num - gi->data.fl[idx]) > 1e-3f) {
            printf("  [FAIL] dIn[%zu]: backward %.6f numeric %.6f @%d\n", idx,
                   gi->data.fl[idx], num, __LINE__);
            din_ok = 0;
        }
    }
    g_pass += din_ok;
    if (!din_ok) g_fail++;

    int db_ok = 1;
    ivf32* B = n->bias->data.fl;
    ivf32* Bsave = (ivf32*)malloc(2 * sizeof(ivf32));
    memcpy(Bsave, B, 2 * sizeof(ivf32));
    for (size_t idx = 0; idx < 2; idx++) {
        memcpy(B, Bsave, 2 * sizeof(ivf32));
        B[idx] += eps; float lp = conv_node_fwd_sum(n, x, out);
        memcpy(B, Bsave, 2 * sizeof(ivf32));
        B[idx] -= eps; float lm = conv_node_fwd_sum(n, x, out);
        memcpy(B, Bsave, 2 * sizeof(ivf32));
        float num = (lp - lm) / (2.0f * eps);
        if (fabsf(num - n->grad_bias->data.fl[idx]) > 1e-3f) {
            printf("  [FAIL] db[%zu]: backward %.6f numeric %.6f (lp=%.6f lm=%.6f b0=%.6f b1=%.6f) @%d\n", idx,
                   n->grad_bias->data.fl[idx], num, lp, lm, B[0], B[1], __LINE__);
            db_ok = 0;
        }
    }
    g_pass += db_ok;
    if (!db_ok) g_fail++;

    free(Wsave);
    free(Xsave);
    free(Bsave);
    fiv_release_tensor((void**)&go);
    fiv_release_tensor((void**)&gi);
    fiv_release_tensor((void**)&x);
    fiv_release_tensor((void**)&out);
    n->base.release_fn(n);
}

/* ---- MAX2D inference: naive reference vs the SIMD fiv_max_2d_node ---- */

static void ref_max2d_plane(ivf32* out, const ivf32* src, size_t H, size_t W)
{
    size_t OH = H / 2;
    size_t OW = W / 2;
    for (size_t oy = 0; oy < OH; oy++)
        for (size_t ox = 0; ox < OW; ox++) {
            ivf32 m = src[(2 * oy) * W + 2 * ox];
            ivf32 v;

            v = src[(2 * oy) * W + 2 * ox + 1];
            if (v > m) m = v;
            v = src[(2 * oy + 1) * W + 2 * ox];
            if (v > m) m = v;
            v = src[(2 * oy + 1) * W + 2 * ox + 1];
            if (v > m) m = v;
            out[oy * OW + ox] = m;
        }
}

static void test_max2d_infer(void)
{
    unsigned seed = 0x20260820u;

    /* random 4D (B,C,H,W): odd/even H,W mixed, W>=8 hits the SIMD main loop */
    for (int t = 0; t < 40; t++) {
        size_t B = 1 + (size_t)(t % 2);
        size_t C = 1 + (size_t)(t % 3);
        size_t H = 2 + (size_t)((t * 5 + 3) % 32);
        size_t W = 2 + (size_t)((t * 7 + 5) % 32);
        size_t sh[4] = { B, C, H, W };
        fiv_tensor4d* in = fiv_create_tensor4d(sh, FIV_32F1);
        CHECK(in != NULL, "max2d alloc in");
        fill_rand(in->data.fl, B * C * H * W, &seed);

        fiv_max_2d_node* n = (fiv_max_2d_node*)fiv_max_2d_node_create(NULL);
        CHECK(n != NULL, "max2d node create");
        fiv_ret r;
        fiv_tensor_hdr* out = (fiv_tensor_hdr*)fiv_max_2d_node_alloc_out(n, in, NULL, &r);
        CHECK(r == FIV_RET_OK && out != NULL, "max2d alloc_out ok");
        CHECK(fiv_max_2d_node_inference(n, out, in) == FIV_RET_OK, "max2d infer ok");

        size_t OH = H / 2;
        size_t OW = W / 2;
        int ok = 1;
        for (size_t c = 0; c < B * C; c++) {
            ivf32 ref[(33 / 2) * (33 / 2)];   /* max OH * OW over the random range */
            ref_max2d_plane(ref, in->data.fl + c * H * W, H, W);
            for (size_t i = 0; i < OH * OW; i++)
                if (fabsf(out->data.fl[c * OH * OW + i] - ref[i]) > 1e-4f) {
                    printf("  [FAIL] max2d infer t=%d ch=%zu idx=%zu got %.6f want %.6f (H=%zu W=%zu) @%d\n",
                           t, c, i, out->data.fl[c * OH * OW + i], ref[i], H, W, __LINE__);
                    ok = 0;
                    break;
                }
            if (!ok) break;
        }
        CHECK(ok, "max2d infer matches naive reference");

        fiv_release_tensor((void**)&out);
        fiv_release_tensor((void**)&in);
        n->base.release_fn(n);
    }

    /* 3D input path */
    {
        size_t sh3[3] = { 2, 9, 7 };
        fiv_tensor3d* in3 = fiv_create_tensor3d(sh3, FIV_32F1);
        fiv_max_2d_node* n = (fiv_max_2d_node*)fiv_max_2d_node_create(NULL);
        fill_rand(in3->data.fl, 2 * 9 * 7, &seed);
        fiv_ret r;
        fiv_tensor_hdr* out3 = (fiv_tensor_hdr*)fiv_max_2d_node_alloc_out(n, in3, NULL, &r);
        CHECK(r == FIV_RET_OK && out3 != NULL, "max2d 3D alloc_out ok");
        CHECK(fiv_max_2d_node_inference(n, out3, in3) == FIV_RET_OK, "max2d 3D infer ok");
        CHECK(((fiv_tensor3d*)out3)->height == 4 && ((fiv_tensor3d*)out3)->width == 3,
              "max2d 3D output shape 4x3");
        int ok3 = 1;
        for (size_t c = 0; c < 2; c++) {
            ivf32 ref[9 * 7 / 4];
            ref_max2d_plane(ref, in3->data.fl + c * 9 * 7, 9, 7);
            for (size_t i = 0; i < 4 * 3; i++)
                if (fabsf(out3->data.fl[c * 12 + i] - ref[i]) > 1e-4f) {
                    printf("  [FAIL] max2d 3D ch=%zu idx=%zu got %.6f want %.6f @%d\n",
                           c, i, out3->data.fl[c * 12 + i], ref[i], __LINE__);
                    ok3 = 0;
                    break;
                }
            if (!ok3) break;
        }
        CHECK(ok3, "max2d 3D matches naive reference");
        fiv_release_tensor((void**)&out3);
        fiv_release_tensor((void**)&in3);
        n->base.release_fn(n);
    }

    /* error paths */
    {
        fiv_max_2d_node* n = (fiv_max_2d_node*)fiv_max_2d_node_create(NULL);
        fiv_ret r;

        fiv_tensor1d* v1 = fiv_create_tensor1d(8, FIV_32F1);
        CHECK(fiv_max_2d_node_alloc_out(n, v1, NULL, &r) == NULL && r == FIV_RET_ERR_PARA,
              "max2d rejects 1D input");
        fiv_release_tensor1d(&v1);

        size_t sh8[4] = { 1, 1, 4, 4 };
        fiv_tensor4d* u8 = fiv_create_tensor4d(sh8, FIV_8U1);
        CHECK(fiv_max_2d_node_alloc_out(n, u8, NULL, &r) == NULL && r == FIV_RET_ERR_PARA,
              "max2d rejects non-32F input");
        fiv_release_tensor4d(&u8);

        size_t sh1[4] = { 1, 1, 1, 4 };   /* H < 2: nothing to downsample */
        fiv_tensor4d* small = fiv_create_tensor4d(sh1, FIV_32F1);
        CHECK(fiv_max_2d_node_alloc_out(n, small, NULL, &r) == NULL && r == FIV_RET_ERR_PARA,
              "max2d rejects H<2 input");
        fiv_release_tensor4d(&small);

        n->base.release_fn(n);
    }
}

int main(void)
{
    test_fixed();
    test_random();
    test_errors();
    test_conv_node();
    test_max2d_infer();
    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
