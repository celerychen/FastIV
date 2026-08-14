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

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>

typedef float ivf32;

void conv2d_v0(ivf32* dst, int Wd, int Hd, int sd,
               ivf32* src, int Ws, int Hs, int ss, ivf32 coef[9]);
void conv2d_v2(ivf32* dst, int Wd, int Hd, int sd,
               ivf32* src, int Ws, int Hs, int ss, ivf32 coef[9]);

static int g_fail = 0;

/* 对一组参数：v2(NEON) 与 v0 逐元素容差一致；并检查 dst 行尾 padding 未被写(越界检测) */
static void run(const char* name, int Ws, int Hs, int Wd, int Hd,
                int ss, int sd, ivf32 coef[9], unsigned seed) {
    size_t src_n = (size_t)ss * Hs;
    size_t dst_n = (size_t)sd * Hd;
    ivf32* src = (ivf32*)malloc(src_n * sizeof(ivf32));
    ivf32* d0  = (ivf32*)malloc(dst_n * sizeof(ivf32));
    ivf32* d2  = (ivf32*)malloc(dst_n * sizeof(ivf32));
    srand(seed);
    for (size_t k = 0; k < src_n; k++) src[k] = ((ivf32)rand()/RAND_MAX)*2.0f - 1.0f;
    for (size_t k = 0; k < dst_n; k++) { d0[k] = -999.0f; d2[k] = -999.0f; }

    conv2d_v0(d0, Wd, Hd, sd, src, Ws, Hs, ss, coef);
    conv2d_v2(d2, Wd, Hd, sd, src, Ws, Hs, ss, coef);

    int mism = 0, overrun = 0;
    for (int j = 0; j < Hd; j++) {
        for (int i = 0; i < Wd; i++) {
            ivf32 a = d0[(size_t)j*sd+i], b = d2[(size_t)j*sd+i];
            /* 容差比较：NEON 内核用 vfmaq(乘累加融合)，与 v0 分支循环相差最多 ~1 ULP */
            if (fabsf(a - b) > 1e-4f * (fabsf(a) + 1.0f)) mism++;
        }
        for (int i = Wd; i < sd; i++)          /* 行尾 padding 槽，检测 SIMD 内核是否越界写 */
            if (d2[(size_t)j*sd+i] != -999.0f) overrun++;
    }
    if (mism)    { printf("  FAIL %s: %d mismatch vs v0\n", name, mism); g_fail++; }
    if (overrun) { printf("  FAIL %s: %d stride overrun\n", name, overrun); g_fail++; }
    printf("%-28s Ws=%d Hs=%d Wd=%d Hd=%d ss=%d sd=%d : %s\n",
           name, Ws, Hs, Wd, Hd, ss, sd, (mism||overrun) ? "FAILED" : "PASS");
    free(src); free(d0); free(d2);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== conv2d_v2 (NEON) vs conv2d_v0 ===\n");

    ivf32 sobel[9] = {1,0,-1, 2,0,-2, 1,0,-1};
    ivf32 box[9]   = {1,1,1, 1,1,1, 1,1,1};
    ivf32 rnd[9];  srand(7); for (int k=0;k<9;k++) rnd[k]=((ivf32)rand()/RAND_MAX)*2-1;

    /* 常规 same 尺寸，含 1x1/2x2 退化(全走边界路径，内部为空) */
    run("1x1 degenerate",   1,1, 1,1, 1,1, sobel, 1);
    run("2x2 degenerate",   2,2, 2,2, 2,2, box,   2);
    run("3x3 min-interior", 3,3, 3,3, 3,3, rnd,   3);
    run("small 5x5",        5,5, 5,5, 5,5, sobel, 4);
    run("small 6x6",        6,6, 6,6, 6,6, box,   40);  /* 内部宽=4 恰好 1 个 NEON 向量 */
    run("small 7x7",        7,7, 7,7, 7,7, rnd,   41);  /* 内部宽=5 = 1向量+1标量收尾 */
    run("small 10x10",      10,10, 10,10, 10,10, sobel, 42); /* 内部宽=8 = 2向量 */
    run("small 11x11",      11,11, 11,11, 11,11, rnd,  43); /* 内部宽=9 = 2向量+1收尾 */
    run("same 37x29",       37,29, 37,29, 37,37, rnd, 5);
    run("same 64x64",       64,64, 64,64, 64,64, box, 6);

    /* 非零 stride(行尾留白)，验证 SIMD 内核不越界写 */
    run("stride 37x29",     37,29, 37,29, 41,44, rnd, 7);
    run("stride 128x96",    128,96, 128,96, 160,150, sobel, 8);

    /* 输出小于输入(左上子区域，i_end/j_end 被 Wd/Hd 截断) */
    run("subregion 8->4",   8,8, 4,4, 8,4, box, 9);
    run("subregion 20->13", 20,20, 13,11, 24,16, rnd, 10);

    /* 非方阵、宽>高 / 高>宽 */
    run("wide 100x10",      100,10, 100,10, 104,104, sobel, 11);
    run("tall 10x100",      10,100, 10,100, 12,100, rnd, 12);

    printf("\n=== %s ===\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return g_fail ? 1 : 0;
}
