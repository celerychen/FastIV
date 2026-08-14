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
/* 6 系数接口：coef[0..2]=行滤波, coef[3..5]=列滤波 */
void conv2d_3x1_1x3_v0(ivf32* dst, int Wd, int Hd, int sd,
                       ivf32* src, int Ws, int Hs, int ss, ivf32 coef[6]);

static int g_fail = 0;

/* 验证方法（按规范）：
 *   取两个 3 维向量 V、H，外积得到 3x3 核 K[r*3+c] = V[r] * H[c]；
 *   用标准 3x3 滤波 conv2d_v0(K) 作参考，与 separable 版本(6 系数 [H.., V..])对比。 */
static void run(const char* name, int Ws, int Hs, int Wd, int Hd,
                int ss, int sd, ivf32 V[3], ivf32 H[3], unsigned src_seed) {
    ivf32 ref9[9], sep6[6];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            ref9[r*3+c] = V[r] * H[c];          /* 外积 → 3x3 核 */
    sep6[0] = H[0]; sep6[1] = H[1]; sep6[2] = H[2];
    sep6[3] = V[0]; sep6[4] = V[1]; sep6[5] = V[2];

    size_t src_n = (size_t)ss * Hs;
    size_t dst_n = (size_t)sd * Hd;
    ivf32* src = (ivf32*)malloc(src_n * sizeof(ivf32));
    ivf32* d0  = (ivf32*)malloc(dst_n * sizeof(ivf32));
    ivf32* d2  = (ivf32*)malloc(dst_n * sizeof(ivf32));
    srand(src_seed);
    for (size_t k = 0; k < src_n; k++) src[k] = ((ivf32)rand()/RAND_MAX)*2.0f - 1.0f;
    for (size_t k = 0; k < dst_n; k++) { d0[k] = -999.0f; d2[k] = -999.0f; }

    conv2d_v0(d0, Wd, Hd, sd, src, Ws, Hs, ss, ref9);            /* 标准 3x3 滤波(参考) */
    conv2d_3x1_1x3_v0(d2, Wd, Hd, sd, src, Ws, Hs, ss, sep6);     /* 可分离 6 系数版(被测) */

    int mism = 0, overrun = 0;
    for (int j = 0; j < Hd; j++) {
        for (int i = 0; i < Wd; i++) {
            ivf32 a = d0[(size_t)j*sd+i], b = d2[(size_t)j*sd+i];
            if (fabsf(a - b) > 1e-4f * (fabsf(a) + 1.0f)) mism++;
        }
        for (int i = Wd; i < sd; i++)
            if (d2[(size_t)j*sd+i] != -999.0f) overrun++;
    }
    if (mism)    { printf("  FAIL %s: %d mismatch vs v0\n", name, mism); g_fail++; }
    if (overrun) { printf("  FAIL %s: %d stride overrun\n", name, overrun); g_fail++; }
    printf("%-30s Ws=%d Hs=%d Wd=%d Hd=%d ss=%d sd=%d : %s\n",
           name, Ws, Hs, Wd, Hd, ss, sd, (mism||overrun) ? "FAILED" : "PASS");
    free(src); free(d0); free(d2);
}

/* 随机生成两个 3 维向量 V、H（kern_seed 控核，src_seed 控输入），再跑外积验证 */
static void run_rand(const char* name, int Ws, int Hs, int Wd, int Hd,
                     int ss, int sd, unsigned kern_seed, unsigned src_seed) {
    ivf32 V[3], H[3];
    srand(kern_seed);
    for (int k = 0; k < 3; k++) {
        V[k] = ((ivf32)rand()/RAND_MAX)*2.0f - 1.0f;
        H[k] = ((ivf32)rand()/RAND_MAX)*2.0f - 1.0f;
    }
    run(name, Ws, Hs, Wd, Hd, ss, sd, V, H, src_seed);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== conv2d_3x1_1x3_v0 (separable, 6 系数) vs conv2d_v0 (外积构造的 3x3 核) ===\n");

    /* 已知可分离核：显式给出 V、H，外积即 3x3 核 */
    ivf32 sobel_V[3] = {1,2,1}, sobel_H[3] = {1,0,-1};
    ivf32 box_V[3]   = {1,1,1}, box_H[3]   = {1,1,1};
    run("sobel", 64,64, 64,64, 64,64, sobel_V, sobel_H, 1);
    run("box",   64,64, 64,64, 64,64, box_V,   box_H,   2);

    /* 随机 3 维向量对 → 外积 3x3 核 → 与标准 3x3 滤波对比 */
    run_rand("1x1 degenerate",   1,1, 1,1, 1,1, 3, 103);
    run_rand("2x2 degenerate",   2,2, 2,2, 2,2, 4, 104);
    run_rand("3x3 min-interior", 3,3, 3,3, 3,3, 5, 105);
    run_rand("small 5x5",        5,5, 5,5, 5,5, 6, 106);
    run_rand("small 10x10",      10,10, 10,10, 10,10, 7, 107);
    run_rand("i8edge 9x9",        9,9, 9,9, 9,9, 8, 108);
    run_rand("i8edge 15x15",      15,15, 15,15, 15,15, 9, 109);
    run_rand("i8edge 17x17",      17,17, 17,17, 17,17, 10, 110);
    run_rand("same 37x29",       37,29, 37,29, 37,37, 11, 111);
    run_rand("same 64x64",       64,64, 64,64, 64,64, 12, 112);
    run_rand("stride 37x29",     37,29, 37,29, 41,44, 13, 113);
    run_rand("stride 128x96",    128,96, 128,96, 160,150, 14, 114);
    run_rand("subregion 8->4",   8,8, 4,4, 8,4, 15, 115);
    run_rand("subregion 20->13", 20,20, 13,11, 24,16, 16, 116);
    run_rand("wide 100x10",      100,10, 100,10, 104,104, 17, 117);
    run_rand("tall 10x100",      10,100, 10,100, 12,100, 18, 118);

    printf("\n=== %s ===\n", g_fail ? "SOME SEPARABLE TESTS FAILED" : "ALL SEPARABLE TESTS PASSED");
    return g_fail ? 1 : 0;
}
