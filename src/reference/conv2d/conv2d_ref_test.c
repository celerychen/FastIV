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
#include <string.h>

typedef float ivf32;

/* ---- 被测函数（原样引入） ---- */
void conv2d_v0(ivf32* dst, int width_dst, int height_dst, int stride_dst,
            ivf32* src, int width_src, int height_src, int stride_src,
            ivf32 coef[9]);

/* ---- 独立参考实现 ----
 * 思路：先显式构造 (Ws+2)x(Hs+2) 的"边缘复制"填充缓冲，再在填充缓冲上做卷积。
 * 索引结构与被测函数不同（被测是在内层循环里按需钳制 src 坐标），
 * 因此两者若结果一致，即可交叉验证填充与系数顺序的正确性。 */
static void conv2d_indep(ivf32* dst, int Wd, int Hd, int sd,
                         ivf32* src, int Ws, int Hs, int ss,
                         ivf32 coef[9]) {
    int PW = Ws + 2, PH = Hs + 2;
    ivf32* pad = (ivf32*)malloc((size_t)PW * PH * sizeof(ivf32));
    for (int y = 0; y < PH; y++) {
        for (int x = 0; x < PW; x++) {
            int sx = x - 1, sy = y - 1;
            if (sx < 0) sx = 0; if (sx >= Ws) sx = Ws - 1;
            if (sy < 0) sy = 0; if (sy >= Hs) sy = Hs - 1;
            pad[y * PW + x] = src[sy * ss + sx];
        }
    }
    for (int j = 0; j < Hd; j++) {
        for (int i = 0; i < Wd; i++) {
            ivf32 s = 0.0f;
            for (int kj = 0; kj < 3; kj++) {
                for (int ki = 0; ki < 3; ki++) {
                    int px = i + ki;       /* 填充缓冲坐标 */
                    int py = j + kj;
                    s += pad[py * PW + px] * coef[kj * 3 + ki];
                }
            }
            dst[j * sd + i] = s;
        }
    }
    free(pad);
}

static int g_fail = 0;
static void check_eq(const char* name, ivf32 got, ivf32 exp) {
    if (got != exp) { /* 同序运算应位级一致 */
        printf("  FAIL %s: got=%f exp=%f\n", name, got, exp);
        g_fail++;
    }
}
static void check_close(const char* name, ivf32 got, ivf32 exp, ivf32 tol) {
    if (fabsf(got - exp) > tol) {
        printf("  FAIL %s: got=%f exp=%f (tol=%g)\n", name, got, exp, tol);
        g_fail++;
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== conv2d_v0 correctness verification ===\n");

    /* ---------- 用例 1: 手工可算的小矩阵 ----------
     * 输入 3x3:            系数(行主序):
     *  1 2 3                1 2 3
     *  4 5 6                4 5 6
     *  7 8 9                7 8 9
     * 同尺寸输出 (Wd=Hd=3) 用 replicate 填充
     * 手算: out(1,1)=285.0 ; out(0,0)=135.0
     */
    {
        int Ws = 3, Hs = 3, Wd = 3, Hd = 3, ss = 3, sd = 3;
        ivf32 src[9] = {1,2,3, 4,5,6, 7,8,9};
        ivf32 coef[9] = {1,2,3, 4,5,6, 7,8,9};
        ivf32 dst[9];
        conv2d_v0(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        check_eq("case1 out(1,1)", dst[1*3+1], 285.0f);
        check_eq("case1 out(0,0)", dst[0*3+0], 135.0f);
        /* 同时与独立参考逐元素比对 */
        ivf32 ref[9];
        conv2d_indep(ref, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        for (int k = 0; k < 9; k++)
            check_eq("case1 indep", dst[k], ref[k]);
        printf("case1 (hand-computed 3x3): %s\n", g_fail ? "FAILED" : "PASS");
    }

    /* ---------- 用例 2: 全 1 输入 + 任意核 => 输出恒为 sum(coef) ---------- */
    {
        int Ws = 5, Hs = 5, Wd = 5, Hd = 5, ss = 5, sd = 5;
        ivf32 src[25]; for (int k=0;k<25;k++) src[k]=1.0f;
        ivf32 coef[9] = {1,2,3, 4,5,6, 7,8,9};
        ivf32 dst[25];
        conv2d_v0(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        ivf32 sum = 45.0f;
        for (int k=0;k<25;k++) check_eq("case2 all-ones", dst[k], sum);
        printf("case2 (all-ones -> sum(coef)): %s\n", g_fail ? "FAILED":"PASS");
    }

    /* ---------- 用例 3: 单位核 => 输出 = 输入(边缘复制) ---------- */
    {
        int Ws = 4, Hs = 4, Wd = 4, Hd = 4, ss = 4, sd = 4;
        ivf32 src[16]; for (int k=0;k<16;k++) src[k]=(ivf32)(k+1);
        ivf32 coef[9] = {0,0,0, 0,1,0, 0,0,0};
        ivf32 dst[16];
        conv2d_v0(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        /* interior 与原值一致; 边界被钳制到最近原值 */
        ivf32 ref[16];
        conv2d_indep(ref, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        for (int k=0;k<16;k++) check_eq("case3 identity", dst[k], ref[k]);
        printf("case3 (identity kernel): %s\n", g_fail ? "FAILED":"PASS");
    }

    /* ---------- 用例 4: 随机大矩阵 + 非零 stride (验证无越界) ---------- */
    {
        int Ws = 37, Hs = 29, Wd = 37, Hd = 29;
        int ss = Ws + 4;     /* src 每行后面留 4 个填充, 验证不越界 */
        int sd = Wd + 7;     /* dst 每行后面留 7 个填充 */
        size_t src_n = (size_t)ss * Hs;
        size_t dst_n = (size_t)sd * Hd;
        ivf32* src = (ivf32*)malloc(src_n * sizeof(ivf32));
        ivf32* dst = (ivf32*)malloc(dst_n * sizeof(ivf32));
        ivf32 coef[9]; for (int k=0;k<9;k++) coef[k]=(ivf32)(k-4); /* -4..4 */
        /* 随机数 */
        srand(12345);
        for (size_t k=0;k<src_n;k++) src[k] = ((ivf32)rand()/RAND_MAX)*2.0f - 1.0f;
        /* 初始化 dst 为哨兵值, 用于检测越界写入 */
        for (size_t k=0;k<dst_n;k++) dst[k] = -999.0f;

        conv2d_v0(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);

        /* 4a: 与独立参考逐元素位级一致 */
        ivf32* ref = (ivf32*)malloc(dst_n * sizeof(ivf32));
        for (size_t k=0;k<dst_n;k++) ref[k] = -999.0f;
        conv2d_indep(ref, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        int mism = 0;
        for (int j=0;j<Hd;j++) for (int i=0;i<Wd;i++) {
            if (dst[j*sd+i] != ref[j*sd+i]) mism++;
        }
        if (mism) { printf("  FAIL case4 mismatch count=%d\n", mism); g_fail++; }

        /* 4b: 检查每行 padding 槽未被写入 (stride 正确, 无越界) */
        int overrun = 0;
        for (int j=0;j<Hd;j++) for (int i=Wd;i<sd;i++)
            if (dst[j*sd+i] != -999.0f) { overrun++; }
        if (overrun) { printf("  FAIL case4 stride overrun slots=%d\n", overrun); g_fail++; }

        printf("case4 (random %dx%d, stride src=%d dst=%d): %s\n",
               Ws, Hs, ss, sd, g_fail ? "FAILED":"PASS");
        free(src); free(dst); free(ref);
    }

    /* ---------- 用例 5: 输出尺寸 < 输入尺寸 (取左上 valid 区域) ---------- */
    {
        int Ws = 8, Hs = 8, Wd = 4, Hd = 4, ss = 8, sd = 4;
        ivf32 src[64]; for (int k=0;k<64;k++) src[k]=(ivf32)(k);
        ivf32 coef[9] = {1,1,1, 1,1,1, 1,1,1}; /* box 求和 */
        ivf32 dst[16];
        conv2d_v0(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        ivf32 ref[16];
        conv2d_indep(ref, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        for (int k=0;k<16;k++) check_eq("case5 subregion", dst[k], ref[k]);
        printf("case5 (Wd<Ws subregion): %s\n", g_fail ? "FAILED":"PASS");
    }

    printf("\n=== %s ===\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return g_fail ? 1 : 0;
}
