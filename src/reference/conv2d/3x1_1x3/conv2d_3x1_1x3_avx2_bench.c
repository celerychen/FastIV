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

/* conv2d_3x1_1x3_avx2_bench —— MSVC 原生基准（_aligned_malloc + QueryPerformanceCounter）
 * 严格照搬原 bench 方法学：
 *   64 字节对齐缓冲 / srand(12345+N) 同种子 / 每函数预热 / 校准 iters>=50ms / best-of-5 取最小
 *   含 max|Δ| 正确性列（v2 vs 标准 3x3 v0，可分离核 H⊗V）
 * 对照：conv2d_v0(标准3x3, 9 MAC/px) / _v1(自动向量化) / _v2(手写SIMD: NEON/AVX2/标量)
 *
 * 本文件为新建 MSVC 原生 bench，不改原 POSIX 版 conv2d_3x1_1x3_v2_bench.c。
 * 编译: cl /O2 /arch:AVX2 /utf-8
 * 输出: 同时打印到控制台(UTF-8 代码页) 与自写 UTF-8+BOM 文件 conv2d_3x1_1x3_bench_utf8.txt
 */
#pragma execution_character_set("utf-8")
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef float ivf32;

static double g_freq;

void conv2d_v0(ivf32* dst, int Wd, int Hd, int sd, ivf32* src, int Ws, int Hs, int ss, ivf32 coef[9]);
void conv2d_3x1_1x3_v1(ivf32* dst, int Wd, int Hd, int sd, ivf32* src, int Ws, int Hs, int ss, ivf32 coef[6]);
void conv2d_3x1_1x3_v2(ivf32* dst, int Wd, int Hd, int sd, ivf32* src, int Ws, int Hs, int ss, ivf32 coef[6]);

/* UTF-8 结果文件（带 BOM），避免 PowerShell/控制台编码导致中文乱码 */
static FILE* g_log = NULL;
#define emit(...) do { printf(__VA_ARGS__); if (g_log) fprintf(g_log, __VA_ARGS__); } while (0)

static ivf32 *g_src, *g_d0, *g_d1, *g_d2;
static ivf32  g_ref9[9], g_sep6[6];
static int    g_N;

static void run_v0(void) { conv2d_v0 (g_d0, g_N, g_N, g_N, g_src, g_N, g_N, g_N, g_ref9); }
static void run_v1(void) { conv2d_3x1_1x3_v1(g_d1, g_N, g_N, g_N, g_src, g_N, g_N, g_N, g_sep6); }
static void run_v2(void) { conv2d_3x1_1x3_v2(g_d2, g_N, g_N, g_N, g_src, g_N, g_N, g_N, g_sep6); }

static double now_ns(void) {
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (double)c.QuadPart / g_freq * 1e9;
}

static double bench_bestof5(void (*fn)(void), int* piters) {
    int iters = *piters;
    fn();
    for (;;) {
        double t0 = now_ns(); for (int k = 0; k < iters; k++) fn(); double t1 = now_ns();
        if (t1 - t0 >= 50e6) break;
        iters *= 2; if (iters > 400000) break;
    }
    *piters = iters;
    double best = 1e30;
    for (int r = 0; r < 5; r++) {
        double t0 = now_ns(); for (int k = 0; k < iters; k++) fn(); double t1 = now_ns();
        double dt = t1 - t0;
        if (dt < best) best = dt;
    }
    return best;
}

#define SIZES 7
static const int Ns[SIZES] = {100, 500, 1000, 2000, 3000, 4000, 5000};

int main(void) {
    SetConsoleOutputCP(CP_UTF8);   /* 控制台以 UTF-8 渲染，确保中文正常显示 */
    g_log = fopen("conv2d_3x1_1x3_bench_utf8.txt", "wb");
    if (g_log) { fputc(0xEF, g_log); fputc(0xBB, g_log); fputc(0xBF, g_log); }  /* UTF-8 BOM */

    {
        LARGE_INTEGER f; QueryPerformanceFrequency(&f);
        g_freq = (double)f.QuadPart;
    }

    ivf32 H[3] = {1, 2, 1}, V[3] = {1, 0, -1};      /* sobel（可分离） */
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) g_ref9[r*3+c] = V[r]*H[c];
    g_sep6[0]=H[0]; g_sep6[1]=H[1]; g_sep6[2]=H[2]; g_sep6[3]=V[0]; g_sep6[4]=V[1]; g_sep6[5]=V[2];

    emit("# conv2d_3x1_1x3_v2 (手写SIMD) vs v1 (自动向量化) vs v0 (标准3x3)\n");
    emit("# NxN, stride=N, 可分离核 H(1x3)⊗V(3x1); Mops: v0=9/px, v1=v2=6/px; 编译 MSVC /O2 /arch:AVX2, best-of-5\n");
    emit("%-6s | %9s %9s %9s | %11s %11s %11s | %7s %7s | %11s\n",
         "N","v0(ms)","v1(ms)","v2(ms)","v0(Mops)","v1(Mops)","v2(Mops)","v2/v1","v2/v0","v2 max|Δ|");
    emit("------+-----------+-----------+-----------+-------------+-------------+-------------+---------+---------+-----------\n");

    for (int s = 0; s < SIZES; s++) {
        int N = Ns[s]; g_N = N;
        g_src = (ivf32*)_aligned_malloc((size_t)N*N*sizeof(ivf32), 64);
        g_d0  = (ivf32*)_aligned_malloc((size_t)N*N*sizeof(ivf32), 64);
        g_d1  = (ivf32*)_aligned_malloc((size_t)N*N*sizeof(ivf32), 64);
        g_d2  = (ivf32*)_aligned_malloc((size_t)N*N*sizeof(ivf32), 64);
        srand(12345 + N);
        for (int i = 0; i < N*N; i++) g_src[i] = ((ivf32)rand()/RAND_MAX)*2.0f - 1.0f;

        int it = 1;
        double dt0 = bench_bestof5(run_v0, &it); double ms0 = dt0/it/1e6;
        double dt1 = bench_bestof5(run_v1, &it); double ms1 = dt1/it/1e6;
        double dt2 = bench_bestof5(run_v2, &it); double ms2 = dt2/it/1e6;

        double mops0 = (double)N*N*9 / ms0 / 1e3;
        double mops1 = (double)N*N*6 / ms1 / 1e3;
        double mops2 = (double)N*N*6 / ms2 / 1e3;

        run_v0(); run_v2();
        float mx = 0.0f;
        for (int i = 0; i < N*N; i++) { float d = fabsf(g_d2[i]-g_d0[i]); if (d>mx) mx=d; }

        emit("%-6d | %9.3f %9.3f %9.3f | %11.1f %11.1f %11.1f | %6.2fx %6.2fx | %11.2e\n",
             N, ms0, ms1, ms2, mops0, mops1, mops2, ms1/ms2, ms0/ms2, mx);

        _aligned_free(g_src); _aligned_free(g_d0); _aligned_free(g_d1); _aligned_free(g_d2);
    }
    if (g_log) fclose(g_log);
    return 0;
}
