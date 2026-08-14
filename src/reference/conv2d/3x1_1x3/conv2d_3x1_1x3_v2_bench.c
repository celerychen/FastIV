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

/* conv2d_3x1_1x3_v2 基准 —— 严格照搬 bench_o2 方法学：
 *   64 字节对齐缓冲 / srand(12345+N) 同种子 / 每函数预热 / 校准 iters>=50ms / best-of-5 取最小
 *   含 max|Δ| 正确性列（v2 vs 标准 3x3 v0，可分离核 H⊗V）
 * 对照：conv2d_v0(标准3x3, 9 MAC/px) / _v1(自动向量化) / _v2(手写NEON intrinsics)
 * 编译: -O2
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>

typedef float ivf32;

void conv2d_v0(ivf32* dst, int Wd, int Hd, int sd, ivf32* src, int Ws, int Hs, int ss, ivf32 coef[9]);
void conv2d_3x1_1x3_v1(ivf32* dst, int Wd, int Hd, int sd, ivf32* src, int Ws, int Hs, int ss, ivf32 coef[6]);
void conv2d_3x1_1x3_v2(ivf32* dst, int Wd, int Hd, int sd, ivf32* src, int Ws, int Hs, int ss, ivf32 coef[6]);

static ivf32 *g_src, *g_d0, *g_d1, *g_d2;
static ivf32  g_ref9[9], g_sep6[6];
static int    g_N;

static void run_v0(void) { conv2d_v0 (g_d0, g_N, g_N, g_N, g_src, g_N, g_N, g_N, g_ref9); }
static void run_v1(void) { conv2d_3x1_1x3_v1(g_d1, g_N, g_N, g_N, g_src, g_N, g_N, g_N, g_sep6); }
static void run_v2(void) { conv2d_3x1_1x3_v2(g_d2, g_N, g_N, g_N, g_src, g_N, g_N, g_N, g_sep6); }

static double now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
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
    ivf32 H[3] = {1, 2, 1}, V[3] = {1, 0, -1};      /* sobel（可分离） */
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) g_ref9[r*3+c] = V[r]*H[c];
    g_sep6[0]=H[0]; g_sep6[1]=H[1]; g_sep6[2]=H[2]; g_sep6[3]=V[0]; g_sep6[4]=V[1]; g_sep6[5]=V[2];

    printf("# conv2d_3x1_1x3_v2 (手写NEON) vs v1 (自动向量化) vs v0 (标准3x3)\n");
    printf("# NxN, stride=N, 可分离核 H(1x3)⊗V(3x1); Mops: v0=9/px, v1=v2=6/px; 编译 -O2, best-of-5\n");
    printf("%-6s | %9s %9s %9s | %11s %11s %11s | %7s %7s | %11s\n",
           "N","v0(ms)","v1(ms)","v2(ms)","v0(Mops)","v1(Mops)","v2(Mops)","v2/v1","v2/v0","v2 max|Δ|");
    printf("------+-----------+-----------+-----------+-------------+-------------+-------------+---------+---------+-----------\n");

    for (int s = 0; s < SIZES; s++) {
        int N = Ns[s]; g_N = N;
        posix_memalign((void**)&g_src, 64, (size_t)N*N*sizeof(ivf32));
        posix_memalign((void**)&g_d0,  64, (size_t)N*N*sizeof(ivf32));
        posix_memalign((void**)&g_d1,  64, (size_t)N*N*sizeof(ivf32));
        posix_memalign((void**)&g_d2,  64, (size_t)N*N*sizeof(ivf32));
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

        printf("%-6d | %9.3f %9.3f %9.3f | %11.1f %11.1f %11.1f | %6.2fx %6.2fx | %11.2e\n",
               N, ms0, ms1, ms2, mops0, mops1, mops2, ms1/ms2, ms0/ms2, mx);

        free(g_src); free(g_d0); free(g_d1); free(g_d2);
    }
    return 0;
}
