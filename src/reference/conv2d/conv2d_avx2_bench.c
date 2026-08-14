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

/* conv2d_avx2_bench.c —— MSVC 原生性能测试（不依赖 POSIX）
 *
 * 说明：原 conv2d_bench.c 使用 posix_memalign / clock_gettime / CLOCK_MONOTONIC，
 * 这些是 POSIX 专有 API，MSVC (cl.exe) 无法编译。此文件用 Windows 原生的
 * _aligned_malloc / _aligned_free 与 QueryPerformanceCounter 计时，覆盖 v0~v6，
 * 复现原 bench 的“迭代校准 + best-of-5”方法，并对每版做 vs v0 的最大绝对误差校验。
 *
 * 编译（在 vcvars64 环境下）：
 *   cl /O2 /arch:AVX2 conv2d_avx2_bench.c conv2d_ref.c conv2d_v1.c conv2d_v2.c \
 *      conv2d_v3.c conv2d_v4.c conv2d_v5.c conv2d_v6.c /Fe:conv2d_avx2_bench.exe
 */
#pragma execution_character_set("utf-8")
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/* 同时输出到控制台(stdout)与一份 UTF-8(带BOM)文件，确保中文跨终端都能正确显示 */
static FILE* g_log = NULL;
#define emit(...) do { printf(__VA_ARGS__); if (g_log) fprintf(g_log, __VA_ARGS__); } while (0)

typedef float ivf32;
void conv2d_v0(ivf32* dst, int wd, int hd, int sd, ivf32* src, int ws, int hs, int ss, ivf32 coef[9]);
void conv2d_v1(ivf32* dst, int wd, int hd, int sd, ivf32* src, int ws, int hs, int ss, ivf32 coef[9]);
void conv2d_v2(ivf32* dst, int wd, int hd, int sd, ivf32* src, int ws, int hs, int ss, ivf32 coef[9]);
void conv2d_v3(ivf32* dst, int wd, int hd, int sd, ivf32* src, int ws, int hs, int ss, ivf32 coef[9]);
void conv2d_v4(ivf32* dst, int wd, int hd, int sd, ivf32* src, int ws, int hs, int ss, ivf32 coef[9]);
void conv2d_v5(ivf32* dst, int wd, int hd, int sd, ivf32* src, int ws, int hs, int ss, ivf32 coef[9]);
void conv2d_v6(ivf32* dst, int wd, int hd, int sd, ivf32* src, int ws, int hs, int ss, ivf32 coef[9]);

typedef void (*ConvFn)(ivf32*,int,int,int,ivf32*,int,int,int,ivf32[9]);

static double g_freq = 0.0;  /* QPC 频率(计数/秒) */
static double now(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / g_freq;
}

static volatile double sink = 0.0;

/* 返回单次调用平均耗时(秒)。iters 由外部校准得到。 */
static double bench(ConvFn fn, ivf32* dst, int wd, int hd, int sd,
                    ivf32* src, int ws, int hs, int ss, ivf32 coef[9], int iters)
{
    double t0 = now();
    for (int k = 0; k < iters; k++) {
        fn(dst, wd, hd, sd, src, ws, hs, ss, coef);
        sink += dst[0];          /* 防止编译器把调用优化掉 */
    }
    double t1 = now();
    return (t1 - t0) / iters;
}

/* best-of-K：取多次测量的最小单次耗时，排除频率调节/调度噪声。 */
static double bench_best(ConvFn fn, ivf32* dst, int wd, int hd, int sd,
                         ivf32* src, int ws, int hs, int ss, ivf32 coef[9],
                         int iters, int trials)
{
    double best = 1e300;
    for (int t = 0; t < trials; t++) {
        double per = bench(fn, dst, wd, hd, sd, src, ws, hs, ss, coef, iters);
        if (per < best) best = per;
    }
    return best;
}

/* 校准迭代数，使单次测量总时长 >= 50ms */
static int calibrate(ConvFn fn, ivf32* dst, int wd, int hd, int sd,
                     ivf32* src, int ws, int hs, int ss, ivf32 coef[9])
{
    int it = 1;
    double per = bench(fn, dst, wd, hd, sd, src, ws, hs, ss, coef, it);
    while (per * it < 0.05 && it < (1 << 22)) {
        it *= 2;
        per = bench(fn, dst, wd, hd, sd, src, ws, hs, ss, coef, it);
    }
    return it;
}

int main(void) {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    g_freq = (double)f.QuadPart;
    SetConsoleOutputCP(CP_UTF8);   /* 控制台以 UTF-8 渲染，确保中文正常显示 */
    g_log = fopen("bench_result_utf8.txt", "wb");
    if (g_log) { fputc(0xEF, g_log); fputc(0xBB, g_log); fputc(0xBF, g_log); }  /* UTF-8 BOM */

    int sizes[] = {100, 500, 1000, 2000, 3000, 4000, 5000};
    int ns = sizeof(sizes) / sizeof(sizes[0]);

    emit("# conv2d AVX2 性能对比 (NxN 同尺寸, stride=N, 3x3, Mops=9次乘加/像素)\n");
    emit("# 编译: MSVC /O2 /arch:AVX2\n");
    emit("%-6s | %9s %9s %9s %9s %9s %9s %9s | %9s %9s %9s %9s %9s %9s | %8s %8s %8s %8s %8s %8s | %10s\n",
           "N", "v0(ms)", "v1(ms)", "v2(ms)", "v3(ms)", "v4(ms)", "v5(ms)", "v6(ms)",
           "v1Mops", "v2Mops", "v3Mops", "v4Mops", "v5Mops", "v6Mops",
           "v2/v0", "v3/v2", "v4/v2", "v5/v4", "v6/v5", "v6/v0", "max|Δ|");
    emit("-------+---------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------------+----------\n");

    for (int s = 0; s < ns; s++) {
        int N = sizes[s];
        int Ws = N, Hs = N, Wd = N, Hd = N, ss = N, sd = N;
        size_t n = (size_t)N * N;

        ivf32 *src = (ivf32*)_aligned_malloc(n * sizeof(ivf32), 64);
        ivf32 *dst = (ivf32*)_aligned_malloc(n * sizeof(ivf32), 64);
        ivf32 *cd0 = (ivf32*)_aligned_malloc(n * sizeof(ivf32), 64);
        ivf32 *cdx = (ivf32*)_aligned_malloc(n * sizeof(ivf32), 64);
        if (!src || !dst || !cd0 || !cdx) { fprintf(stderr, "alloc failed\n"); return 1; }
        ivf32 coef[9];

        srand(12345 + N);
        for (size_t k = 0; k < n; k++) src[k] = (ivf32)rand() / RAND_MAX;
        for (int k = 0; k < 9; k++) coef[k] = (ivf32)rand() / RAND_MAX * 2 - 1;

        /* 预热：每个函数各调用一次，把首触碰(page fault/冷缓存)排除出计时 */
        conv2d_v0(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        conv2d_v1(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        conv2d_v2(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        conv2d_v3(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        conv2d_v4(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        conv2d_v5(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        conv2d_v6(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);

        int it0 = calibrate(conv2d_v0, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        int it1 = calibrate(conv2d_v1, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        int it2 = calibrate(conv2d_v2, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        int it3 = calibrate(conv2d_v3, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        int it4 = calibrate(conv2d_v4, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        int it5 = calibrate(conv2d_v5, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        int it6 = calibrate(conv2d_v6, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);

        double per0 = bench_best(conv2d_v0, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it0, 5);
        double per1 = bench_best(conv2d_v1, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it1, 5);
        double per2 = bench_best(conv2d_v2, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it2, 5);
        double per3 = bench_best(conv2d_v3, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it3, 5);
        double per4 = bench_best(conv2d_v4, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it4, 5);
        double per5 = bench_best(conv2d_v5, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it5, 5);
        double per6 = bench_best(conv2d_v6, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it6, 5);

        double mops1 = (double)n * 9.0 / per1 / 1e6;
        double mops2 = (double)n * 9.0 / per2 / 1e6;
        double mops3 = (double)n * 9.0 / per3 / 1e6;
        double mops4 = (double)n * 9.0 / per4 / 1e6;
        double mops5 = (double)n * 9.0 / per5 / 1e6;
        double mops6 = (double)n * 9.0 / per6 / 1e6;
        double sp2  = per0 / per2;   /* v2 相对 v0 */
        double sp32 = per2 / per3;   /* v3 相对 v2 */
        double sp42 = per2 / per4;   /* v4 相对 v2 */
        double sp54 = per4 / per5;   /* v5 相对 v4 */
        double sp65 = per5 / per6;   /* v6 相对 v5 */
        double sp60 = per0 / per6;   /* v6 相对 v0 */

        /* 正确性校验：取 v2~v6 各自与 v0 的最大绝对差里的最大者 */
        conv2d_v0(cd0, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        double maxd = 0.0;
        ConvFn vs[5] = { conv2d_v2, conv2d_v3, conv2d_v4, conv2d_v5, conv2d_v6 };
        for (int vi = 0; vi < 5; vi++) {
            vs[vi](cdx, Wd, Hd, sd, src, Ws, Hs, ss, coef);
            for (size_t k = 0; k < n; k++) {
                double d = fabs((double)cd0[k] - (double)cdx[k]);
                if (d > maxd) maxd = d;
            }
        }

        emit("%-6d | %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f | %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f | %7.2fx %7.2fx %7.2fx %7.2fx %7.2fx %7.2fx | %10.2e\n",
               N, per0 * 1e3, per1 * 1e3, per2 * 1e3, per3 * 1e3, per4 * 1e3, per5 * 1e3, per6 * 1e3,
               mops1, mops2, mops3, mops4, mops5, mops6,
               sp2, sp32, sp42, sp54, sp65, sp60, maxd);

        _aligned_free(src); _aligned_free(dst); _aligned_free(cd0); _aligned_free(cdx);
    }
    emit("sink=%f (防DCE)\n", sink);
    return 0;
}
