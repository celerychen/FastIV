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

/*
 * bench_all.c —— 全版本内积性能对比（严格测量）。
 *
 * 参与对比的 5 个版本：
 *   basic        : 标量 C（/fp:fast 下会被 MSVC 自动向量化）
 *   avx          : FMA，4 累加器 4 路展开
 *   avx_v2       : dp(VDPPS)，4 累加器 4 路展开
 *   avx_sp       : FMA + 软件流水线（load-ahead 错位）
 *   avx_v2_sp    : dp  + 软件流水线（load-ahead 错位）
 *
 * 方法学：min-of-REPEAT（取最短耗时，抗调度/降频噪声）、交错测量、
 *   n 覆盖 L1→L2→L3→RAM 全谱、返回值累加防止死代码消除。
 *   speedup 一律以 basic 为分母（>1 即比标量基线快）。
 */
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "dotproduct.h"

static double now_sec(void) {
#ifdef _WIN32
    static LARGE_INTEGER f = {0};
    LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

static uint32_t rs = 123456789u;
static float rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return (float)((rs >> 8) & 0xFFFF) / 32768.0f - 1.0f; }

typedef float (*dot_fn)(const float*, const float*, size_t);

/* 该函数在 n 上 min-of-repeats 的耗时(秒/次调用) */
static double bench(dot_fn fn, const float* a, const float* b, size_t n, int iters, int repeat, volatile float* sink) {
    double best = 1e30;
    for (int r = 0; r < repeat; r++) {
        double t0 = now_sec();
        float acc = 0.0f;
        for (int it = 0; it < iters; it++) acc += fn(a, b, n);
        double t1 = now_sec();
        *sink += acc;
        double per = (t1 - t0) / iters;
        if (per < best) best = per;
    }
    return best;
}

int main(void) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    size_t ns[] = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 262144, 1048576, 4194304 };
    int    nN   = (int)(sizeof(ns) / sizeof(ns[0]));
    int    repeat = 9;
    volatile float sink = 0.0f;

    printf("CPU: Intel i7-6700HQ (Skylake)  L1D=32KB L2=256KB L3=6MB\n");
    printf("min-of-%d, Mop/s = 2*n*iters/t ；speedup 以 basic 为分母\n\n", repeat);
    printf("%9s %6s | %8s %8s %8s %8s %8s | %8s %8s %8s %8s\n",
           "n", "region", "basic", "avx", "avx_v2", "avx_sp", "v2_sp",
           "avx/b", "v2/b", "sp/b", "v2sp/b");
    printf("---------------------------------------------------------------------------------------------------------\n");

    for (int k = 0; k < nN; k++) {
        size_t n = ns[k];
        float* a = (float*)malloc(n * sizeof(float));
        float* b = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) { a[i] = rnd(); b[i] = rnd(); }

        int iters = (int)(80000000ull / n);
        if (iters < 20) iters = 20;

        /* warmup */
        for (int w = 0; w < 3; w++) {
            sink += dot_basic(a,b,n); sink += dot_avx(a,b,n); sink += dot_avx_v2(a,b,n);
            sink += dot_avx_sp(a,b,n); sink += dot_avx_v2_sp(a,b,n);
        }

        double tb  = bench(dot_basic,     a, b, n, iters, repeat, &sink);
        double tf  = bench(dot_avx,       a, b, n, iters, repeat, &sink);
        double td  = bench(dot_avx_v2,    a, b, n, iters, repeat, &sink);
        double tfs = bench(dot_avx_sp,    a, b, n, iters, repeat, &sink);
        double tds = bench(dot_avx_v2_sp, a, b, n, iters, repeat, &sink);

        double mb  = 2.0 * n / tb  / 1e6;
        double mf  = 2.0 * n / tf  / 1e6;
        double md  = 2.0 * n / td  / 1e6;
        double mfs = 2.0 * n / tfs / 1e6;
        double mds = 2.0 * n / tds / 1e6;

        size_t bytes = n * 2 * sizeof(float);
        const char* region = bytes <= 32*1024 ? "L1" : bytes <= 256*1024 ? "L2" : bytes <= 6*1024*1024 ? "L3" : "RAM";

        printf("%9zu %6s | %8.0f %8.0f %8.0f %8.0f %8.0f | %7.2fx %7.2fx %7.2fx %7.2fx\n",
               n, region, mb, mf, md, mfs, mds,
               mb > 0 ? mf/mb : 0, mb > 0 ? md/mb : 0, mb > 0 ? mfs/mb : 0, mb > 0 ? mds/mb : 0);
        free(a); free(b);
    }
    printf("\n(sink=%g)\n", (double)sink);
    return 0;
}
