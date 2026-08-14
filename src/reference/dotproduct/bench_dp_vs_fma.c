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
 * bench_dp_vs_fma.c —— 严格对比 dp 版(dot_avx_v2) vs FMA 版(dot_avx)。
 *
 * 方法学（尽量消噪、公平）：
 *   - min-of-repeats：每个 n 对每个函数重复测 REPEAT 次，取「最短耗时」
 *     （最少受调度/中断/降频干扰，代表稳定峰值）。
 *   - 交错测量：同一轮里紧挨着测 fma / dp / basic，尽量让三者处于同样热状态。
 *   - n 覆盖 L1(32KB)→L2(256KB)→L3(6MB)→RAM 全谱，看清 compute-bound 与
 *     memory-bound 两种区间。
 *   - 用返回值做累加防止编译器把整段计算优化掉。
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

/* 天真 FMA：单累加器 + 强制 precise（禁止编译器重结合成多累加器），
 * 使其退回真正的串行依赖链、被 FMA 的 4 周期延迟卡住（latency-bound）。
 * 这是「被刻意削弱的 FMA」基线，用来展示 dp 唯一能赢的场景。 */
#include <immintrin.h>
#pragma float_control(precise, on, push)
static float dot_fma1(const float* a, const float* b, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1), lo = _mm256_castps256_ps128(acc);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(1,1,1,1)));
    float r = _mm_cvtss_f32(s);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
}
#pragma float_control(pop)

/* 返回该函数在 n 上 min-of-repeats 的耗时(秒/次调用) */
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
    printf("min-of-%d, Mop/s = 2*n*iters/t\n\n", repeat);
    printf("%9s %6s | %9s %11s %9s | %8s %9s\n",
           "n", "region", "fma4(优)", "fma1(precise)", "dp(v2)", "dp/fma4", "dp/fma1");
    printf("-------------------------------------------------------------------------------\n");

    for (int k = 0; k < nN; k++) {
        size_t n = ns[k];
        float* a = (float*)malloc(n * sizeof(float));
        float* b = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) { a[i] = rnd(); b[i] = rnd(); }

        /* iters 随 n 缩放，保证每次测量总工作量足够大 */
        int iters = (int)(80000000ull / n);
        if (iters < 20) iters = 20;

        /* warmup */
        for (int w = 0; w < 3; w++) { sink += dot_fma1(a,b,n); sink += dot_avx(a,b,n); sink += dot_avx_v2(a,b,n); }

        double tf4 = bench(dot_avx,    a, b, n, iters, repeat, &sink);
        double tf1 = bench(dot_fma1,   a, b, n, iters, repeat, &sink);
        double td  = bench(dot_avx_v2, a, b, n, iters, repeat, &sink);

        double mf4 = 2.0 * n / tf4 / 1e6;
        double mf1 = 2.0 * n / tf1 / 1e6;
        double md  = 2.0 * n / td  / 1e6;

        size_t bytes = n * 2 * sizeof(float);
        const char* region = bytes <= 32*1024 ? "L1" : bytes <= 256*1024 ? "L2" : bytes <= 6*1024*1024 ? "L3" : "RAM";

        printf("%9zu %6s | %9.0f %11.0f %9.0f | %7.2fx %8.2fx\n",
               n, region, mf4, mf1, md, mf4 > 0 ? md / mf4 : 0, mf1 > 0 ? md / mf1 : 0);
        free(a); free(b);
    }
    printf("\n(sink=%g)\n", (double)sink);
    return 0;
}
