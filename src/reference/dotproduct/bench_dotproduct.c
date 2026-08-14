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

#include "dotproduct.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* 计时：Windows 用 QueryPerformanceCounter，否则用 clock_gettime(CLOCK_MONOTONIC) */
#ifdef _WIN32
#include <windows.h>
static double now_sec(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
#endif

/* CPU 是否支持 AVX2+FMA（GCC/Clang 内置；其它编译器默认当支持） */
static int has_fma(void) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 1;
#endif
}

static unsigned int rng_state = 0x12345678u;
static float rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state & 0xffffffu) / (float)(1u << 24)) * 20.0f - 10.0f;
}

/*
 * 对单个版本计时：先 1 次 warmup（避免冷缓存/分支预测抖动），
 * 再取所有重复耗时的算术平均（avg）。每个 FMA = 一个 a[i]*b[i]，故
 * 以 n 计 throughput，单位 Mop/s。
 */
static double bench_one(const char* name,
                        float (*fn)(const float*, const float*, size_t),
                        const float* a, const float* b,
                        size_t n, int iters) {
    fn(a, b, n);                       /* warmup */
    double total = 0.0;
    for (int it = 0; it < iters; it++) {
        double t0 = now_sec();
        float r = fn(a, b, n);
        double t1 = now_sec();
        total += (t1 - t0);
        (void)r;                       /* 防优化掉调用 */
    }
    double avg = total / (double)iters;
    double mops = (double)n / avg / 1e6;
    printf("  %-10s  avg=%9.3f us   %8.2f Mop/s\n", name, avg * 1e6, mops);
    return avg;
}

int main(void) {
    /* 把控制台输出代码页切到 UTF-8，避免中文乱码（源码按 UTF-8 编译） */
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    printf("CPU AVX2+FMA support: %s\n",
           has_fma() ? "yes" : "no (avx run may crash on this machine)");

    size_t sizes[] = {1024, 4096, 65536, 262144};
    int    iters[] = {2000, 1000, 300, 100};

    for (int s = 0; s < 4; s++) {
        size_t n = sizes[s];
        int it = iters[s];
        float* a = (float*)malloc(n * sizeof(float));
        float* b = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) { a[i] = rnd(); b[i] = rnd(); }

        printf("\n--- n = %zu, iters = %d ---\n", n, it);
        double t_basic = bench_one("basic",    dot_basic,  a, b, n, it);
        double t_avx   = bench_one("avx",      dot_avx,    a, b, n, it);
        double t_avx2  = bench_one("avx_v2",   dot_avx_v2, a, b, n, it);
        printf("  speedup avx/basic    = %.2fx\n",      t_basic / t_avx);
        printf("  speedup avx_v2/basic = %.2fx   (DPPS 版)\n", t_basic / t_avx2);
        printf("  avx_v2 / avx        = %.2fx   (DPPS vs FMA)\n", t_avx / t_avx2);

        free(a); free(b);
    }
    return 0;
}
