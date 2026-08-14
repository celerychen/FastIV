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

#include "softmax.h"
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

/* CPU 是否支持 AVX2（GCC/Clang 内置；其它编译器默认当支持） */
static int has_avx2(void) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
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
 * 对单个版本计时：取所有重复耗时的算术平均（avg）。
 * 返回 avg 耗时（秒）。
 */
static double bench_one(const char* name,
                        void (*fn)(float*, size_t),
                        const float* src,
                        size_t n,
                        int iters) {
    float* buf = (float*)malloc(n * sizeof(float));
    double total = 0.0;
    for (int it = 0; it < iters; it++) {
        memcpy(buf, src, n * sizeof(float));
        double t0 = now_sec();
        fn(buf, n);
        double t1 = now_sec();
        total += (t1 - t0);
    }
    free(buf);
    double avg = total / (double)iters;
    double mops = (double)n / avg / 1e6;
    printf("  %-12s  avg=%9.3f us   %8.2f Mop/s\n", name, avg * 1e6, mops);
    return avg;
}

int main(void) {
    /* 把控制台输出代码页切到 UTF-8，避免中文乱码（源码按 UTF-8 编译） */
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
#endif
    printf("CPU AVX2 support: %s\n",
           has_avx2() ? "yes" : "no (avx2 run may crash on this machine)");

    size_t sizes[] = {1024, 4096, 65536, 262144};
    int    iters[] = {2000, 1000, 300, 100};

    for (int s = 0; s < 4; s++) {
        size_t n = sizes[s];
        int it = iters[s];
        float* src = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) src[i] = rnd();

        printf("\n--- n = %zu, iters = %d ---\n", n, it);
        double t_basic = bench_one("basic",      softmax_basic,     src, n, it);
        double t_avx   = bench_one("avx_v1",     softmax_avx_v1,    src, n, it);
        double t_avx2  = bench_one("avx_v1_2",   softmax_avx_v1_2,  src, n, it);
        double t_avx3  = bench_one("avx_v1_3",   softmax_avx_v1_3,  src, n, it);
        double t_avx4  = bench_one("avx_v1_4",   softmax_avx_v1_4,  src, n, it);
        printf("  speedup avx_v1/basic    = %.2fx\n",   t_basic / t_avx);
        printf("  speedup avx_v1_2/basic  = %.2fx\n",   t_basic / t_avx2);
        printf("  speedup avx_v1_3/basic  = %.2fx\n",   t_basic / t_avx3);
        printf("  speedup avx_v1_4/basic  = %.2fx\n",   t_basic / t_avx4);
        printf("  avx_v1_2 / avx_v1       = %.2fx   (双向量交错 exp vs 单向量)\n", t_avx / t_avx2);
        printf("  avx_v1_3 / avx_v1       = %.2fx   (系数打包+vpermilps vs 单向量)\n", t_avx / t_avx3);
        printf("  avx_v1_4 / avx_v1_2     = %.2fx   (双向量交错+系数打包 vs 仅双向量交错)\n", t_avx2 / t_avx4);

        free(src);
    }
    return 0;
}
