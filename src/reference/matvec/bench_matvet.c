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
 * bench_matvet.c - 性能对比基准 (独立于正确性测试)
 *
 * 计时实现选择:
 *    _WIN32   -> QueryPerformanceCounter (高精度, 子微秒)
 *    其它      -> clock_gettime(CLOCK_MONOTONIC)
 *
 * 对比三种实现的吞吐 (GFLOP/s, 1 FMA = 2 FLOP):
 *   mat_mul_vet_real32      (纯 C)
 *   mat_mul_vet_real32_avx  (AVX2+FMA, 仅 __AVX2__ 编译时)
 *   mat_mul_vet_real32_neon (NEON, 仅 ARM 目标)
 *   以及对应的 mat^T · vec 版本。
 *
 * 本机 x86 下 NEON 自动跳过; 需在 ARM 目标编译才得到 NEON 数据。
 */

#include "matvet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef IVF32_DEFINED
typedef float ivf32;
#endif

#if defined(_WIN32)
#include <windows.h>
static double now_sec(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

/* 所有实现签名统一, 用函数指针做泛型计时 */
typedef void (*matmul_fn)(ivf32*, const ivf32*, int, int, int, const ivf32*);

/* 计时: 先 warmup 1 次, 再跑 iters 次, 返回单次平均毫秒 */
static double bench_run(matmul_fn fn, ivf32* dst, const ivf32* mat,
                        int rows, int cols, int stride, const ivf32* vec, int iters)
{
    fn(dst, mat, rows, cols, stride, vec);          /* warmup, 也可能触发页错误 */
    double t0 = now_sec();
    for (int k = 0; k < iters; k++)
        fn(dst, mat, rows, cols, stride, vec);
    double t1 = now_sec();
    return (t1 - t0) / iters * 1e3;                 /* ms / call */
}

static void bench_variant(const char* label, matmul_fn fn,
                          int rows, int cols, int stride, int iters,
                          const ivf32* mat, const ivf32* vec, int dsize)
{
    ivf32* dst = (ivf32*)malloc((size_t)dsize * sizeof(ivf32));
    double ms = bench_run(fn, dst, mat, rows, cols, stride, vec, iters);
    double flops_per_call = 2.0 * (double)rows * (double)cols;   /* 1 FMA = 2 FLOP */
    double gflops = flops_per_call / (ms * 1e-3) / 1e9;
    printf("  %-12s rows=%-5d cols=%-5d stride=%-5d  %9.4f ms/call  %8.2f GFLOP/s\n",
           label, rows, cols, stride, ms, gflops);
    free(dst);
}

int main(void)
{
    int sizes[][2] = { {512,512}, {1024,1024}, {2048,2048}, {4096,4096} };
    int ns = sizeof(sizes) / sizeof(sizes[0]);
    int iters = 200;

    printf("=== benchmark: dst = mat · vec ===\n");
    for (int s = 0; s < ns; s++)
    {
        int rows = sizes[s][0], cols = sizes[s][1], stride = cols;
        int cap = rows * stride;
        ivf32* mat = (ivf32*)malloc((size_t)cap * sizeof(ivf32));
        ivf32* vec = (ivf32*)malloc((size_t)cols * sizeof(ivf32));
        for (int k = 0; k < cap;  k++) mat[k] = (ivf32)((k % 7) - 3);
        for (int k = 0; k < cols; k++) vec[k] = (ivf32)((k % 5) - 2);

        bench_variant("C",        mat_mul_vet_real32,      rows, cols, stride, iters, mat, vec, rows);
#ifdef __AVX2__
        bench_variant("AVX2+FMA", mat_mul_vet_real32_avx,  rows, cols, stride, iters, mat, vec, rows);
#endif
#ifdef __ARM_NEON
        bench_variant("NEON",     mat_mul_vet_real32_neon, rows, cols, stride, iters, mat, vec, rows);
#endif
        free(mat); free(vec);
    }

    printf("\n=== benchmark: dst = mat^T · vec ===\n");
    for (int s = 0; s < ns; s++)
    {
        int rows = sizes[s][0], cols = sizes[s][1], stride = cols;
        int cap = rows * stride;
        ivf32* mat = (ivf32*)malloc((size_t)cap * sizeof(ivf32));
        ivf32* vec = (ivf32*)malloc((size_t)rows * sizeof(ivf32));
        for (int k = 0; k < cap;  k++) mat[k] = (ivf32)((k % 7) - 3);
        for (int k = 0; k < rows; k++) vec[k] = (ivf32)((k % 5) - 2);

        bench_variant("C",        mat_t_mul_vet_real32,      rows, cols, stride, iters, mat, vec, cols);
#ifdef __AVX2__
        bench_variant("AVX2+FMA", mat_t_mul_vet_real32_avx,  rows, cols, stride, iters, mat, vec, cols);
#endif
#ifdef __ARM_NEON
        bench_variant("NEON",     mat_t_mul_vet_real32_neon, rows, cols, stride, iters, mat, vec, cols);
#endif
        free(mat); free(vec);
    }

    printf("\nGFLOP/s = 2 * rows * cols / (ms_per_call / 1e3) / 1e9  (1 FMA 计 2 FLOP)\n");
#if defined(__AVX2__)
    printf("build: AVX2+FMA active\n");
#else
    printf("build: AVX2+FMA inactive (用 /arch:AVX2 重新编译可得 AVX 数据)\n");
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
    printf("build: NEON active\n");
#else
    printf("build: NEON inactive (x86 预期跳过, 需在 ARM 目标编译)\n");
#endif
    return 0;
}
