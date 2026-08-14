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
 * conv2d_3x1_1x3_v1 基准 —— 严格遵循 conv2d/conv2d_bench.c (bench_o2) 的规范:
 *   - 尺寸 {100,500,1000,2000,3000,4000,5000}, NxN, stride=N
 *   - 64 字节对齐缓冲 (posix_memalign)
 *   - 种子 srand(12345+N); src 用 rand/RAND_MAX 填充 (与 bench_o2 同序列)
 *   - 每个函数先预热一次
 *   - 自动校准迭代数使单次测量 >= 50ms
 *   - best-of-5 取最小单次耗时, 排除大小核迁移/频率调节噪声
 *   - 输出含 ms / Mops / 加速比 / v1 相对 v0 的 max|Δ| 正确性校验
 * 差异: 可分离 v1 仅对可分离核成立, 故核由 H[3]⊗V[3] 生成; v0 用完整 9 系数对照。
 *        v0 每像素 9 次乘加, v1 每像素 6 次乘加 (3 行 + 3 列), Mops 分别按 9/6 计。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

typedef float ivf32;
void conv2d_v0(ivf32* dst, int wd, int hd, int sd, ivf32* src, int ws, int hs, int ss, ivf32 coef[9]);
void conv2d_3x1_1x3_v1(ivf32* dst, int wd, int hd, int sd, ivf32* src, int ws, int hs, int ss, ivf32 coef[6]);

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static volatile double sink = 0.0;

/* coef[9] 与 coef[6] 作函数参数都退化为 ivf32*, 故可用同一函数指针类型计时两套实现 */
typedef void (*fn_t)(ivf32*, int, int, int, ivf32*, int, int, int, ivf32*);

static double bench(fn_t fn, ivf32* dst, int wd, int hd, int sd,
                    ivf32* src, int ws, int hs, int ss, ivf32* coef, int iters) {
    double t0 = now();
    for (int k = 0; k < iters; k++) {
        fn(dst, wd, hd, sd, src, ws, hs, ss, coef);
        sink += dst[0];          /* 防止编译器把调用优化掉 */
    }
    double t1 = now();
    return (t1 - t0) / iters;
}

/* best-of-5：取多次测量的最小单次耗时（微基准里最小值最接近内核真实吞吐） */
static double bench_best(fn_t fn, ivf32* dst, int wd, int hd, int sd,
                         ivf32* src, int ws, int hs, int ss, ivf32* coef, int iters, int trials) {
    double best = 1e300;
    for (int t = 0; t < trials; t++) {
        double per = bench(fn, dst, wd, hd, sd, src, ws, hs, ss, coef, iters);
        if (per < best) best = per;
    }
    return best;
}

int main(void) {
    int sizes[] = {100, 500, 1000, 2000, 3000, 4000, 5000};
    int ns = sizeof(sizes) / sizeof(sizes[0]);

    printf("# conv2d_3x1_1x3_v1 性能对比 (NxN 同尺寸, stride=N, 可分离核 H(1x3)⊗V(3x1), Mops: v0=9次乘加/像素, v1=6次乘加/像素)\n");
    printf("# 编译: %s\n", __OPTIMIZE__ ? "已优化(-O2)" : "未优化");
    printf("%-6s | %9s %9s | %11s %11s | %8s | %12s\n",
           "N", "v0(ms)", "v1(ms)", "v0(Mops)", "v1(Mops)", "v1/v0", "v1 max|Δ|");
    printf("-------+-----------+-----------+-------------+-------------+----------+------------\n");

    for (int s = 0; s < ns; s++) {
        int N = sizes[s];
        int Ws = N, Hs = N, Wd = N, Hd = N, ss = N, sd = N;
        size_t n = (size_t)N * N;
        ivf32 *src = NULL, *dst = NULL, *cd0 = NULL, *cd1 = NULL;
        if (posix_memalign((void**)&src, 64, n * sizeof(ivf32)) ||
            posix_memalign((void**)&dst, 64, n * sizeof(ivf32)) ||
            posix_memalign((void**)&cd0, 64, n * sizeof(ivf32)) ||
            posix_memalign((void**)&cd1, 64, n * sizeof(ivf32))) {
            fprintf(stderr, "alloc failed\n"); return 1;
        }
        ivf32 coef9[9];
        ivf32 coef6[6];

        srand(12345 + N);
        for (size_t k = 0; k < n; k++) src[k] = (ivf32)rand() / RAND_MAX;
        /* 生成可分离核: H[3], V[3] 来自同一随机流, coef9 = V⊗H, coef6 = [H, V] */
        ivf32 H[3], V[3];
        for (int k = 0; k < 3; k++) H[k] = (ivf32)rand() / RAND_MAX * 2 - 1;
        for (int k = 0; k < 3; k++) V[k] = (ivf32)rand() / RAND_MAX * 2 - 1;
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++) coef9[a * 3 + b] = V[a] * H[b];
        for (int b = 0; b < 3; b++) coef6[b] = H[b];
        for (int a = 0; a < 3; a++) coef6[3 + a] = V[a];

        /* 预热：每个函数先各调用一次(写入同一 dst)，排除首触碰冷缓存 */
        conv2d_v0(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef9);
        conv2d_3x1_1x3_v1(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef6);

        /* 校准迭代数，使单次测量总时长 >= 50ms */
        int it0 = 1, it1 = 1;
        double per0 = bench(conv2d_v0, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef9, it0);
        while (per0 * it0 < 0.05 && it0 < (1 << 22)) { it0 *= 2; per0 = bench(conv2d_v0, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef9, it0); }
        double per1 = bench(conv2d_3x1_1x3_v1, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef6, it1);
        while (per1 * it1 < 0.05 && it1 < (1 << 22)) { it1 *= 2; per1 = bench(conv2d_3x1_1x3_v1, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef6, it1); }

        /* best-of-5：用校准好的迭代数各测 5 轮取最小 */
        per0 = bench_best(conv2d_v0, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef9, it0, 5);
        per1 = bench_best(conv2d_3x1_1x3_v1, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef6, it1, 5);

        double mops0 = (double)n * 9.0 / per0 / 1e6;
        double mops1 = (double)n * 6.0 / per1 / 1e6;
        double sp = per0 / per1;

        /* 正确性校验：v1 与 v0 在独立缓冲上逐元素最大绝对差 */
        conv2d_v0(cd0, Wd, Hd, sd, src, Ws, Hs, ss, coef9);
        conv2d_3x1_1x3_v1(cd1, Wd, Hd, sd, src, Ws, Hs, ss, coef6);
        double maxd = 0.0;
        for (size_t k = 0; k < n; k++) {
            double d = fabs((double)cd0[k] - (double)cd1[k]);
            if (d > maxd) maxd = d;
        }

        printf("%-6d | %9.3f %9.3f | %11.1f %11.1f | %7.2fx | %12.2e\n",
               N, per0 * 1e3, per1 * 1e3, mops0, mops1, sp, maxd);

        free(src); free(dst); free(cd0); free(cd1);
    }
    printf("sink=%f (防DCE)\n", sink);
    return 0;
}
