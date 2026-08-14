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
#include <math.h>
#include <time.h>
#include <stdint.h>

typedef float ivf32;
void conv2d_v0(ivf32* dst, int wd, int hd, int sd, ivf32* src, int ws, int hs, int ss, ivf32 coef[9]);
void conv2d_v1(ivf32* dst, int wd, int hd, int sd, ivf32* src, int ws, int hs, int ss, ivf32 coef[9]);
void conv2d_v2(ivf32* dst, int wd, int hd, int sd, ivf32* src, int ws, int hs, int ss, ivf32 coef[9]);
void conv2d_v3(ivf32* dst, int wd, int hd, int sd, ivf32* src, int ws, int hs, int ss, ivf32 coef[9]);
void conv2d_v4(ivf32* dst, int wd, int hd, int sd, ivf32* src, int ws, int hs, int ss, ivf32 coef[9]);
void conv2d_v5(ivf32* dst, int wd, int hd, int sd, ivf32* src, int ws, int hs, int ss, ivf32 coef[9]);

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static volatile double sink = 0.0;

/* 返回单次调用平均耗时(秒)。iters 由外部校准得到。 */
static double bench(void (*fn)(ivf32*,int,int,int,ivf32*,int,int,int,ivf32[9]),
                    ivf32* dst, int wd, int hd, int sd,
                    ivf32* src, int ws, int hs, int ss, ivf32 coef[9],
                    int iters)
{
    double t0 = now();
    for (int k = 0; k < iters; k++) {
        fn(dst, wd, hd, sd, src, ws, hs, ss, coef);
        sink += dst[0];          /* 防止编译器把调用优化掉 */
    }
    double t1 = now();
    return (t1 - t0) / iters;
}

/* best-of-K：取多次测量的最小单次耗时，排除频率调节/大小核迁移等调度噪声。
   微基准里最小值最接近内核真实吞吐(噪声只会让测量变慢，不会变快)。 */
static double bench_best(void (*fn)(ivf32*,int,int,int,ivf32*,int,int,int,ivf32[9]),
                         ivf32* dst, int wd, int hd, int sd,
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

int main(void) {
    int sizes[] = {100, 500, 1000, 2000, 3000, 4000, 5000};
    int ns = sizeof(sizes) / sizeof(sizes[0]);

    printf("# conv2d 性能对比 (NxN 同尺寸, stride=N, 3x3, Mops=9次乘加/像素)\n");
    printf("# 编译: %s\n", __OPTIMIZE__ ? "已优化(-O2/-O3)" : "未优化");
    printf("%-6s | %9s %9s %9s %9s %9s %9s | %11s %11s %11s %11s %11s %11s | %8s %8s %8s %8s %8s | %10s\n",
           "N", "v0(ms)", "v1(ms)", "v2(ms)", "v3(ms)", "v4(ms)", "v5(ms)",
           "v0(Mops)", "v1(Mops)", "v2(Mops)", "v3(Mops)", "v4(Mops)", "v5(Mops)",
           "v1/v0", "v2/v0", "v3/v2", "v4/v2", "v5/v4", "v5 max|Δ|");
    printf("-------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+-------------+-------------+-------------+-------------+-------------+----------+----------+----------+----------+----------+----------\n");

    for (int s = 0; s < ns; s++) {
        int N = sizes[s];
        int Ws = N, Hs = N, Wd = N, Hd = N, ss = N, sd = N;
        size_t n = (size_t)N * N;
        ivf32 *src = NULL, *dst = NULL, *cd0 = NULL, *cd2 = NULL, *cd3 = NULL, *cd5 = NULL;
        /* 关键：v0/v1/v2/v3/v4 计时共用同一块 64 字节对齐的 dst 缓冲，
           这样 src↔dst 的内存关系对三者完全一致，只剩代码差异——
           消除之前 v1(dst1)/v2(dst2) 偏移不同导致的 cache 组冲突/伪别名噪声。 */
        if (posix_memalign((void**)&src, 64, n * sizeof(ivf32)) ||
            posix_memalign((void**)&dst, 64, n * sizeof(ivf32)) ||
            posix_memalign((void**)&cd0, 64, n * sizeof(ivf32)) ||
            posix_memalign((void**)&cd2, 64, n * sizeof(ivf32)) ||
            posix_memalign((void**)&cd3, 64, n * sizeof(ivf32)) ||
            posix_memalign((void**)&cd5, 64, n * sizeof(ivf32))) {
            fprintf(stderr, "alloc failed\n"); return 1;
        }
        ivf32 coef[9];

        srand(12345 + N);
        for (size_t k = 0; k < n; k++) src[k] = (ivf32)rand() / RAND_MAX;
        for (int k = 0; k < 9; k++) coef[k] = (ivf32)rand() / RAND_MAX * 2 - 1;

        /* 预热：每个函数先各调用一次(写入同一 dst)，把首触碰(page fault/冷缓存)
           排除出计时，否则最后计时的函数会被首次页错误拖慢，产生假性掉速。 */
        conv2d_v0(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        conv2d_v1(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        conv2d_v2(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        conv2d_v3(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        conv2d_v4(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        conv2d_v5(dst, Wd, Hd, sd, src, Ws, Hs, ss, coef);

        /* 校准迭代数，使单次测量总时长 >= 50ms */
        int it0 = 1, it1 = 1, it2 = 1, it3 = 1, it4 = 1, it5 = 1;
        double per0 = bench(conv2d_v0, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it0);
        while (per0 * it0 < 0.05 && it0 < (1 << 22)) { it0 *= 2; per0 = bench(conv2d_v0, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it0); }
        double per1 = bench(conv2d_v1, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it1);
        while (per1 * it1 < 0.05 && it1 < (1 << 22)) { it1 *= 2; per1 = bench(conv2d_v1, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it1); }
        double per2 = bench(conv2d_v2, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it2);
        while (per2 * it2 < 0.05 && it2 < (1 << 22)) { it2 *= 2; per2 = bench(conv2d_v2, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it2); }
        double per3 = bench(conv2d_v3, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it3);
        while (per3 * it3 < 0.05 && it3 < (1 << 22)) { it3 *= 2; per3 = bench(conv2d_v3, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it3); }
        double per4 = bench(conv2d_v4, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it4);
        while (per4 * it4 < 0.05 && it4 < (1 << 22)) { it4 *= 2; per4 = bench(conv2d_v4, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it4); }
        double per5 = bench(conv2d_v5, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it5);
        while (per5 * it5 < 0.05 && it5 < (1 << 22)) { it5 *= 2; per5 = bench(conv2d_v5, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it5); }

        /* best-of-5：用校准好的迭代数各测 5 轮取最小，消除调度/频率噪声 */
        per0 = bench_best(conv2d_v0, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it0, 5);
        per1 = bench_best(conv2d_v1, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it1, 5);
        per2 = bench_best(conv2d_v2, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it2, 5);
        per3 = bench_best(conv2d_v3, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it3, 5);
        per4 = bench_best(conv2d_v4, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it4, 5);
        per5 = bench_best(conv2d_v5, dst, Wd, Hd, sd, src, Ws, Hs, ss, coef, it5, 5);

        double mops0 = (double)n * 9.0 / per0 / 1e6;
        double mops1 = (double)n * 9.0 / per1 / 1e6;
        double mops2 = (double)n * 9.0 / per2 / 1e6;
        double mops3 = (double)n * 9.0 / per3 / 1e6;
        double mops4 = (double)n * 9.0 / per4 / 1e6;
        double mops5 = (double)n * 9.0 / per5 / 1e6;
        double sp1 = per0 / per1;
        double sp2 = per0 / per2;
        double sp32 = per2 / per3;   /* v3 相对 v2 的加速比(>1 表示 v3 更快) */
        double sp42 = per2 / per4;   /* v4 相对 v2 的加速比(>1 表示 v4 更快) */
        double sp54 = per4 / per5;   /* v5 相对 v4 的加速比(>1 表示 v5 更快) */

        /* 正确性校验：用独立 scratch 缓冲，v5 与 v0 逐元素最大绝对差 */
        conv2d_v0(cd0, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        conv2d_v5(cd5, Wd, Hd, sd, src, Ws, Hs, ss, coef);
        double maxd = 0.0;
        for (size_t k = 0; k < n; k++) {
            double d = fabs((double)cd0[k] - (double)cd5[k]);
            if (d > maxd) maxd = d;
        }

        printf("%-6d | %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f | %11.1f %11.1f %11.1f %11.1f %11.1f %11.1f | %7.2fx %7.2fx %7.2fx %7.2fx %7.2fx | %10.2e\n",
               N, per0 * 1e3, per1 * 1e3, per2 * 1e3, per3 * 1e3, per4 * 1e3, per5 * 1e3,
               mops0, mops1, mops2, mops3, mops4, mops5, sp1, sp2, sp32, sp42, sp54, maxd);

        free(src); free(dst); free(cd0); free(cd2); free(cd3); free(cd5);
    }
    printf("sink=%f (防DCE)\n", sink);
    return 0;
}
