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

/* bench.c —— mean/variance 各实现的性能对比
 *
 * 方法学要点：
 *   - 绑定到 P-core（Alder Lake 混合架构下不绑核数据完全不可用）
 *   - HIGH_PRIORITY_CLASS + THREAD_PRIORITY_HIGHEST
 *   - QPC 作墙钟，TSC 作周期计数，开跑前标定 TSC 实际频率
 *   - 每档规模自适应重复次数，保证单次测量 >= MIN_TIME_SEC
 *   - 丢弃预热，多轮取中位数，同时报告最小值与相对标准差
 *   - 内联汇编屏障阻止编译器消除调用 / 提出循环
 *   - 规模覆盖 L1(48KB) -> L2(1.25MB) -> L3(25MB) -> DRAM
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mean_var.h"
#include "bench_util.h"

#define MIN_TIME_SEC 0.10
#define TRIALS_SMALL 7
#define TRIALS_LARGE 3
#define LARGE_THRESHOLD (16u * 1024u * 1024u)

typedef Stats (*impl_fn)(const float*, size_t);

typedef struct {
    const char* name;
    const char* desc;
    impl_fn     fn;
} Impl;

static Stats w_serial(const float* a, size_t n)    { return mean_variance_serial(a, n); }
static Stats w_avx2(const float* a, size_t n)      { return mean_variance_avx2_welford(a, n); }
static Stats w_sumsq(const float* a, size_t n)     { return mean_variance_scalar_sumsq(a, n); }
static Stats w_serial_av(const float* a, size_t n) { return mean_variance_serial_autovec(a, n); }
static Stats w_avx2_o3(const float* a, size_t n)   { return mean_variance_avx2_welford_o3(a, n); }
static Stats w_opt_f(const float* a, size_t n)     { return mean_variance_avx2_welford_opt_f(a, n); }
static Stats w_opt_d(const float* a, size_t n)     { return mean_variance_avx2_welford_opt_d(a, n); }
static Stats w_fma1(const float* a, size_t n)      { return mean_variance_avx2_welford_fma(a, n); }

static Impl g_impls[] = {
    { "serial_welford",  "原始串行 Welford (-O2)",              w_serial    },
    { "avx2_welford",    "原始 AVX2 Welford (-O2 -mavx2)",      w_avx2      },
    { "scalar_sumsq",    "标量 sum/sumsq 单趟 (-O2, 参照基线)", w_sumsq     },
    { "serial_autovec",  "同源标量码 (-O3 -march=native)",      w_serial_av },
    { "avx2_welford_o3", "原始 AVX2 (-O3 -march=native)",       w_avx2_o3   },
    { "avx2_welford_opt_f", "优化 Welford: FMA+快速倒数+2路展开(float)", w_opt_f },
    { "avx2_welford_opt_d", "优化 Welford: double 累加器",       w_opt_d     },
    { "avx2_welford_fma1",  "单路 AVX2+FMA (隔离实验)",          w_fma1     },
};
#define NIMPL ((int)(sizeof(g_impls)/sizeof(g_impls[0])))

/* 规模扫描：清晰体现 L1 / L2 / L3 / DRAM 台阶 */
static const size_t g_sizes[] = {
    8, 32, 128, 512,
    2048, 8192, 12288,          /* 12288 floats = 48KB ≈ L1d */
    32768, 131072, 327680,      /* 327680 floats = 1.25MB ≈ L2 */
    1048576, 4194304, 6553600,  /* 6553600 floats = 25MB ≈ L3 */
    16777216, 33554432, 67108864, 134217728   /* -> DRAM, 最大 512MB */
};
#define NSIZE ((int)(sizeof(g_sizes)/sizeof(g_sizes[0])))

typedef struct {
    double ns_per_call;
    double ns_per_elem;
    double gb_s;
    double cyc_per_elem;
    double rsd;         /* 相对标准差 % */
    double best_ns;
} Meas;

/* 测一个 (实现, 规模) 组合 */
static Meas measure(impl_fn fn, const float* arr, size_t n, int trials)
{
    Meas m;
    memset(&m, 0, sizeof(m));

    /* --- 预热 --- */
    for (int i = 0; i < 3; i++) {
        const float* p = arr;
        BU_ESCAPE_PTR(p);
        Stats s = fn(p, n);
        BU_CONSUME_F(s.mean);
        BU_CONSUME_F(s.variance);
    }

    /* --- 估算单次耗时，决定重复次数 --- */
    int64_t q0 = bu_qpc();
    int probe = 0;
    do {
        const float* p = arr;
        BU_ESCAPE_PTR(p);
        Stats s = fn(p, n);
        BU_CONSUME_F(s.mean);
        BU_CONSUME_F(s.variance);
        probe++;
    } while (probe < 4 && bu_qpc_to_sec(bu_qpc() - q0) < 0.002);
    double per_call = bu_qpc_to_sec(bu_qpc() - q0) / probe;
    if (per_call <= 0.0) per_call = 1e-9;

    long long reps = (long long)(MIN_TIME_SEC / per_call);
    if (reps < 1) reps = 1;
    if (reps > 200000000LL) reps = 200000000LL;

    /* --- 正式测量 --- */
    double* samples = (double*)malloc(sizeof(double) * trials);
    double* cycsam  = (double*)malloc(sizeof(double) * trials);

    for (int t = 0; t < trials; t++) {
        BU_BARRIER();
        uint64_t c0 = bu_tsc();
        int64_t  t0 = bu_qpc();

        for (long long r = 0; r < reps; r++) {
            const float* p = arr;
            BU_ESCAPE_PTR(p);          /* 阻止把调用提出循环 */
            Stats s = fn(p, n);
            BU_CONSUME_F(s.mean);      /* 阻止结果被消除 */
            BU_CONSUME_F(s.variance);
        }

        int64_t  t1 = bu_qpc();
        uint64_t c1 = bu_tsc();
        BU_BARRIER();

        samples[t] = bu_qpc_to_sec(t1 - t0) / (double)reps;   /* 秒/次 */
        cycsam[t]  = (double)(c1 - c0) / (double)reps;        /* TSC ticks/次 */
    }

    /* 统计 */
    double best = samples[0];
    for (int t = 1; t < trials; t++) if (samples[t] < best) best = samples[t];

    double mean = 0.0;
    for (int t = 0; t < trials; t++) mean += samples[t];
    mean /= trials;
    double sd = 0.0;
    for (int t = 0; t < trials; t++) { double d = samples[t] - mean; sd += d * d; }
    sd = (trials > 1) ? sqrt(sd / (trials - 1)) : 0.0;

    double med  = bu_median(samples, trials);
    double medc = bu_median(cycsam, trials);

    m.ns_per_call  = med * 1e9;
    m.ns_per_elem  = med * 1e9 / (double)n;
    m.gb_s         = ((double)n * sizeof(float)) / med / 1e9;
    /* TSC 频率即为标定出的实际有效频率 */
    m.cyc_per_elem = medc / (double)n;
    m.rsd          = (mean > 0.0) ? (sd / mean * 100.0) : 0.0;
    m.best_ns      = best * 1e9;

    free(samples);
    free(cycsam);
    return m;
}

int main(void)
{
    printf("=====================================================================\n");
    printf("  mean/variance 性能基准测试\n");
    printf("=====================================================================\n");

    bu_calibrate();
    bu_pin_to_p_core();
    bu_boost_priority();

    printf("[bench] QPC 频率 = %.3f MHz | TSC 标定频率 = %.3f GHz\n",
           bu_qpc_freq / 1e6, bu_tsc_freq / 1e9);
    printf("[bench] 规模档位 %d 个，实现 %d 个\n\n", NSIZE, NIMPL);

    size_t maxn = g_sizes[NSIZE - 1];
    printf("[bench] 分配 %zu 个 float (%.0f MB) ...\n",
           maxn, (double)(maxn * sizeof(float)) / (1024.0 * 1024.0));
    float* arr = bu_alloc_floats(maxn);

    /* 正态分布数据，避免 denormal 慢路径干扰性能测量 */
    uint64_t st = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < maxn; i++) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        double u = (double)(st >> 11) * (1.0 / 9007199254740992.0);
        arr[i] = (float)(u * 2.0 - 1.0);
    }
    printf("[bench] 数据就绪，开始测量\n\n");

    FILE* csv = fopen("results.csv", "w");
    fprintf(csv, "n,bytes,impl,ns_per_call,ns_per_elem,gb_per_s,cycles_per_elem,rsd_pct,best_ns_per_call\n");

    for (int si = 0; si < NSIZE; si++) {
        size_t n = g_sizes[si];
        int trials = (n >= LARGE_THRESHOLD) ? TRIALS_LARGE : TRIALS_SMALL;
        double kb = (double)(n * sizeof(float)) / 1024.0;

        printf("--- n = %zu 个元素 (%.1f KB) ---\n", n, kb);
        printf("  %-18s %12s %10s %10s %10s %8s  %s\n",
               "实现", "ns/次", "ns/元素", "GB/s", "周期/元素", "RSD%", "加速比");
        printf("  %s\n",
               "---------------------------------------------------------------------------------");

        double base_ns = 0.0;
        for (int ii = 0; ii < NIMPL; ii++) {
            Meas m = measure(g_impls[ii].fn, arr, n, trials);
            if (ii == 0) base_ns = m.ns_per_call;

            double speedup = (m.ns_per_call > 0.0) ? base_ns / m.ns_per_call : 0.0;
            printf("  %-18s %12.1f %10.4f %10.2f %10.3f %8.1f  %6.2fx%s\n",
                   g_impls[ii].name, m.ns_per_call, m.ns_per_elem, m.gb_s,
                   m.cyc_per_elem, m.rsd, speedup,
                   (m.rsd > 5.0) ? "  <-噪声偏大" : "");

            fprintf(csv, "%zu,%zu,%s,%.4f,%.6f,%.4f,%.6f,%.2f,%.4f\n",
                    n, n * sizeof(float), g_impls[ii].name,
                    m.ns_per_call, m.ns_per_elem, m.gb_s,
                    m.cyc_per_elem, m.rsd, m.best_ns);
            fflush(csv);
        }
        printf("\n");
    }

    fclose(csv);
    bu_free_floats(arr);

    printf("=====================================================================\n");
    printf("  完成，原始数据已写入 results.csv\n");
    printf("  说明: 加速比以 serial_welford 为基准 1.00x\n");
    printf("=====================================================================\n");
    return 0;
}
