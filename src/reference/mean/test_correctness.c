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

/* test_correctness.c —— mean/variance 实现的正确性与数值精度测试
 *
 * 参考基准：long double（MinGW 上是 x87 80-bit，64 位尾数）两趟法 +
 * Kahan 补偿求和。相对 float32 的 24 位尾数有 40 位余量，可作 ground truth。
 *
 * 覆盖：
 *   - 25 组 n（重点打 8 的倍数 ±1，覆盖主循环/尾循环交界）
 *   - 10 种数据分布（含灾难性抵消、swamping、denormal 等病态输入）
 *   - 解析解校验（常量数组方差必须为 0；1..n 方差 = (n^2-1)/12）
 *   - 非对齐指针路径
 *   - 尾部保护页越界读检测（子进程模式）
 *
 * 用法：
 *   test_correctness.exe              跑全部测试
 *   test_correctness.exe --oob <impl> <n>   子进程模式，仅做越界探测
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

#include "mean_var.h"
#include "bench_util.h"

#define EPS_F32 1.1920928955078125e-7   /* 2^-23 */

/* ================================================================== */
/* 被测实现表                                                          */
/* ================================================================== */

typedef Stats (*impl_fn)(const float*, size_t);

typedef struct {
    const char* name;
    impl_fn     fn;
    int         expect_stable;   /* 是否期望它数值稳定 */
} Impl;

static Stats w_serial(const float* a, size_t n)        { return mean_variance_serial(a, n); }
static Stats w_avx2(const float* a, size_t n)          { return mean_variance_avx2_welford(a, n); }
static Stats w_sumsq(const float* a, size_t n)         { return mean_variance_scalar_sumsq(a, n); }
static Stats w_serial_av(const float* a, size_t n)     { return mean_variance_serial_autovec(a, n); }
static Stats w_avx2_o3(const float* a, size_t n)       { return mean_variance_avx2_welford_o3(a, n); }
static Stats w_opt_f(const float* a, size_t n)         { return mean_variance_avx2_welford_opt_f(a, n); }
static Stats w_opt_d(const float* a, size_t n)         { return mean_variance_avx2_welford_opt_d(a, n); }
static Stats w_fma1(const float* a, size_t n)          { return mean_variance_avx2_welford_fma(a, n); }

static Impl g_impls[] = {
    { "serial_welford",   w_serial,    1 },
    { "avx2_welford",     w_avx2,      1 },
    { "scalar_sumsq",     w_sumsq,     0 },
    { "serial_autovec",   w_serial_av, 1 },
    { "avx2_welford_o3",  w_avx2_o3,   1 },
    { "avx2_welford_opt_f", w_opt_f,   1 },
    { "avx2_welford_opt_d", w_opt_d,   1 },
    { "avx2_welford_fma1",  w_fma1,    1 },
};
#define NIMPL ((int)(sizeof(g_impls)/sizeof(g_impls[0])))

/* ================================================================== */
/* 高精度参考实现                                                      */
/* ================================================================== */

typedef struct {
    long double mean;
    long double var;
    long double kappa;    /* 条件数 ||x||_2 / sqrt(n*var) */
} Ref;

static Ref reference_stats(const float* a, size_t n)
{
    Ref r = { 0.0L, 0.0L, 1.0L };
    if (n == 0) return r;

    /* 第一趟：Kahan 求和得到均值 */
    long double s = 0.0L, c = 0.0L;
    for (size_t i = 0; i < n; i++) {
        long double y = (long double)a[i] - c;
        long double t = s + y;
        c = (t - s) - y;
        s = t;
    }
    long double mean = s / (long double)n;

    /* 第二趟：Kahan 求偏差平方和；同时累计 ||x||_2^2 */
    long double ss = 0.0L, c2 = 0.0L;
    long double nrm = 0.0L, c3 = 0.0L;
    for (size_t i = 0; i < n; i++) {
        long double x = (long double)a[i];
        long double d = x - mean;

        long double y = d * d - c2;
        long double t = ss + y;
        c2 = (t - ss) - y;
        ss = t;

        long double y3 = x * x - c3;
        long double t3 = nrm + y3;
        c3 = (t3 - nrm) - y3;
        nrm = t3;
    }

    r.mean = mean;
    r.var  = ss / (long double)n;
    r.kappa = (ss > 0.0L) ? sqrtl(nrm) / sqrtl(ss) : 1.0L;
    return r;
}

/* ================================================================== */
/* 数据生成器                                                          */
/* ================================================================== */

static uint64_t rng_state = 0x243F6A8885A308D3ULL;

static void rng_reset(void) { rng_state = 0x243F6A8885A308D3ULL; }

static uint64_t xorshift64(void)
{
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng_state = x;
    return x;
}

/* [0,1) 均匀 */
static double urand(void) { return (double)(xorshift64() >> 11) * (1.0 / 9007199254740992.0); }

/* 标准正态（Box-Muller） */
static double nrand(void)
{
    double u1 = urand(); if (u1 < 1e-300) u1 = 1e-300;
    double u2 = urand();
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

typedef void (*gen_fn)(float*, size_t);

typedef struct {
    const char* name;
    gen_fn      fn;
    const char* purpose;
} Gen;

static void g_zeros(float* a, size_t n)     { for (size_t i=0;i<n;i++) a[i]=0.0f; }
static void g_const(float* a, size_t n)     { for (size_t i=0;i<n;i++) a[i]=3.25f; }
static void g_ramp(float* a, size_t n)      { for (size_t i=0;i<n;i++) a[i]=(float)(i+1); }
static void g_uniform(float* a, size_t n)   { for (size_t i=0;i<n;i++) a[i]=(float)(urand()*2.0-1.0); }
static void g_normal(float* a, size_t n)    { for (size_t i=0;i<n;i++) a[i]=(float)nrand(); }
static void g_offset(float* a, size_t n)    { for (size_t i=0;i<n;i++) a[i]=(float)(1.0e6 + nrand()); }
static void g_altbig(float* a, size_t n)    { for (size_t i=0;i<n;i++) a[i]=(i&1)?-1.0e6f:1.0e6f; }
static void g_outlier(float* a, size_t n)   { for (size_t i=0;i<n;i++) a[i]=(float)nrand(); if(n) a[n/2]=1.0e7f; }
static void g_dynrange(float* a, size_t n)  { for (size_t i=0;i<n;i++) a[i]=(float)((i%3==0)?1e-20:((i%3==1)?1.0:1e20)); }
static void g_denorm(float* a, size_t n)    { for (size_t i=0;i<n;i++) a[i]=(float)(FLT_MIN*1e-3*(1.0+urand())); }

static Gen g_gens[] = {
    { "zeros",     g_zeros,    "除零 / NaN" },
    { "const",     g_const,    "解析解：方差必须为 0" },
    { "ramp",      g_ramp,     "解析解：方差 = (n^2-1)/12" },
    { "uniform",   g_uniform,  "常规路径" },
    { "normal",    g_normal,   "常规路径" },
    { "offset1e6", g_offset,   "灾难性抵消（mean=1e6, sigma=1）" },
    { "alt_pm1e6", g_altbig,   "符号抵消" },
    { "outlier",   g_outlier,  "M2 swamping" },
    { "dynrange",  g_dynrange, "大动态范围 1e-20 ~ 1e20" },
    { "denormal",  g_denorm,   "非规格化数慢路径" },
};
#define NGEN ((int)(sizeof(g_gens)/sizeof(g_gens[0])))

/* ================================================================== */
/* n 覆盖列表：重点打 8 的倍数 ±1                                      */
/* ================================================================== */

static const size_t g_ns[] = {
    0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33,
    63, 64, 65, 127, 128, 129, 1000, 1023, 1024, 1025, 100000
};
#define NNS ((int)(sizeof(g_ns)/sizeof(g_ns[0])))

/* ================================================================== */
/* 判定                                                                */
/* ================================================================== */

static double rel_err(long double got, long double ref)
{
    long double d = fabsl(got - ref);
    long double m = fabsl(ref);
    if (m < 1e-30L) return (double)(d < 1e-30L ? 0.0L : d);   /* ref≈0 时看绝对误差 */
    return (double)(d / m);
}

typedef struct {
    double worst_mean_err;
    double worst_var_err;
    char   worst_case[128];
    int    hard_fails;
    int    cases;
} ImplScore;

static ImplScore g_score[NIMPL];

static int g_total_fail = 0;

static void check_one(int ii, const char* gname, size_t n, int unaligned,
                      const float* data, const Ref* ref)
{
    Stats s = g_impls[ii].fn(data, n);
    ImplScore* sc = &g_score[ii];
    sc->cases++;

    int hard = 0;
    const char* reason = NULL;

    if (isnan(s.mean) || isnan(s.variance)) { hard = 1; reason = "NaN"; }
    else if (isinf(s.mean) || isinf(s.variance)) {
        /* dynrange 用例本身会溢出到 inf，参考值同样是 inf 才算正常 */
        if (!(isinf((double)ref->var) || isinf((double)ref->mean))) { hard = 1; reason = "Inf"; }
    }
    else if (s.variance < 0.0f) { hard = 1; reason = "负方差"; }

    double me = 0.0, ve = 0.0;
    if (!hard && n > 0) {
        me = rel_err((long double)s.mean, ref->mean);
        ve = rel_err((long double)s.variance, ref->var);
        if (!isinf((double)ref->var)) {
            if (me > sc->worst_mean_err) sc->worst_mean_err = me;
            if (ve > sc->worst_var_err) {
                sc->worst_var_err = ve;
                snprintf(sc->worst_case, sizeof(sc->worst_case), "%s/n=%zu%s",
                         gname, n, unaligned ? "/unaligned" : "");
            }
            /* 完全崩坏才算硬失败 */
            if (ve > 1.0 && ref->var > 1e-20L) { hard = 1; reason = "方差相对误差 > 100%"; }
            if (me > 1.0 && fabsl(ref->mean) > 1e-20L) { hard = 1; reason = "均值相对误差 > 100%"; }
        }
    }

    if (n == 0) {
        if (s.mean != 0.0f || s.variance != 0.0f) { hard = 1; reason = "n=0 未返回 {0,0}"; }
    }

    if (hard) {
        sc->hard_fails++;
        g_total_fail++;
        if (sc->hard_fails <= 3) {
            printf("  [FAIL] %-16s %-10s n=%-7zu%s  %s  (mean=%g var=%g, ref_mean=%.10Lg ref_var=%.10Lg)\n",
                   g_impls[ii].name, gname, n, unaligned ? " unaligned" : "",
                   reason ? reason : "?", s.mean, s.variance, ref->mean, ref->var);
        }
    }
}

/* 解析解专项校验 */
static void analytic_checks(void)
{
    printf("\n--- 解析解校验 ---\n");
    size_t ns[] = { 8, 9, 16, 17, 100, 1000, 1024 };
    for (int k = 0; k < (int)(sizeof(ns)/sizeof(ns[0])); k++) {
        size_t n = ns[k];
        float* a = bu_alloc_floats(n);

        /* 常量数组：方差必须恰为 0 */
        g_const(a, n);
        for (int ii = 0; ii < NIMPL; ii++) {
            Stats s = g_impls[ii].fn(a, n);
            if (s.variance != 0.0f) {
                printf("  [WARN] %-16s const n=%-5zu 方差非零: %g\n",
                       g_impls[ii].name, n, s.variance);
            }
        }

        /* 1..n：方差 = (n^2-1)/12 */
        g_ramp(a, n);
        double expect = ((double)n * (double)n - 1.0) / 12.0;
        for (int ii = 0; ii < NIMPL; ii++) {
            Stats s = g_impls[ii].fn(a, n);
            double e = fabs(s.variance - expect) / expect;
            if (e > 1e-4) {
                printf("  [WARN] %-16s ramp  n=%-5zu 方差=%.6f 期望=%.6f 相对误差=%.2e\n",
                       g_impls[ii].name, n, s.variance, expect, e);
            }
        }
        bu_free_floats(a);
    }
    printf("  常量数组方差为 0、等差数列方差 = (n^2-1)/12 —— 未列出者均通过\n");
}

/* ================================================================== */
/* 越界读检测：子进程 + 尾部保护页                                     */
/* ================================================================== */

static int oob_probe(int impl_index, size_t n)
{
    GuardedBuf gb;
    if (!bu_guarded_alloc(&gb, n)) return 3;
    for (size_t i = 0; i < n; i++) gb.data[i] = (float)(i % 17) * 0.5f;

    Stats s = g_impls[impl_index].fn(gb.data, n);
    BU_CONSUME_F(s.mean);
    BU_CONSUME_F(s.variance);

    bu_guarded_free(&gb);
    return 0;
}

static void run_oob_tests(const char* exe)
{
    printf("\n--- 越界读检测（数组末尾紧贴 PAGE_NOACCESS 保护页）---\n");
    size_t ns[] = { 1, 7, 8, 9, 15, 31, 33, 1000, 4095, 4096, 4097 };
    int nfail = 0;
    for (int ii = 0; ii < NIMPL; ii++) {
        int impl_fail = 0;
        for (int k = 0; k < (int)(sizeof(ns)/sizeof(ns[0])); k++) {
            char cmd[512];
            snprintf(cmd, sizeof(cmd), "\"%s\" --oob %d %zu >nul 2>&1", exe, ii, ns[k]);
            int rc = system(cmd);
            if (rc != 0) {
                printf("  [FAIL] %-16s n=%-6zu 越界访问保护页 (exit=0x%08X)\n",
                       g_impls[ii].name, ns[k], (unsigned)rc);
                impl_fail = 1;
                nfail++;
                g_total_fail++;
            }
        }
        printf("  %-16s %s\n", g_impls[ii].name, impl_fail ? "存在越界读" : "无越界读");
    }
    if (nfail == 0) printf("  全部实现均未越界读取\n");
}

/* ================================================================== */
/* 主流程                                                              */
/* ================================================================== */

int main(int argc, char** argv)
{
    /* 子进程探测模式 */
    if (argc >= 4 && strcmp(argv[1], "--oob") == 0) {
        int ii = atoi(argv[2]);
        size_t n = (size_t)strtoull(argv[3], NULL, 10);
        if (ii < 0 || ii >= NIMPL) return 4;
        return oob_probe(ii, n);
    }

    printf("=====================================================================\n");
    printf("  mean/variance 正确性与数值精度测试\n");
    printf("  参考基准: long double (x87 80-bit) 两趟法 + Kahan 补偿求和\n");
    printf("  被测实现: %d 个 | n 取值: %d 组 | 数据分布: %d 种 | 含非对齐路径\n",
           NIMPL, NNS, NGEN);
    printf("=====================================================================\n");

    memset(g_score, 0, sizeof(g_score));

    size_t maxn = 0;
    for (int k = 0; k < NNS; k++) if (g_ns[k] > maxn) maxn = g_ns[k];

    /* +1 用于非对齐测试 */
    float* buf = bu_alloc_floats(maxn + 1);

    printf("\n--- 逐用例扫描（仅打印失败项）---\n");

    for (int gi = 0; gi < NGEN; gi++) {
        rng_reset();
        for (int ni = 0; ni < NNS; ni++) {
            size_t n = g_ns[ni];

            /* 对齐路径 */
            g_gens[gi].fn(buf, n);
            Ref ref = reference_stats(buf, n);
            for (int ii = 0; ii < NIMPL; ii++)
                check_one(ii, g_gens[gi].name, n, 0, buf, &ref);

            /* 非对齐路径：从 buf+1 起算，考察 loadu */
            if (n > 0) {
                g_gens[gi].fn(buf, n + 1);
                Ref ref2 = reference_stats(buf + 1, n);
                for (int ii = 0; ii < NIMPL; ii++)
                    check_one(ii, g_gens[gi].name, n, 1, buf + 1, &ref2);
            }
        }
    }
    if (g_total_fail == 0) printf("  （无失败项）\n");

    bu_free_floats(buf);

    /* 汇总表 */
    printf("\n--- 精度汇总（相对误差最大值，越小越好）---\n");
    printf("  %-18s %-12s %-12s %-8s %s\n",
           "实现", "均值误差", "方差误差", "硬失败", "最差用例");
    printf("  %s\n", "------------------------------------------------------------------------------");
    for (int ii = 0; ii < NIMPL; ii++) {
        printf("  %-18s %-12.3e %-12.3e %-8d %s\n",
               g_impls[ii].name,
               g_score[ii].worst_mean_err,
               g_score[ii].worst_var_err,
               g_score[ii].hard_fails,
               g_score[ii].worst_case[0] ? g_score[ii].worst_case : "-");
    }

    analytic_checks();
    run_oob_tests(argv[0]);

    /* 分布说明 */
    printf("\n--- 数据分布说明 ---\n");
    for (int gi = 0; gi < NGEN; gi++)
        printf("  %-12s %s\n", g_gens[gi].name, g_gens[gi].purpose);

    printf("\n=====================================================================\n");
    if (g_total_fail == 0)
        printf("  结论: 全部通过，未发现硬失败\n");
    else
        printf("  结论: 共 %d 项硬失败，详见上方 [FAIL]\n", g_total_fail);
    printf("=====================================================================\n");

    return g_total_fail ? 1 : 0;
}
