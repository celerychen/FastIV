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

/* mean_var_opt.c —— 优化版 Welford（算法本体与原 mean_var.c 完全一致）
 *
 * 核心原则：这里做的全部是「性能 / 数值工程」层面的优化，
 * Welford 在线递推公式（mean += delta/k; M2 += delta*(x-mean)）
 * 一行都没改。具体优化点：
 *
 * 1. 【FMA】_mm256_fmadd_*: 把乘加融合成单条指令，缩短
 *    loop-carried 依赖链，且 FMA 是正确舍入（比 mul+add 少一次舍入）。
 *
 * 2. 【FMA 抵消除法延迟】原实现热循环里 _mm256_div_ps(1, cnt) 延迟 ~13 周期。
 *    曾尝试用 _mm256_rcp_ps + 2 次牛顿迭代( r <- r*(2 - v*r) )替换它，
 *    但 float 牛顿在 cnt=1 处收敛到 1-2^-24(差 1 ULP 而非精确 1.0)，
 *    乘上大均值后把 M2 放大到灾难量级(offset1e6 方差从 0.87 算成 62500)。
 *    故保留正确舍入的 _mm256_div_ps（整数 cnt 下 IEEE 除法即正确舍入的 1/cnt），
 *    仅以 FMA + 2 路展开兑现提速。这印证了「优化 Welford 不能牺牲除法精度」。
 *
 * 3. 【2 路独立累加器展开】两组 8 宽 Welford 状态 (m0/M0/c0) 与
 *    (m1/M1/c1) 各管一半数据，打断依赖链、提升乱序调度的 ILP。
 *    末尾用 welford_merge 把两段部分统计合并（Chan 并行算法，仍是 Welford）。
 *
 * 4. 【size_t 计数】修复原实现 int 计数在 n >= 2^31 时有符号溢出 UB；
 *    合并权重用 double 计算后再转 float，比原版 (float)(n1*n2/total)
 *    更精确（原版在 n 大时 n1*n2 已超出 float 精确整数区间）。
 *
 * 5. 【运行时 CPU 分发】mean_variance_dispatch 用 __builtin_cpu_supports
 *    检测 AVX2/FMA，缺失时回退标量 Welford，避免直接 SIGILL。
 *
 * 6. 【double 累加器变体】mean_variance_avx2_welford_opt_d 把 mean/M2
 *    放在 __m256d 里累加（仍是同一套递推），只在入口把 float 转 double、
 *    出口转回 float。这修复了「均值远大于方差」时 (x-mean) 在 float 下
 *    的灾难性抵消——但请注意：这是 Welford 在更高精度下的同一算法，
 *    不是换算法。接口返回 float，所以绝对溢出（如 dynrange 方差 2.5e39）
 *    仍会 Inf。
 *
 * 与原 mean_var.c 的唯一「算法」差异：
 *   - 原版是单路 8 宽交错 Welford；本版是 2 路 16 宽交错 + 合并。
 *     两者数学等价（都是把全集拆成不相交子集各自 Welford 再合并），
 *     数值上仅因合并顺序不同而有 < 1 ulp 量级的微小差异，属于同类算法。
 */

#include "mean_var.h"
#include <immintrin.h>

/* 部分统计：与原版 BlockStat 同义，但计数用 size_t 杜绝溢出 */
typedef struct {
    float  mean;
    float  M2;
    size_t cnt;
} BlockStat;

/* 与原始 welford_merge 等价；合并权重用 double 计算以保精度 */
static inline void welford_merge(BlockStat* restrict a, const BlockStat* restrict b)
{
    if (b->cnt == 0) return;
    if (a->cnt == 0) { *a = *b; return; }
    size_t n1 = a->cnt;
    size_t n2 = b->cnt;
    size_t total = n1 + n2;
    float delta = b->mean - a->mean;
    a->mean = a->mean + delta * ((float)n2 / (float)total);
    /* n1*n2/total 用 double 求，避免 float 在 n 大时失精 */
    double w = (double)n1 * (double)n2 / (double)total;
    a->M2 = a->M2 + b->M2 + delta * delta * (float)w;
    a->cnt = total;
}

/* 把一组 8 宽 Welford 状态归约成单个 BlockStat */
static inline BlockStat reduce_track(__m256 m, __m256 M, __m256i c)
{
    float mb[8], Mb[8];
    int   cb[8];
    _mm256_storeu_ps(mb, m);
    _mm256_storeu_ps(Mb, M);
    _mm256_storeu_si256((__m256i*)cb, c);
    BlockStat g = { 0.0f, 0.0f, 0 };
    for (int l = 0; l < 8; l++) {
        BlockStat b = { mb[l], Mb[l], (size_t)cb[l] };
        welford_merge(&g, &b);
    }
    return g;
}

/* ================================================================== */
/* 优化版 A：float 向量累加的 Welford（性能优先，与原始同数值特性）    */
/* ================================================================== */

__attribute__((target("avx2,fma")))
Stats mean_variance_avx2_welford_opt_f(const float* arr, size_t n)
{
    Stats res = { 0.0f, 0.0f };
    if (n == 0) return res;

    __m256 m0 = _mm256_setzero_ps(), m1 = _mm256_setzero_ps();
    __m256 M0 = _mm256_setzero_ps(), M1 = _mm256_setzero_ps();
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i one = _mm256_set1_epi32(1);
    __m256 one_ps = _mm256_set1_ps(1.0f);

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        /* 关键：先把两条 track 的载荷都发出去，让内存子系统与硬件
         * 预取尽早并行取数，盖掉后续 load->use 的延迟。否则 track1 的
         * load 要等 track0 整段算完才发，DRAM 场景下延迟暴露无遗。 */
        __m256 x0  = _mm256_loadu_ps(arr + i);
        __m256 x1  = _mm256_loadu_ps(arr + i + 8);

        c0 = _mm256_add_epi32(c0, one);
        c1 = _mm256_add_epi32(c1, one);
        /* 用正确舍入的 div_ps 取 1/cnt：Welford 对倒数误差敏感，
         * 近似倒数（rcp+牛顿）在 cnt=1 处会差 1 个 ULP，乘上大均值后
         * 把 M2 放大到灾难量级。原始 avx2_welford 也用 div_ps 正是此理。 */
        __m256 inv0 = _mm256_div_ps(one_ps, _mm256_cvtepi32_ps(c0));
        __m256 inv1 = _mm256_div_ps(one_ps, _mm256_cvtepi32_ps(c1));

        /* ---- Track 0：元素 [i, i+8) ---- */
        __m256 d0   = _mm256_sub_ps(x0, m0);
        __m256 m0n  = _mm256_fmadd_ps(d0, inv0, m0);
        __m256 d0n  = _mm256_sub_ps(x0, m0n);
        M0 = _mm256_fmadd_ps(d0, d0n, M0);
        m0 = m0n;

        /* ---- Track 1：元素 [i+8, i+16) ---- */
        __m256 d1   = _mm256_sub_ps(x1, m1);
        __m256 m1n  = _mm256_fmadd_ps(d1, inv1, m1);
        __m256 d1n  = _mm256_sub_ps(x1, m1n);
        M1 = _mm256_fmadd_ps(d1, d1n, M1);
        m1 = m1n;
    }

    /* 合并两段部分统计 */
    BlockStat g0 = reduce_track(m0, M0, c0);
    BlockStat g1 = reduce_track(m1, M1, c1);
    BlockStat g  = { 0.0f, 0.0f, 0 };
    welford_merge(&g, &g0);
    welford_merge(&g, &g1);

    /* 尾部标量续推（仍是 Welford 递推） */
    float mean = g.mean;
    float M2   = g.M2;
    size_t count = g.cnt;
    for (; i < n; i++) {
        float x = arr[i];
        size_t k = count + 1;
        float delta  = x - mean;
        mean += delta / (float)k;
        float delta2 = x - mean;
        M2  += delta * delta2;
        count++;
    }
    res.mean = mean;
    res.variance = M2 / (float)count;
    return res;
}

/* ================================================================== */
/* 单路 FMA 版 Welford（隔离实验用）                                    */
/* 与原始 avx2_welford 同结构：单条 8 宽 lane，末尾 8 路归约 + 尾部续推 */
/* 仅把 mul+add 替换为 fmadd（融合乘加，少一次舍入、缩短依赖链）。      */
/* 目的：隔离验证「FMA」与「2 路展开」各自的加速贡献——                  */
/*   若 单路FMA ≈ opt_f(2路FMA)，说明加速主要来自 FMA；                */
/*   若 单路FMA ≈ 原始 avx2_welford，说明 2 路展开才是关键。           */
/* ================================================================== */

__attribute__((target("avx2,fma")))
Stats mean_variance_avx2_welford_fma(const float* arr, size_t n)
{
    Stats res = { 0.0f, 0.0f };
    if (n == 0) return res;

    __m256  v_mean = _mm256_setzero_ps();
    __m256  v_M2   = _mm256_setzero_ps();
    __m256i v_cnt  = _mm256_setzero_si256();

    const __m256 one_ps = _mm256_set1_ps(1.0f);
    const __m256i one_epi32 = _mm256_set1_epi32(1);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(arr + i);
        v_cnt = _mm256_add_epi32(v_cnt, one_epi32);
        __m256 cnt_f = _mm256_cvtepi32_ps(v_cnt);
        __m256 inv_cnt = _mm256_div_ps(one_ps, cnt_f);
        __m256 delta  = _mm256_sub_ps(x, v_mean);
        __m256 m_new  = _mm256_fmadd_ps(delta, inv_cnt, v_mean);
        __m256 delta2 = _mm256_sub_ps(x, m_new);
        v_M2   = _mm256_fmadd_ps(delta, delta2, v_M2);
        v_mean = m_new;
    }

    BlockStat lanes[8] = {0};
    float mean_buf[8], m2_buf[8];
    int cnt_buf[8];
    _mm256_storeu_ps(mean_buf, v_mean);
    _mm256_storeu_ps(m2_buf, v_M2);
    _mm256_storeu_si256((__m256i*)cnt_buf, v_cnt);
    for (int l = 0; l < 8; l++) {
        lanes[l].cnt  = cnt_buf[l];
        lanes[l].mean = mean_buf[l];
        lanes[l].M2   = m2_buf[l];
    }

    BlockStat global = {0, 0.0f, 0};
    for (int l = 0; l < 8; l++) welford_merge(&global, &lanes[l]);

    float mean = global.mean;
    float M2   = global.M2;
    size_t count = global.cnt;

    for (; i < n; i++) {
        float x = arr[i];
        size_t k = count + 1;
        float delta  = x - mean;
        mean += delta / (float)k;
        float delta2 = x - mean;
        M2  += delta * delta2;
        count++;
    }

    res.mean = mean;
    res.variance = M2 / (float)count;
    return res;
}

/* ================================================================== */
/* 优化版 B：double 累加器的 Welford（数值稳定优先，同一套递推）       */
/* ================================================================== */

/* double 版部分统计 + 合并（合并权重同样用 double） */
typedef struct {
    double mean;
    double M2;
    size_t cnt;
} DBlock;

static inline void welford_merge_d(DBlock* restrict a, const DBlock* restrict b)
{
    if (b->cnt == 0) return;
    if (a->cnt == 0) { *a = *b; return; }
    size_t n1 = a->cnt, n2 = b->cnt, total = n1 + n2;
    double delta = b->mean - a->mean;
    a->mean = a->mean + delta * ((double)n2 / (double)total);
    a->M2 = a->M2 + b->M2 + delta * delta * ((double)n1 * (double)n2 / (double)total);
    a->cnt = total;
}

static inline DBlock reduce_track_d(__m256d m, __m256d M, size_t nb)
{
    double mb[4], Mb[4];
    _mm256_storeu_pd(mb, m);
    _mm256_storeu_pd(Mb, M);
    DBlock g = { 0.0, 0.0, 0 };
    for (int l = 0; l < 4; l++) {
        DBlock b = { mb[l], Mb[l], nb };
        welford_merge_d(&g, &b);
    }
    return g;
}

__attribute__((target("avx2,fma")))
Stats mean_variance_avx2_welford_opt_d(const float* arr, size_t n)
{
    Stats res = { 0.0f, 0.0f };
    if (n == 0) return res;

    __m256d m0 = _mm256_setzero_pd(), m1 = _mm256_setzero_pd();
    __m256d M0 = _mm256_setzero_pd(), M1 = _mm256_setzero_pd();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xf = _mm256_loadu_ps(arr + i);
        __m256d x0 = _mm256_cvtps_pd(_mm256_castps256_ps128(xf));
        __m256d x1 = _mm256_cvtps_pd(_mm256_extractf128_ps(xf, 1));

        size_t k = (i >> 3) + 1;          /* 该块内每条 lane 的运行计数 */
        double inv = 1.0 / (double)k;
        __m256d invv = _mm256_set1_pd(inv);

        __m256d d0  = _mm256_sub_pd(x0, m0);
        __m256d m0n = _mm256_fmadd_pd(d0, invv, m0);
        __m256d d0n = _mm256_sub_pd(x0, m0n);
        M0 = _mm256_fmadd_pd(d0, d0n, M0);
        m0 = m0n;

        __m256d d1  = _mm256_sub_pd(x1, m1);
        __m256d m1n = _mm256_fmadd_pd(d1, invv, m1);
        __m256d d1n = _mm256_sub_pd(x1, m1n);
        M1 = _mm256_fmadd_pd(d1, d1n, M1);
        m1 = m1n;
    }

    /* 8 个 double lane 归约：每条 lane 处理过 nb = floor(n/8) 个块，
       合并 8 段部分统计得到 SIMD 覆盖部分（计数 = 8*nb = n - n%8）。 */
    size_t nb = i >> 3;
    DBlock g0 = reduce_track_d(m0, M0, nb);
    DBlock g1 = reduce_track_d(m1, M1, nb);
    DBlock g  = { 0.0, 0.0, 0 };
    welford_merge_d(&g, &g0);
    welford_merge_d(&g, &g1);

    /* 尾部标量续推（double Welford） */
    double mean = g.mean;
    double M2   = g.M2;
    size_t count = g.cnt;
    for (; i < n; i++) {
        double x = (double)arr[i];
        size_t k = count + 1;
        double delta  = x - mean;
        mean += delta / (double)k;
        double delta2 = x - mean;
        M2  += delta * delta2;
        count++;
    }
    res.mean = (float)mean;
    res.variance = (float)(M2 / (double)count);
    return res;
}

/* ================================================================== */
/* 标量回退 + 运行时分发                                               */
/* ================================================================== */

static Stats welford_scalar_fallback(const float* arr, size_t n)
{
    Stats r = { 0.0f, 0.0f };
    if (n == 0) return r;
    float mean = 0.0f;
    float M2   = 0.0f;
    size_t count = 0;
    for (size_t i = 0; i < n; i++) {
        float x = arr[i];
        size_t k = count + 1;
        float delta  = x - mean;
        mean += delta / (float)k;
        float delta2 = x - mean;
        M2  += delta * delta2;
        count++;
    }
    r.mean = mean;
    r.variance = M2 / (float)count;
    return r;
}

Stats mean_variance_dispatch(const float* arr, size_t n)
{
    static int checked = 0;
    static int has_avx2fma = 0;
    if (!checked) {
        __builtin_cpu_init();
        has_avx2fma = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
        checked = 1;
    }
    if (has_avx2fma) return mean_variance_avx2_welford_opt_f(arr, n);
    return welford_scalar_fallback(arr, n);
}
