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
#include <math.h>
#include <immintrin.h>

/* ================================================================== */
/*  向量化 exp 近似（仅用 FMA + 整数移位，无除法、无 libcall）          */
/*                                                                    */
/*  思路：区间约化到 [-ln2/2, ln2/2] 再做 6 阶泰勒，收敛极快。          */
/*    t = x / ln2                                                     */
/*    k = round(t)                 (x/ln2 最近整数)                    */
/*    r = x - k * ln2  ∈  [-0.3466, 0.3466]                            */
/*    exp(x) = 2^k * exp(r)                                          */
/*    exp(r) 用 Horner 形式的 6 阶泰勒（全 FMA）：                      */
/*      1 + r*(1/1! + r*(1/2! + r*(1/3! + r*(1/4! + r*(1/5! + r/6!))))) */
/*    2^k 不做乘法，直接把 (k+127) 放进浮点"指数域"（左移 23 位）。     */
/*  该区间上 7 阶余项 < 1.2e-7，单精度误差远低于 1e-3。                 */
/*  入参先 clamp 到 [-88, 88]，保证 exp 不溢出/下溢成 inf/nan。         */
/* ================================================================== */
static inline __m256 exp256_ps(__m256 x) {
    const __m256 c_invln2 = _mm256_set1_ps(1.4426950408889634f); /* 1/ln2  */
    const __m256 c_ln2    = _mm256_set1_ps(0.6931471805599453f); /* ln2    */
    const __m256 c_half   = _mm256_set1_ps(0.5f);

    /* 区间裁剪，避免 exp 溢出/下溢 */
    x = _mm256_min_ps(x, _mm256_set1_ps( 88.0f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));

    /* k = round(x / ln2)， r = x - k*ln2 */
    __m256 k = _mm256_floor_ps(_mm256_fmadd_ps(x, c_invln2, c_half));
    __m256 r = _mm256_fnmadd_ps(k, c_ln2, x);

    /* 6 阶泰勒展开 exp(r)，Horner 全 FMA（务必以常数项 1 收尾） */
    __m256 y = _mm256_set1_ps(0.001388888888888889f);                  /* 1/6! */
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(0.008333333333333333f));  /* +1/5! */
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(0.041666666666666664f));  /* +1/4! */
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(0.16666666666666666f));   /* +1/3! */
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(0.5f));                   /* +1/2! */
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f));                   /* +1/1! */
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f));                   /* +1/0! (常数项) */

    /* 2^k = ldexp(1, k)：把 (k+127) 移到指数域（无乘法） */
    __m256i ki  = _mm256_cvtps_epi32(k);
    __m256i exp = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(exp));
}

/* 256-bit 向量水平最大值 */
static inline float hmax256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m = _mm_max_ps(hi, lo);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(m);
}

/* 256-bit 向量水平求和 */
static inline float hsum256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(s);
}

/*
 * AVX2 加速版 softmax v1（原地，float only）。数学同 softmax_basic。
 *
 * 优化要点：
 *   - 求 max 与 exp 两趟的主循环均「4 路展开」（每次迭代 32 个 float），
 *     用 4 个独立的向量累加器（max / sum）容纳 4 条相互独立的依赖链。
 *     exp256_ps 内部是 ~6 级 FMA 串行链（长延迟），单链吞吐受限；同时飞
 *     行 4 条链可把 FMA 端口（port0/1）吃满，掩盖延迟、逼近计算吞吐上限。
 *   - 归一化趟 4 路展开（每次 32 个 float），load/mul/store 交织隐藏访存。
 *   - 全程 float 累加（末次 hsum，相对误差 ~1e-5，足够），避免 double 转换开销。
 *   - 归一化用精确 1/sum 除法（非 rcp 近似），保数值正确。
 *
 * 这是原地「精确」softmax 的最少内存趟数：max 依赖使 exp 趟须等 max 趟先
 * 完成；exp 趟写回 exp 值并累加 sum，末趟除以 sum 归一化（3 趟）。
 */
void softmax_avx_v1(float* x, size_t n) {
    if (n == 0) return;

    /* 趟1：最大值（4 路展开，32 float/迭代，4 个独立 max 累加器） */
    float max_val;
    if (n >= 8) {
        __m256 vmax0 = _mm256_loadu_ps(x);
        __m256 vmax1 = vmax0;
        __m256 vmax2 = vmax0;
        __m256 vmax3 = vmax0;
        size_t i = 8;
        for (; i + 32 <= n; i += 32) {
            vmax0 = _mm256_max_ps(vmax0, _mm256_loadu_ps(x + i));
            vmax1 = _mm256_max_ps(vmax1, _mm256_loadu_ps(x + i + 8));
            vmax2 = _mm256_max_ps(vmax2, _mm256_loadu_ps(x + i + 16));
            vmax3 = _mm256_max_ps(vmax3, _mm256_loadu_ps(x + i + 24));
        }
        __m256 vmax = _mm256_max_ps(_mm256_max_ps(vmax0, vmax1),
                                    _mm256_max_ps(vmax2, vmax3));
        for (; i + 8 <= n; i += 8)
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + i));
        max_val = hmax256_ps(vmax);
        for (; i < n; i++)
            max_val = fmaxf(max_val, x[i]);
    } else {
        max_val = x[0];
        for (size_t i = 1; i < n; i++)
            max_val = fmaxf(max_val, x[i]);
    }

    /* 趟2：exp(x-m) 写回 + 累加 sum（4 路展开，4 条独立 exp 链重叠飞行） */
    const __m256 vmax_bc = _mm256_set1_ps(max_val);
    __m256 vs0 = _mm256_setzero_ps();
    __m256 vs1 = _mm256_setzero_ps();
    __m256 vs2 = _mm256_setzero_ps();
    __m256 vs3 = _mm256_setzero_ps();
    size_t j = 0;
    for (; j + 32 <= n; j += 32) {
        __m256 e0 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x + j),      vmax_bc));
        __m256 e1 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x + j + 8),  vmax_bc));
        __m256 e2 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x + j + 16), vmax_bc));
        __m256 e3 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x + j + 24), vmax_bc));
        _mm256_storeu_ps(x + j,      e0);
        _mm256_storeu_ps(x + j + 8,  e1);
        _mm256_storeu_ps(x + j + 16, e2);
        _mm256_storeu_ps(x + j + 24, e3);
        vs0 = _mm256_add_ps(vs0, e0);
        vs1 = _mm256_add_ps(vs1, e1);
        vs2 = _mm256_add_ps(vs2, e2);
        vs3 = _mm256_add_ps(vs3, e3);
    }
    __m256 vsum = _mm256_add_ps(_mm256_add_ps(vs0, vs1), _mm256_add_ps(vs2, vs3));
    for (; j + 8 <= n; j += 8) {
        __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x + j), vmax_bc));
        _mm256_storeu_ps(x + j, e);
        vsum = _mm256_add_ps(vsum, e);
    }
    float sum = hsum256_ps(vsum);
    for (; j < n; j++) {
        float e = expf(x[j] - max_val);
        x[j] = e;
        sum += e;
    }

    /* 趟3：原地除以 sum（精确除法，4 路展开） */
    const float inv = 1.0f / sum;
    const __m256 vinv = _mm256_set1_ps(inv);
    size_t k = 0;
    for (; k + 32 <= n; k += 32) {
        _mm256_storeu_ps(x + k,      _mm256_mul_ps(_mm256_loadu_ps(x + k),      vinv));
        _mm256_storeu_ps(x + k + 8,  _mm256_mul_ps(_mm256_loadu_ps(x + k + 8),  vinv));
        _mm256_storeu_ps(x + k + 16, _mm256_mul_ps(_mm256_loadu_ps(x + k + 16), vinv));
        _mm256_storeu_ps(x + k + 24, _mm256_mul_ps(_mm256_loadu_ps(x + k + 24), vinv));
    }
    for (; k + 8 <= n; k += 8)
        _mm256_storeu_ps(x + k, _mm256_mul_ps(_mm256_loadu_ps(x + k), vinv));
    for (; k < n; k++)
        x[k] *= inv;
}
