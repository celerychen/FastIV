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
/*  exp256_ps3：单向量版，与 softmax_avx_v1 的 exp256_ps 数学完全一致， */
/*  差别仅在「多项式系数如何喂给 Horner 链」：                          */
/*    - v1 的 exp256_ps 每个系数都用 _mm256_set1_ps 从内存广播          */
/*      （vbroadcastss，每次占一个临时寄存器 + 一次访存端口）；           */
/*    - 本版把 6 个系数打包进「2 个 __m256 寄存器」，每步用              */
/*      vpermilps（_mm256_permute_ps，立即数控制，零额外寄存器、零内存） */
/*      把目标 lane 广播到全 8 路。常数寄存器从 6+ 降到 2，且把 6 次     */
/*      内存广播 load 换成 2 次加载 + 6 次洗牌（洗牌走 port5，不占       */
/*      访存端口）——在大 n 内存带宽受限时尤为有利。                     */
/*                                                                    */
/*  打包约定：vpermilps 不跨 128 位边界，故低半与高半放相同系数序，      */
/*  这样「半内广播」即等价于「全 8 lane 广播」。                         */
/*    K0 = [1/6!, 1/5!, 1/4!, 1/3! | 同上]                              */
/*    K1 = [1/2!,   1,    1,    1   | 同上]                              */
/* ================================================================== */
static inline __m256 exp256_ps3(__m256 x) {
    /* 区间裁剪：clamp 常数 inline 临时加载，用完即释放寄存器 */
    x = _mm256_min_ps(x, _mm256_set1_ps( 88.0f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));

    /* k = round(x / ln2)， r = x - k*ln2 */
    __m256 k = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.4426950408889634f),
                                               _mm256_set1_ps(0.5f)));
    __m256 r = _mm256_fnmadd_ps(k, _mm256_set1_ps(0.6931471805599453f), x);

    /* 多项式系数打包：仅 2 个常数寄存器常驻 Horner 全程 */
    const __m256 K0 = _mm256_set_ps(
        0.16666666666666666f, 0.041666666666666664f, 0.008333333333333333f, 0.001388888888888889f,
        0.16666666666666666f, 0.041666666666666664f, 0.008333333333333333f, 0.001388888888888889f);
    const __m256 K1 = _mm256_set_ps(
        1.0f, 1.0f, 1.0f, 0.5f,
        1.0f, 1.0f, 1.0f, 0.5f);

    /* 6 阶 Horner：每步用 vpermilps（立即数）从 K0/K1 取系数并广播到全 lane */
    __m256 y = _mm256_permute_ps(K0, _MM_SHUFFLE(0, 0, 0, 0));                /* 1/6! */
    y = _mm256_fmadd_ps(y, r, _mm256_permute_ps(K0, _MM_SHUFFLE(1, 1, 1, 1))); /* 1/5! */
    y = _mm256_fmadd_ps(y, r, _mm256_permute_ps(K0, _MM_SHUFFLE(2, 2, 2, 2))); /* 1/4! */
    y = _mm256_fmadd_ps(y, r, _mm256_permute_ps(K0, _MM_SHUFFLE(3, 3, 3, 3))); /* 1/3! */
    y = _mm256_fmadd_ps(y, r, _mm256_permute_ps(K1, _MM_SHUFFLE(0, 0, 0, 0))); /* 1/2! */
    y = _mm256_fmadd_ps(y, r, _mm256_permute_ps(K1, _MM_SHUFFLE(1, 1, 1, 1))); /* 1/1! */
    y = _mm256_fmadd_ps(y, r, _mm256_permute_ps(K1, _MM_SHUFFLE(1, 1, 1, 1))); /* 1/0! 常数项 */

    /* 2^k = ldexp(1, k)：把 (k+127) 移到指数域（无乘法） */
    __m256i ex = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(k),
                                                     _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(ex));
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
 * AVX2 加速版 softmax v1_3（原地，float only）。数学同 softmax_basic。
 *
 * 与 softmax_avx_v1 的唯一差别：exp 趟用「系数打包 + vpermilps」的
 * exp256_ps3，其余（4 路展开、3 趟、全程 float）完全一致。用于对比
 * 「常数打包」对寄存器压力与访存端口的改善。
 */
void softmax_avx_v1_3(float* x, size_t n) {
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
        __m256 e0 = exp256_ps3(_mm256_sub_ps(_mm256_loadu_ps(x + j),      vmax_bc));
        __m256 e1 = exp256_ps3(_mm256_sub_ps(_mm256_loadu_ps(x + j + 8),  vmax_bc));
        __m256 e2 = exp256_ps3(_mm256_sub_ps(_mm256_loadu_ps(x + j + 16), vmax_bc));
        __m256 e3 = exp256_ps3(_mm256_sub_ps(_mm256_loadu_ps(x + j + 24), vmax_bc));
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
        __m256 e = exp256_ps3(_mm256_sub_ps(_mm256_loadu_ps(x + j), vmax_bc));
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
