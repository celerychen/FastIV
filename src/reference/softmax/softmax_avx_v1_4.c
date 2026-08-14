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
/*  本文件 = softmax_avx_v1_2（双向量交错）与                          */
/*          softmax_avx_v1_3（系数打包 + vpermilps）的叠加：           */
/*                                                                    */
/*  exp256_ps4(__m256 x, __m256 y)：双向量交错版，但 Horner 的 6 个     */
/*    多项式系数不再为每条链各做一次 _mm256_set1_ps 广播（v1_2 每次     */
/*    exp256_ps2 调用要 12 次广播 load），而是打包进「2 个 __m256 寄存器 */
/*    K0/K1」，x、y 两条链共用这一组常数，每步用 vpermilps（立即数、    */
/*    零额外寄存器、零内存广播）从 K0/K1 取系数并广播到全 8 lane。      */
/*    常数 load 从 12 降到 2，省下的是「双向量版特有的、每条链各一份」 */
/*    的冗余广播。返回 {exp(x), exp(y)}。                              */
/*                                                                    */
/*  打包约定：vpermilps 不跨 128 位边界，故低半与高半放相同系数序，      */
/*  这样「半内广播」即等价于「全 8 lane 广播」。                        */
/*    K0 = [1/6!, 1/5!, 1/4!, 1/3! | 同上]                              */
/*    K1 = [1/2!,   1,    1,    1   | 同上]                              */
/* ================================================================== */

/* 单向量 exp（inline 常数），供 <8 元素尾部与 scalar 路径复用 */
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
    __m256i ex = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(ex));
}

/* 双向量结果打包 */
typedef struct { __m256 a; __m256 b; } m256x2;

/*
 * 双向量交错 + 系数打包版 exp：x、y 两条 Horner 链逐拍交错，
 * 共用打包常数 K0/K1（vpermilps 取用）。返回 {exp(x), exp(y)}。
 */
static inline m256x2 exp256_ps4(__m256 x, __m256 y) {
    /* 只命名跨步用到的 3 个常数（k/r 计算后即死） */
    const __m256 c_invln2 = _mm256_set1_ps(1.4426950408889634f);
    const __m256 c_ln2    = _mm256_set1_ps(0.6931471805599453f);
    const __m256 c_half   = _mm256_set1_ps(0.5f);

    /* 区间裁剪（inline 常数，不全程占寄存器） */
    x = _mm256_min_ps(x, _mm256_set1_ps( 88.0f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    y = _mm256_min_ps(y, _mm256_set1_ps( 88.0f));
    y = _mm256_max_ps(y, _mm256_set1_ps(-88.0f));

    /* k = round(x/ln2); r = x - k*ln2 —— x、y 交错 */
    __m256 kx = _mm256_floor_ps(_mm256_fmadd_ps(x, c_invln2, c_half));
    __m256 ky = _mm256_floor_ps(_mm256_fmadd_ps(y, c_invln2, c_half));
    __m256 rx = _mm256_fnmadd_ps(kx, c_ln2, x);
    __m256 ry = _mm256_fnmadd_ps(ky, c_ln2, y);

    /* 多项式系数打包：仅 2 个常数寄存器常驻 Horner 全程，x/y 两链共用 */
    const __m256 K0 = _mm256_set_ps(
        0.16666666666666666f, 0.041666666666666664f, 0.008333333333333333f, 0.001388888888888889f,
        0.16666666666666666f, 0.041666666666666664f, 0.008333333333333333f, 0.001388888888888889f);
    const __m256 K1 = _mm256_set_ps(
        1.0f, 1.0f, 1.0f, 0.5f,
        1.0f, 1.0f, 1.0f, 0.5f);

    /* 6 阶 Horner，x/y 两链逐拍交错：每步用 vpermilps（立即数）从 K0/K1
       取系数并广播到全 lane（零额外寄存器、零内存广播）。
       两条链共用同一组打包常数，把 v1_2 每调用一次 12 次广播 load
       降成 2 次加载 + 12 次洗牌（洗牌走 port5，不占访存端口）。 */
    __m256 yx = _mm256_permute_ps(K0, _MM_SHUFFLE(0, 0, 0, 0));                  /* 1/6! */
    __m256 yy = _mm256_permute_ps(K0, _MM_SHUFFLE(0, 0, 0, 0));
    yx = _mm256_fmadd_ps(yx, rx, _mm256_permute_ps(K0, _MM_SHUFFLE(1, 1, 1, 1)));/* 1/5! */
    yy = _mm256_fmadd_ps(yy, ry, _mm256_permute_ps(K0, _MM_SHUFFLE(1, 1, 1, 1)));
    yx = _mm256_fmadd_ps(yx, rx, _mm256_permute_ps(K0, _MM_SHUFFLE(2, 2, 2, 2)));/* 1/4! */
    yy = _mm256_fmadd_ps(yy, ry, _mm256_permute_ps(K0, _MM_SHUFFLE(2, 2, 2, 2)));
    yx = _mm256_fmadd_ps(yx, rx, _mm256_permute_ps(K0, _MM_SHUFFLE(3, 3, 3, 3)));/* 1/3! */
    yy = _mm256_fmadd_ps(yy, ry, _mm256_permute_ps(K0, _MM_SHUFFLE(3, 3, 3, 3)));
    yx = _mm256_fmadd_ps(yx, rx, _mm256_permute_ps(K1, _MM_SHUFFLE(0, 0, 0, 0)));/* 1/2! */
    yy = _mm256_fmadd_ps(yy, ry, _mm256_permute_ps(K1, _MM_SHUFFLE(0, 0, 0, 0)));
    yx = _mm256_fmadd_ps(yx, rx, _mm256_permute_ps(K1, _MM_SHUFFLE(1, 1, 1, 1)));/* 1/1! */
    yy = _mm256_fmadd_ps(yy, ry, _mm256_permute_ps(K1, _MM_SHUFFLE(1, 1, 1, 1)));
    yx = _mm256_fmadd_ps(yx, rx, _mm256_permute_ps(K1, _MM_SHUFFLE(1, 1, 1, 1)));/* 1/0! */
    yy = _mm256_fmadd_ps(yy, ry, _mm256_permute_ps(K1, _MM_SHUFFLE(1, 1, 1, 1)));

    /* 2^k（指数域移位，无乘法） */
    __m256i ex = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(kx),
                                                    _mm256_set1_epi32(127)), 23);
    __m256i ey = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(ky),
                                                    _mm256_set1_epi32(127)), 23);

    m256x2 r;
    r.a = _mm256_mul_ps(yx, _mm256_castsi256_ps(ex));
    r.b = _mm256_mul_ps(yy, _mm256_castsi256_ps(ey));
    return r;
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
 * AVX2 加速版 softmax v1_4（原地，float only）。数学同 softmax_basic。
 *
 * 与 softmax_avx_v1_2 的差别仅在「exp 趟用的双向量 exp」：
 *   v1_2 用 exp256_ps2（每链各一份 inline 常数广播）；
 *   本版用 exp256_ps4（双向量交错 + 系数打包共用，减少冗余广播 load）。
 * max 趟、归一化趟与 v1_2 完全一致（4 路展开）。用于对比
 * 「双向量交错 + 系数打包」组合是否比单纯的双向量交错更快。
 */
void softmax_avx_v1_4(float* x, size_t n) {
    if (n == 0) return;

    /* 趟1：最大值（4 路展开，32 float/迭代） */
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

    /* 趟2：exp(x-m) 写回 + 累加 sum。
       双向量交错 + 系数打包版 exp256_ps4 一次处理 2 块（16 float），
       两条 exp 链交错飞行、共用打包常数。 */
    const __m256 vmax_bc = _mm256_set1_ps(max_val);
    __m256 vs0 = _mm256_setzero_ps();
    __m256 vs1 = _mm256_setzero_ps();
    __m256 vs2 = _mm256_setzero_ps();
    __m256 vs3 = _mm256_setzero_ps();
    size_t j = 0;
    for (; j + 32 <= n; j += 32) {
        __m256 b0 = _mm256_sub_ps(_mm256_loadu_ps(x + j),      vmax_bc);
        __m256 b1 = _mm256_sub_ps(_mm256_loadu_ps(x + j + 8),  vmax_bc);
        __m256 b2 = _mm256_sub_ps(_mm256_loadu_ps(x + j + 16), vmax_bc);
        __m256 b3 = _mm256_sub_ps(_mm256_loadu_ps(x + j + 24), vmax_bc);
        m256x2 e01 = exp256_ps4(b0, b1);
        m256x2 e23 = exp256_ps4(b2, b3);
        _mm256_storeu_ps(x + j,      e01.a);
        _mm256_storeu_ps(x + j + 8,  e01.b);
        _mm256_storeu_ps(x + j + 16, e23.a);
        _mm256_storeu_ps(x + j + 24, e23.b);
        vs0 = _mm256_add_ps(vs0, e01.a);
        vs1 = _mm256_add_ps(vs1, e01.b);
        vs2 = _mm256_add_ps(vs2, e23.a);
        vs3 = _mm256_add_ps(vs3, e23.b);
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
