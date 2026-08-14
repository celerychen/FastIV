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

#include "dotproduct.h"
#include <immintrin.h>

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
 * dot_avx_v2：用 256-bit dot-product 指令 _mm256_dp_ps（AVX 的 VDPPS ymm）实现，
 * 采用 4 累加器 4 路展开。
 *
 * VDPPS ymm 语义：每个 128-bit lane 内各做一次 4 元素点积。imm8=0xF1：
 *   高 4 位 0xF -> lane 内 4 个元素全参与相乘；
 *   低 4 位 0x1 -> 乘积和只写回该 lane 的第 0 位，其余置 0。
 * 于是每条得到 [s_lo,0,0,0, s_hi,0,0,0]，可直接 _mm256_add_ps 垂直累加。
 *
 * 展开次数：4 条独立依赖链。VDPPS ymm 延迟 ~13 周期、倒数吞吐 ~1.5 周期/条，
 * 完全喂满端口需约 9 条在飞的链；4 路展开无法压到吞吐上限，但结构简洁、
 * 寄存器占用小，符合「只展开 4 次」的要求。
 */
float dot_avx_v2(const float* a, const float* b, size_t n) {
    if (n == 0) return 0.0f;

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t i = 0;
    /* 主循环：每次 32 个 float，4 条独立 VDPPS 依赖链重叠飞行 */
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_add_ps(acc0, _mm256_dp_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      0xF1));
        acc1 = _mm256_add_ps(acc1, _mm256_dp_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  0xF1));
        acc2 = _mm256_add_ps(acc2, _mm256_dp_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), 0xF1));
        acc3 = _mm256_add_ps(acc3, _mm256_dp_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), 0xF1));
    }
    /* 每次 8 个 float，收整块 */
    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_add_ps(acc0, _mm256_dp_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), 0xF1));

    /* 合并 4 个累加器，末尾仅一次水平归约 */
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    float s = hsum256_ps(acc0);

    /* 不足 8 个：标量收尾 */
    for (; i < n; i++)
        s += a[i] * b[i];
    return s;
}
