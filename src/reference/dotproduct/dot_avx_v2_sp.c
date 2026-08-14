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
 * dot_avx_v2_sp：dp 版（_mm256_dp_ps / VDPPS ymm）+ 软件流水线（load-ahead）。
 *
 * 结构与 dot_avx_sp 完全一致，只把 compute 从 FMA 换成
 *   acc = _mm256_add_ps(acc, _mm256_dp_ps(a, b, 0xF1))。
 * imm8=0xF1：lane 内 4 元素全参与相乘，和写回该 lane 第 0 位（其余置 0），
 * 故可直接垂直 add 累加，末尾一次水平归约。
 *
 *   序幕 : load 第 0 块（a/b 各 4 个 __m256）。
 *   稳态 : 先对已 load 的 4 块各做一次 VDPPS+add，再提前 load 下一块。
 *   尾声 : 补算最后一块。
 *
 * 4 累加器 4 路展开，与 dot_avx_v2 对齐。VDPPS 延迟高（~13 周期），软件
 * 流水线理论上有机会把 load 延迟藏起来，但 VDPPS 自身吞吐（~1.5 周期/条）
 * 才是瓶颈，收益以实测为准。
 */
float dot_avx_v2_sp(const float* a, const float* b, size_t n) {
    if (n == 0) return 0.0f;

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t i = 0;
    size_t nb = n / 32;              /* 完整 32-float 块的个数 */

    if (nb >= 1) {
        /* --- 序幕：load 第 0 块 --- */
        __m256 a0 = _mm256_loadu_ps(a +  0), b0 = _mm256_loadu_ps(b +  0);
        __m256 a1 = _mm256_loadu_ps(a +  8), b1 = _mm256_loadu_ps(b +  8);
        __m256 a2 = _mm256_loadu_ps(a + 16), b2 = _mm256_loadu_ps(b + 16);
        __m256 a3 = _mm256_loadu_ps(a + 24), b3 = _mm256_loadu_ps(b + 24);

        /* --- 稳态：compute(k) 与 load(k+1) 错位重叠 --- */
        for (size_t k = 1; k < nb; k++) {
            const float* pa = a + k * 32;
            const float* pb = b + k * 32;
            /* 先算当前块 */
            acc0 = _mm256_add_ps(acc0, _mm256_dp_ps(a0, b0, 0xF1));
            acc1 = _mm256_add_ps(acc1, _mm256_dp_ps(a1, b1, 0xF1));
            acc2 = _mm256_add_ps(acc2, _mm256_dp_ps(a2, b2, 0xF1));
            acc3 = _mm256_add_ps(acc3, _mm256_dp_ps(a3, b3, 0xF1));
            /* 再提前 load 下一块 */
            a0 = _mm256_loadu_ps(pa +  0); b0 = _mm256_loadu_ps(pb +  0);
            a1 = _mm256_loadu_ps(pa +  8); b1 = _mm256_loadu_ps(pb +  8);
            a2 = _mm256_loadu_ps(pa + 16); b2 = _mm256_loadu_ps(pb + 16);
            a3 = _mm256_loadu_ps(pa + 24); b3 = _mm256_loadu_ps(pb + 24);
        }

        /* --- 尾声：补算最后一块 --- */
        acc0 = _mm256_add_ps(acc0, _mm256_dp_ps(a0, b0, 0xF1));
        acc1 = _mm256_add_ps(acc1, _mm256_dp_ps(a1, b1, 0xF1));
        acc2 = _mm256_add_ps(acc2, _mm256_dp_ps(a2, b2, 0xF1));
        acc3 = _mm256_add_ps(acc3, _mm256_dp_ps(a3, b3, 0xF1));

        i = nb * 32;
    }

    /* 合并 4 个累加器 */
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    /* 剩余整 8 块 */
    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_add_ps(acc0, _mm256_dp_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), 0xF1));

    float s = hsum256_ps(acc0);
    for (; i < n; i++)
        s += a[i] * b[i];
    return s;
}
