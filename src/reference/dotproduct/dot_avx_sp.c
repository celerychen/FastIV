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
 * dot_avx_sp：FMA 版 + 软件流水线（software pipelining, load-ahead）。
 *
 * 与 dot_avx 的唯一区别在指令调度：把「下一块的 load」提前到「当前块的
 * compute」之前发起，使 load 的内存延迟被 FMA 计算掩盖（错位一格）。
 *
 *   序幕(prologue) : 先 load 第 0 块的 8 个向量寄存器（a/b 各 4）。
 *   稳态(steady)   : 循环体内先用「上一轮已 load 好」的寄存器做 4 条 FMA，
 *                    随后立刻 load 下一块 —— compute(k) 与 load(k+1) 重叠。
 *   尾声(epilogue) : 循环退出后，最后一块已 load 但未 compute，补算一次。
 *
 * 块大小 = 32 float（4×8），4 累加器 4 条独立 FMA 依赖链，与 dot_avx 对齐，
 * 便于「同展开度下，软件流水线是否带来额外收益」的公平对比。
 *
 * 注意：现代乱序(OoO) CPU 的 load buffer / 调度器本就会自动做类似重排，
 * 故在 Skylake + /O2 上本版相对 dot_avx 的收益可能很小甚至被硬件吃掉，
 * 以实测为准。
 */
float dot_avx_sp(const float* a, const float* b, size_t n) {
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
            /* 先算当前块（数据在上一轮已 load 好） */
            acc0 = _mm256_fmadd_ps(a0, b0, acc0);
            acc1 = _mm256_fmadd_ps(a1, b1, acc1);
            acc2 = _mm256_fmadd_ps(a2, b2, acc2);
            acc3 = _mm256_fmadd_ps(a3, b3, acc3);
            /* 再提前 load 下一块 */
            a0 = _mm256_loadu_ps(pa +  0); b0 = _mm256_loadu_ps(pb +  0);
            a1 = _mm256_loadu_ps(pa +  8); b1 = _mm256_loadu_ps(pb +  8);
            a2 = _mm256_loadu_ps(pa + 16); b2 = _mm256_loadu_ps(pb + 16);
            a3 = _mm256_loadu_ps(pa + 24); b3 = _mm256_loadu_ps(pb + 24);
        }

        /* --- 尾声：补算最后一块 --- */
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        acc2 = _mm256_fmadd_ps(a2, b2, acc2);
        acc3 = _mm256_fmadd_ps(a3, b3, acc3);

        i = nb * 32;
    }

    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                               _mm256_add_ps(acc2, acc3));

    /* 剩余整 8 块 */
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);

    float s = hsum256_ps(acc);
    for (; i < n; i++)
        s += a[i] * b[i];
    return s;
}
