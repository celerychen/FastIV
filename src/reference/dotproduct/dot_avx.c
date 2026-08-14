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
 * AVX2+FMA 加速版内积（float only，非原地，只读 a/b）。
 *
 * 优化要点：
 *   - 主循环「4 路展开」：每次迭代处理 32 个 float，用 4 个独立的向量
 *     累加器 acc0..3 容纳 4 条相互独立的 FMA 依赖链
 *     （acc = fma(a,b,acc) = a*b+acc）。单条 FMA 链延迟约 4 周期、吞吐
 *     1/周期；同时飞行 4 条链可把 FMA 端口（port0/1）吃满，掩盖延迟、
 *     逼近计算吞吐上限。每条链只做「垂直」指令（load/fma/store 没有分支、
 *     没有水平归约），水平归约只在循环末尾做一次。
 *   - 用 FMA（_mm256_fmadd_ps）替代「先 vmul 再 vadd」：少一条指令、
 *     且 a*b+c 只在末尾舍入一次（比 mul 后 add 两次舍入更准）。
 *   - 全程 float 累加（末次 hsum），与 basic 的 float 语义一致，便于
 *     直接对比。
 *   - 尾部：先按 8 对齐（单累加器再吃若干 256-bit 块），再标量收尾，
 *     对未对齐 / 任意长度 n 都安全（loadu）。
 *
 * 内存趟数：只读 a、b 各一遍 + 写回无（结果是一个标量），对带宽不友好的
 * 小 n 也能靠计算并行度取胜。
 */
float dot_avx(const float* a, const float* b, size_t n) {
    if (n == 0) return 0.0f;

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                               _mm256_add_ps(acc2, acc3));

    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);

    float s = hsum256_ps(acc);
    for (; i < n; i++)
        s += a[i] * b[i];
    return s;
}
