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

#include <stdio.h>
#include <immintrin.h>
#include <stddef.h>

// 最终输出统计结果
typedef struct {
    float mean;
    float variance;
} Stats;

// 单路Welford统计块，用于多路合并
typedef struct {
    int cnt;        // 单块元素数，32位整型单通道最大支持21亿，8通道总168亿，足够绝大多数场景
    float mean;
    float M2;
} BlockStat;

// Welford 合并公式：a = merge(a, b)，数学上严格等价于全集统计
static inline void welford_merge(BlockStat* restrict a, const BlockStat* restrict b)
{
    if (b->cnt == 0) return;
    if (a->cnt == 0) {
        *a = *b;
        return;
    }

    int n1 = a->cnt;
    int n2 = b->cnt;
    int total = n1 + n2;
    float delta = b->mean - a->mean;

    a->mean = a->mean + delta * ((float)n2 / total);
    a->M2 = a->M2 + b->M2 + delta * delta * ((float)n1 * n2 / total);
    a->cnt = total;
}

// AVX2 向量化Welford：8路独立并行，单趟遍历，保持Welford数值稳定性
Stats mean_variance_avx2_welford(const float* arr, size_t n)
{
    Stats res = {0.0f, 0.0f};
    if (n == 0) return res;

    // 8路独立Welford状态：均值、平方和累加量、整型计数（精确）
    __m256  v_mean = _mm256_setzero_ps();
    __m256  v_M2   = _mm256_setzero_ps();
    __m256i v_cnt  = _mm256_setzero_si256();  // 核心修复：整型计数，完全精确

    const __m256 one_ps = _mm256_set1_ps(1.0f);
    const __m256i one_epi32 = _mm256_set1_epi32(1);

    size_t i = 0;
    // 主循环：每次处理8个连续元素，分发到8路独立Welford通道
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(arr + i);

        // 计数+1（整型加法，无精度损失）
        v_cnt = _mm256_add_epi32(v_cnt, one_epi32);
        __m256 cnt_f = _mm256_cvtepi32_ps(v_cnt);

        // Welford 递推（向量并行执行8路）
        __m256 delta = _mm256_sub_ps(x, v_mean);
        __m256 inv_cnt = _mm256_div_ps(one_ps, cnt_f);
        v_mean = _mm256_add_ps(v_mean, _mm256_mul_ps(delta, inv_cnt));

        __m256 delta2 = _mm256_sub_ps(x, v_mean);
        v_M2 = _mm256_add_ps(v_M2, _mm256_mul_ps(delta, delta2));
    }

    // 将8路向量状态导出为独立块
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

    // 合并8路统计结果
    BlockStat global = {0, 0.0f, 0.0f};
    for (int l = 0; l < 8; l++) {
        welford_merge(&global, &lanes[l]);
    }

    // 处理尾部不足8个的元素（串行Welford）
    float mean = global.mean;
    float M2   = global.M2;
    int count  = global.cnt;

    for (; i < n; i++) {
        float x = arr[i];
        int k = count + 1;
        float delta = x - mean;
        mean += delta / (float)k;
        float delta2 = x - mean;
        M2 += delta * delta2;
        count++;
    }

    res.mean = mean;
    res.variance = M2 / (float)count;  // 总体方差；如需样本方差改为 / (count-1)
    return res;
}

// 基准：标准串行Welford，用于正确性对比
Stats mean_variance_serial(const float arr[], size_t n)
{
    Stats result = {0.0f, 0.0f};
    if (n == 0) return result;

    float mean = 0.0f;
    float M2 = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float x = arr[i];
        int k = (int)(i + 1);
        float delta = x - mean;
        mean += delta / (float)k;
        float delta2 = x - mean;
        M2 += delta * delta2;
    }

    result.mean = mean;
    result.variance = M2 / (float)n;
    return result;
}

// 简单测试
#ifndef MEANVAR_NO_MAIN
int main(void)
{
    float test[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    size_t n = sizeof(test) / sizeof(test[0]);

    Stats s_ser = mean_variance_serial(test, n);
    Stats s_avx = mean_variance_avx2_welford(test, n);

    printf("串行Welford:  mean=%.6f  variance=%.6f\n", s_ser.mean, s_ser.variance);
    printf("AVX2 Welford: mean=%.6f  variance=%.6f\n", s_avx.mean, s_avx.variance);

    return 0;
}
#endif // MEANVAR_NO_MAIN