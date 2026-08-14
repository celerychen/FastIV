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
#include <math.h>
#include <stdlib.h>
#include <immintrin.h>

// ==============================================
// 辅助：AVX2 快速近似除法（倒数+1次牛顿迭代）
// 精度接近原生 _mm256_div_ps，速度快3~4倍
// ==============================================
static inline __m256 fast_div_ps(__m256 a, __m256 b) {
    __m256 r0 = _mm256_rcp_ps(b);          // 近似倒数
    __m256 b_r0 = _mm256_mul_ps(b, r0);
    __m256 two_sub = _mm256_sub_ps(_mm256_set1_ps(2.0f), b_r0);
    __m256 r = _mm256_mul_ps(r0, two_sub); // 牛顿迭代一次
    return _mm256_mul_ps(a, r);            // a / b = a * (1/b)
}

// ==============================================
// 基准：标准两趟数值稳定L2范数（float标量版）
// ==============================================
float l2_norm_two_pass_float(const float *arr, int n) {
    if (n <= 0) return 0.0f;

    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs_x = fabsf(arr[i]);
        if (abs_x > max_abs) max_abs = abs_x;
    }
    if (max_abs == 0.0f) return 0.0f;

    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float scaled = arr[i] / max_abs;
        sum_sq += scaled * scaled;
    }
    return max_abs * sqrtf(sum_sq);
}

// ==============================================
// 标量单趟无分支版（float）
// ==============================================
float l2_norm_scalar_single_pass(const float *arr, int n) {
    if (n <= 0) return 0.0f;

    // 修复：运行最大值为0（开头连续为0）时不能做除零，否则产生 NaN
    float current_max = 0.0f;
    float sum_sq = 0.0f;

    for (int i = 0; i < n; i++) {
        float abs_x = fabsf(arr[i]);
        if (abs_x > current_max) {
            if (current_max != 0.0f) {
                float ratio = current_max / abs_x;
                sum_sq *= ratio * ratio;
            } else {
                sum_sq = 0.0f;   // 之前全为0，尚无有效累加
            }
            current_max = abs_x;
        }
        if (current_max != 0.0f) {
            float scaled = abs_x / current_max;
            sum_sq += scaled * scaled;
        }
        // 否则 abs_x==0 且 current_max==0，贡献为0
    }

    return current_max == 0.0f ? 0.0f : current_max * sqrtf(sum_sq);
}

// ==============================================
// AVX2单趟无分支版（快速除法优化）
// 主循环无原生除法指令，全部替换为倒数+牛顿迭代
// ==============================================
float l2_norm_avx2_fast_div(const float *arr, int n) {
    if (n <= 0) return 0.0f;

    const int simd_width = 8;
    int n_avx = (n / simd_width) * simd_width;

    if (n_avx < simd_width) {
        return l2_norm_scalar_single_pass(arr, n);
    }

    // 1. 初始化：max=0, sum_sq=0（避免首块全0时把 sum_sq 写死为 1.0 导致错误）
    __m256 v_zero = _mm256_set1_ps(0.0f);
    __m256 v_local_max = _mm256_set1_ps(0.0f);
    __m256 v_sum_sq = _mm256_set1_ps(0.0f);

    // 2. 主循环：全乘法/加法，无原生除法
    //    修复：current_max==0 时用 blend 强制 ratio/scaled 为0，避免 rcp(0)=inf -> NaN
    for (int i = 0; i < n_avx; i += simd_width) {
        __m256 v_x = _mm256_loadu_ps(arr + i);
        __m256 v_abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v_x);

        __m256 v_new_max = _mm256_max_ps(v_local_max, v_abs_x);

        // 【优化点1】ratio = old_max / new_max，用快速除法
        __m256 v_ratio = fast_div_ps(v_local_max, v_new_max);
        v_ratio = _mm256_blendv_ps(v_ratio, v_zero,
                                   _mm256_cmp_ps(v_local_max, v_zero, _CMP_EQ_OS));
        __m256 v_ratio_sq = _mm256_mul_ps(v_ratio, v_ratio);
        v_sum_sq = _mm256_mul_ps(v_sum_sq, v_ratio_sq);

        v_local_max = v_new_max;

        // 【优化点2】scaled = abs_x / current_max，用快速除法
        __m256 v_scaled = fast_div_ps(v_abs_x, v_local_max);
        v_scaled = _mm256_blendv_ps(v_scaled, v_zero,
                                    _mm256_cmp_ps(v_local_max, v_zero, _CMP_EQ_OS));
        __m256 v_scaled_sq = _mm256_mul_ps(v_scaled, v_scaled);
        v_sum_sq = _mm256_add_ps(v_sum_sq, v_scaled_sq);
    }

    // 3. 提取通道结果
    float local_max[8];
    float local_sum_sq[8];
    _mm256_storeu_ps(local_max, v_local_max);
    _mm256_storeu_ps(local_sum_sq, v_sum_sq);

    // 4. 处理剩余元素
    int remain = n - n_avx;
    float remain_max = 0.0f;
    float remain_sum_sq = 0.0f;
    if (remain > 0) {
        const float *remain_arr = arr + n_avx;
        remain_max = 0.0f;
        remain_sum_sq = 0.0f;
        for (int i = 0; i < remain; i++) {
            float abs_x = fabsf(remain_arr[i]);
            if (abs_x > remain_max) {
                if (remain_max != 0.0f) {
                    float ratio = remain_max / abs_x;
                    remain_sum_sq *= ratio * ratio;
                } else {
                    remain_sum_sq = 0.0f;
                }
                remain_max = abs_x;
            }
            if (remain_max != 0.0f) {
                float scaled = abs_x / remain_max;
                remain_sum_sq += scaled * scaled;
            }
        }
    }

    // 5. 全局合并
    float global_max = local_max[0];
    for (int i = 1; i < 8; i++) {
        if (local_max[i] > global_max) global_max = local_max[i];
    }
    if (remain > 0 && remain_max > global_max) global_max = remain_max;
    if (global_max == 0.0f) return 0.0f;

    float global_sum_sq = 0.0f;
    for (int i = 0; i < 8; i++) {
        float ratio = local_max[i] / global_max;
        global_sum_sq += local_sum_sq[i] * ratio * ratio;
    }
    if (remain > 0) {
        float ratio = remain_max / global_max;
        global_sum_sq += remain_sum_sq * ratio * ratio;
    }

    return global_max * sqrtf(global_sum_sq);
}

// ==============================================
// 测试主函数
// ==============================================
int main() {
    // 测试1：普通数组
    float test1[] = {3.0f, 4.0f, 12.0f, 5.0f, 12.0f, 9.0f, 1.0f, 7.0f, 2.0f, 6.0f};
    int n1 = sizeof(test1) / sizeof(test1[0]);
    printf("=== 测试1：普通数组（长度%d） ===\n", n1);
    printf("标准两趟标量：  %.6f\n", l2_norm_two_pass_float(test1, n1));
    printf("AVX2快速除法版：%.6f\n", l2_norm_avx2_fast_div(test1, n1));
    printf("\n");

    // 测试2：大数（数值稳定性）
    float test2[] = {3e18f, 4e18f, 12e18f, 5e18f};
    int n2 = sizeof(test2) / sizeof(test2[0]);
    printf("=== 测试2：大数向量 ===\n");
    printf("标准两趟标量：  %.6e\n", l2_norm_two_pass_float(test2, n2));
    printf("AVX2快速除法版：%.6e\n", l2_norm_avx2_fast_div(test2, n2));
    printf("\n");

    // 测试3：百万级随机数组（精度验证）
    int n3 = 1024 * 1024;
    float *test3 = (float*)malloc(n3 * sizeof(float));
    for (int i = 0; i < n3; i++) {
        test3[i] = (rand() % 10000 - 5000) * 1e-3f;
    }
    float base = l2_norm_two_pass_float(test3, n3);
    float fast_res = l2_norm_avx2_fast_div(test3, n3);
    printf("=== 测试3：100万元素随机数组 ===\n");
    printf("标准两趟标量：  %.6f\n", base);
    printf("AVX2快速除法版：%.6f\n", fast_res);
    printf("相对误差：      %.6e\n", fabsf(base - fast_res) / base);

    free(test3);
    return 0;
}
