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
// ==============================================
static inline __m256 fast_div_ps(__m256 a, __m256 b) {
    __m256 r0 = _mm256_rcp_ps(b);
    __m256 b_r0 = _mm256_mul_ps(b, r0);
    __m256 two_sub = _mm256_sub_ps(_mm256_set1_ps(2.0f), b_r0);
    __m256 r = _mm256_mul_ps(r0, two_sub);
    return _mm256_mul_ps(a, r);
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
// AVX2单趟无分支版（快速除法优化）
// ==============================================
float l2_norm_avx2_fast_div(const float *arr, int n) {
    if (n <= 0) return 0.0f;
    const int simd_width = 8;
    int n_avx = (n / simd_width) * simd_width;
    if (n_avx < simd_width) return 0.0f; // debug: skip scalar fallback

    __m256 v_first = _mm256_loadu_ps(arr);
    __m256 v_local_max = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v_first);
    __m256 v_sum_sq = _mm256_set1_ps(1.0f);

    for (int i = simd_width; i < n_avx; i += simd_width) {
        __m256 v_x = _mm256_loadu_ps(arr + i);
        __m256 v_abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v_x);
        __m256 v_new_max = _mm256_max_ps(v_local_max, v_abs_x);
        __m256 v_ratio = fast_div_ps(v_local_max, v_new_max);
        __m256 v_ratio_sq = _mm256_mul_ps(v_ratio, v_ratio);
        v_sum_sq = _mm256_mul_ps(v_sum_sq, v_ratio_sq);
        v_local_max = v_new_max;
        __m256 v_scaled = fast_div_ps(v_abs_x, v_local_max);
        __m256 v_scaled_sq = _mm256_mul_ps(v_scaled, v_scaled);
        v_sum_sq = _mm256_add_ps(v_sum_sq, v_scaled_sq);
    }

    float local_max[8]; float local_sum_sq[8];
    _mm256_storeu_ps(local_max, v_local_max);
    _mm256_storeu_ps(local_sum_sq, v_sum_sq);

    int remain = n - n_avx;
    float remain_max = 0.0f; float remain_sum_sq = 0.0f;
    if (remain > 0) {
        const float *remain_arr = arr + n_avx;
        remain_max = fabsf(remain_arr[0]);
        remain_sum_sq = (remain_max == 0.0f) ? 0.0f : 1.0f;
        for (int i = 1; i < remain; i++) {
            float abs_x = fabsf(remain_arr[i]);
            float new_max = fmaxf(remain_max, abs_x);
            float ratio = remain_max / new_max;
            remain_sum_sq *= ratio * ratio;
            remain_max = new_max;
            float scaled = abs_x / remain_max;
            remain_sum_sq += scaled * scaled;
        }
    }

    float global_max = local_max[0];
    for (int i = 1; i < 8; i++) if (local_max[i] > global_max) global_max = local_max[i];
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
// AVX2单趟无分支版（原生除法，用于对照）
// ==============================================
float l2_norm_avx2_native_div(const float *arr, int n) {
    if (n <= 0) return 0.0f;
    const int simd_width = 8;
    int n_avx = (n / simd_width) * simd_width;
    if (n_avx < simd_width) return 0.0f;

    __m256 v_first = _mm256_loadu_ps(arr);
    __m256 v_local_max = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v_first);
    __m256 v_sum_sq = _mm256_set1_ps(1.0f);

    for (int i = simd_width; i < n_avx; i += simd_width) {
        __m256 v_x = _mm256_loadu_ps(arr + i);
        __m256 v_abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v_x);
        __m256 v_new_max = _mm256_max_ps(v_local_max, v_abs_x);
        __m256 v_ratio = _mm256_div_ps(v_local_max, v_new_max);
        __m256 v_ratio_sq = _mm256_mul_ps(v_ratio, v_ratio);
        v_sum_sq = _mm256_mul_ps(v_sum_sq, v_ratio_sq);
        v_local_max = v_new_max;
        __m256 v_scaled = _mm256_div_ps(v_abs_x, v_local_max);
        __m256 v_scaled_sq = _mm256_mul_ps(v_scaled, v_scaled);
        v_sum_sq = _mm256_add_ps(v_sum_sq, v_scaled_sq);
    }

    float local_max[8]; float local_sum_sq[8];
    _mm256_storeu_ps(local_max, v_local_max);
    _mm256_storeu_ps(local_sum_sq, v_sum_sq);

    int remain = n - n_avx;
    float remain_max = 0.0f; float remain_sum_sq = 0.0f;
    if (remain > 0) {
        const float *remain_arr = arr + n_avx;
        remain_max = fabsf(remain_arr[0]);
        remain_sum_sq = (remain_max == 0.0f) ? 0.0f : 1.0f;
        for (int i = 1; i < remain; i++) {
            float abs_x = fabsf(remain_arr[i]);
            float new_max = fmaxf(remain_max, abs_x);
            float ratio = remain_max / new_max;
            remain_sum_sq *= ratio * ratio;
            remain_max = new_max;
            float scaled = abs_x / remain_max;
            remain_sum_sq += scaled * scaled;
        }
    }

    float global_max = local_max[0];
    for (int i = 1; i < 8; i++) if (local_max[i] > global_max) global_max = local_max[i];
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
// 对照实验主函数
// ==============================================
int main() {
    int n3 = 1024 * 1024;
    float *test3 = (float*)malloc(n3 * sizeof(float));
    for (int i = 0; i < n3; i++) {
        test3[i] = (rand() % 10000 - 5000) * 1e-3f;
    }
    float base      = l2_norm_two_pass_float(test3, n3);
    float fast_res  = l2_norm_avx2_fast_div(test3, n3);
    float native_res= l2_norm_avx2_native_div(test3, n3);

    printf("[DEBUG] reference two-pass : %.6f\n", base);
    printf("[DEBUG] avx2 fast-div      : %.6f  rel_err = %.6e\n", fast_res,   fabsf(base - fast_res)   / base);
    printf("[DEBUG] avx2 native-div    : %.6f  rel_err = %.6e\n", native_res, fabsf(base - native_res) / base);
    printf("[DEBUG] fast vs native diff: %.6e\n", fabsf(fast_res - native_res));

    // 小规模一致性
    float t1[] = {3.0f,4.0f,12.0f,5.0f,12.0f,9.0f,1.0f,7.0f,2.0f,6.0f};
    int n1 = 10;
    printf("[DEBUG] t1 fast=%.6f native=%.6f ref=%.6f\n",
        l2_norm_avx2_fast_div(t1,n1), l2_norm_avx2_native_div(t1,n1), l2_norm_two_pass_float(t1,n1));

    free(test3);
    return 0;
}
