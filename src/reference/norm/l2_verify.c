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
#include <string.h>
#include <immintrin.h>

static inline __m256 fast_div_ps(__m256 a, __m256 b) {
    __m256 r0 = _mm256_rcp_ps(b);
    __m256 b_r0 = _mm256_mul_ps(b, r0);
    __m256 two_sub = _mm256_sub_ps(_mm256_set1_ps(2.0f), b_r0);
    __m256 r = _mm256_mul_ps(r0, two_sub);
    return _mm256_mul_ps(a, r);
}

// 1) 标量两趟（参考实现）
float l2_two_pass(const float *arr, int n) {
    if (n <= 0) return 0.0f;
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(arr[i]); if (a > max_abs) max_abs = a; }
    if (max_abs == 0.0f) return 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) { float s = arr[i] / max_abs; sum_sq += s * s; }
    return max_abs * sqrtf(sum_sq);
}

// 2) 标量单趟（修复：运行最大值为0时不做除零）
float l2_scalar_single(const float *arr, int n) {
    if (n <= 0) return 0.0f;
    float cur = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(arr[i]);
        if (a > cur) {
            if (cur != 0.0f) { float ratio = cur / a; sum_sq *= ratio * ratio; }
            else sum_sq = 0.0f;   // 之前全为0，尚无有效累加
            cur = a;
        }
        if (cur != 0.0f) { float s = a / cur; sum_sq += s * s; }
        // 否则 a==0 且 cur==0，贡献为0
    }
    return (cur == 0.0f) ? 0.0f : cur * sqrtf(sum_sq);
}

// 3) AVX2 单趟 + fast_div（修复：max==0 分支用 blend 强制为0，避免 rcp(0)=NaN）
float l2_avx2_fast(const float *arr, int n) {
    if (n <= 0) return 0.0f;
    const int W = 8;
    int n_avx = (n / W) * W;
    if (n_avx < W) return l2_scalar_single(arr, n);
    __m256 v_zero = _mm256_set1_ps(0.0f);
    __m256 v_max = _mm256_set1_ps(0.0f);
    __m256 v_sum = _mm256_set1_ps(0.0f);
    for (int i = 0; i < n_avx; i += W) {
        __m256 v_x = _mm256_loadu_ps(arr + i);
        __m256 v_a = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v_x);
        __m256 v_nm = _mm256_max_ps(v_max, v_a);
        __m256 v_r = fast_div_ps(v_max, v_nm);
        v_r = _mm256_blendv_ps(v_r, v_zero, _mm256_cmp_ps(v_max, v_zero, _CMP_EQ_OS));
        v_sum = _mm256_mul_ps(v_sum, _mm256_mul_ps(v_r, v_r));
        v_max = v_nm;
        __m256 v_s = fast_div_ps(v_a, v_max);
        v_s = _mm256_blendv_ps(v_s, v_zero, _mm256_cmp_ps(v_max, v_zero, _CMP_EQ_OS));
        v_sum = _mm256_add_ps(v_sum, _mm256_mul_ps(v_s, v_s));
    }
    float lm[8], ls[8];
    _mm256_storeu_ps(lm, v_max); _mm256_storeu_ps(ls, v_sum);
    int rem = n - n_avx; float rm = 0.0f, rs = 0.0f;
    for (int i = 0; i < rem; i++) {
        float a = fabsf(arr[n_avx + i]);
        if (a > rm) {
            if (rm != 0.0f) { float ratio = rm / a; rs *= ratio * ratio; }
            else rs = 0.0f;
            rm = a;
        }
        if (rm != 0.0f) { float s = a / rm; rs += s * s; }
    }
    float gmax = lm[0];
    for (int i = 1; i < 8; i++) if (lm[i] > gmax) gmax = lm[i];
    if (rem > 0 && rm > gmax) gmax = rm;
    if (gmax == 0.0f) return 0.0f;
    float gsum = 0.0f;
    for (int i = 0; i < 8; i++) { float r = lm[i]/gmax; gsum += ls[i]*r*r; }
    if (rem > 0) { float r = rm/gmax; gsum += rs*r*r; }
    return gmax * sqrtf(gsum);
}

// 4) AVX2 单趟 + 原生除法（同样修复除零）
float l2_avx2_native(const float *arr, int n) {
    if (n <= 0) return 0.0f;
    const int W = 8;
    int n_avx = (n / W) * W;
    if (n_avx < W) return l2_scalar_single(arr, n);
    __m256 v_zero = _mm256_set1_ps(0.0f);
    __m256 v_max = _mm256_set1_ps(0.0f);
    __m256 v_sum = _mm256_set1_ps(0.0f);
    for (int i = 0; i < n_avx; i += W) {
        __m256 v_x = _mm256_loadu_ps(arr + i);
        __m256 v_a = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v_x);
        __m256 v_nm = _mm256_max_ps(v_max, v_a);
        __m256 v_r = _mm256_div_ps(v_max, v_nm);
        v_r = _mm256_blendv_ps(v_r, v_zero, _mm256_cmp_ps(v_max, v_zero, _CMP_EQ_OS));
        v_sum = _mm256_mul_ps(v_sum, _mm256_mul_ps(v_r, v_r));
        v_max = v_nm;
        __m256 v_s = _mm256_div_ps(v_a, v_max);
        v_s = _mm256_blendv_ps(v_s, v_zero, _mm256_cmp_ps(v_max, v_zero, _CMP_EQ_OS));
        v_sum = _mm256_add_ps(v_sum, _mm256_mul_ps(v_s, v_s));
    }
    float lm[8], ls[8];
    _mm256_storeu_ps(lm, v_max); _mm256_storeu_ps(ls, v_sum);
    int rem = n - n_avx; float rm = 0.0f, rs = 0.0f;
    for (int i = 0; i < rem; i++) {
        float a = fabsf(arr[n_avx + i]);
        if (a > rm) {
            if (rm != 0.0f) { float ratio = rm / a; rs *= ratio * ratio; }
            else rs = 0.0f;
            rm = a;
        }
        if (rm != 0.0f) { float s = a / rm; rs += s * s; }
    }
    float gmax = lm[0];
    for (int i = 1; i < 8; i++) if (lm[i] > gmax) gmax = lm[i];
    if (rem > 0 && rm > gmax) gmax = rm;
    if (gmax == 0.0f) return 0.0f;
    float gsum = 0.0f;
    for (int i = 0; i < 8; i++) { float r = lm[i]/gmax; gsum += ls[i]*r*r; }
    if (rem > 0) { float r = rm/gmax; gsum += rs*r*r; }
    return gmax * sqrtf(gsum);
}

// 5) 两趟 AVX2（v0，原生除法）
float l2_two_pass_avx_v0(const float *arr, int n) {
    if (n <= 0) return 0.0f;
    const int W = 8;
    int n_avx = (n / W) * W;
    __m256 v_max = _mm256_set1_ps(0.0f);
    int i = 0;
    for (; i < n_avx; i += W) {
        __m256 v = _mm256_loadu_ps(arr + i);
        v = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v);
        v_max = _mm256_max_ps(v_max, v);
    }
    float lm[8]; _mm256_storeu_ps(lm, v_max);
    float max_abs = 0.0f;
    for (int k = 0; k < 8; k++) if (lm[k] > max_abs) max_abs = lm[k];
    for (; i < n; i++) { float a = fabsf(arr[i]); if (a > max_abs) max_abs = a; }
    if (max_abs == 0.0f) return 0.0f;
    __m256 v_maxb = _mm256_set1_ps(max_abs);
    __m256 v_sum = _mm256_set1_ps(0.0f);
    int j = 0;
    for (; j < n_avx; j += W) {
        __m256 v = _mm256_loadu_ps(arr + j);
        __m256 s = _mm256_div_ps(v, v_maxb);
        v_sum = _mm256_add_ps(v_sum, _mm256_mul_ps(s, s));
    }
    float ls[8]; _mm256_storeu_ps(ls, v_sum);
    float sum_sq = 0.0f;
    for (int k = 0; k < 8; k++) sum_sq += ls[k];
    for (; j < n; j++) { float s = arr[j] / max_abs; sum_sq += s * s; }
    return max_abs * sqrtf(sum_sq);
}

// ===== double 真值 =====
double l2_true(const float *arr, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) { double x = (double)arr[i]; s += x * x; }
    return sqrt(s);
}

typedef float (*fn_t)(const float*, int);
fn_t methods[] = { l2_two_pass, l2_two_pass_avx_v0, l2_scalar_single, l2_avx2_fast, l2_avx2_native };
const char *names[] = { "two_pass(scalar)", "two_pass_avx_v0", "scalar_single", "avx2_fast_div", "avx2_native_div" };
const int N_M = 5;

int main(void) {
    int sizes[] = { 1, 3, 7, 8, 9, 100, 1000, 1024*1024 };
    int n_sizes = sizeof(sizes)/sizeof(sizes[0]);

    printf("===== 正确性校验：各方法 vs double 真值 =====\n");
    printf("%-18s | %-12s | %s\n", "size", "true(double)", "各方法相对误差 (method=true? 标记*)");

    int serious = 0; // 记录严重偏差
    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        float *arr = (float*)malloc((size_t)n * sizeof(float));
        srand(12345 + n);
        for (int i = 0; i < n; i++) arr[i] = ((float)(rand()%10000-5000)) * 1e-3f;
        double t = l2_true(arr, n);

        printf("\n[n=%d] true=%.6f\n", n, t);
        for (int m = 0; m < N_M; m++) {
            float r = methods[m](arr, n);
            double rel = (t == 0.0) ? 0.0 : fabs((double)r - t) / t;
            int bad = (rel > 1e-2) || isnan(r) || isinf(r);
            if (bad) serious++;
            printf("   %-18s r=%.6f  rel=%.3e%s\n", names[m], r, rel, bad?"  <-- 严重!":"");
        }

        // 专门检查：two_pass_avx_v0 与 avx2_native_div 是否逐字节相同（之前 bench 里 rel 完全相同）
        float a = l2_two_pass_avx_v0(arr, n);
        float b = l2_avx2_native(arr, n);
        if (memcmp(&a, &b, sizeof(float)) == 0)
            printf("   [!] two_pass_avx_v0 与 avx2_native_div 结果逐字节相同! (a=b=%.6f)\n", a);
        free(arr);
    }

    // ---- 边界用例 ----
    printf("\n===== 边界用例 =====\n");
    // 全零
    {
        int n = 16; float z[16] = {0};
        for (int m = 0; m < N_M; m++) {
            float r = methods[m](z, n);
            printf("   all-zero  %-18s r=%.6f %s\n", names[m], r, (r==0.0f)?"ok":"BAD");
        }
    }
    // 单元素
    {
        int n = 1; float o[1] = { -3.0f };
        for (int m = 0; m < N_M; m++) {
            float r = methods[m](o, n);
            printf("   single(-3) %-18s r=%.6f %s\n", names[m], r, (fabs(r-3.0f)<1e-5f)?"ok":"BAD");
        }
    }
    // 大数（1e18 量级）
    {
        float big[4] = { 3e18f, 4e18f, 12e18f, 5e18f };
        int n = 4;
        double t = l2_true(big, n);
        for (int m = 0; m < N_M; m++) {
            float r = methods[m](big, n);
            double rel = fabs((double)r - t)/t;
            printf("   big(1e18) %-18s r=%.6e rel=%.3e %s\n", names[m], r, rel, (rel<1e-3)?"ok":"CHECK");
        }
    }
    // 含 0 的混合
    {
        float mix[8] = { 0.0f, 3.0f, 0.0f, -4.0f, 0.0f, 12.0f, 5.0f, 0.0f };
        int n = 8; double t = l2_true(mix, n);
        for (int m = 0; m < N_M; m++) {
            float r = methods[m](mix, n);
            double rel = fabs((double)r - t)/t;
            printf("   mix+zeros %-18s r=%.6f rel=%.3e %s\n", names[m], r, rel, (rel<1e-4)?"ok":"CHECK");
        }
    }
    // 前8个元素全为0（触发 AVX 单趟版 v_sum_sq 写死=1.0 与 fast_div 除0=NaN 的隐患）
    {
        float z[16];
        for (int i = 0; i < 8; i++) z[i] = 0.0f;
        float t1[8] = { 3.0f,4.0f,12.0f,5.0f,12.0f,9.0f,1.0f,7.0f };
        for (int i = 0; i < 8; i++) z[8+i] = t1[i];
        int n = 16; double t = l2_true(z, n);
        printf("   [前8个为0] true=%.6f\n", t);
        for (int m = 0; m < N_M; m++) {
            float r = methods[m](z, n);
            double rel = (t==0.0)?0.0:fabs((double)r - t)/t;
            int bad = (rel > 1e-4) || isnan(r) || isinf(r);
            printf("      %-18s r=%.6f rel=%.3e %s\n", names[m], r, rel, bad?"<-- 严重!":"ok");
        }
    }
    // 全部为负且前8个为0（再探一次）
    {
        float z[16];
        for (int i = 0; i < 8; i++) z[i] = 0.0f;
        float t1[8] = { -3.0f,-4.0f,-12.0f,-5.0f,-12.0f,-9.0f,-1.0f,-7.0f };
        for (int i = 0; i < 8; i++) z[8+i] = t1[i];
        int n = 16; double t = l2_true(z, n);
        printf("   [前8个为0,负] true=%.6f\n", t);
        for (int m = 0; m < N_M; m++) {
            float r = methods[m](z, n);
            double rel = (t==0.0)?0.0:fabs((double)r - t)/t;
            int bad = (rel > 1e-4) || isnan(r) || isinf(r);
            printf("      %-18s r=%.6f rel=%.3e %s\n", names[m], r, rel, bad?"<-- 严重!":"ok");
        }
    }
    // 16个前导0（证明 AVX 单趟版在主循环里遇到全零块会 rcp(0)=NaN）
    {
        float z[24];
        for (int i = 0; i < 16; i++) z[i] = 0.0f;
        float t1[8] = { 3.0f,4.0f,12.0f,5.0f,12.0f,9.0f,1.0f,7.0f };
        for (int i = 0; i < 8; i++) z[16+i] = t1[i];
        int n = 24; double t = l2_true(z, n);
        printf("   [16个前导0] true=%.6f\n", t);
        for (int m = 0; m < N_M; m++) {
            float r = methods[m](z, n);
            double rel = (t==0.0)?0.0:fabs((double)r - t)/t;
            int bad = (rel > 1e-4) || isnan(r) || isinf(r);
            printf("      %-18s r=%.6f rel=%.3e %s\n", names[m], r, rel, bad?"<-- 严重!":"ok");
        }
    }

    printf("\n严重偏差计数(rel>1%% 或 NaN/Inf): %d\n", serious);
    return 0;
}
