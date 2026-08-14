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
#include <windows.h>

// ============ 高精度计时 ============
static inline double now_sec(void) {
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return (double)c.QuadPart / (double)f.QuadPart;
}

// ============ fast_div（rcp + 1次牛顿） ============
static inline __m256 fast_div_ps(__m256 a, __m256 b) {
    __m256 r0 = _mm256_rcp_ps(b);
    __m256 b_r0 = _mm256_mul_ps(b, r0);
    __m256 two_sub = _mm256_sub_ps(_mm256_set1_ps(2.0f), b_r0);
    __m256 r = _mm256_mul_ps(r0, two_sub);
    return _mm256_mul_ps(a, r);
}

// ============ 1) 标准两趟标量 ============
float l2_two_pass(const float *arr, int n) {
    if (n <= 0) return 0.0f;
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(arr[i]); if (a > max_abs) max_abs = a; }
    if (max_abs == 0.0f) return 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) { float s = arr[i] / max_abs; sum_sq += s * s; }
    return max_abs * sqrtf(sum_sq);
}

// ============ 2) 标量单趟无分支（修复除零） ============
float l2_scalar_single(const float *arr, int n) {
    if (n <= 0) return 0.0f;
    float cur = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(arr[i]);
        if (a > cur) {
            if (cur != 0.0f) { float ratio = cur / a; sum_sq *= ratio * ratio; }
            else sum_sq = 0.0f;
            cur = a;
        }
        if (cur != 0.0f) { float s = a / cur; sum_sq += s * s; }
    }
    return (cur == 0.0f) ? 0.0f : cur * sqrtf(sum_sq);
}

// ============ 3) AVX2 单趟 + fast_div（修复除零） ============
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
    for (int i = 0; i < 8; i++) { float r = lm[i] / gmax; gsum += ls[i] * r * r; }
    if (rem > 0) { float r = rm / gmax; gsum += rs * r * r; }
    return gmax * sqrtf(gsum);
}

// ============ 4) AVX2 单趟 + 原生除法（修复除零） ============
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
    for (int i = 0; i < 8; i++) { float r = lm[i] / gmax; gsum += ls[i] * r * r; }
    if (rem > 0) { float r = rm / gmax; gsum += rs * r * r; }
    return gmax * sqrtf(gsum);
}

// ============ 5) 两趟算法的 AVX2 版本（v0：原生除法，保精度） ============
float l2_two_pass_avx_v0(const float *arr, int n) {
    if (n <= 0) return 0.0f;
    const int W = 8;
    int n_avx = (n / W) * W;

    // ---- Pass 1: 求 max_abs（向量化归约） ----
    __m256 v_max = _mm256_set1_ps(0.0f);
    int i = 0;
    for (; i < n_avx; i += W) {
        __m256 v = _mm256_loadu_ps(arr + i);
        v = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v);   // abs
        v_max = _mm256_max_ps(v_max, v);
    }
    float lm[8];
    _mm256_storeu_ps(lm, v_max);
    float max_abs = 0.0f;
    for (int k = 0; k < 8; k++) if (lm[k] > max_abs) max_abs = lm[k];
    for (; i < n; i++) { float a = fabsf(arr[i]); if (a > max_abs) max_abs = a; }
    if (max_abs == 0.0f) return 0.0f;

    // ---- Pass 2: sum((x/max)^2)（向量化累加） ----
    __m256 v_maxb = _mm256_set1_ps(max_abs);
    __m256 v_sum = _mm256_set1_ps(0.0f);
    int j = 0;
    for (; j < n_avx; j += W) {
        __m256 v = _mm256_loadu_ps(arr + j);
        __m256 s = _mm256_div_ps(v, v_maxb);
        v_sum = _mm256_add_ps(v_sum, _mm256_mul_ps(s, s));
    }
    float ls[8];
    _mm256_storeu_ps(ls, v_sum);
    float sum_sq = 0.0f;
    for (int k = 0; k < 8; k++) sum_sq += ls[k];
    for (; j < n; j++) { float s = arr[j] / max_abs; sum_sq += s * s; }

    return max_abs * sqrtf(sum_sq);
}

// ============ 基准框架 ============
typedef float (*l2fn)(const float*, int);

static void bench(const char *name, l2fn fn, const float *arr, int n,
                  int iters, float ref, double ref_ms) {
    float r0 = fn(arr, n);                       // 正确性参照
    volatile float sink = r0; (void)sink;
    double t0 = now_sec();
    float acc = 0.0f;
    for (int i = 0; i < iters; i++) acc += fn(arr, n);
    double t1 = now_sec();
    double per_call = (t1 - t0) / iters;
    double meps = (double)n / per_call / 1e6;    // 百万元素/秒
    double rel = fabsf(r0 - ref) / ref;
    double speedup = ref_ms / per_call;
    printf("  %-22s %9.3f ms  %9.1f M/s  %6.2fx  rel=%.3e\n",
           name, per_call * 1e3, meps, speedup, rel);
    if (acc == 0.0f) printf(""); // 防止死代码消除
}

int main(void) {
    int sizes[] = { 1<<20, 1<<22, 1<<24 };       // 1M / 4M / 16M 元素
    const char *slabels[] = { "1M (4 MB)", "4M (16 MB)", "16M (64 MB)" };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("===== L2 范数 性能/精度对比 (AVX2, /O2) =====\n");
    printf("方法                    耗时        吞吐      提速   相对误差\n");

    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        float *arr = (float*)malloc((size_t)n * sizeof(float));
        for (int i = 0; i < n; i++) arr[i] = ((float)(rand() % 10000 - 5000)) * 1e-3f;

        // 参考值（单次两趟结果，不可被计时循环污染）
        float ref = l2_two_pass(arr, n);
        // 单独测两趟基准耗时（结果累加进独立变量，避免改写 ref）
        volatile float ref_sink = 0.0f;
        double t0 = now_sec();
        for (int i = 0; i < 15; i++) ref_sink += l2_two_pass(arr, n);
        double t1 = now_sec();
        double ref_ms = (t1 - t0) / 15;
        (void)ref_sink;

        printf("\n[%s]\n", slabels[s]);
        bench("two-pass(基准)",      l2_two_pass,       arr, n, 15, ref, ref_ms);
        bench("two-pass-avx-v0",     l2_two_pass_avx_v0,arr, n, 15, ref, ref_ms);
        bench("scalar-single",       l2_scalar_single,  arr, n, 15, ref, ref_ms);
        bench("avx2-single+fast_div",l2_avx2_fast,      arr, n, 15, ref, ref_ms);
        bench("avx2-single+native",  l2_avx2_native,    arr, n, 15, ref, ref_ms);

        free(arr);
    }
    printf("\n(注：提速 = two-pass 耗时 / 本方法耗时；rel = 相对 two-pass 参考的相对误差)\n");
    return 0;
}
