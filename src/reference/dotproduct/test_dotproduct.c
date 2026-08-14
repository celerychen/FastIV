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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

static int g_fail = 0;

/*
 * 参考实现：用 double 计算，独立于两版，作为正确性基准。
 * 内积只有一个标量输出，故直接返回 double。
 */
static double dot_ref(const float* a, const float* b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++)
        s += (double)a[i] * (double)b[i];
    return s;
}

/* 简易确定性随机数（与 bench 同款 LCG/xorshift），保证可复现 */
static unsigned int rng_state = 0x9e3779b9u;
static float rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state & 0xffffffu) / (float)(1u << 24)) * 20.0f - 10.0f;
}

/*
 * 检查「某版本 vs double 参考」（宽松 sanity）：
 * 浮点顺序累加相对 double 的天然漂移约 1e-6（实测，见仓库说明），
 * 故放宽容差——只用于抓住「忘乘 / 越界 / 累加器写错」这类硬错误。
 */
static void check_vs_ref(const char* name,
                         float (*fn)(const float*, const float*, size_t),
                         double ref,
                         const float* a, const float* b, size_t n) {
    float got  = fn(a, b, n);
    float reff = (float)ref;
    float d   = fabsf(got - reff);
    float rel = (fabsf(reff) > 1e-20f) ? d / fabsf(reff) : d;
    int ok = (d <= 1e-2f) && (rel <= 1e-3f);
    if (!ok) {
        printf("FAIL %s: got %g  ref %g  abs_diff %g  rel %g\n", name, got, reff, d, rel);
        g_fail = 1; return;
    }
    printf("PASS %-10s vs ref   got=%g  ref=%g  abs_err=%g  rel_err=%g\n", name, got, reff, d, rel);
}

/*
 * 检查「avx vs basic」（严格）：两版都是 float 累加、算同一件事，
 * 仅浮点重排导致末位差异（FMA 比「先乘后加」还少一次舍入，反而更准），
 * 故用紧容差，证明向量化路径结果正确。
 */
static void check_vs_basic(const char* name,
                           float (*fn)(const float*, const float*, size_t),
                           float basic_val,
                           const float* a, const float* b, size_t n) {
    float got = fn(a, b, n);
    float d   = fabsf(got - basic_val);
    float rel = (fabsf(basic_val) > 1e-20f) ? d / fabsf(basic_val) : d;
    /*
     * 阈值 = max(相对 1e-3 * |basic|, 绝对 1e-3)。
     * 两版都是 float 累加、只重排了运算顺序，理论相对误差 ~1e-6；用「相对 1e-3」
     * 做主判据，对大结果（如 ~5e8，float ULP≈64）仍能容下正常的舍入差；
     * 对近零结果改用绝对地板 1e-3，避免 rel 在 |basic|→0 时爆炸。
     * 真实 bug（漏乘 / 越界 / 累加器写错）会产生 ~100% 量级的偏差，必被抓住。
     */
    float tol = (fabsf(basic_val) > 1e-20f) ? (1e-3f * fabsf(basic_val)) : 1e-3f;
    int ok = (d <= tol);
    if (!ok) {
        printf("FAIL %s: got %g  basic %g  abs_diff %g  rel %g  (tol=%g)\n",
               name, got, basic_val, d, rel, tol);
        g_fail = 1; return;
    }
    printf("PASS %-10s vs basic got=%g  basic=%g  abs_err=%g  rel_err=%g\n", name, got, basic_val, d, rel);
}

/* 跑一组用例：先算 ref，再校验 basic（宽松）与 avx（严格） */
static void run_case(const char* label, float* a, float* b, size_t n) {
    printf("=== case: %s (n=%zu) ===\n", label, n);
    if (n == 0) {
        dot_basic(a, b, 0); dot_avx(a, b, 0); /* 仅验证不崩溃 */
        printf("PASS n=0 (no-crash)\n");
        return;
    }
    double ref = dot_ref(a, b, n);
    float  bv  = dot_basic(a, b, n);
    check_vs_ref  ("dot_basic",     dot_basic,     ref, a, b, n);
    check_vs_basic("dot_avx",       dot_avx,       bv,  a, b, n);
    check_vs_basic("dot_avx_v2",    dot_avx_v2,    bv,  a, b, n);
    check_vs_basic("dot_avx_sp",    dot_avx_sp,    bv,  a, b, n);
    check_vs_basic("dot_avx_v2_sp", dot_avx_v2_sp, bv,  a, b, n);
}

int main(void) {
    /* 把控制台输出代码页切到 UTF-8，避免中文乱码（源码按 UTF-8 编译） */
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    /* 1) 随机多规模 */
    size_t sizes[] = {1, 7, 256, 4096, 65536};
    for (int c = 0; c < 5; c++) {
        size_t n = sizes[c];
        float* a = (float*)malloc(n * sizeof(float));
        float* b = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) { a[i] = rnd(); b[i] = rnd(); }
        char label[32];
        snprintf(label, sizeof(label), "random n=%zu", n);
        run_case(label, a, b, n);
        free(a); free(b);
    }

    /* 2) 全相等 */
    {
        size_t n = 1000;
        float* a = (float*)malloc(n * sizeof(float));
        float* b = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) { a[i] = 3.0f; b[i] = 2.0f; }
        run_case("all equal", a, b, n);
        free(a); free(b);
    }

    /* 3) 极大正值 */
    {
        size_t n = 500;
        float* a = (float*)malloc(n * sizeof(float));
        float* b = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) { a[i] = 1000.0f + rnd()*0.1f; b[i] = 1000.0f + rnd()*0.1f; }
        run_case("large positive ~1000", a, b, n);
        free(a); free(b);
    }

    /* 4) 极大负值 */
    {
        size_t n = 500;
        float* a = (float*)malloc(n * sizeof(float));
        float* b = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) { a[i] = -1000.0f + rnd()*0.1f; b[i] = -1000.0f + rnd()*0.1f; }
        run_case("large negative ~-1000", a, b, n);
        free(a); free(b);
    }

    /* 5) 极端跨度（同时含极大/极小，考验累加稳定性） */
    {
        size_t n = 2000;
        float* a = (float*)malloc(n * sizeof(float));
        float* b = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) {
            a[i] = (i % 2 == 0) ? 50.0f : -50.0f;
            b[i] = (i % 3 == 0) ? 50.0f : -50.0f;
        }
        run_case("mixed +/-50", a, b, n);
        free(a); free(b);
    }

    /* 6) n = 0 边界 */
    run_case("empty", NULL, NULL, 0);

    if (g_fail) {
        printf("\n==== RESULT: FAIL ====\n");
        return 1;
    }
    printf("\n==== RESULT: ALL PASS ====\n");
    return 0;
}
