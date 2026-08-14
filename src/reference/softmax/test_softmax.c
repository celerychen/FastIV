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

#include "softmax.h"
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
 */
static void softmax_ref(const float* x, float* y, size_t n) {
    if (n == 0) return;
    double m = (double)x[0];
    for (size_t i = 1; i < n; i++)
        if ((double)x[i] > m) m = (double)x[i];
    double s = 0.0;
    for (size_t i = 0; i < n; i++) {
        double e = exp((double)x[i] - m);
        y[i] = (float)e;
        s += e;
    }
    for (size_t i = 0; i < n; i++)
        y[i] = (float)((double)y[i] / s);
}

/* 简易确定性随机数 */
static unsigned int rng_state = 0x9e3779b9u;
static float rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state & 0xffffffu) / (float)(1u << 24)) * 20.0f - 10.0f;
}

/*
 * 对一个版本做检查：复制输入 -> 调 fn -> 与 ref 比对 -> 校验和为 1。
 */
static void check_version(const char* name,
                          void (*fn)(float*, size_t),
                          const float* ref,
                          const float* in,
                          size_t n,
                          float tol) {
    float* buf = (float*)malloc(n * sizeof(float));
    if (!buf) { printf("FAIL %s: OOM\n", name); g_fail = 1; return; }
    memcpy(buf, in, n * sizeof(float));

    fn(buf, n);

    float max_abs = 0.0f, max_rel = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(buf[i] - ref[i]);
        if (d > max_abs) max_abs = d;
        float r = (ref[i] != 0.0f) ? d / fabsf(ref[i]) : d;
        if (r > max_rel) max_rel = r;
        if (d > tol) {
            printf("FAIL %s: idx %zu  got %g  ref %g  diff %g\n",
                   name, i, buf[i], ref[i], d);
            g_fail = 1;
            free(buf);
            return;
        }
    }
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += buf[i];
    /* 单元素误差上限 tol 已覆盖正确性；float SIMD 多项式 exp 下，
       大 n 时“和为 1”的绝对偏差可能达 ~1e-4，故此处容差放宽到 1e-4 */
    if (fabsf(s - 1.0f) > 1e-4f) {
        printf("FAIL %s: sum = %g (expected ~1)\n", name, s);
        g_fail = 1;
        free(buf);
        return;
    }
    printf("PASS %s   (max_abs_err=%g  max_rel_err=%g)\n", name, max_abs, max_rel);
    free(buf);
}

/* 跑一组用例（给定输入 x），分别校验两版 */
static void run_case(const char* label, float* x, size_t n) {
    printf("=== case: %s (n=%zu) ===\n", label, n);
    if (n == 0) {
        /* 仅验证不崩溃 */
        softmax_basic(x, 0);
        softmax_avx_v1(x, 0);
        softmax_avx_v1_2(x, 0);
        softmax_avx_v1_3(x, 0);
        softmax_avx_v1_4(x, 0);
        printf("PASS n=0 (no-crash)\n");
        return;
    }
    float* ref  = (float*)malloc(n * sizeof(float)); /* double 参考，仅供 basic 做 sanity */
    float* in   = (float*)malloc(n * sizeof(float));
    float* cout = (float*)malloc(n * sizeof(float)); /* C 版（系统 expf）输出，作为 AVX 的对比基准 */
    if (!ref || !in || !cout) { printf("FAIL: OOM\n"); g_fail = 1; free(ref); free(in); free(cout); return; }

    softmax_ref(x, ref, n);
    memcpy(cout, x, n * sizeof(float));
    softmax_basic(cout, n);                 /* 纯 C 版输出 = 对比基准 */

    memcpy(in, x, n * sizeof(float));
    /* basic 对 double 参考只做 sanity（它用系统标准 expf，本就该对得上） */
    check_version("softmax_basic", softmax_basic, ref, in, n, 1e-5f);

    memcpy(in, x, n * sizeof(float));
    /* AVX v1 的对比对象 = 纯 C 版（系统 expf）输出：检验的是多项式 exp256_ps
       相对标准 expf 引入的误差（~1e-7），而非与 double 的差距。 */
    check_version("softmax_avx_v1", softmax_avx_v1, cout, in, n, 1e-7f);

    memcpy(in, x, n * sizeof(float));
    /* v1_2 同样对比纯 C 版（系统 expf）输出：校验双向量交错 exp256_ps2 的精度 */
    check_version("softmax_avx_v1_2", softmax_avx_v1_2, cout, in, n, 1e-7f);

    memcpy(in, x, n * sizeof(float));
    /* v1_3 对比纯 C 版（系统 expf）输出：校验「系数打包 + vpermilps」exp256_ps3 的精度 */
    check_version("softmax_avx_v1_3", softmax_avx_v1_3, cout, in, n, 1e-7f);

    memcpy(in, x, n * sizeof(float));
    /* v1_4 对比纯 C 版（系统 expf）输出：校验「双向量交错 + 系数打包」exp256_ps4 的精度 */
    check_version("softmax_avx_v1_4", softmax_avx_v1_4, cout, in, n, 1e-7f);
    free(ref); free(in); free(cout);
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
        float* x = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) x[i] = rnd();
        char label[32];
        snprintf(label, sizeof(label), "random n=%zu", n);
        run_case(label, x, n);
        free(x);
    }

    /* 2) 全相等 */
    {
        size_t n = 1000;
        float* x = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) x[i] = 3.0f;
        run_case("all equal = 3.0", x, n);
        free(x);
    }

    /* 3) 极大正值 */
    {
        size_t n = 500;
        float* x = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) x[i] = 1000.0f + rnd() * 0.1f;
        run_case("large positive ~1000", x, n);
        free(x);
    }

    /* 4) 极大负值（会下溢到 0，需数值稳定） */
    {
        size_t n = 500;
        float* x = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) x[i] = -1000.0f + rnd() * 0.1f;
        run_case("large negative ~-1000", x, n);
        free(x);
    }

    /* 5) 极端跨度（同时含极大/极小，考验减最大值稳定性） */
    {
        size_t n = 2000;
        float* x = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++)
            x[i] = (i % 2 == 0) ? 50.0f : -50.0f;
        run_case("mixed +/-50", x, n);
        free(x);
    }

    /* 6) n = 0 边界 */
    run_case("empty", NULL, 0);

    if (g_fail) {
        printf("\n==== RESULT: FAIL ====\n");
        return 1;
    }
    printf("\n==== RESULT: ALL PASS ====\n");
    return 0;
}
