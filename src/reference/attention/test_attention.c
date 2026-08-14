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

/*
 * test_attention.c — 单头自注意力参考实现 正确性测试
 *
 * 校验策略（全程仅 float，不引入 double）：
 *   1) 朴素 float 参考实现（plain nested loops，不使用 attn_gemm）作为对照，
 *      与被测实现互相独立（不同索引/求和顺序），能真正暴露算法/索引/转置错误。
 *   2) 一个“手算可验证”的精确用例做硬性锚定（期望值全为 0 / 0.5 / 1.0，bit 级可比对）。
 */

#include "attention.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

static int g_fail = 0;

/* ---------------- 朴素 float 参考（独立重写，plain loops，不使用 attn_gemm） ---------------- */

static void naive_softmax(float* x, size_t n)
{
    if (n == 0) return;
    float m = x[0];
    for (size_t i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    for (size_t i = 0; i < n; i++) x[i] /= s;
}

/* C = A·B, A:(m,k) B:(k,n)，行主序、连续存储 */
static void naive_matmul(const float* A, const float* B, float* C, int m, int k, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int p = 0; p < k; p++)
                s += A[(size_t)i * k + p] * B[(size_t)p * n + j];
            C[(size_t)i * n + j] = s;
        }
}

/* C = A·Bᵀ, A:(m,k) B:(n,k) -> (m,n) */
static void naive_matmul_a_bT(const float* A, const float* B, float* C, int m, int k, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int t = 0; t < k; t++)
                s += A[(size_t)i * k + t] * B[(size_t)j * k + t];
            C[(size_t)i * n + j] = s;
        }
}

/* 与 attn_forward 同算法，朴素 float 实现（全程不使用 attn_gemm） */
static void attn_ref(const float* X, const float* Wq, const float* Wk,
                     const float* Wv, const float* Wo,
                     float* Y, int T, int d_model, int d_k, int d_v, int causal)
{
    float* Q = (float*)malloc((size_t)T * d_k * sizeof(float));
    float* K = (float*)malloc((size_t)T * d_k * sizeof(float));
    float* V = (float*)malloc((size_t)T * d_v * sizeof(float));
    float* S = (float*)malloc((size_t)T * T  * sizeof(float));
    float* O = (float*)malloc((size_t)T * d_v * sizeof(float));

    naive_matmul(X, Wq, Q, T, d_model, d_k);
    naive_matmul(X, Wk, K, T, d_model, d_k);
    naive_matmul(X, Wv, V, T, d_model, d_v);

    float scale = 1.0f / sqrtf((float)d_k);
    naive_matmul_a_bT(Q, K, S, T, d_k, T);
    for (size_t i = 0; i < (size_t)T * T; i++) S[i] *= scale;

    for (int i = 0; i < T; i++) {
        if (causal)
            for (int j = i + 1; j < T; j++) S[(size_t)i * T + j] = -INFINITY;
        naive_softmax(&S[(size_t)i * T], (size_t)T);
    }

    naive_matmul(S, V, O, T, T, d_v);

    if (Wo) naive_matmul(O, Wo, Y, T, d_v, d_model);
    else    memcpy(Y, O, (size_t)T * d_v * sizeof(float));

    free(Q); free(K); free(V); free(S); free(O);
}

/* ---------------- 简单确定性随机数（xorshift） ---------------- */
static unsigned int rng_state = 0x9e3779b9u;
static float rndf(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state & 0xffffffu) / (float)(1u << 24)) * 4.0f - 2.0f;
}

/* ---------------- 随机用例：与朴素 float 参考比对 ---------------- */
static void check_case(const char* label, int T, int d_model, int d_k, int d_v, int causal, int use_wo)
{
    size_t nX  = (size_t)T * d_model;
    size_t nQ  = (size_t)d_model * d_k;
    size_t nK  = (size_t)d_model * d_k;
    size_t nV  = (size_t)d_model * d_v;
    size_t out = use_wo ? (size_t)T * d_model : (size_t)T * d_v;

    float* X  = (float*)malloc(nX * sizeof(float));
    float* Wq = (float*)malloc(nQ * sizeof(float));
    float* Wk = (float*)malloc(nK * sizeof(float));
    float* Wv = (float*)malloc(nV * sizeof(float));
    float* Wo = use_wo ? (float*)malloc((size_t)d_v * d_model * sizeof(float)) : NULL;
    float* Y  = (float*)malloc(out * sizeof(float));
    float* Yr = (float*)malloc(out * sizeof(float));

    for (size_t i = 0; i < nX; i++) X[i]  = rndf();
    for (size_t i = 0; i < nQ; i++) Wq[i] = rndf();
    for (size_t i = 0; i < nK; i++) Wk[i] = rndf();
    for (size_t i = 0; i < nV; i++) Wv[i] = rndf();
    if (use_wo) for (size_t i = 0; i < (size_t)d_v * d_model; i++) Wo[i] = rndf();

    int rc = attn_forward(X, Wq, Wk, Wv, Wo, Y, T, d_model, d_k, d_v, causal);
    if (rc != 0) { printf("FAIL %s: attn_forward returned %d\n", label, rc); g_fail = 1; }
    attn_ref(X, Wq, Wk, Wv, Wo, Yr, T, d_model, d_k, d_v, causal);

    double max_abs = 0.0, denom = 1.0;
    for (size_t i = 0; i < out; i++) {
        double d = fabsf(Y[i] - Yr[i]);
        if (d > max_abs) max_abs = d;
        double a = fabsf(Yr[i]);
        if (a > denom) denom = a;
    }
    double norm_err = max_abs / denom;
    /* 两路均为 float，不同求和顺序可能带来微小舍入；容忍相对 1e-3，
     * 或绝对差极小（近零输出）亦视为通过。 */
    int ok = (norm_err <= 1e-3) || (max_abs <= 1e-5);
    if (!ok) {
        printf("FAIL %s: norm_rel_err=%g (max_abs=%.3e) exceeds tol\n", label, norm_err, max_abs);
        g_fail = 1;
    } else {
        printf("PASS %-34s T=%d d_model=%d d_k=%d d_v=%d causal=%d%s  (max_abs=%.3e norm_rel=%.3e)\n",
               label, T, d_model, d_k, d_v, causal, use_wo ? "" : " no_Wo", max_abs, norm_err);
    }

    free(X); free(Wq); free(Wk); free(Wv); free(Wo); free(Y); free(Yr);
}

/* ---------------- 手算精确锚定用例 ---------------- */
static void check_exact(void)
{
    /* T=2, d_model=2, d_k=1, d_v=2, causal=1
     *   X = I, Wq=Wk=[1;1], Wv=Wo=I, scale=1/√1=1
     *   Q=K=[1;1], S=[[1,1],[1,1]] -> 因果+softmax -> A=[[1,0],[0.5,0.5]]
     *   V=I, O=A·V=[[1,0],[0.5,0.5]], Y=O·Wo = [[1,0],[0.5,0.5]]
     * 期望值全为 0 / 0.5 / 1.0，参与运算的中间量均为可精确表示的数，可作 bit 级锚定。 */
    int T = 2, d_model = 2, d_k = 1, d_v = 2, causal = 1;
    float X[4]  = { 1,0, 0,1 };
    float Wq[2] = { 1,1 };          /* (d_model=2, d_k=1) */
    float Wk[2] = { 1,1 };
    float Wv[4] = { 1,0, 0,1 };     /* (d_model=2, d_v=2) */
    float Wo[4] = { 1,0, 0,1 };     /* (d_v=2, d_model=2) */
    float Y[4];
    float exp[4] = { 1.0f, 0.0f, 0.5f, 0.5f };

    int rc = attn_forward(X, Wq, Wk, Wv, Wo, Y, T, d_model, d_k, d_v, causal);
    if (rc != 0) { printf("FAIL exact: attn_forward returned %d\n", rc); g_fail = 1; return; }

    double max_abs = 0.0;
    for (int i = 0; i < 4; i++) {
        double d = fabsf(Y[i] - exp[i]);
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > 1e-5f) {
        printf("FAIL exact: max_abs=%.3e  got=[%g %g %g %g] exp=[%g %g %g %g]\n",
               max_abs, Y[0], Y[1], Y[2], Y[3], exp[0], exp[1], exp[2], exp[3]);
        g_fail = 1;
    } else {
        printf("PASS %-34s 手算精确锚定 (max_abs=%.3e)\n", "exact T=2 dk=1", max_abs);
    }
}

int main(void)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    printf("==== attention 参考实现 自测（float only，无 double） ====\n");

    check_exact();

    /* 含输出投影 Wo 的各种规模与因果开关 */
    check_case("tiny T=1",           1,  4,  4,  4, 1, 1);
    check_case("small T=4",          4,  8,  8,  8, 1, 1);
    check_case("square T=16 dk=8",  16, 16,  8,  8, 1, 1);
    check_case("causal T=32 dk=16", 32, 16, 16, 16, 1, 1);
    check_case("non-causal T=16",   16, 16, 16, 16, 0, 1);
    check_case("dv!=dk T=24",       24, 12,  8, 10, 1, 1);
    check_case("large T=64 dk=32",  64, 32, 32, 32, 1, 1);

    /* 无输出投影 Wo（Y 形状 T×d_v） */
    check_case("no_Wo T=8",   8,  8,  8,  8, 1, 0);
    check_case("no_Wo T=32", 32, 16, 12, 10, 1, 0);

    if (g_fail) {
        printf("\n==== RESULT: FAIL ====\n");
        return 1;
    }
    printf("\n==== RESULT: ALL PASS ====\n");
    return 0;
}
