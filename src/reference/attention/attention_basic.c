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
 * attention_basic.c — Decoder-only 单头自注意力 参考实现（标量 C，float）
 *
 * 设计参考（详见 attention.h 头部注释）：
 *   - GEMM 接口约定借鉴 matmul/fiv_small_matrix_mul_matrix_real32
 *     （行主序 + stride；并含 A·Bᵀ 变体用于 Q·Kᵀ）。因 attention 的矩阵乘从不清零累加，
 *     去掉 alpha/beta，仅做 C = A·B / C = A·Bᵀ。
 *     matmul 原版带 AVX2/FMA 优化，这里仅保留标量参考内核，定位与 softmax_basic 一致。
 *   - softmax 直接复用 softmax/softmax_basic（数值稳定：先减最大值再 exp）。
 */

#include "attention.h"
#include "softmax.h"   /* 复用 softmax_basic */
#include <math.h>
#include <stdlib.h>

/*
 * 参考 GEMM：C = A · B
 * 纯标量三循环；调用前不依赖 C 原有内容（从零写）。
 */
void attn_gemm(const float* A, int m, int k, int lda,
               const float* B, int n, int ldb,
               float* C, int ldc)
{
    for (int i = 0; i < m; i++) {
        const float* a = A + (size_t)i * lda;
        float* c = C + (size_t)i * ldc;

        for (int j = 0; j < n; j++) c[j] = 0.0f;   /* 先清零，再累加 */

        for (int p = 0; p < k; p++) {
            float av = a[p];
            const float* b = B + (size_t)p * ldb;
            for (int j = 0; j < n; j++)
                c[j] += av * b[j];
        }
    }
}

/*
 * 参考 GEMM（转置 B）：C = A · Bᵀ
 * 即 C[r][c] = Σ_t A[r][t] * B[c][t]
 * 用于分数矩阵 S = Q · Kᵀ。
 */
void attn_gemm_a_bT(const float* A, int m, int k, int lda,
                    const float* B, int n, int ldb,
                    float* C, int ldc)
{
    for (int i = 0; i < m; i++) {
        const float* a = A + (size_t)i * lda;
        float* c = C + (size_t)i * ldc;

        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            const float* b = B + (size_t)j * ldb;   /* B[j][*] = 转置后的第 j 列 */
            for (int t = 0; t < k; t++)
                s += a[t] * b[t];
            c[j] = s;
        }
    }
}

int attn_forward(const float* X,
                 const float* Wq, const float* Wk, const float* Wv, const float* Wo,
                 float* Y,
                 int T, int d_model, int d_k, int d_v,
                 int causal)
{
    if (T <= 0 || d_model <= 0 || d_k <= 0 || d_v <= 0)
        return -1;

    float* Q = (float*)malloc((size_t)T * d_k * sizeof(float));
    float* K = (float*)malloc((size_t)T * d_k * sizeof(float));
    float* V = (float*)malloc((size_t)T * d_v * sizeof(float));
    float* S = (float*)malloc((size_t)T * T * sizeof(float));
    float* O = (float*)malloc((size_t)T * d_v * sizeof(float));
    if (!Q || !K || !V || !S || !O) {
        free(Q); free(K); free(V); free(S); free(O);
        return -2;
    }

    /* 1) 投影：Q = X·Wq, K = X·Wk, V = X·Wv */
    attn_gemm(X, T, d_model, d_model, Wq, d_k, d_k, Q, d_k);
    attn_gemm(X, T, d_model, d_model, Wk, d_k, d_k, K, d_k);
    attn_gemm(X, T, d_model, d_model, Wv, d_v, d_v, V, d_v);

    /* 2) 分数 S = (Q · Kᵀ) / √d_k */
    attn_gemm_a_bT(Q, T, d_k, d_k, K, T, d_k, S, T);
    float scale = 1.0f / sqrtf((float)d_k);
    for (size_t i = 0; i < (size_t)T * T; i++) S[i] *= scale;

    /* 3) 因果掩码 + 逐行 softmax */
    for (int i = 0; i < T; i++) {
        if (causal) {
            /* decoder-only：位置 i 只能看到 j <= i，j > i 置 -∞（被 softmax 压成 0） */
            for (int j = i + 1; j < T; j++)
                S[(size_t)i * T + j] = -INFINITY;
        }
        softmax_basic(&S[(size_t)i * T], (size_t)T);
    }

    /* 4) 上下文 O = A · V */
    attn_gemm(S, T, T, T, V, d_v, d_v, O, d_v);

    /* 5) 输出投影 Y = O · Wo（可选） */
    if (Wo) {
        attn_gemm(O, T, d_v, d_v, Wo, d_model, d_model, Y, d_model);
    } else {
        for (size_t t = 0; t < (size_t)T * d_v; t++)
            Y[t] = O[t];
    }

    free(Q); free(K); free(V); free(S); free(O);
    return 0;
}
