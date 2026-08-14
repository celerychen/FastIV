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

#ifndef ATTENTION_H
#define ATTENTION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Decoder-only 单头（single-head）自注意力 参考实现（标量 C，float）。
 *
 * 设计参考：
 *   - 矩阵乘法借鉴 matmul/fiv_small_matrix_mul_matrix_real32 的接口约定
 *     （行主序、stride，以及 A·Bᵀ 变体），此处只保留标量参考实现
 *     （无 SIMD）。因 attention 的矩阵乘从不累加到已有矩阵，故去掉 alpha/beta，
 *     仅做 C = A·B / C = A·Bᵀ；与 softmax 目录里 softmax_basic 的定位一致。
 *   - softmax 直接复用 softmax/softmax_basic（数值稳定：先减最大值再 exp）。
 *
 * 计算流程（自注意力，X 为输入序列，T 为序列长度）：
 *   Q = X · Wq          (T, d_k)
 *   K = X · Wk          (T, d_k)
 *   V = X · Wv          (T, d_v)
 *   S = (Q · Kᵀ) / √d_k (T, T)             ← attn_gemm_a_bT
 *   if causal: S[i][j] = -∞  for j > i     ← decoder-only 因果掩码（下三角）
 *   A = softmax(S, 逐行)  (T, T)           ← 逐行调用 softmax_basic
 *   O = A · V           (T, d_v)           ← attn_gemm
 *   Y = O · Wo          (T, d_model)       ← 输出投影（Wo 可省，省则 Y=O）
 *
 * 说明：本实现为“参考实现”，目标是正确的算法与清晰的写法，不追求极致性能；
 *       多头（multi-head）不在本文件考虑范围内。
 */

/*
 * C = A · B
 *   A: (m, k) 行主序，行距 lda（>= k）
 *   B: (k, n) 行主序，行距 ldb（>= n）
 *   C: (m, n) 行主序，行距 ldc（>= n），调用前无需清零（此处从零写）
 */
void attn_gemm(const float* A, int m, int k, int lda,
               const float* B, int n, int ldb,
               float* C, int ldc);

/*
 * C = A · Bᵀ
 *   A: (m, k)    行主序，行距 lda（>= k）
 *   B: (n, k)    行主序（按行主序视为 n×k，故 Bᵀ 为 k×n），行距 ldb（>= k）
 *   C: (m, n)    行主序，行距 ldc（>= n），调用前无需清零
 * 等价： C[r][c] = Σ_t A[r][t] * B[c][t]
 * 该变体用于计算分数矩阵 S = Q · Kᵀ（对应 matmul 里的 matrix_mul_matrix_t）。
 */
void attn_gemm_a_bT(const float* A, int m, int k, int lda,
                    const float* B, int n, int ldb,
                    float* C, int ldc);

/*
 * 单头自注意力前向（decoder-only，可选因果掩码）
 *
 *   X        : (T, d_model) 行主序，输入序列
 *   Wq/Wk/Wv : (d_model, d_k) / (d_model, d_k) / (d_model, d_v) 行主序
 *   Wo       : (d_v, d_model) 行主序，输出投影；传 NULL 则跳过（此时 Y 形状为 T×d_v）
 *   Y        : 输出缓冲，至少容纳 (T, d_model)（有 Wo）或 (T, d_v)（无 Wo），行主序
 *   T, d_model, d_k, d_v : 维度
 *   causal   : 非零表示施加下三角因果掩码（decoder-only 默认开启；传 0 则为全连接注意力）
 *
 * 返回 0 表示成功；非 0（如 -1 维度非法、-2 内存分配失败）表示失败。
 */
int attn_forward(const float* X,
                 const float* Wq, const float* Wk, const float* Wv, const float* Wo,
                 float* Y,
                 int T, int d_model, int d_k, int d_v,
                 int causal);

#ifdef __cplusplus
}
#endif

#endif /* ATTENTION_H */
