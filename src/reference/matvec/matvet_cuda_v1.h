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
 * matvet_cuda_v1.h - 优化版 CUDA 矩阵-向量乘法 (v1)
 *
 * v1 优化点 (相比朴素版 matvet_cuda):
 *   - 合并访存: mat·vec 改为"一个 block 负责一行, 线程按列分片",
 *     使对 mat / vec 的加载在 warp 内连续 -> 合并 (朴素版 mat 跨步读, 极慢)
 *   - 向量化加载: 用 float4 + __ldg 一次搬 16 字节, 指令/事务数降到 1/4
 *   - 块内归约: 用 warp shuffle (__shfl_down_sync) 在寄存器内完成, 不碰 shared memory
 *   - mat^T·vec: 朴素版本身已合并, v1 仅对 vec 内循环做 float4 向量化加载
 *
 * 语义同朴素版:
 *   mat_mul_vet_real32_cuda_v1    dst = mat · vec
 *   mat_t_mul_vet_real32_cuda_v1  dst = mat^T · vec
 */

#ifndef MATVET_CUDA_V1_H
#define MATVET_CUDA_V1_H

#ifndef IVF32_DEFINED
typedef float ivf32;
#define IVF32_DEFINED
#endif

#ifdef __cplusplus
extern "C" {
#endif

void mat_mul_vet_real32_cuda_v1 (ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);
void mat_t_mul_vet_real32_cuda_v1(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);

void matvec_v1_launch(const float* d_mat, const float* d_vec, float* d_dst, int rows, int cols, int stride);
void mat_t_vec_v1_launch(const float* d_mat, const float* d_vec, float* d_dst, int rows, int cols, int stride);

#ifdef __cplusplus
}
#endif

#endif /* MATVET_CUDA_V1_H */
