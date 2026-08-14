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
 * matvet_cuda_v4.h - CUDA 矩阵-向量乘法 (v4: 共享内存 tile + float4 向量化叠加)
 *
 * v4 = v3 的共享内存 tile 加载 与 v1 的 float4 向量化 叠加:
 *   - 向量 x 按 256 一块 tile 协作搬进共享内存, 搬运用 float4 (64 线程各搬一个 float4);
 *   - 内层点积同样用 float4 一次吃 4 个连续列/行, 减少指令数与访存事务;
 *   - 仅当 指针 16B 对齐 且 stride 为 4 的倍数 时走 float4 路径, 否则退回 v3 标量等价,
 *     保证任意 stride / 任意尺寸都正确 (不依赖 malloc 对齐运气)。
 *   mat·vec   : 一个 block 负责一行, 每线程管 4 连续列, 块内归约, tid==0 写 dst[row]。
 *   mat^T·vec : 一个 block 负责 256 列, 每线程一列, x 按 tile 用 float4 搬共享, 内层 unroll-4。
 *
 *   mat_mul_vet_real32_cuda_v4    dst = mat · vec
 *   mat_t_mul_vet_real32_cuda_v4  dst = mat^T · vec
 */

#ifndef MATVET_CUDA_V4_H
#define MATVET_CUDA_V4_H

#ifndef IVF32_DEFINED
typedef float ivf32;
#define IVF32_DEFINED
#endif

#ifdef __cplusplus
extern "C" {
#endif

void mat_mul_vet_real32_cuda_v4 (ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);
void mat_t_mul_vet_real32_cuda_v4(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);

void matvec_v4_launch(const float* d_mat, const float* d_vec, float* d_dst, int rows, int cols, int stride);
void mat_t_vec_v4_launch(const float* d_mat, const float* d_vec, float* d_dst, int rows, int cols, int stride);

#ifdef __cplusplus
}
#endif

#endif /* MATVET_CUDA_V4_H */
