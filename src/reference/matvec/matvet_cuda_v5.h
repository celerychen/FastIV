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
 * matvet_cuda_v5.h - CUDA 矩阵-向量乘法 (v5)
 *
 * v5 重点解决 mat^T·vec 在高瘦矩阵(如 8192x1024)上远落后 cuBLAS 的问题:
 *   根因 = 旧版按"输出列"并行 (一个 block 管 256 列), 高瘦形状 cols 小 -> block 极少
 *          -> 绝大多数 SM 空转 -> 访存流水线饿死。
 *   改法 = 把 GEMV 当 GEMM 做, 2D 分块: grid=(colTiles, rowTiles),
 *          block 内共享内存暂存 vec 段与 A 子块(按行读->列连续->合并),
 *          每线程管 4 列做点积, 写"部分积"到 part[rowTile*cols+col],
 *          再由一个极轻归约核求和得 dst。part 缓冲只有 A 的 1/rowTile, 带宽代价可忽略。
 *
 *   mat_mul_vet_real32_cuda_v5    dst = mat · vec   (复用 v4 的 float4 tile 实现, 已达 cuBLAS 水平)
 *   mat_t_mul_vet_real32_cuda_v5  dst = mat^T · vec (v5 新 2D 分块实现)
 */

#ifndef MATVET_CUDA_V5_H
#define MATVET_CUDA_V5_H

#ifndef IVF32_DEFINED
typedef float ivf32;
#define IVF32_DEFINED
#endif

#ifdef __cplusplus
extern "C" {
#endif

void mat_mul_vet_real32_cuda_v5 (ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);
void mat_t_mul_vet_real32_cuda_v5(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);

void matvec_v5_launch(const float* d_mat, const float* d_vec, float* d_dst, int rows, int cols, int stride);
void mat_t_vec_v5_launch(const float* d_mat, const float* d_vec, float* d_dst, int rows, int cols, int stride);

#ifdef __cplusplus
}
#endif

#endif /* MATVET_CUDA_V5_H */
