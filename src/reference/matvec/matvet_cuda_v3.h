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
 * matvet_cuda_v3.h - CUDA 矩阵-向量乘法 (v3: 严格照 SGEMV 指南 sgemv_opt)
 *
 * 共享内存是固定 __shared__ float xs[256] / vs[256], 不是跟向量一样大的缓存;
 * 向量按 256 一块 tile 协作搬进共享内存, 内层只扫当前 tile。
 *   mat·vec   : 一个 block 负责一行, 块内归约, tid==0 写 dst[row]。
 *   mat^T·vec : 一个 block 负责一列, 块内归约, tid==0 写 dst[col]。
 *
 *   mat_mul_vet_real32_cuda_v3    dst = mat · vec
 *   mat_t_mul_vet_real32_cuda_v3  dst = mat^T · vec
 */

#ifndef MATVET_CUDA_V3_H
#define MATVET_CUDA_V3_H

#ifndef IVF32_DEFINED
typedef float ivf32;
#define IVF32_DEFINED
#endif

#ifdef __cplusplus
extern "C" {
#endif

void mat_mul_vet_real32_cuda_v3 (ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);
void mat_t_mul_vet_real32_cuda_v3(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);

void matvec_v3_launch(const float* d_mat, const float* d_vec, float* d_dst, int rows, int cols, int stride);
void mat_t_vec_v3_launch(const float* d_mat, const float* d_vec, float* d_dst, int rows, int cols, int stride);

#ifdef __cplusplus
}
#endif

#endif /* MATVET_CUDA_V3_H */
