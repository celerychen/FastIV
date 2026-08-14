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
 * matvet_cuda_cublas.h - 以 cuBLAS (cublasSgemv) 作为对标基线
 *
 * 行主序 mat (rows x cols, leading dim = stride) 如何喂给 cuBLAS:
 *   cuBLAS 默认把传入数组当列主序。我们的行主序矩阵 mat (rows x cols, lda=stride)
 *   当作列主序矩阵 reinterpret 时, 其"值"等价于一个 cols x rows 的列主序矩阵:
 *     dst = mat·vec   : m=cols, n=rows, op=T  -> op(A)=A^T 才是 rows x cols 的乘法
 *     dst = mat^T·vec : m=cols, n=rows, op=N  -> A 本身就是 cols x rows (= mat^T 维度)
 *   两种都令 lda=stride, 从而 lda(=stride=cols) >= m(=cols), 满足 cuBLAS 约束。
 *   alpha=1, beta=0。
 *
 *   mat_mul_vet_real32_cuda_cublas    dst = mat · vec
 *   mat_t_mul_vet_real32_cuda_cublas  dst = mat^T · vec
 */

#ifndef MATVET_CUDA_CUBLAS_H
#define MATVET_CUDA_CUBLAS_H

#ifndef IVF32_DEFINED
typedef float ivf32;
#define IVF32_DEFINED
#endif

#ifdef __cplusplus
extern "C" {
#endif

void mat_mul_vet_real32_cuda_cublas (ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);
void mat_t_mul_vet_real32_cuda_cublas(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);

void cublas_matvec_launch(const float* d_mat, const float* d_vec, float* d_dst, int rows, int cols, int stride);
void cublas_mat_t_vec_launch(const float* d_mat, const float* d_vec, float* d_dst, int rows, int cols, int stride);

#ifdef __cplusplus
}
#endif

#endif /* MATVET_CUDA_CUBLAS_H */
