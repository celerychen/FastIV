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
 * matvet_cuda.h - 朴素 CUDA 版矩阵-向量乘法 (声明)
 *
 * 与 matvet.h 同接口风格, 后缀 _cuda 区分; ivf32 占位守卫与 matvet.h 一致
 * (两处都 #define IVF32_DEFINED, 故同一翻译单元同时包含二者也不会重复 typedef)。
 *
 * 语义 (与 CPU 版完全相同):
 *   mat_mul_vet_real32_cuda      dst = mat · vec,      dst 长度 = rows, vec 长度 = cols
 *   mat_t_mul_vet_real32_cuda    dst = mat^T · vec,    dst 长度 = cols, vec 长度 = rows
 * 这些封装在内部完成 设备内存分配/拷贝/启动/拷回/释放, 调用方只给主机指针即可。
 */

#ifndef MATVET_CUDA_H
#define MATVET_CUDA_H

#ifndef IVF32_DEFINED
typedef float ivf32;
#define IVF32_DEFINED
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* dst = mat · vec */
void mat_mul_vet_real32_cuda (ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);
/* dst = mat^T · vec */
void mat_t_mul_vet_real32_cuda(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);

/* 仅启动核函数 (供 benchmark; 设备指针由调用方管理) */
void matvec_naive_launch(const float* d_mat, const float* d_vec, float* d_dst, int rows, int cols, int stride);
void mat_t_vec_naive_launch(const float* d_mat, const float* d_vec, float* d_dst, int rows, int cols, int stride);

#ifdef __cplusplus
}
#endif

#endif /* MATVET_CUDA_H */
