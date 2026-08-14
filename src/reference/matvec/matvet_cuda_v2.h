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
 * matvet_cuda_v2.h - 优化版 v2 声明 (shared-memory + block reduction, 参照 SGEMV 指南 sgemv_opt)
 *
 * 与 matvet_cuda.h / v1.h 同接口风格, 后缀 _cuda_v2 区分。
 *   mat_mul_vet_real32_cuda_v2      dst = mat · vec
 *   mat_t_mul_vet_real32_cuda_v2    dst = mat^T · vec
 */
#ifndef MATVET_CUDA_V2_H
#define MATVET_CUDA_V2_H

#ifndef IVF32_DEFINED
typedef float ivf32;
#define IVF32_DEFINED
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* dst = mat · vec */
void mat_mul_vet_real32_cuda_v2 (ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);
/* dst = mat^T · vec */
void mat_t_mul_vet_real32_cuda_v2(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);

/* 仅启动核函数 (供 benchmark; 设备指针由调用方管理) */
void matvec_v2_launch(const float* d_mat, const float* d_vec, float* d_dst, int rows, int cols, int stride);
void mat_t_vec_v2_launch(const float* d_mat, const float* d_vec, float* d_dst, int rows, int cols, int stride);

#ifdef __cplusplus
}
#endif

#endif /* MATVET_CUDA_V2_H */
