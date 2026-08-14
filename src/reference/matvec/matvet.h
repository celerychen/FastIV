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
 * matvet.h - 矩阵-向量乘法 (纯 C / AVX2+FMA / NEON 三套独立实现)
 *
 * 约定:
 *   - ivf32 在本项目为 float (标量)。若你的工程已定义 ivf32, 在包含本头文件前
 *     #define IVF32_DEFINED, 本文件不再重复定义。
 *   - 所有维度整数均以 float 元素为单位。
 *   - mat 行主序存储, 第 i 行起始于 mat + i*mat_stride, mat_stride >= cols (可含 padding)。
 *   - mat_mul_vet_real32*:        dst = mat · vec,      dst 长度 = rows, vec 长度 = cols。
 *   - mat_t_mul_vet_real32*:      dst = mat^T · vec,    dst 长度 = cols, vec 长度 = rows。
 *   调用方须保证 dst / vec 缓冲区尺寸足够; 本实现不做越界检查 (与原始接口一致)。
 *
 * 实现选择:
 *   *          纯 C 标量/循环展开版 (无特殊指令)
 *   *_avx      AVX2 + FMA (乘累加 _mm256_fmadd_ps), 仅在编译目标支持 AVX2 时可用
 *   *_neon     ARM NEON (乘累加 vmlaq_f32), 仅在 ARM 目标可用
 * 不同指令集拆成不同函数, 由调用方按平台显式选择 (或自行为其加运行时分派)。
 */

#ifndef MATVET_H
#define MATVET_H

#ifndef IVF32_DEFINED
typedef float ivf32;
#define IVF32_DEFINED
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============ mat · vec : dst[0..rows) = mat[rows×cols] · vec[0..cols) ============ */
void mat_mul_vet_real32      (ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);
void mat_mul_vet_real32_avx  (ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);
void mat_mul_vet_real32_neon (ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);

/* ============ mat^T · vec : dst[0..cols) = mat[rows×cols]^T · vec[0..rows) ============ */
void mat_t_mul_vet_real32      (ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);
void mat_t_mul_vet_real32_avx  (ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);
void mat_t_mul_vet_real32_neon (ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec);

#ifdef __cplusplus
}
#endif

#endif /* MATVET_H */
