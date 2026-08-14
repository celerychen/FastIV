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

#ifndef MEAN_VAR_H
#define MEAN_VAR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 与 mean_var.c 中定义保持严格一致 */
typedef struct {
    float mean;
    float variance;
} Stats;

/* ---- 原始实现（mean_var.c） ---- */
Stats mean_variance_serial(const float arr[], size_t n);
Stats mean_variance_avx2_welford(const float* arr, size_t n);

/* ---- 新增基线（baselines.c） ---- */
/* 标量单趟 sum / sum-of-squares，float 累加 */
Stats mean_variance_scalar_sumsq(const float* arr, size_t n);
/* 与 mean_variance_serial 完全同源，但用 -O3 -march=native 编译，考察自动向量化。
 * 由 mean_var.c 经 -Dmean_variance_serial=... 重命名生成，无独立源文件。 */
Stats mean_variance_serial_autovec(const float arr[], size_t n);
/* 原始 AVX2 实现用 -O3 -march=native 重编译的版本（允许 FMA 收缩） */
Stats mean_variance_avx2_welford_o3(const float* arr, size_t n);

/* ---- 优化版（mean_var_opt.c）：仍是 Welford，仅做工程优化 ---- */
/* AVX2 + FMA + 快速倒数 + 2 路展开，float 累加（与原始同数值特性） */
Stats mean_variance_avx2_welford_opt_f(const float* arr, size_t n);
/* 同上，但 mean/M2 用 double 累加（同一套递推，修复大均值抵消） */
Stats mean_variance_avx2_welford_opt_d(const float* arr, size_t n);
/* 单条 8 宽 lane + FMA（隔离实验：对比「FMA」与「2 路展开」的加速贡献） */
Stats mean_variance_avx2_welford_fma(const float* arr, size_t n);
/* 带运行时 CPU 特性分发的对外入口 */
Stats mean_variance_dispatch(const float* arr, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* MEAN_VAR_H */
