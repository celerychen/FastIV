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

#ifndef DOTPRODUCT_H
#define DOTPRODUCT_H

#include <stddef.h>

/*
 * 两个 float 数组的内积（点积 / inner product）：sum_{i} a[i]*b[i]。
 * 仅支持 float 输入、float 累加、返回 float。
 *
 *   dot_basic : 参考实现（标量 C）。一次循环一个乘加 s += a[i]*b[i]。
 *   dot_avx   : AVX2+FMA 向量化加速版。256-bit 一次吃 8 个 float，
 *               主循环「4 路展开」（每次迭代 32 个 float，4 条独立 FMA
 *               依赖链重叠飞行），吃满 FMA 端口、掩盖长延迟；尾部按
 *               8 对齐 + 标量收尾。需 CPU 支持 AVX2+FMA，否则触发
 *               非法指令（ILLEGAL INSTRUCTION）。数学结果与 basic 等价
 *               （浮点重排导致的末位误差在 1e-5 量级，见 test_dotproduct）。
 */
float dot_basic(const float* a, const float* b, size_t n);
float dot_avx(const float* a, const float* b, size_t n);
float dot_avx_v2(const float* a, const float* b, size_t n);
float dot_avx_sp(const float* a, const float* b, size_t n);     /* FMA + 软件流水线 */
float dot_avx_v2_sp(const float* a, const float* b, size_t n);  /* dp  + 软件流水线 */

#endif /* DOTPRODUCT_H */
