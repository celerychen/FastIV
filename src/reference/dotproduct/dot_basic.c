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

#include "dotproduct.h"

/*
 * 参考实现（标量 C）。最朴素的写法：一次循环一个乘加。
 *   s = 0; for i: s += a[i]*b[i];
 * 全程 float 累加，单条串行依赖链，吞吐受限于一条 FADD/FMUL 链。
 */
float dot_basic(const float* a, const float* b, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}
