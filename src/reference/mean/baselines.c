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

/* 参照基线实现
 *
 * mean_variance_scalar_sumsq:
 *   单趟 sum / sum-of-squares，float 累加，无除法在热循环内。
 *   目的是把「Welford 每元素一次除法的开销」与「SIMD 宽度带来的收益」
 *   两个因素拆开——否则拿含除法的串行版直接对比 AVX2，会得到远超
 *   8x 的虚高加速比。
 *   注意：这个算法数值上是最差的（经典灾难性抵消），正确性测试里
 *   会明确暴露这一点，它只作性能参照，不作推荐实现。
 */

#include "mean_var.h"

Stats mean_variance_scalar_sumsq(const float* arr, size_t n)
{
    Stats r = {0.0f, 0.0f};
    if (n == 0) return r;

    float s = 0.0f;
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float x = arr[i];
        s  += x;
        ss += x * x;
    }

    float mean = s / (float)n;
    float var  = ss / (float)n - mean * mean;
    if (var < 0.0f) var = 0.0f;   /* 抵消导致的负方差，钳到 0 */

    r.mean = mean;
    r.variance = var;
    return r;
}
