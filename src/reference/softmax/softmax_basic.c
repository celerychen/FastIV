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

#include "softmax.h"
#include <math.h>

/*
 * 参考实现（原地）。数值稳定：先减去最大值 m，使 exp 入参 <= 0，避免溢出。
 *   趟1：求 m = max(x)
 *   趟2：e_i = exp(x_i - m)，累加 sum，并原地写回 x
 *   趟3：x_i = e_i / sum
 * 固定缓冲下每个元素只调用 1 次 expf。
 */
void softmax_basic(float* x, size_t n) {
    if (n == 0) return;

    float m = x[0];
    for (size_t i = 1; i < n; i++)
        if (x[i] > m) m = x[i];

    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);
        sum += x[i];
    }

    for (size_t i = 0; i < n; i++)
        x[i] /= sum;
}
