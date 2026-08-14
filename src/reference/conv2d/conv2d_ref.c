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

#include <stddef.h>
#include <stdint.h>

typedef float ivf32;

void conv2d_v0(ivf32* dst, int width_dst, int height_dst, int stride_dst,
            ivf32* src, int width_src, int height_src, int stride_src,
            ivf32 coef[9])
{
    int i, j, ki, kj;
    
    for (j = 0; j < height_dst; j++) {
        for (i = 0; i < width_dst; i++) {
            ivf32 sum = 0.0f;
            
            for (kj = 0; kj < 3; kj++) {
                for (ki = 0; ki < 3; ki++) {
                    int src_x = i + ki - 1;
                    int src_y = j + kj - 1;
                    
                    // Replicate padding
                    if (src_x < 0) src_x = 0;
                    if (src_x >= width_src) src_x = width_src - 1;
                    if (src_y < 0) src_y = 0;
                    if (src_y >= height_src) src_y = height_src - 1;
                    
                    sum += src[src_y * stride_src + src_x] * coef[kj * 3 + ki];
                }
            }
            dst[j * stride_dst + i] = sum;
        }
    }
}
