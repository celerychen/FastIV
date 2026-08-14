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

/* 边界像素：带钳制(replicate)，只用于 4 条边界带，不进热循环 */
static ivf32 conv2d_px_clamp(ivf32* src, int Ws, int Hs, int ss,
                             ivf32 coef[9], int i, int j)
{
    ivf32 sum = 0.0f;
    for (int kj = 0; kj < 3; kj++) {
        for (int ki = 0; ki < 3; ki++) {
            int sx = i + ki - 1;
            int sy = j + kj - 1;
            if (sx < 0) sx = 0;
            if (sx >= Ws) sx = Ws - 1;
            if (sy < 0) sy = 0;
            if (sy >= Hs) sy = Hs - 1;
            sum += src[sy * ss + sx] * coef[kj * 3 + ki];
        }
    }
    return sum;
}

void conv2d_v1(ivf32* dst, int width_dst, int height_dst, int stride_dst,
            ivf32* src, int width_src, int height_src, int stride_src,
            ivf32 coef[9])
{
    /* 系数提前取到局部变量 */
    ivf32 c0 = coef[0], c1 = coef[1], c2 = coef[2];
    ivf32 c3 = coef[3], c4 = coef[4], c5 = coef[5];
    ivf32 c6 = coef[6], c7 = coef[7], c8 = coef[8];

    /* 内部区域(无需钳制)的坐标范围:
     *   列 i: 需 i-1>=0 且 i+1<=Ws-1  =>  i ∈ [1, Ws-2]，同时 i<Wd
     *   行 j: 需 j-1>=0 且 j+1<=Hs-1  =>  j ∈ [1, Hs-2]，同时 j<Hd
     * 用半开区间 [start, end) 表示，若 end<=start 则内部为空。 */
    int i_start = 1;
    int i_end   = width_src  - 1;  if (i_end > width_dst)  i_end = width_dst;
    int j_start = 1;
    int j_end   = height_src - 1;  if (j_end > height_dst) j_end = height_dst;
    if (i_end < i_start) i_end = i_start;
    if (j_end < j_start) j_end = j_start;

    /* ---------- 1) 内部区域：无分支 + 9 项手动展开 ---------- */
    for (int j = j_start; j < j_end; j++) {
        ivf32* r0 = src + (size_t)(j - 1) * stride_src;
        ivf32* r1 = src + (size_t) j      * stride_src;
        ivf32* r2 = src + (size_t)(j + 1) * stride_src;
        ivf32* d  = dst + (size_t) j      * stride_dst;
        for (int i = i_start; i < i_end; i++) {
            ivf32 sum =
                r0[i - 1] * c0 + r0[i] * c1 + r0[i + 1] * c2 +
                r1[i - 1] * c3 + r1[i] * c4 + r1[i + 1] * c5 +
                r2[i - 1] * c6 + r2[i] * c7 + r2[i + 1] * c8;
            d[i] = sum;
        }
    }

    /* ---------- 2) 四条边界带：单独用带钳制的路径 ---------- */
    /* 上边：行 [0, j_start) 的整行 */
    for (int j = 0; j < j_start && j < height_dst; j++)
        for (int i = 0; i < width_dst; i++)
            dst[(size_t)j * stride_dst + i] =
                conv2d_px_clamp(src, width_src, height_src, stride_src, coef, i, j);

    /* 下边：行 [j_end, height_dst) 的整行 */
    for (int j = j_end; j < height_dst; j++)
        for (int i = 0; i < width_dst; i++)
            dst[(size_t)j * stride_dst + i] =
                conv2d_px_clamp(src, width_src, height_src, stride_src, coef, i, j);

    /* 左右边：仅内部行 [j_start, j_end)，列 [0, i_start) 和 [i_end, width_dst) */
    for (int j = j_start; j < j_end; j++) {
        for (int i = 0; i < i_start && i < width_dst; i++)
            dst[(size_t)j * stride_dst + i] =
                conv2d_px_clamp(src, width_src, height_src, stride_src, coef, i, j);
        for (int i = i_end; i < width_dst; i++)
            dst[(size_t)j * stride_dst + i] =
                conv2d_px_clamp(src, width_src, height_src, stride_src, coef, i, j);
    }
}
