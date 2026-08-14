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
#include <stdlib.h>

typedef float ivf32;

/* ============================================================================
 * conv2d_3x1_1x3_v0 —— 可分离卷积(基础标量实现)
 *
 * 输入为 6 个系数：coef[0..2] = 行滤波(横向 1x3)，coef[3..5] = 列滤波(纵向 3x1)。
 * 计算：
 *   1) 行滤波(横)：tmp[y][x] = coef[0]*src[y][x-1] + coef[1]*src[y][x] + coef[2]*src[y][x+1]
 *   2) 列滤波(纵)：dst[j][i] = coef[3]*tmp[j-1][i] + coef[4]*tmp[j][i] + coef[5]*tmp[j+1][i]
 *
 * 无 pivot、无分解、直接算。对可分离核 coef=V⊗H 等价于直接 3x3 卷积
 * （每行/列各自独立做 replicate 钳制，数学上等价于 v0 的二维独立钳制）。
 * 每输出像素仅 6 次乘累加，比 v0 的 9 次少 1/3。
 *
 * 注：若实际核不可分离，调用方应在外部把 3x3 核近似成 (行,列) 一对 1D 滤波器，
 *     再传入这 6 个系数；本函数只做可分离计算，不负责分解。
 * ==========================================================================*/

void conv2d_3x1_1x3_v0(ivf32* dst, int width_dst, int height_dst, int stride_dst,
                       ivf32* src, int width_src, int height_src, int stride_src,
                       ivf32 coef[6])
{
    /* ---- 直接取 6 个系数：前 3 个行滤波，后 3 个列滤波 ---- */
    ivf32 h0 = coef[0], h1 = coef[1], h2 = coef[2];   /* 行滤波(横向) */
    ivf32 v0 = coef[3], v1 = coef[4], v2 = coef[5];   /* 列滤波(纵向) */

    /* ---- 1) 行滤波：对整张源图做横向 1D，结果写入 tmp(全源尺寸) ---- */
    ivf32* tmp = (ivf32*)malloc((size_t)height_src * width_src * sizeof(ivf32));

    for (int y = 0; y < height_src; y++) {
        ivf32* srow = src + (size_t)y * stride_src;
        ivf32* trow = tmp + (size_t)y * width_src;
        for (int x = 0; x < width_src; x++) {
            int xl = x - 1; if (xl < 0) xl = 0; if (xl >= width_src) xl = width_src - 1;
            int xr = x + 1; if (xr < 0) xr = 0; if (xr >= width_src) xr = width_src - 1;
            trow[x] = h0 * srow[xl] + h1 * srow[x] + h2 * srow[xr];
        }
    }

    /* ---- 2) 列滤波：在 tmp 上做纵向 1D，写入 dst ---- */
    for (int j = 0; j < height_dst; j++) {
        int yt = j - 1; if (yt < 0) yt = 0; if (yt >= height_src) yt = height_src - 1;
        int ym = j;     if (ym < 0) ym = 0; if (ym >= height_src) ym = height_src - 1;
        int yb = j + 1; if (yb < 0) yb = 0; if (yb >= height_src) yb = height_src - 1;
        ivf32* tT = tmp + (size_t)yt * width_src;
        ivf32* tM = tmp + (size_t)ym * width_src;
        ivf32* tB = tmp + (size_t)yb * width_src;
        ivf32* drow = dst + (size_t)j * stride_dst;
        for (int i = 0; i < width_dst; i++) {
            drow[i] = v0 * tT[i] + v1 * tM[i] + v2 * tB[i];
        }
    }

    free(tmp);
}
