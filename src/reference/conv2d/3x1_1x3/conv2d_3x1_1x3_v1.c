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
 * conv2d_3x1_1x3_v1 —— 可分离卷积(去大 buffer · in-place · 一次 4 行 · 热路径零分支)
 *
 * 输入为 6 个系数：coef[0..2] = 行滤波(横向 1x3)，coef[3..5] = 列滤波(纵向 3x1)。
 *
 * 改造要点：
 *   1) 去掉整图 tmp 大 buffer，改用 6 行小缓冲 buf（行滤波结果的滑动窗口）；
 *   2) 支持 src == dst 的 in-place 模式；
 *   3) 列滤波合并进同一 for y 行滤波循环，不再有独立第二段；
 *   4) 外层循环 y += 4：每次迭代行滤波 4 个源行，并用 6 行 buffer 一次性列滤波
 *      出最多 4 行输出（4 个独立累加器展开）。
 *
 * 分支处理：热路径（行滤波内部循环、列滤波内部循环）零分支；仅行首/行尾两列、
 *   顶/底输出行、图像底少数源行的装载做一次性钳制，非逐像素判断。
 *
 * 纯标量，无 NEON/汇编。每输出像素 6 次乘累加，等价于可分离 3x3 二维卷积。
 * ==========================================================================*/

/* 6 行环形缓冲里，源行 j（已行滤波）所在槽位；j 越界按 replicate 钳制。
 * w = 最近一次写入的源行 last_y 的槽位，故源行 j 的槽位 = (w - (last_y - j) + 6) % 6。 */
static inline ivf32* tmp_row(ivf32* buf, int w, int last_y, int ws, int hs, int j) {
    if (j < 0)   j = 0;
    if (j >= hs) j = hs - 1;
    int slot = (w - (last_y - j) + 6) % 6;
    return buf + (size_t)slot * ws;
}

/* 行滤波核心：对一条源行做横向 1x3，写入 brow。
 * 横向 replicate 钳制只发生在左右边界；内部列用原生 x-1/x+1，最内层零分支。 */
static inline void row_filter(ivf32* brow, const ivf32* srow, int W,
                              ivf32 h0, ivf32 h1, ivf32 h2) {
    if (W == 1) {                                   /* 退化：邻居全钳到自身 */
        brow[0] = (h0 + h1 + h2) * srow[0];
        return;
    }
    brow[0]   = (h0 + h1) * srow[0] + h2 * srow[1];             /* 左边界：xl 钳到 0   */
    for (int x = 1; x < W - 1; x++)                            /* 内部：无钳制        */
        brow[x] = h0 * srow[x - 1] + h1 * srow[x] + h2 * srow[x + 1];
    brow[W-1] = h0 * srow[W - 2] + (h1 + h2) * srow[W - 1];     /* 右边界：xr 钳到 W-1 */
}

/* 行滤波一个源行 yr 入 buf[ring]，并推进 ring / last_y（取代原 RF 宏）。 */
static inline void rf(ivf32* buf, int* ring, int* last_y, int yr,
                      const ivf32* src, int ws, int ss,
                      ivf32 h0, ivf32 h1, ivf32 h2) {
    int w = *ring;
    row_filter(buf + (size_t)w * ws, src + (size_t)yr * ss, ws, h0, h1, h2);
    *ring = (w + 1) % 6;
    *last_y = yr;
}

void conv2d_3x1_1x3_v1(ivf32* dst, int width_dst, int height_dst, int stride_dst,
                       ivf32* src, int width_src, int height_src, int stride_src,
                       ivf32 coef[6])
{
    /* 直接取 6 个系数：前 3 个行滤波，后 3 个列滤波 */
    ivf32 h0 = coef[0], h1 = coef[1], h2 = coef[2];   /* 行滤波(横向) */
    ivf32 v0 = coef[3], v1 = coef[4], v2 = coef[5];   /* 列滤波(纵向) */

    /* 6 行小缓冲（行滤波结果滑动窗口），取代整图 tmp */
    ivf32* buf = (ivf32*)malloc((size_t)6 * width_src * sizeof(ivf32));

    int ring = 0;        /* 下一个写入槽 */
    int w = 0;           /* 最近一次写入的源行 last_y 的槽位 */
    int last_y = -1;     /* 最近一次写入的源行号（收尾用） */
    int next_out = 0;    /* 下一个待输出的输出行 */

    /* 主循环：y += 4，每次处理 4 个源行，发射最多 4 行输出 */
    for (int y = 0; y < height_src; y += 4) {
        /* ---- 行滤波 4 个源行（越界跳过） ---- */
        rf(buf, &ring, &last_y, y, src, width_src, stride_src, h0, h1, h2);
        if (y + 1 < height_src) rf(buf, &ring, &last_y, y + 1, src, width_src, stride_src, h0, h1, h2);
        if (y + 2 < height_src) rf(buf, &ring, &last_y, y + 2, src, width_src, stride_src, h0, h1, h2);
        if (y + 3 < height_src) rf(buf, &ring, &last_y, y + 3, src, width_src, stride_src, h0, h1, h2);
        w = (ring - 1 + 6) % 6;            /* w = 最近写入源行 last_y 的槽位 */

        /* ---- 列滤波：buf 就绪后一次发射最多 4 行 [next_out .. next_out+3] ---- */
        int ready = last_y - 1;                   /* 当前可发射的最高输出行（含） */
        if (ready > height_dst - 1) ready = height_dst - 1;
        if (ready < 0) ready = -1;
        int avail = height_dst - next_out;        /* 剩余待输出行数 */
        if (avail > 4) avail = 4;
        if (avail > ready - next_out + 1) avail = ready - next_out + 1;  /* 不超过就绪范围 */
        if (avail > 0) {
            int j0 = next_out;
            ivf32* t0 = tmp_row(buf, w, last_y, width_src, height_src, j0 - 1);
            ivf32* t1 = tmp_row(buf, w, last_y, width_src, height_src, j0);
            ivf32* t2 = tmp_row(buf, w, last_y, width_src, height_src, j0 + 1);
            ivf32* t3 = tmp_row(buf, w, last_y, width_src, height_src, j0 + 2);
            ivf32* t4 = tmp_row(buf, w, last_y, width_src, height_src, j0 + 3);
            ivf32* t5 = tmp_row(buf, w, last_y, width_src, height_src, j0 + 4);
            ivf32* d0 = dst + (size_t)(j0 + 0) * stride_dst;
            ivf32* d1 = dst + (size_t)(j0 + 1) * stride_dst;
            ivf32* d2 = dst + (size_t)(j0 + 2) * stride_dst;
            ivf32* d3 = dst + (size_t)(j0 + 3) * stride_dst;
            /* 循环外提(loop unswitching)：avail 每 4 行块只算一次，在此按值分派，
             * 最内层 for(x) 循环变为纯直线 FMA，零分支。 */
            int W = width_dst;
            if (avail >= 4) {
                for (int x = 0; x < W; x++) {
                    d0[x] = v0 * t0[x] + v1 * t1[x] + v2 * t2[x];
                    d1[x] = v0 * t1[x] + v1 * t2[x] + v2 * t3[x];
                    d2[x] = v0 * t2[x] + v1 * t3[x] + v2 * t4[x];
                    d3[x] = v0 * t3[x] + v1 * t4[x] + v2 * t5[x];
                }
            } else if (avail == 3) {
                for (int x = 0; x < W; x++) {
                    d0[x] = v0 * t0[x] + v1 * t1[x] + v2 * t2[x];
                    d1[x] = v0 * t1[x] + v1 * t2[x] + v2 * t3[x];
                    d2[x] = v0 * t2[x] + v1 * t3[x] + v2 * t4[x];
                }
            } else if (avail == 2) {
                for (int x = 0; x < W; x++) {
                    d0[x] = v0 * t0[x] + v1 * t1[x] + v2 * t2[x];
                    d1[x] = v0 * t1[x] + v1 * t2[x] + v2 * t3[x];
                }
            } else { /* avail == 1 */
                for (int x = 0; x < W; x++) {
                    d0[x] = v0 * t0[x] + v1 * t1[x] + v2 * t2[x];
                }
            }
            next_out += avail;
        }
    }

    /* ---- 收尾：输出剩余行（用 buf 现存行 + 钳制） ---- */
    while (next_out < height_dst) {
        int j = next_out;
        ivf32* tT = tmp_row(buf, w, last_y, width_src, height_src, j - 1);
        ivf32* tM = tmp_row(buf, w, last_y, width_src, height_src, j);
        ivf32* tB = tmp_row(buf, w, last_y, width_src, height_src, j + 1);
        ivf32* drow = dst + (size_t)j * stride_dst;
        for (int x = 0; x < width_dst; x++) {
            drow[x] = v0 * tT[x] + v1 * tM[x] + v2 * tB[x];
        }
        next_out++;
    }

    free(buf);
}
