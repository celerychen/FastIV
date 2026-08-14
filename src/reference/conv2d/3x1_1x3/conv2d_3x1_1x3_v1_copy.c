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
#include <string.h>

typedef float ivf32;

/* ============================================================================
 * conv2d_3x1_1x3_v1 —— 可分离卷积(去大 buffer · in-place · 一次 4 行 · 零逐像素分支)
 *
 * 输入为 6 个系数：coef[0..2] = 行滤波(横向 1x3, H)，coef[3..5] = 列滤波(纵向 3x1, V)。
 *
 * 缓冲布局（按需求 redesign）：
 *   - 6 行环形缓冲 buf[6]，每行宽 width_src + 2。
 *   - 载入一行源数据时：slot[0] = srow[0](左复制)，memcpy(slot+1, srow, W)(中间整段)，
 *     slot[W+1] = srow[W-1](右复制)。这样水平滤波读 x-1/x/x+1 永远合法，无逐像素判断。
 *   - 水平滤波原地算（单寄存器延迟），结果直接填回该 slot。
 *
 * 环形复用：
 *   - 一次迭代载入若干源行做水平滤波入环形缓冲（绝不从 src 重新 copy 已算过的行），
 *     随后发射最多 4 行输出。缓冲恒为 6 行，内存 O(W) 而非 O(W*H)。
 *
 * 4 次纵向滤波需 5 行水平结果，槽位按滑动窗口：
 *   输出行 oo 取缓冲槽 {oo-1, oo, oo+1}(顶/底越界 replicate 钳到自身)。
 *   稳态下一个块 4 行输出即 {1,1,2},{1,2,3},{2,3,4},{3,4,5}(物理槽经环形偏移)。
 *
 * in-place 安全性：
 *   每次迭代先行滤波若干源行入 buf，后发射输出行。发射输出 oo 前已保证 src[oo+1]
 *   入环形缓冲，此后 src[oo] 不再从 src 读取(纵向滤波只读 buf)，故 dst[oo] 覆盖
 *   src[oo] 安全。src == dst 可正确运行。
 *
 * 热路径：行滤波(含载入)与列滤波的最内层循环均零分支；仅每行输出有 1 次顶/底
 *   replicate 钳制（每输出行一次，非逐像素），由编译器以条件移动实现，无预测惩罚。
 * ==========================================================================*/

/* 水平滤波一行 src -> slot（slot 行宽 = W+2，已含左右复制边界）。
 * 行首/行尾复制保证水平滤波读 x-1/x+1 不越界；水平滤波原地算，无逐像素分支。 */
static inline void row_hfilter(ivf32* slot, const ivf32* srow, int W,
                               ivf32 h0, ivf32 h1, ivf32 h2) {
    slot[0] = srow[0];                                       /* 左边界复制 */
    memcpy(slot + 1, srow, (size_t)W * sizeof(ivf32));       /* 中间整段 copy */
    slot[W + 1] = srow[W - 1];                               /* 右边界复制 */
    ivf32 saved = slot[0];                                  /* raw[-1] = 左复制 = raw[0] */
    for (int c = 1; c <= W; c++) {                          /* 原地水平滤波，零逐像素判断 */
        ivf32 rc = slot[c];
        ivf32 rcp1 = slot[c + 1];
        slot[c] = h0 * saved + h1 * rc + h2 * rcp1;
        saved = rc;
    }
}

void conv2d_3x1_1x3_v1(ivf32* dst, int width_dst, int height_dst, int stride_dst,
                       ivf32* src, int width_src, int height_src, int stride_src,
                       ivf32 coef[6])
{
    ivf32 h0 = coef[0], h1 = coef[1], h2 = coef[2];   /* 行滤波(横向 H) */
    ivf32 v0 = coef[3], v1 = coef[4], v2 = coef[5];   /* 列滤波(纵向 V) */

    int W = width_src;
    int roww = W + 2;                                 /* 每行带左右复制边界 */
    ivf32* buf = (ivf32*)malloc((size_t)6 * roww * sizeof(ivf32));

    int ring = 0;        /* 下一个写入的物理槽 */
    int last_y = -1;     /* 已载入的最高源行号 */

    int o = 0;
    while (o < height_dst) {
        /* 载入源行到 o+4（或图像底），保证本块 4 行输出的 +1 邻居就绪 */
        int ymax = o + 4;
        if (ymax > height_src - 1) ymax = height_src - 1;
        while (last_y < ymax) {
            int y = last_y + 1;
            int slot = ring;
            row_hfilter(buf + (size_t)slot * roww,
                        src + (size_t)y * stride_src, W, h0, h1, h2);
            ring = (ring + 1) % 6;
            last_y = y;
        }

        /* 本块内 源行 y 的物理槽 = (base + y) % 6；base 为块内不变的偏移 */
        int base = (ring - 1 - last_y + 6) % 6;

        /* 发射最多 4 行输出：oo 取缓冲槽 {oo-1, oo, oo+1}（顶/底 replicate） */
        int nout = height_dst - o;
        if (nout > 4) nout = 4;
        for (int k = 0; k < nout; k++) {
            int oo = o + k;
            int yt = oo - 1; if (yt < 0)            yt = 0;            /* 顶复制 */
            int yb = oo + 1; if (yb > height_src - 1) yb = height_src - 1; /* 底复制 */
            ivf32* bt = buf + (size_t)((base + yt) % 6) * roww;
            ivf32* bm = buf + (size_t)((base + oo) % 6) * roww;
            ivf32* bb = buf + (size_t)((base + yb) % 6) * roww;
            ivf32* d  = dst + (size_t)oo * stride_dst;
            for (int i = 0; i < width_dst; i++)                     /* 纵向滤波：零分支 */
                d[i] = v0 * bt[i + 1] + v1 * bm[i + 1] + v2 * bb[i + 1];
        }
        o += 4;
    }

    free(buf);
}
