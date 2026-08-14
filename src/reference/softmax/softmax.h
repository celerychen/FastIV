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

#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <stddef.h>

/*
 * 原地（in-place）softmax，仅支持 float。
 * x 既是输入也是输出；返回后 x[0..n-1] 满足 x[i] >= 0 且 sum(x) == 1（浮点精度内）。
 *
 *   softmax_basic  : 参考实现（标量 C）。先求 max，再算 exp 与 sum，最后归一化。
 *   softmax_avx_v1 : AVX2+FMA 向量化（256-bit，8 个 float 并行）的最高效版本。
 *                    数学同 basic。要求 CPU 支持 AVX2+FMA，否则运行会触发非法指令
 *                    （ILLEGAL INSTRUCTION）。优化点：全程 float 累加；求 max 与 exp
 *                    主循环 4 路展开（一次 32 个 float，4 条独立 exp 依赖链重叠飞行）
 *                    填满 FMA 端口、掩盖 exp 长延迟；归一化趟 4 路展开。exp 近似等
 *                    辅助函数内联在同文件中（无额外依赖）。
 *   softmax_avx_v1_2: 与 v1 同结构，仅 exp 趟改用双向量交错版 exp256_ps2
 *                    （一次吃两块、两条 exp 依赖链交错发射、常数共用），用于和 v1 对比。
 *   softmax_avx_v1_3: 与 v1 同结构，仅 exp 趟改用「系数打包 + vpermilps」版 exp256_ps3
 *                    （6 个 Horner 系数打包进 2 个寄存器，立即数洗牌取用，不占额外寄存器、
 *                    不产生内存广播 load），用于和 v1/v1_2 对比常数装载策略。
 *   softmax_avx_v1_4: 与 v1_2 同结构（双向量交错），但 exp 趟改用 exp256_ps4——
 *                    双向量交错 + 系数打包共用（K0/K1 仅加载一次，x/y 两链共用，
 *                    每步 vpermilps 取用）。用于对比「双向量交错 + 系数打包」组合
 *                    是否比单纯的双向量交错（v1_2）更快。
 */
void softmax_basic(float* x, size_t n);
void softmax_avx_v1(float* x, size_t n);
void softmax_avx_v1_2(float* x, size_t n);
void softmax_avx_v1_3(float* x, size_t n);
void softmax_avx_v1_4(float* x, size_t n);

#endif /* SOFTMAX_H */
