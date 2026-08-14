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

#ifdef __aarch64__
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

typedef float ivf32;

/* 边界像素：带钳制(replicate)，标量，只用于 4 条边界带（NEON 不加速这里） */
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
            sum += src[(size_t)sy * ss + sx] * coef[kj * 3 + ki];
        }
    }
    return sum;
}

void conv2d_v6(ivf32* dst, int width_dst, int height_dst, int stride_dst,
            ivf32* src, int width_src, int height_src, int stride_src,
            ivf32 coef[9])
{
    ivf32 c0 = coef[0], c1 = coef[1], c2 = coef[2];
    ivf32 c3 = coef[3], c4 = coef[4], c5 = coef[5];
    ivf32 c6 = coef[6], c7 = coef[7], c8 = coef[8];

    /* 内部区域(无需钳制)的半开区间 [start, end)，退化时为空 */
    int i_start = 1;
    int i_end   = width_src  - 1;  if (i_end > width_dst)  i_end = width_dst;
    int j_start = 1;
    int j_end   = height_src - 1;  if (j_end > height_dst) j_end = height_dst;
    if (i_end < i_start) i_end = i_start;
    if (j_end < j_start) j_end = j_start;

    /* ---------- 1) 内部区域 ---------- */
#ifdef __aarch64__
    {
        /* v6 优化点：9 个系数不再各 dup 成 9 个向量寄存器，而是打包进
           3 个 4-lane 向量(每行 3 个系数装一个)，用 vfmaq_laneq_f32 做
           “矢量 × 标量(lane)”乘累加——系数寄存器由 9 个减为 3 个。
           这对 v5 内核至关重要：v5 同时活跃 4 累加器 + 每行 6 次源 load，
           原 9 个 dup 系数极易因寄存器不够而 spill 到栈；减到 3 个后
           寄存器压力大幅下降，更利于调度器隐藏 FMA/load 延迟。
           lane 映射：vcoef0={c0,c1,c2}, vcoef1={c3,c4,c5}, vcoef2={c6,c7,c8}；
           某源行归属的系数向量：源行 = 窗口第 0/1/2 行 -> vcoef0/1/2；
           列偏移 -1/0/+1 -> lane 0/1/2。 */
        float32x4_t vcoef0 = (float32x4_t){c0, c1, c2, 0.0f};
        float32x4_t vcoef1 = (float32x4_t){c3, c4, c5, 0.0f};
        float32x4_t vcoef2 = (float32x4_t){c6, c7, c8, 0.0f};

        int i_neon_end = i_start + ((i_end - i_start) / 8) * 8;  /* 8 列对齐(2×4-lane，无 vext) */
        int j_neon_end = j_start + ((j_end - j_start) / 2) * 2;  /* 2 行对齐(纵向展开) */

        /* 最外层 j += 2：一次算 2 行输出，纵向复用重叠的中间行(同 v4)。
           最内层 i += 8：一次算 8 列。每行 8 像素拆成 2 个 4-像素子块
           blockA(像素 i..i+3) / blockB(像素 i+4..i+7)，各用独立累加器，
           全部用普通 vld1q 直接加载(不用 vext)。每行 6 次 load/8px。
           4 个累加器(accA0/accA1 行 j，accB0/accB1 行 j+1)提升 ILP。
           与 v5 的唯一区别：乘累加改用 vfmaq_laneq_f32(矢量×标量)，
           系数寄存器 9 -> 3。 */
        for (int j = j_start; j < j_neon_end; j += 2) {
            ivf32* r0 = src + (size_t)(j - 1) * stride_src;  /* -> 窗口第 0 行源 */
            ivf32* r1 = src + (size_t) j      * stride_src;  /* 共享 */
            ivf32* r2 = src + (size_t)(j + 1) * stride_src;  /* 共享 */
            ivf32* r3 = src + (size_t)(j + 2) * stride_src;  /* -> 窗口第 2 行源(输出行 j+1 用) */
            ivf32* dA = dst + (size_t) j      * stride_dst;
            ivf32* dB = dst + (size_t)(j + 1) * stride_dst;

            int i;
            for (i = i_start; i < i_neon_end; i += 8) {
                /* ===== 输出行 j：blockA(accA0) / blockB(accA1) ===== */
                /* 源行 r0/r1/r2 分别对应系数 vcoef0/1/2；列偏移 -1/0/+1 对应 lane 0/1/2 */
                float32x4_t accA0 = vmulq_laneq_f32(vld1q_f32(r0 + i),     vcoef0, 1);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r0 + i - 1), vcoef0, 0);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r0 + i + 1), vcoef0, 2);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r1 + i - 1), vcoef1, 0);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r1 + i),     vcoef1, 1);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r1 + i + 1), vcoef1, 2);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r2 + i - 1), vcoef2, 0);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r2 + i),     vcoef2, 1);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r2 + i + 1), vcoef2, 2);

                float32x4_t accA1 = vmulq_laneq_f32(vld1q_f32(r0 + i + 4), vcoef0, 1);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r0 + i + 3), vcoef0, 0);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r0 + i + 5), vcoef0, 2);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r1 + i + 3), vcoef1, 0);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r1 + i + 4), vcoef1, 1);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r1 + i + 5), vcoef1, 2);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r2 + i + 3), vcoef2, 0);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r2 + i + 4), vcoef2, 1);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r2 + i + 5), vcoef2, 2);

                /* ===== 输出行 j+1：源行 r1/r2/r3 对应系数 vcoef0/1/2 ===== */
                float32x4_t accB0 = vmulq_laneq_f32(vld1q_f32(r1 + i),     vcoef0, 1);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r1 + i - 1), vcoef0, 0);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r1 + i + 1), vcoef0, 2);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r2 + i - 1), vcoef1, 0);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r2 + i),     vcoef1, 1);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r2 + i + 1), vcoef1, 2);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r3 + i - 1), vcoef2, 0);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r3 + i),     vcoef2, 1);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r3 + i + 1), vcoef2, 2);

                float32x4_t accB1 = vmulq_laneq_f32(vld1q_f32(r1 + i + 4), vcoef0, 1);
                accB1 = vfmaq_laneq_f32(accB1, vld1q_f32(r1 + i + 3), vcoef0, 0);
                accB1 = vfmaq_laneq_f32(accB1, vld1q_f32(r1 + i + 5), vcoef0, 2);
                accB1 = vfmaq_laneq_f32(accB1, vld1q_f32(r2 + i + 3), vcoef1, 0);
                accB1 = vfmaq_laneq_f32(accB1, vld1q_f32(r2 + i + 4), vcoef1, 1);
                accB1 = vfmaq_laneq_f32(accB1, vld1q_f32(r2 + i + 5), vcoef1, 2);
                accB1 = vfmaq_laneq_f32(accB1, vld1q_f32(r3 + i + 3), vcoef2, 0);
                accB1 = vfmaq_laneq_f32(accB1, vld1q_f32(r3 + i + 4), vcoef2, 1);
                accB1 = vfmaq_laneq_f32(accB1, vld1q_f32(r3 + i + 5), vcoef2, 2);

                vst1q_f32(dA + i,     accA0);
                vst1q_f32(dA + i + 4, accA1);
                vst1q_f32(dB + i,     accB0);
                vst1q_f32(dB + i + 4, accB1);
            }
            /* 末尾不足 8 列的标量收尾（仍是内部，无钳制），2 行一起算 */
            for (; i < i_end; i++) {
                dA[i] = r0[i-1]*c0 + r0[i]*c1 + r0[i+1]*c2
                      + r1[i-1]*c3 + r1[i]*c4 + r1[i+1]*c5
                      + r2[i-1]*c6 + r2[i]*c7 + r2[i+1]*c8;
                dB[i] = r1[i-1]*c0 + r1[i]*c1 + r1[i+1]*c2
                      + r2[i-1]*c3 + r2[i]*c4 + r2[i+1]*c5
                      + r3[i-1]*c6 + r3[i]*c7 + r3[i+1]*c8;
            }
        }
        /* 内部区域余下奇数行（j_end - j_neon_end == 1）走标量 9 项展开 */
        for (int j = j_neon_end; j < j_end; j++) {
            ivf32* r0 = src + (size_t)(j - 1) * stride_src;
            ivf32* r1 = src + (size_t) j      * stride_src;
            ivf32* r2 = src + (size_t)(j + 1) * stride_src;
            ivf32* d  = dst + (size_t) j      * stride_dst;
            for (int i = i_start; i < i_end; i++) {
                d[i] = r0[i-1]*c0 + r0[i]*c1 + r0[i+1]*c2
                     + r1[i-1]*c3 + r1[i]*c4 + r1[i+1]*c5
                     + r2[i-1]*c6 + r2[i]*c7 + r2[i+1]*c8;
            }
        }
    }
#elif defined(__AVX2__)
    {
        /* v6 优化点(AVX2 版)：不再把 9 个系数各 set1 到 9 个 YMM 常量寄存器，
           而是用 _mm256_broadcast_ss(&coef[k]) 在使用点从内存 load-broadcast，
           系数不长期占用寄存器——对应 NEON v6 “系数寄存器 9->3” 的等价思路：
           降低寄存器压力。x86 有 16 个 YMM，v5 内核同时活跃 4 累加器 + 每行 6
           次源 load，把常量交给 broadcast(可与 FMA 融合的 memory 操作数)后，
           更利于调度器隐藏 FMA/load 延迟。
           内核结构与 v5 完全一致：j+=2 纵向复用 + i+=16 横向、4 累加器。 */
        const ivf32* cf = coef;   /* 供 broadcast 的系数基址 */

        int i_simd_end = i_start + ((i_end - i_start) / 16) * 16; /* 16 列对齐(2×8-lane) */
        int j_pair_end = j_start + ((j_end - j_start) / 2) * 2;   /* 2 行对齐(纵向展开) */

        for (int j = j_start; j < j_pair_end; j += 2) {
            ivf32* r0 = src + (size_t)(j - 1) * stride_src;
            ivf32* r1 = src + (size_t) j      * stride_src;
            ivf32* r2 = src + (size_t)(j + 1) * stride_src;
            ivf32* r3 = src + (size_t)(j + 2) * stride_src;
            ivf32* dA = dst + (size_t) j      * stride_dst;
            ivf32* dB = dst + (size_t)(j + 1) * stride_dst;

            int i;
            for (i = i_start; i < i_simd_end; i += 16) {
                /* ===== 输出行 j：blockA(accA0) / blockB(accA1) ===== */
                __m256 accA0 = _mm256_mul_ps(_mm256_loadu_ps(r0 + i),     _mm256_broadcast_ss(cf + 1));
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i - 1), _mm256_broadcast_ss(cf + 0), accA0);
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i + 1), _mm256_broadcast_ss(cf + 2), accA0);
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i - 1), _mm256_broadcast_ss(cf + 3), accA0);
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i),     _mm256_broadcast_ss(cf + 4), accA0);
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 1), _mm256_broadcast_ss(cf + 5), accA0);
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i - 1), _mm256_broadcast_ss(cf + 6), accA0);
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i),     _mm256_broadcast_ss(cf + 7), accA0);
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 1), _mm256_broadcast_ss(cf + 8), accA0);

                __m256 accA1 = _mm256_mul_ps(_mm256_loadu_ps(r0 + i + 8), _mm256_broadcast_ss(cf + 1));
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i + 7), _mm256_broadcast_ss(cf + 0), accA1);
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i + 9), _mm256_broadcast_ss(cf + 2), accA1);
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 7), _mm256_broadcast_ss(cf + 3), accA1);
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 8), _mm256_broadcast_ss(cf + 4), accA1);
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 9), _mm256_broadcast_ss(cf + 5), accA1);
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 7), _mm256_broadcast_ss(cf + 6), accA1);
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 8), _mm256_broadcast_ss(cf + 7), accA1);
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 9), _mm256_broadcast_ss(cf + 8), accA1);

                /* ===== 输出行 j+1：源行 r1/r2/r3 ===== */
                __m256 accB0 = _mm256_mul_ps(_mm256_loadu_ps(r1 + i),     _mm256_broadcast_ss(cf + 1));
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i - 1), _mm256_broadcast_ss(cf + 0), accB0);
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 1), _mm256_broadcast_ss(cf + 2), accB0);
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i - 1), _mm256_broadcast_ss(cf + 3), accB0);
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i),     _mm256_broadcast_ss(cf + 4), accB0);
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 1), _mm256_broadcast_ss(cf + 5), accB0);
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i - 1), _mm256_broadcast_ss(cf + 6), accB0);
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i),     _mm256_broadcast_ss(cf + 7), accB0);
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i + 1), _mm256_broadcast_ss(cf + 8), accB0);

                __m256 accB1 = _mm256_mul_ps(_mm256_loadu_ps(r1 + i + 8), _mm256_broadcast_ss(cf + 1));
                accB1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 7), _mm256_broadcast_ss(cf + 0), accB1);
                accB1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 9), _mm256_broadcast_ss(cf + 2), accB1);
                accB1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 7), _mm256_broadcast_ss(cf + 3), accB1);
                accB1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 8), _mm256_broadcast_ss(cf + 4), accB1);
                accB1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 9), _mm256_broadcast_ss(cf + 5), accB1);
                accB1 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i + 7), _mm256_broadcast_ss(cf + 6), accB1);
                accB1 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i + 8), _mm256_broadcast_ss(cf + 7), accB1);
                accB1 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i + 9), _mm256_broadcast_ss(cf + 8), accB1);

                _mm256_storeu_ps(dA + i,     accA0);
                _mm256_storeu_ps(dA + i + 8, accA1);
                _mm256_storeu_ps(dB + i,     accB0);
                _mm256_storeu_ps(dB + i + 8, accB1);
            }
            /* 末尾不足 16 列的标量收尾（仍是内部，无钳制），2 行一起算 */
            for (; i < i_end; i++) {
                dA[i] = r0[i-1]*c0 + r0[i]*c1 + r0[i+1]*c2
                      + r1[i-1]*c3 + r1[i]*c4 + r1[i+1]*c5
                      + r2[i-1]*c6 + r2[i]*c7 + r2[i+1]*c8;
                dB[i] = r1[i-1]*c0 + r1[i]*c1 + r1[i+1]*c2
                      + r2[i-1]*c3 + r2[i]*c4 + r2[i+1]*c5
                      + r3[i-1]*c6 + r3[i]*c7 + r3[i+1]*c8;
            }
        }
        /* 内部区域余下奇数行（j_end - j_pair_end == 1）走标量 9 项展开 */
        for (int j = j_pair_end; j < j_end; j++) {
            ivf32* r0 = src + (size_t)(j - 1) * stride_src;
            ivf32* r1 = src + (size_t) j      * stride_src;
            ivf32* r2 = src + (size_t)(j + 1) * stride_src;
            ivf32* d  = dst + (size_t) j      * stride_dst;
            for (int i = i_start; i < i_end; i++) {
                d[i] = r0[i-1]*c0 + r0[i]*c1 + r0[i+1]*c2
                     + r1[i-1]*c3 + r1[i]*c4 + r1[i+1]*c5
                     + r2[i-1]*c6 + r2[i]*c7 + r2[i+1]*c8;
            }
        }
    }
#else
    /* 非 aarch64/AVX2：直接走标量 9 项展开（与 v1 同构） */
    for (int j = j_start; j < j_end; j++) {
        ivf32* r0 = src + (size_t)(j - 1) * stride_src;
        ivf32* r1 = src + (size_t) j      * stride_src;
        ivf32* r2 = src + (size_t)(j + 1) * stride_src;
        ivf32* d  = dst + (size_t) j      * stride_dst;
        for (int i = i_start; i < i_end; i++) {
            d[i] = r0[i-1]*c0 + r0[i]*c1 + r0[i+1]*c2
                 + r1[i-1]*c3 + r1[i]*c4 + r1[i+1]*c5
                 + r2[i-1]*c6 + r2[i]*c7 + r2[i+1]*c8;
        }
    }
#endif

    /* ---------- 2) 四条边界带：标量带钳制（NEON 不加速） ---------- */
    for (int j = 0; j < j_start && j < height_dst; j++)
        for (int i = 0; i < width_dst; i++)
            dst[(size_t)j * stride_dst + i] =
                conv2d_px_clamp(src, width_src, height_src, stride_src, coef, i, j);

    for (int j = j_end; j < height_dst; j++)
        for (int i = 0; i < width_dst; i++)
            dst[(size_t)j * stride_dst + i] =
                conv2d_px_clamp(src, width_src, height_src, stride_src, coef, i, j);

    for (int j = j_start; j < j_end; j++) {
        for (int i = 0; i < i_start && i < width_dst; i++)
            dst[(size_t)j * stride_dst + i] =
                conv2d_px_clamp(src, width_src, height_src, stride_src, coef, i, j);
        for (int i = i_end; i < width_dst; i++)
            dst[(size_t)j * stride_dst + i] =
                conv2d_px_clamp(src, width_src, height_src, stride_src, coef, i, j);
    }
}
