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

void conv2d_v2(ivf32* dst, int width_dst, int height_dst, int stride_dst,
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
        /* 系数广播成 4-lane 向量，供 vfmaq 乘累加 */
        float32x4_t k0 = vdupq_n_f32(c0), k1 = vdupq_n_f32(c1), k2 = vdupq_n_f32(c2);
        float32x4_t k3 = vdupq_n_f32(c3), k4 = vdupq_n_f32(c4), k5 = vdupq_n_f32(c5);
        float32x4_t k6 = vdupq_n_f32(c6), k7 = vdupq_n_f32(c7), k8 = vdupq_n_f32(c8);

        int i_neon_end = i_start + ((i_end - i_start) / 4) * 4;  /* 4 列对齐(4-lane) */

        for (int j = j_start; j < j_end; j++) {
            ivf32* r0 = src + (size_t)(j - 1) * stride_src;
            ivf32* r1 = src + (size_t) j      * stride_src;
            ivf32* r2 = src + (size_t)(j + 1) * stride_src;
            ivf32* d  = dst + (size_t) j      * stride_dst;

            int i;
            /* 照抄编译器为 v1 生成的向量循环结构(LBB0_12)：
               每 4 像素每行做 3 次重叠非对齐 load(i-1/i/i+1)，共 9 次 vld1q；
               单累加器一条 fmul + 8×fmla；load 穿插在 fma 间。
               无跨迭代依赖，靠 CPU 乱序把相邻迭代重叠、隐藏 FMA 延迟。 */
            for (i = i_start; i < i_neon_end; i += 4) {
                float32x4_t s01 = vld1q_f32(r0 + i);
                float32x4_t s00 = vld1q_f32(r0 + i - 1);
                float32x4_t acc = vmulq_f32(s01, k1);
                acc = vfmaq_f32(acc, s00, k0);
                float32x4_t s02 = vld1q_f32(r0 + i + 1);
                acc = vfmaq_f32(acc, s02, k2);

                float32x4_t s10 = vld1q_f32(r1 + i - 1);
                acc = vfmaq_f32(acc, s10, k3);
                float32x4_t s11 = vld1q_f32(r1 + i);
                float32x4_t s12 = vld1q_f32(r1 + i + 1);
                acc = vfmaq_f32(acc, s11, k4);
                acc = vfmaq_f32(acc, s12, k5);

                float32x4_t s20 = vld1q_f32(r2 + i - 1);
                acc = vfmaq_f32(acc, s20, k6);
                float32x4_t s21 = vld1q_f32(r2 + i);
                acc = vfmaq_f32(acc, s21, k7);
                float32x4_t s22 = vld1q_f32(r2 + i + 1);
                acc = vfmaq_f32(acc, s22, k8);

                vst1q_f32(d + i, acc);
            }
            /* 末尾不足 4 列的标量收尾（仍是内部，无钳制） */
            for (; i < i_end; i++) {
                d[i] = r0[i-1]*c0 + r0[i]*c1 + r0[i+1]*c2
                     + r1[i-1]*c3 + r1[i]*c4 + r1[i+1]*c5
                     + r2[i-1]*c6 + r2[i]*c7 + r2[i+1]*c8;
            }
        }
    }
#elif defined(__AVX2__)
    {
        /* AVX2：8-wide __m256，把 NEON 的 4-lane 内核放大到 8-lane。
           系数广播成 8-lane 向量，供 _mm256_fmadd_ps 乘累加。
           照抄 v2 的单累加器结构：每 8 像素每行做 3 次重叠非对齐 loadu
           (i-1/i/i+1)，共 9 次 loadu；单累加器 1 mul + 8 fmadd。
           无跨迭代依赖，靠 CPU 乱序把相邻迭代重叠、隐藏 FMA 延迟。 */
        __m256 k0 = _mm256_set1_ps(c0), k1 = _mm256_set1_ps(c1), k2 = _mm256_set1_ps(c2);
        __m256 k3 = _mm256_set1_ps(c3), k4 = _mm256_set1_ps(c4), k5 = _mm256_set1_ps(c5);
        __m256 k6 = _mm256_set1_ps(c6), k7 = _mm256_set1_ps(c7), k8 = _mm256_set1_ps(c8);

        int i_simd_end = i_start + ((i_end - i_start) / 8) * 8;  /* 8 列对齐(8-lane) */

        for (int j = j_start; j < j_end; j++) {
            ivf32* r0 = src + (size_t)(j - 1) * stride_src;
            ivf32* r1 = src + (size_t) j      * stride_src;
            ivf32* r2 = src + (size_t)(j + 1) * stride_src;
            ivf32* d  = dst + (size_t) j      * stride_dst;

            int i;
            for (i = i_start; i < i_simd_end; i += 8) {
                __m256 s01 = _mm256_loadu_ps(r0 + i);
                __m256 s00 = _mm256_loadu_ps(r0 + i - 1);
                __m256 acc = _mm256_mul_ps(s01, k1);
                acc = _mm256_fmadd_ps(s00, k0, acc);
                __m256 s02 = _mm256_loadu_ps(r0 + i + 1);
                acc = _mm256_fmadd_ps(s02, k2, acc);

                __m256 s10 = _mm256_loadu_ps(r1 + i - 1);
                acc = _mm256_fmadd_ps(s10, k3, acc);
                __m256 s11 = _mm256_loadu_ps(r1 + i);
                __m256 s12 = _mm256_loadu_ps(r1 + i + 1);
                acc = _mm256_fmadd_ps(s11, k4, acc);
                acc = _mm256_fmadd_ps(s12, k5, acc);

                __m256 s20 = _mm256_loadu_ps(r2 + i - 1);
                acc = _mm256_fmadd_ps(s20, k6, acc);
                __m256 s21 = _mm256_loadu_ps(r2 + i);
                acc = _mm256_fmadd_ps(s21, k7, acc);
                __m256 s22 = _mm256_loadu_ps(r2 + i + 1);
                acc = _mm256_fmadd_ps(s22, k8, acc);

                _mm256_storeu_ps(d + i, acc);
            }
            /* 末尾不足 8 列的标量收尾（仍是内部，无钳制） */
            for (; i < i_end; i++) {
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
