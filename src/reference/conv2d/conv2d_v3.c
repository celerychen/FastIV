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

void conv2d_v3(ivf32* dst, int width_dst, int height_dst, int stride_dst,
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

        int i_neon_end = i_start + ((i_end - i_start) / 8) * 8;  /* 8 列对齐(2×4-lane 展开) */

        for (int j = j_start; j < j_end; j++) {
            ivf32* r0 = src + (size_t)(j - 1) * stride_src;
            ivf32* r1 = src + (size_t) j      * stride_src;
            ivf32* r2 = src + (size_t)(j + 1) * stride_src;
            ivf32* d  = dst + (size_t) j      * stride_dst;

            int i;
            /* 8 像素/迭代：不使用 vext（此前实测 vext 更慢）。
               改为按 2 倍展开 v2 的 i+=4 内核：两个 4-像素子块 A/B 各用普通
               vld1q 直接加载自己的 3×3 窗口列（每行 6 次 load，18 load/8px，
               与 v2 单位加载量相同），并各自维护独立累加器 accA/accB。
               收益来自双累加链带来的更高 ILP 与更少的循环开销，而非减少加载。
                 blockA -> 像素 i..i+3，列偏移 i-1, i, i+1
                 blockB -> 像素 i+4..i+7，列偏移 i+3, i+4, i+5 */
            for (i = i_start; i < i_neon_end; i += 8) {
                /* ---- row 0 ---- */
                float32x4_t accA = vmulq_f32(vld1q_f32(r0 + i),     k1);
                accA = vfmaq_f32(accA, vld1q_f32(r0 + i - 1), k0);
                accA = vfmaq_f32(accA, vld1q_f32(r0 + i + 1), k2);
                float32x4_t accB = vmulq_f32(vld1q_f32(r0 + i + 4), k1);
                accB = vfmaq_f32(accB, vld1q_f32(r0 + i + 3), k0);
                accB = vfmaq_f32(accB, vld1q_f32(r0 + i + 5), k2);

                /* ---- row 1 ---- */
                accA = vfmaq_f32(accA, vld1q_f32(r1 + i - 1), k3);
                accA = vfmaq_f32(accA, vld1q_f32(r1 + i),     k4);
                accA = vfmaq_f32(accA, vld1q_f32(r1 + i + 1), k5);
                accB = vfmaq_f32(accB, vld1q_f32(r1 + i + 3), k3);
                accB = vfmaq_f32(accB, vld1q_f32(r1 + i + 4), k4);
                accB = vfmaq_f32(accB, vld1q_f32(r1 + i + 5), k5);

                /* ---- row 2 ---- */
                accA = vfmaq_f32(accA, vld1q_f32(r2 + i - 1), k6);
                accA = vfmaq_f32(accA, vld1q_f32(r2 + i),     k7);
                accA = vfmaq_f32(accA, vld1q_f32(r2 + i + 1), k8);
                accB = vfmaq_f32(accB, vld1q_f32(r2 + i + 3), k6);
                accB = vfmaq_f32(accB, vld1q_f32(r2 + i + 4), k7);
                accB = vfmaq_f32(accB, vld1q_f32(r2 + i + 5), k8);

                vst1q_f32(d + i,     accA);
                vst1q_f32(d + i + 4, accB);
            }
            /* 末尾不足 8 列的标量收尾（仍是内部，无钳制） */
            for (; i < i_end; i++) {
                d[i] = r0[i-1]*c0 + r0[i]*c1 + r0[i+1]*c2
                     + r1[i-1]*c3 + r1[i]*c4 + r1[i+1]*c5
                     + r2[i-1]*c6 + r2[i]*c7 + r2[i+1]*c8;
            }
        }
    }
#elif defined(__AVX2__)
    {
        /* AVX2：8-wide __m256。横向 2× 展开 v2 的 i+=8 内核 -> i+=16。
           两个 8-像素子块 A/B 各用普通 loadu 加载自己的 3×3 窗口列，
           各自维护独立累加器 accA/accB。收益来自双累加链带来的更高 ILP
           与更少的循环开销。
             blockA -> 像素 i..i+7，列偏移 i-1, i, i+1
             blockB -> 像素 i+8..i+15，列偏移 i+7, i+8, i+9 */
        __m256 k0 = _mm256_set1_ps(c0), k1 = _mm256_set1_ps(c1), k2 = _mm256_set1_ps(c2);
        __m256 k3 = _mm256_set1_ps(c3), k4 = _mm256_set1_ps(c4), k5 = _mm256_set1_ps(c5);
        __m256 k6 = _mm256_set1_ps(c6), k7 = _mm256_set1_ps(c7), k8 = _mm256_set1_ps(c8);

        int i_simd_end = i_start + ((i_end - i_start) / 16) * 16;  /* 16 列对齐(2×8-lane 展开) */

        for (int j = j_start; j < j_end; j++) {
            ivf32* r0 = src + (size_t)(j - 1) * stride_src;
            ivf32* r1 = src + (size_t) j      * stride_src;
            ivf32* r2 = src + (size_t)(j + 1) * stride_src;
            ivf32* d  = dst + (size_t) j      * stride_dst;

            int i;
            for (i = i_start; i < i_simd_end; i += 16) {
                /* ---- row 0 ---- */
                __m256 accA = _mm256_mul_ps(_mm256_loadu_ps(r0 + i),     k1);
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i - 1), k0, accA);
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i + 1), k2, accA);
                __m256 accB = _mm256_mul_ps(_mm256_loadu_ps(r0 + i + 8), k1);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i + 7), k0, accB);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i + 9), k2, accB);

                /* ---- row 1 ---- */
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i - 1), k3, accA);
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i),     k4, accA);
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 1), k5, accA);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 7), k3, accB);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 8), k4, accB);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 9), k5, accB);

                /* ---- row 2 ---- */
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i - 1), k6, accA);
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i),     k7, accA);
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 1), k8, accA);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 7), k6, accB);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 8), k7, accB);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 9), k8, accB);

                _mm256_storeu_ps(d + i,     accA);
                _mm256_storeu_ps(d + i + 8, accB);
            }
            /* 末尾不足 16 列的标量收尾（仍是内部，无钳制） */
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
