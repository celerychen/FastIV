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

void conv2d_v4(ivf32* dst, int width_dst, int height_dst, int stride_dst,
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
        int j_neon_end = j_start + ((j_end - j_start) / 2) * 2;  /* 2 行对齐(纵向展开) */

        /* 最外层 j += 2：一次算 2 行输出，纵向复用重叠的中间行。
           输出行 j 用源行 j-1/j/j+1；输出行 j+1 用源行 j/j+1/j+2；
           两者共享 j、j+1 两行，故一轮只 load 4 条不同源行(相对两轮 j++ 的 6 条
           减少 2 条/行对)，即“纵向避免了部分数据的加载”。
           两行各用独立累加器 accA(行 j) / accB(行 j+1) 做乘累加，保持 ILP；
           每个 4 像素窗口的 load 结构照抄编译器 LBB0_12(3 次重叠 load + 单链 fma)。 */
        for (int j = j_start; j < j_neon_end; j += 2) {
            ivf32* r0 = src + (size_t)(j - 1) * stride_src;  /* -> 输出行 j 的 k0..k2 行 */
            ivf32* r1 = src + (size_t) j      * stride_src;  /* 共享：行 j 的 k3..k5 / 行 j+1 的 k0..k2 */
            ivf32* r2 = src + (size_t)(j + 1) * stride_src;  /* 共享：行 j 的 k6..k8 / 行 j+1 的 k3..k5 */
            ivf32* r3 = src + (size_t)(j + 2) * stride_src;  /* -> 输出行 j+1 的 k6..k8 行 */
            ivf32* dA = dst + (size_t) j      * stride_dst;
            ivf32* dB = dst + (size_t)(j + 1) * stride_dst;

            int i;
            for (i = i_start; i < i_neon_end; i += 4) {
                /* ---- 输出行 j 累加器 accA ---- */
                float32x4_t accA = vmulq_f32(vld1q_f32(r0 + i),     k1);
                accA = vfmaq_f32(accA, vld1q_f32(r0 + i - 1), k0);
                accA = vfmaq_f32(accA, vld1q_f32(r0 + i + 1), k2);
                accA = vfmaq_f32(accA, vld1q_f32(r1 + i - 1), k3);
                accA = vfmaq_f32(accA, vld1q_f32(r1 + i),     k4);
                accA = vfmaq_f32(accA, vld1q_f32(r1 + i + 1), k5);
                accA = vfmaq_f32(accA, vld1q_f32(r2 + i - 1), k6);
                accA = vfmaq_f32(accA, vld1q_f32(r2 + i),     k7);
                accA = vfmaq_f32(accA, vld1q_f32(r2 + i + 1), k8);

                /* ---- 输出行 j+1 累加器 accB ---- */
                float32x4_t accB = vmulq_f32(vld1q_f32(r1 + i),     k1);
                accB = vfmaq_f32(accB, vld1q_f32(r1 + i - 1), k0);
                accB = vfmaq_f32(accB, vld1q_f32(r1 + i + 1), k2);
                accB = vfmaq_f32(accB, vld1q_f32(r2 + i - 1), k3);
                accB = vfmaq_f32(accB, vld1q_f32(r2 + i),     k4);
                accB = vfmaq_f32(accB, vld1q_f32(r2 + i + 1), k5);
                accB = vfmaq_f32(accB, vld1q_f32(r3 + i - 1), k6);
                accB = vfmaq_f32(accB, vld1q_f32(r3 + i),     k7);
                accB = vfmaq_f32(accB, vld1q_f32(r3 + i + 1), k8);

                vst1q_f32(dA + i, accA);
                vst1q_f32(dB + i, accB);
            }
            /* 末尾不足 4 列的标量收尾（仍是内部，无钳制），2 行一起算 */
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
        /* AVX2：8-wide __m256。纵向 j+=2 复用重叠的中间行(同 NEON v4)。
           输出行 j 用源行 j-1/j/j+1；输出行 j+1 用源行 j/j+1/j+2；两者共享
           r1/r2 两行 -> 一轮只 load 4 条不同源行。两行各用独立累加器
           accA(行 j)/accB(行 j+1) 做 fmadd，保持 ILP。i+=8(8-lane)。 */
        __m256 k0 = _mm256_set1_ps(c0), k1 = _mm256_set1_ps(c1), k2 = _mm256_set1_ps(c2);
        __m256 k3 = _mm256_set1_ps(c3), k4 = _mm256_set1_ps(c4), k5 = _mm256_set1_ps(c5);
        __m256 k6 = _mm256_set1_ps(c6), k7 = _mm256_set1_ps(c7), k8 = _mm256_set1_ps(c8);

        int i_simd_end = i_start + ((i_end - i_start) / 8) * 8;  /* 8 列对齐(8-lane) */
        int j_pair_end = j_start + ((j_end - j_start) / 2) * 2;  /* 2 行对齐(纵向展开) */

        for (int j = j_start; j < j_pair_end; j += 2) {
            ivf32* r0 = src + (size_t)(j - 1) * stride_src;  /* -> 输出行 j 的 k0..k2 行 */
            ivf32* r1 = src + (size_t) j      * stride_src;  /* 共享 */
            ivf32* r2 = src + (size_t)(j + 1) * stride_src;  /* 共享 */
            ivf32* r3 = src + (size_t)(j + 2) * stride_src;  /* -> 输出行 j+1 的 k6..k8 行 */
            ivf32* dA = dst + (size_t) j      * stride_dst;
            ivf32* dB = dst + (size_t)(j + 1) * stride_dst;

            int i;
            for (i = i_start; i < i_simd_end; i += 8) {
                /* ---- 输出行 j 累加器 accA ---- */
                __m256 accA = _mm256_mul_ps(_mm256_loadu_ps(r0 + i),     k1);
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i - 1), k0, accA);
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i + 1), k2, accA);
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i - 1), k3, accA);
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i),     k4, accA);
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 1), k5, accA);
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i - 1), k6, accA);
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i),     k7, accA);
                accA = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 1), k8, accA);

                /* ---- 输出行 j+1 累加器 accB ---- */
                __m256 accB = _mm256_mul_ps(_mm256_loadu_ps(r1 + i),     k1);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i - 1), k0, accB);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 1), k2, accB);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i - 1), k3, accB);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i),     k4, accB);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 1), k5, accB);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i - 1), k6, accB);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i),     k7, accB);
                accB = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i + 1), k8, accB);

                _mm256_storeu_ps(dA + i, accA);
                _mm256_storeu_ps(dB + i, accB);
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
