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

/* 3x3 kernel plane convolutions (STD fast paths). Split out of fiv_nn_conv2d.c. */

#include "fiv_nn_3x3_conv2d.h"
#include <stddef.h>
#include <stdlib.h>
#include "fiv_common.h"

static ivf32 fiv_conv2d_px_pad_3x3(const ivf32* src, int width_src, int height_src, int stride_src,
                               const ivf32 coef[9], int ox, int oy, int st,
                               int pt, int pl, int zero_pad)
{
    ivf32 sum = 0.0f;
    for (int kj = 0; kj < 3; kj++) {
        int sy = oy * st - pt + kj;
        if (sy < 0 || sy >= height_src) {
            if (zero_pad) continue;
            if (sy < 0) sy = 0;
            if (sy >= height_src) sy = height_src - 1;
        }
        const ivf32* row = src + (size_t)sy * stride_src;
        for (int ki = 0; ki < 3; ki++) {
            int sx = ox * st - pl + ki;
            if (sx < 0 || sx >= width_src) {
                if (zero_pad) continue;
                if (sx < 0) sx = 0;
                if (sx >= width_src) sx = width_src - 1;
            }
            sum += row[sx] * coef[kj * 3 + ki];
        }
    }
    return sum;
}

/* stride-1 scalar boundary pixel (SAME pad pt=pl=1). */
static ivf32 fiv_conv2d_px_pad_3x3_s1(const ivf32* src, int width_src, int height_src, int stride_src,
                                  const ivf32 coef[9], int ox, int oy, int zero_pad)
{
    return fiv_conv2d_px_pad_3x3(src, width_src, height_src, stride_src, coef, ox, oy, 1, 1, 1, zero_pad);
}

/* stride-2 scalar boundary pixel (SAME pad pt=pl=1). */
static ivf32 fiv_conv2d_px_pad_3x3_s2(const ivf32* src, int width_src, int height_src, int stride_src,
                                  const ivf32 coef[9], int ox, int oy, int zero_pad)
{
    return fiv_conv2d_px_pad_3x3(src, width_src, height_src, stride_src, coef, ox, oy, 2, 1, 1, zero_pad);
}

void fiv_conv2d_plane_3x3_s1(ivf32* dst, int width_dst, int height_dst, int stride_dst,
                             const ivf32* src, int width_src, int height_src, int stride_src,
                             const ivf32 coef[9], int accumulate, int zero_pad)
{
    ivf32 c0 = coef[0], c1 = coef[1], c2 = coef[2];
    ivf32 c3 = coef[3], c4 = coef[4], c5 = coef[5];
    ivf32 c6 = coef[6], c7 = coef[7], c8 = coef[8];

    /* interior region (no padding involved): half-open [start, end), empty if degenerate */
    int i_start = 1;
    int i_end   = width_src  - 1;  if (i_end > width_dst)  i_end = width_dst;
    int j_start = 1;
    int j_end   = height_src - 1;  if (j_end > height_dst) j_end = height_dst;
    if (i_end < i_start) i_end = i_start;
    if (j_end < j_start) j_end = j_start;

    /* ---------- 1) interior ---------- */
#if defined(FIV_USE_ARM_NEON)
    {
        /* 9 coefficients packed into 3 four-lane vectors (one per source row);
           lane 0/1/2 = column offset -1/0/+1. */
        float32x4_t vcoef0 = (float32x4_t){c0, c1, c2, 0.0f};
        float32x4_t vcoef1 = (float32x4_t){c3, c4, c5, 0.0f};
        float32x4_t vcoef2 = (float32x4_t){c6, c7, c8, 0.0f};

        int i_neon_end = i_start + ((i_end - i_start) / 8) * 8;  /* 8 cols = 2x4 lanes */
        int j_neon_end = j_start + ((j_end - j_start) / 2) * 2;  /* 2 rows vertical reuse */

        for (int j = j_start; j < j_neon_end; j += 2) {
            const ivf32* r0 = src + (size_t)(j - 1) * stride_src;
            const ivf32* r1 = src + (size_t) j      * stride_src;
            const ivf32* r2 = src + (size_t)(j + 1) * stride_src;
            const ivf32* r3 = src + (size_t)(j + 2) * stride_src;
            ivf32* dA = dst + (size_t) j      * stride_dst;
            ivf32* dB = dst + (size_t)(j + 1) * stride_dst;

            int i;
            for (i = i_start; i < i_neon_end; i += 8) {
                /* output row j: blockA / blockB */
                float32x4_t accA0 = accumulate
                                       ? vfmaq_laneq_f32(vld1q_f32(dA + i), vld1q_f32(r0 + i), vcoef0, 1)
                                       : vmulq_laneq_f32(vld1q_f32(r0 + i), vcoef0, 1);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r0 + i - 1), vcoef0, 0);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r0 + i + 1), vcoef0, 2);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r1 + i - 1), vcoef1, 0);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r1 + i),     vcoef1, 1);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r1 + i + 1), vcoef1, 2);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r2 + i - 1), vcoef2, 0);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r2 + i),     vcoef2, 1);
                accA0 = vfmaq_laneq_f32(accA0, vld1q_f32(r2 + i + 1), vcoef2, 2);

                float32x4_t accA1 = accumulate
                                       ? vfmaq_laneq_f32(vld1q_f32(dA + i + 4), vld1q_f32(r0 + i + 4), vcoef0, 1)
                                       : vmulq_laneq_f32(vld1q_f32(r0 + i + 4), vcoef0, 1);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r0 + i + 3), vcoef0, 0);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r0 + i + 5), vcoef0, 2);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r1 + i + 3), vcoef1, 0);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r1 + i + 4), vcoef1, 1);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r1 + i + 5), vcoef1, 2);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r2 + i + 3), vcoef2, 0);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r2 + i + 4), vcoef2, 1);
                accA1 = vfmaq_laneq_f32(accA1, vld1q_f32(r2 + i + 5), vcoef2, 2);

                /* output row j+1: source rows r1/r2/r3 */
                float32x4_t accB0 = accumulate
                                       ? vfmaq_laneq_f32(vld1q_f32(dB + i), vld1q_f32(r1 + i), vcoef0, 1)
                                       : vmulq_laneq_f32(vld1q_f32(r1 + i), vcoef0, 1);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r1 + i - 1), vcoef0, 0);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r1 + i + 1), vcoef0, 2);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r2 + i - 1), vcoef1, 0);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r2 + i),     vcoef1, 1);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r2 + i + 1), vcoef1, 2);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r3 + i - 1), vcoef2, 0);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r3 + i),     vcoef2, 1);
                accB0 = vfmaq_laneq_f32(accB0, vld1q_f32(r3 + i + 1), vcoef2, 2);

                float32x4_t accB1 = accumulate
                                       ? vfmaq_laneq_f32(vld1q_f32(dB + i + 4), vld1q_f32(r1 + i + 4), vcoef0, 1)
                                       : vmulq_laneq_f32(vld1q_f32(r1 + i + 4), vcoef0, 1);
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
            /* scalar tail (interior, no padding), 2 rows together */
            for (; i < i_end; i++) {
                ivf32 sa = r0[i-1]*c0 + r0[i]*c1 + r0[i+1]*c2
                         + r1[i-1]*c3 + r1[i]*c4 + r1[i+1]*c5
                         + r2[i-1]*c6 + r2[i]*c7 + r2[i+1]*c8;
                ivf32 sb = r1[i-1]*c0 + r1[i]*c1 + r1[i+1]*c2
                         + r2[i-1]*c3 + r2[i]*c4 + r2[i+1]*c5
                         + r3[i-1]*c6 + r3[i]*c7 + r3[i+1]*c8;
                dA[i] = accumulate ? dA[i] + sa : sa;
                dB[i] = accumulate ? dB[i] + sb : sb;
            }
        }
        /* remaining odd interior row (j_end - j_neon_end == 1) */
        for (int j = j_neon_end; j < j_end; j++) {
            const ivf32* r0 = src + (size_t)(j - 1) * stride_src;
            const ivf32* r1 = src + (size_t) j      * stride_src;
            const ivf32* r2 = src + (size_t)(j + 1) * stride_src;
            ivf32* d = dst + (size_t) j * stride_dst;
            for (int i = i_start; i < i_end; i++) {
                ivf32 s = r0[i-1]*c0 + r0[i]*c1 + r0[i+1]*c2
                        + r1[i-1]*c3 + r1[i]*c4 + r1[i+1]*c5
                        + r2[i-1]*c6 + r2[i]*c7 + r2[i+1]*c8;
                d[i] = accumulate ? d[i] + s : s;
            }
        }
    }
#elif defined(FIV_USE_AVX2)
    {
        /* coefficients load-broadcast at use point (no dedicated vector registers) */
        const ivf32* cf = coef;

        int i_simd_end = i_start + ((i_end - i_start) / 16) * 16; /* 16 cols = 2x8 lanes */
        int j_pair_end = j_start + ((j_end - j_start) / 2) * 2;

        for (int j = j_start; j < j_pair_end; j += 2) {
            const ivf32* r0 = src + (size_t)(j - 1) * stride_src;
            const ivf32* r1 = src + (size_t) j      * stride_src;
            const ivf32* r2 = src + (size_t)(j + 1) * stride_src;
            const ivf32* r3 = src + (size_t)(j + 2) * stride_src;
            ivf32* dA = dst + (size_t) j      * stride_dst;
            ivf32* dB = dst + (size_t)(j + 1) * stride_dst;

            int i;
            for (i = i_start; i < i_simd_end; i += 16) {
                /* output row j: blockA / blockB */
                __m256 accA0 = accumulate
                                   ? _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i), _mm256_broadcast_ss(cf + 1), _mm256_loadu_ps(dA + i))
                                   : _mm256_mul_ps(_mm256_loadu_ps(r0 + i), _mm256_broadcast_ss(cf + 1));
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i - 1), _mm256_broadcast_ss(cf + 0), accA0);
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i + 1), _mm256_broadcast_ss(cf + 2), accA0);
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i - 1), _mm256_broadcast_ss(cf + 3), accA0);
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i),     _mm256_broadcast_ss(cf + 4), accA0);
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 1), _mm256_broadcast_ss(cf + 5), accA0);
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i - 1), _mm256_broadcast_ss(cf + 6), accA0);
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i),     _mm256_broadcast_ss(cf + 7), accA0);
                accA0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 1), _mm256_broadcast_ss(cf + 8), accA0);

                __m256 accA1 = accumulate
                                   ? _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i + 8), _mm256_broadcast_ss(cf + 1), _mm256_loadu_ps(dA + i + 8))
                                   : _mm256_mul_ps(_mm256_loadu_ps(r0 + i + 8), _mm256_broadcast_ss(cf + 1));
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i + 7), _mm256_broadcast_ss(cf + 0), accA1);
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i + 9), _mm256_broadcast_ss(cf + 2), accA1);
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 7), _mm256_broadcast_ss(cf + 3), accA1);
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 8), _mm256_broadcast_ss(cf + 4), accA1);
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 9), _mm256_broadcast_ss(cf + 5), accA1);
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 7), _mm256_broadcast_ss(cf + 6), accA1);
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 8), _mm256_broadcast_ss(cf + 7), accA1);
                accA1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 9), _mm256_broadcast_ss(cf + 8), accA1);

                /* output row j+1: source rows r1/r2/r3 */
                __m256 accB0 = accumulate
                                   ? _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i), _mm256_broadcast_ss(cf + 1), _mm256_loadu_ps(dB + i))
                                   : _mm256_mul_ps(_mm256_loadu_ps(r1 + i), _mm256_broadcast_ss(cf + 1));
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i - 1), _mm256_broadcast_ss(cf + 0), accB0);
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 1), _mm256_broadcast_ss(cf + 2), accB0);
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i - 1), _mm256_broadcast_ss(cf + 3), accB0);
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i),     _mm256_broadcast_ss(cf + 4), accB0);
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 1), _mm256_broadcast_ss(cf + 5), accB0);
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i - 1), _mm256_broadcast_ss(cf + 6), accB0);
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i),     _mm256_broadcast_ss(cf + 7), accB0);
                accB0 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i + 1), _mm256_broadcast_ss(cf + 8), accB0);

                __m256 accB1 = accumulate
                                   ? _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 8), _mm256_broadcast_ss(cf + 1), _mm256_loadu_ps(dB + i + 8))
                                   : _mm256_mul_ps(_mm256_loadu_ps(r1 + i + 8), _mm256_broadcast_ss(cf + 1));
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
            /* scalar tail (interior, no padding), 2 rows together */
            for (; i < i_end; i++) {
                ivf32 sa = r0[i-1]*c0 + r0[i]*c1 + r0[i+1]*c2
                         + r1[i-1]*c3 + r1[i]*c4 + r1[i+1]*c5
                         + r2[i-1]*c6 + r2[i]*c7 + r2[i+1]*c8;
                ivf32 sb = r1[i-1]*c0 + r1[i]*c1 + r1[i+1]*c2
                         + r2[i-1]*c3 + r2[i]*c4 + r2[i+1]*c5
                         + r3[i-1]*c6 + r3[i]*c7 + r3[i+1]*c8;
                dA[i] = accumulate ? dA[i] + sa : sa;
                dB[i] = accumulate ? dB[i] + sb : sb;
            }
        }
        /* remaining odd interior row */
        for (int j = j_pair_end; j < j_end; j++) {
            const ivf32* r0 = src + (size_t)(j - 1) * stride_src;
            const ivf32* r1 = src + (size_t) j      * stride_src;
            const ivf32* r2 = src + (size_t)(j + 1) * stride_src;
            ivf32* d = dst + (size_t) j * stride_dst;
            for (int i = i_start; i < i_end; i++) {
                ivf32 s = r0[i-1]*c0 + r0[i]*c1 + r0[i+1]*c2
                        + r1[i-1]*c3 + r1[i]*c4 + r1[i+1]*c5
                        + r2[i-1]*c6 + r2[i]*c7 + r2[i+1]*c8;
                d[i] = accumulate ? d[i] + s : s;
            }
        }
    }
#else
    /* non-aarch64/AVX2: scalar 9-term unroll (same structure as v1) */
    for (int j = j_start; j < j_end; j++) {
        const ivf32* r0 = src + (size_t)(j - 1) * stride_src;
        const ivf32* r1 = src + (size_t) j      * stride_src;
        const ivf32* r2 = src + (size_t)(j + 1) * stride_src;
        ivf32* d = dst + (size_t) j * stride_dst;
        for (int i = i_start; i < i_end; i++) {
            ivf32 s = r0[i-1]*c0 + r0[i]*c1 + r0[i+1]*c2
                    + r1[i-1]*c3 + r1[i]*c4 + r1[i+1]*c5
                    + r2[i-1]*c6 + r2[i]*c7 + r2[i+1]*c8;
            d[i] = accumulate ? d[i] + s : s;
        }
    }
#endif

    /* ---------- 2) boundary bands: scalar, padding-aware ---------- */
    for (int j = 0; j < j_start && j < height_dst; j++)
        for (int i = 0; i < width_dst; i++) {
            ivf32 p = fiv_conv2d_px_pad_3x3_s1(src, width_src, height_src, stride_src, coef, i, j, zero_pad);
            dst[(size_t)j * stride_dst + i] =
                accumulate ? dst[(size_t)j * stride_dst + i] + p : p;
        }

    for (int j = j_end; j < height_dst; j++)
        for (int i = 0; i < width_dst; i++) {
            ivf32 p = fiv_conv2d_px_pad_3x3_s1(src, width_src, height_src, stride_src, coef, i, j, zero_pad);
            dst[(size_t)j * stride_dst + i] =
                accumulate ? dst[(size_t)j * stride_dst + i] + p : p;
        }

    for (int j = j_start; j < j_end; j++) {
        for (int i = 0; i < i_start && i < width_dst; i++) {
            ivf32 p = fiv_conv2d_px_pad_3x3_s1(src, width_src, height_src, stride_src, coef, i, j, zero_pad);
            dst[(size_t)j * stride_dst + i] =
                accumulate ? dst[(size_t)j * stride_dst + i] + p : p;
        }
        for (int i = i_end; i < width_dst; i++) {
            ivf32 p = fiv_conv2d_px_pad_3x3_s1(src, width_src, height_src, stride_src, coef, i, j, zero_pad);
            dst[(size_t)j * stride_dst + i] =
                accumulate ? dst[(size_t)j * stride_dst + i] + p : p;
        }
    }
}

/* boundary bands (scalar, padding-aware) for the stride-2 plane: fills the
   complement of the interior [i_start,i_end) x [j_start,j_end). */
static void fiv_conv2d_plane_3x3_s2_bands(ivf32* dst, int ow, int oh,
        const ivf32* src, int width, int height, const ivf32 coef[9],
        int accumulate, int zero_pad, int i_start, int i_end, int j_start, int j_end)
{
    for (int j = 0; j < j_start && j < oh; j++)
        for (int i = 0; i < ow; i++) {
            ivf32 p = fiv_conv2d_px_pad_3x3_s2(src, width, height, width, coef, i, j, zero_pad);
            dst[(size_t)j * ow + i] = accumulate ? dst[(size_t)j * ow + i] + p : p;
        }
    for (int j = j_end; j < oh; j++)
        for (int i = 0; i < ow; i++) {
            ivf32 p = fiv_conv2d_px_pad_3x3_s2(src, width, height, width, coef, i, j, zero_pad);
            dst[(size_t)j * ow + i] = accumulate ? dst[(size_t)j * ow + i] + p : p;
        }
    for (int j = j_start; j < j_end; j++) {
        for (int i = 0; i < i_start && i < ow; i++) {
            ivf32 p = fiv_conv2d_px_pad_3x3_s2(src, width, height, width, coef, i, j, zero_pad);
            dst[(size_t)j * ow + i] = accumulate ? dst[(size_t)j * ow + i] + p : p;
        }
        for (int i = i_end; i < ow; i++) {
            ivf32 p = fiv_conv2d_px_pad_3x3_s2(src, width, height, width, coef, i, j, zero_pad);
            dst[(size_t)j * ow + i] = accumulate ? dst[(size_t)j * ow + i] + p : p;
        }
    }
}

void fiv_conv2d_plane_3x3_s2(ivf32* dst, int ow, int oh, int stride_dst,
                             const ivf32* src, int width, int height, int stride_src,
                             const ivf32 coef[9], int accumulate, int zero_pad)
{
    ivf32 c0 = coef[0], c1 = coef[1], c2 = coef[2];
    ivf32 c3 = coef[3], c4 = coef[4], c5 = coef[5];
    ivf32 c6 = coef[6], c7 = coef[7], c8 = coef[8];

    /* interior: output pixels whose 3x3 stride-2 window stays in bounds */
    int i_start = 1;
    int i_end   = ((width - 2) / 2) + 1;  if (i_end > ow) i_end = ow;
    int j_start = 1;
    int j_end   = ((height - 2) / 2) + 1;  if (j_end > oh) j_end = oh;
    if (i_end < i_start) i_end = i_start;
    if (j_end < j_start) j_end = j_start;

#if defined(FIV_USE_ARM_NEON)
    {
        int i_neon_end = i_start + (((i_end - 1) - i_start) / 4) * 4;
        if (i_neon_end < i_start) i_neon_end = i_start;
        for (int j = j_start; j < j_end; j++) {
            const ivf32* R0 = src + (size_t)(2 * j - 1) * width;
            const ivf32* R1 = src + (size_t)(2 * j)     * width;
            const ivf32* R2 = src + (size_t)(2 * j + 1) * width;
            ivf32* d = dst + (size_t)j * ow;
            int i;
            for (i = i_start; i < i_neon_end; i += 4) {
                float32x4x2_t L0_r0 = vld2q_f32(R0 + (2 * i - 1));
                float32x4x2_t L1_r0 = vld2q_f32(R0 + (2 * i + 1));
                float32x4x2_t L0_r1 = vld2q_f32(R1 + (2 * i - 1));
                float32x4x2_t L1_r1 = vld2q_f32(R1 + (2 * i + 1));
                float32x4x2_t L0_r2 = vld2q_f32(R2 + (2 * i - 1));
                float32x4x2_t L1_r2 = vld2q_f32(R2 + (2 * i + 1));
                float32x4_t oL = vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(L0_r0.val[0], c0),
                                                         L0_r1.val[0], c3),
                                             L0_r2.val[0], c6);
                float32x4_t eC = vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(L0_r0.val[1], c1),
                                                         L0_r1.val[1], c4),
                                             L0_r2.val[1], c7);
                float32x4_t oR = vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(L1_r0.val[0], c2),
                                                         L1_r1.val[0], c5),
                                             L1_r2.val[0], c8);
                float32x4_t acc = vaddq_f32(vaddq_f32(oL, eC), oR);
                if (accumulate) acc = vaddq_f32(acc, vld1q_f32(d + i));
                vst1q_f32(d + i, acc);
            }
            for (; i < i_end; i++) {
                ivf32 p = fiv_conv2d_px_pad_3x3_s2(src, width, height, width, coef, i, j, zero_pad);
                d[i] = accumulate ? d[i] + p : p;
            }
        }
    }
#elif defined(FIV_USE_AVX2)
    {
        /* Leaner AVX2 tap extraction (adopted from the reference d:\work kernel):
           one 256 load covers the 16 source cols [2i-1, 2i+14]; permutevar8x32
           gathers the left / center / right even-picks in a single shuffle each,
           and the lone out-of-window right tap (col 2i+15) is blended in. Same
           interior bounds, scalar tail and fmadd order as the previous
           fiv_s2_tap_even version, so results stay bit-identical while the
           instruction count per row drops (3 loads + 9 permutes + 3 blends vs the
           old 18 loads + 18 permute2f128 + 9 shuffles). */
        const __m256i I_L = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
        const __m256i I_C = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);
        const __m256i I_R = _mm256_setr_epi32(2, 4, 6, 8, 10, 12, 14, 14);
        __m256 vc0 = _mm256_set1_ps(c0), vc1 = _mm256_set1_ps(c1), vc2 = _mm256_set1_ps(c2);
        __m256 vc3 = _mm256_set1_ps(c3), vc4 = _mm256_set1_ps(c4), vc5 = _mm256_set1_ps(c5);
        __m256 vc6 = _mm256_set1_ps(c6), vc7 = _mm256_set1_ps(c7), vc8 = _mm256_set1_ps(c8);
        const int nv = 8;
        int i_vec_end = i_start + ((i_end - i_start) / nv) * nv;
        for (int j = j_start; j < j_end; j++) {
            const ivf32* R0 = src + (size_t)(2 * j - 1) * width;
            const ivf32* R1 = src + (size_t)(2 * j)     * width;
            const ivf32* R2 = src + (size_t)(2 * j + 1) * width;
            ivf32* d = dst + (size_t)j * ow;
            int i;
            for (i = i_start; i < i_vec_end; i += nv) {
                int b = 2 * i - 1;
                __m256 w0 = _mm256_loadu_ps(R0 + b);
                __m256 l0 = _mm256_permutevar8x32_ps(w0, I_L);
                __m256 m0 = _mm256_permutevar8x32_ps(w0, I_C);
                __m256 r0 = _mm256_permutevar8x32_ps(w0, I_R);
                r0 = _mm256_blend_ps(r0, _mm256_set1_ps(R0[b + 16]), 0x80);

                __m256 w1 = _mm256_loadu_ps(R1 + b);
                __m256 l1 = _mm256_permutevar8x32_ps(w1, I_L);
                __m256 m1 = _mm256_permutevar8x32_ps(w1, I_C);
                __m256 r1 = _mm256_permutevar8x32_ps(w1, I_R);
                r1 = _mm256_blend_ps(r1, _mm256_set1_ps(R1[b + 16]), 0x80);

                __m256 w2 = _mm256_loadu_ps(R2 + b);
                __m256 l2 = _mm256_permutevar8x32_ps(w2, I_L);
                __m256 m2 = _mm256_permutevar8x32_ps(w2, I_C);
                __m256 r2 = _mm256_permutevar8x32_ps(w2, I_R);
                r2 = _mm256_blend_ps(r2, _mm256_set1_ps(R2[b + 16]), 0x80);

                __m256 acc = _mm256_mul_ps(l0, vc0);
                acc = _mm256_fmadd_ps(m0, vc1, acc);
                acc = _mm256_fmadd_ps(r0, vc2, acc);
                acc = _mm256_fmadd_ps(l1, vc3, acc);
                acc = _mm256_fmadd_ps(m1, vc4, acc);
                acc = _mm256_fmadd_ps(r1, vc5, acc);
                acc = _mm256_fmadd_ps(l2, vc6, acc);
                acc = _mm256_fmadd_ps(m2, vc7, acc);
                acc = _mm256_fmadd_ps(r2, vc8, acc);
                if (accumulate) acc = _mm256_add_ps(acc, _mm256_loadu_ps(d + i));
                _mm256_storeu_ps(d + i, acc);
            }
            for (; i < i_end; i++) {
                ivf32 p = fiv_conv2d_px_pad_3x3_s2(src, width, height, width, coef, i, j, zero_pad);
                d[i] = accumulate ? d[i] + p : p;
            }
        }
    }
#else
    {
        /* scalar interior as above: only the interior region, the boundary bands
           are completed once by fiv_conv2d_plane_3x3_s2_bands below. Filling the
           whole plane here double-counts the bands under accumulate. */
        for (int j = j_start; j < j_end; j++)
            for (int i = i_start; i < i_end; i++) {
                ivf32 p = fiv_conv2d_px_pad_3x3_s2(src, width, height, width, coef, i, j, zero_pad);
                dst[(size_t)j * ow + i] = accumulate ? dst[(size_t)j * ow + i] + p : p;
            }
    }
#endif

    fiv_conv2d_plane_3x3_s2_bands(dst, ow, oh, src, width, height, coef, accumulate,
                                zero_pad, i_start, i_end, j_start, j_end);
}

/* ---- Winograd F(2,3) dense 3x3 stride-1 layer (inference only) ----
   A 2x2 output tile is computed from a 4x4 input tile and a 3x3 kernel using
   only m = (2+3-1)^2 = 16 elementwise multiplies per tile (4 per output pixel)
   instead of the direct 9, at the cost of cheap add-only transforms:
        U = G g G^T            (weight, 3x3 -> 4x4, done once per (oc,ic))
        V = B^T d B            (input,  4x4 -> 4x4)
        M = sum_ic V (*) U     (elementwise in transform domain)
        Y = A^T M A            (output, 4x4 -> 2x2)
   Matrices (from Winograd F(2,3), scaled to avoid the /2 by keeping 6 taps):
        G   = [[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]]
        B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
        A^T = [[1,1,1,0],[0,1,-1,-1]]
   The sum over ic happens in the transform domain and the output transform is
   linear, so the layer is mathematically equivalent to the direct conv; in
   float it differs at ~1e-6 level (not bit-identical). Boundary tiles gather
   from a zero/replicate-padded view, exactly like the direct kernel's
   px_pad. Kept scalar for the transforms (gather dominates) with the per-ic
   multiply-accumulate vectorized over the 16 transform points. */

/* read src plane pixel (sx,sy) with SAME pad handling (pad=1 each side) */
static ivf32 fiv_wino_pick(const ivf32* plane, int width, int height,
                           int sx, int sy, int zero_pad)
{
    if (sx < 0 || sx >= width || sy < 0 || sy >= height) {
        if (zero_pad) return 0.0f;
        if (sx < 0) sx = 0;          else if (sx >= width)  sx = width  - 1;
        if (sy < 0) sy = 0;          else if (sy >= height) sy = height - 1;
    }
    return plane[(size_t)sy * (size_t)width + (size_t)sx];
}

/* U = G g G^T : 3x3 kernel g[9] (row-major) -> 4x4 u[16]. */
static void fiv_wino_transform_weight(const ivf32 g[9], ivf32 u[16])
{
    ivf32 w[4][3];
    for (int c = 0; c < 3; c++) {
        ivf32 a = g[c], b = g[3 + c], cc = g[6 + c];
        w[0][c] = a;
        w[1][c] = 0.5f * (a + b + cc);
        w[2][c] = 0.5f * (a - b + cc);
        w[3][c] = cc;
    }
    for (int r = 0; r < 4; r++) {
        ivf32 a = w[r][0], b = w[r][1], cc = w[r][2];
        u[r * 4 + 0] = a;
        u[r * 4 + 1] = 0.5f * (a + b + cc);
        u[r * 4 + 2] = 0.5f * (a - b + cc);
        u[r * 4 + 3] = cc;
    }
}

/* V = B^T d B : 4x4 input tile d[16] -> transformed v[16]. Two separable
   1D passes (rows then columns) using add/subtract only. */
static void fiv_wino_transform_input(const ivf32 d[16], ivf32 v[16])
{
    ivf32 T[16];
    for (int c = 0; c < 4; c++) {
        ivf32 d0 = d[0 * 4 + c], d1 = d[1 * 4 + c], d2 = d[2 * 4 + c], d3 = d[3 * 4 + c];
        T[0 * 4 + c] = d0 - d2;
        T[1 * 4 + c] = d1 + d2;
        T[2 * 4 + c] = d2 - d1;
        T[3 * 4 + c] = d1 - d3;
    }
    for (int r = 0; r < 4; r++) {
        ivf32 t0 = T[r * 4 + 0], t1 = T[r * 4 + 1], t2 = T[r * 4 + 2], t3 = T[r * 4 + 3];
        v[r * 4 + 0] = t0 - t2;
        v[r * 4 + 1] = t1 + t2;
        v[r * 4 + 2] = t2 - t1;
        v[r * 4 + 3] = t1 - t3;
    }
}

/* Y = A^T M A : 4x4 M[16] -> 2x2 out[4] (row-major). */
static void fiv_wino_inverse(const ivf32 M[16], ivf32 out[4])
{
    ivf32 R0[4], R1[4];
    for (int c = 0; c < 4; c++) {
        R0[c] = M[c] + M[4 + c] + M[8 + c];       /* A^T row 0 */
        R1[c] = M[4 + c] - M[8 + c] - M[12 + c];  /* A^T row 1 */
    }
    out[0] = R0[0] + R0[1] + R0[2];
    out[1] = R0[1] - R0[2] - R0[3];
    out[2] = R1[0] + R1[1] + R1[2];
    out[3] = R1[1] - R1[2] - R1[3];
}

/* one-time env switch so the direct kernel stays the default and Winograd can
   be A/B benchmarked (FIV_WINO=1) without touching the dispatch. */
static int fiv_wino_3x3_s1_enabled(void)
{
    static int init = 0;
    static int on   = 0;
    if (!init) { init = 1; const char* e = getenv("FIV_WINO"); on = e && e[0] == '1'; }
    return on;
}

void fiv_conv2d_std_3x3_s1_wino(ivf32* d, const ivf32* s, const ivf32* w,
                                int c_in, int c_out, int width, int height,
                                int zero_pad)
{
    const size_t hw    = (size_t)height * width;
    const size_t kpair = (size_t)c_in * 16;

    ivf32* U = (ivf32*)fiv_malloc(sizeof(ivf32) * (size_t)c_in * (size_t)c_out * 16);
    ivf32* V = (ivf32*)fiv_malloc(sizeof(ivf32) * (size_t)c_in * 16);
    if (!U || !V) { fiv_free(U); fiv_free(V); return; }

    /* weight transforms: U[oc][ic][16] */
    for (int oc = 0; oc < c_out; oc++)
        for (int ic = 0; ic < c_in; ic++)
            fiv_wino_transform_weight(w + ((size_t)oc * c_in + ic) * 9,
                                      U + ((size_t)oc * kpair + ic * 16));

    const int nty = (height + 1) / 2;          /* output 2x2 tile rows */
    const int ntx = (width  + 1) / 2;

    for (int ty = 0; ty < nty; ty++) {
        const int sy0 = 2 * ty - 1;
        for (int tx = 0; tx < ntx; tx++) {
            const int sx0 = 2 * tx - 1;

            /* gather + input-transform every input channel once (reused across oc) */
            for (int ic = 0; ic < c_in; ic++) {
                const ivf32* plane = s + (size_t)ic * hw;
                ivf32 d_[16];
                for (int r = 0; r < 4; r++)
                    for (int cc = 0; cc < 4; cc++)
                        d_[r * 4 + cc] = fiv_wino_pick(plane, width, height,
                                                       sx0 + cc, sy0 + r, zero_pad);
                fiv_wino_transform_input(d_, V + (size_t)ic * 16);
            }

            for (int oc = 0; oc < c_out; oc++) {
                ivf32 M[16];
#if defined(FIV_USE_AVX2)
                __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
                const ivf32* Uoc = U + (size_t)oc * kpair;
                for (int ic = 0; ic < c_in; ic++) {
                    const ivf32* v = V + (size_t)ic * 16;
                    const ivf32* u = Uoc + (size_t)ic * 16;
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(v),     _mm256_loadu_ps(u),     acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(v + 8), _mm256_loadu_ps(u + 8), acc1);
                }
                _mm256_storeu_ps(M,     acc0);
                _mm256_storeu_ps(M + 8, acc1);
#else
                for (int k = 0; k < 16; k++) M[k] = 0.0f;
                const ivf32* Uoc = U + (size_t)oc * kpair;
                for (int ic = 0; ic < c_in; ic++) {
                    const ivf32* v = V + (size_t)ic * 16;
                    const ivf32* u = Uoc + (size_t)ic * 16;
                    for (int k = 0; k < 16; k++) M[k] += v[k] * u[k];
                }
#endif
                ivf32 out[4];
                fiv_wino_inverse(M, out);

                ivf32* dstp = d + (size_t)oc * hw;
                const int oy0 = sy0 + 1, ox0 = sx0 + 1;   /* output pixel for this tile */
                if (oy0 < height) {
                    if (ox0 < width)     dstp[(size_t)oy0 * width + ox0]     = out[0];
                    if (ox0 + 1 < width) dstp[(size_t)oy0 * width + ox0 + 1] = out[1];
                }
                if (oy0 + 1 < height) {
                    if (ox0 < width)     dstp[(size_t)(oy0 + 1) * width + ox0]     = out[2];
                    if (ox0 + 1 < width) dstp[(size_t)(oy0 + 1) * width + ox0 + 1] = out[3];
                }
            }
        }
    }
    fiv_free(U);
    fiv_free(V);
}

/* ---- multi-channel STD wrappers (own the oc / ic loops) ----
   These loop over output channels / input channels and add each input plane's
   contribution into the output plane, so the sum over ic matches the scalar
   reference bit-for-bit (accumulate = (ic > 0)). Both loops live here so the
   module can be tuned (e.g. reuse the input plane across output channels)
   without touching the dispatch file. stride-1 keeps output size == input; the
   stride-2 variants halve it. */

void fiv_conv2d_std_3x3_s1(ivf32* d, const ivf32* s, const ivf32* w,
                           int c_in, int c_out, int width, int height,
                           int zero_pad)
{
    /* optional Winograd F(2,3) fast path (FIV_WINO=1) for A/B benchmarking.
       Stays OFF by default: it is ~1e-6 off the direct kernel, which would
       fail the landmark e2e's 1e-6 px threshold if enabled globally. */
    if (fiv_wino_3x3_s1_enabled()) {
        fiv_conv2d_std_3x3_s1_wino(d, s, w, c_in, c_out, width, height, zero_pad);
        return;
    }
    const size_t hw = (size_t)height * width;
    for (int oc = 0; oc < c_out; oc++)
        for (int ic = 0; ic < c_in; ic++)
            fiv_conv2d_plane_3x3_s1(d + (size_t)oc * hw, width, height, width,
                                    s + (size_t)ic * hw, width, height, width,
                                    w + ((size_t)oc * c_in + ic) * 9, ic > 0, zero_pad);
}

void fiv_conv2d_std_3x3_s2(ivf32* d, const ivf32* s, const ivf32* w,
                           int c_in, int c_out, int width, int height,
                           int oh, int ow, int zero_pad)
{
    const size_t o_hw = (size_t)oh * ow;
    const size_t hw   = (size_t)height * width;
    for (int oc = 0; oc < c_out; oc++)
        for (int ic = 0; ic < c_in; ic++)
            fiv_conv2d_plane_3x3_s2(d + (size_t)oc * o_hw, ow, oh, ow,
                                    s + (size_t)ic * hw, width, height, width,
                                    w + ((size_t)oc * c_in + ic) * 9, ic > 0, zero_pad);
}

/* ---- DEPTHWISE wrappers (own the output-channel loop) ----
   One 3x3 coef per output channel, applied to a single input channel
   ic = oc / mult (output group -> input channel). No accumulation across
   input channels, so each plane is written outright (accumulate = 0). */
void fiv_conv2d_dw_3x3_s1(ivf32* d, const ivf32* s, const ivf32* w,
                          int c_out, int mult, int width, int height,
                          int zero_pad)
{
    const size_t hw = (size_t)height * width;
    for (int oc = 0; oc < c_out; oc++) {
        size_t ic = (size_t)oc / (size_t)mult;
        fiv_conv2d_plane_3x3_s1(d + (size_t)oc * hw, width, height, width,
                                s + ic * hw, width, height, width,
                                w + (size_t)oc * 9, 0, zero_pad);
    }
}

void fiv_conv2d_dw_3x3_s2(ivf32* d, const ivf32* s, const ivf32* w,
                          int c_out, int mult, int width, int height,
                          int oh, int ow, int zero_pad)
{
    const size_t o_hw = (size_t)oh * ow;
    const size_t hw   = (size_t)height * width;
    for (int oc = 0; oc < c_out; oc++) {
        size_t ic = (size_t)oc / (size_t)mult;
        fiv_conv2d_plane_3x3_s2(d + (size_t)oc * o_hw, ow, oh, ow,
                                s + ic * hw, width, height, width,
                                w + (size_t)oc * 9, 0, zero_pad);
    }
}
