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

#include "fiv_nn_conv2d.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "fiv_common.h"

/* SIMD headers are auto-included and detected by api/fiv_data_typedefs.h
   (FIV_USE_ARM_NEON / FIV_USE_AVX2 / FIV_USE_X86_SIMD). */

/* ---- 3x3 plane convolution core (ported from src/reference/conv2d/conv2d_v6.c).
   One src plane x one 3x3 coef -> one dst plane, same size (stride 1).
   zero_pad:    0 = replicate edge (clamp), 1 = zero padding (skip out-of-range).
   accumulate:  1 = dst += result, 0 = dst = result (multi-channel sum). ---- */

/* stride-aware scalar pixel: out[oy][ox] = sum_{kj,ki} src[oy*st-pt+kj][ox*st-pl+ki] * coef.
   Shared core; the s1/s2 wrappers below pin the stride. */
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

static void fiv_conv2d_plane_3x3_s1(ivf32* dst, int width_dst, int height_dst, int stride_dst,
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

/* ---- 3x3 stride-2 plane convolution (same SIMD idea as the stride-1 plane) ----
   Output is half-sized; the sampling is strided, so the horizontal lane trick of
   stride-1 does not apply. Each output column's taps split into even/odd source
   streams: the kernel center lands on even source cols, the side taps on odd
   (out[ox] = c0*s_odd[ox] + c1*s_even[ox] + c2*s_odd[ox+1]). NEON deinterleaves
   with vld2; AVX2 with permutevar8x32 over a [src0..src15] vector. Interior is
   the whole plane minus the top row and left column (the only padded pixels for
   pt=pl=1), so the scalar boundary bands are tiny. zero_pad/accumulate as stride-1. */
static void fiv_conv2d_plane_3x3_s2(ivf32* dst, int ow, int oh, int stride_dst,
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
        __m256i I_L = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
        __m256i I_E = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);
        __m256i I_R = _mm256_setr_epi32(2, 4, 6, 8, 10, 12, 14, 0);
        int i_neon_end = i_start + ((((width - 16) / 2) - i_start) / 8) * 8;
        if (i_neon_end < i_start) i_neon_end = i_start;
        for (int j = j_start; j < j_end; j++) {
            const ivf32* R0 = src + (size_t)(2 * j - 1) * width;
            const ivf32* R1 = src + (size_t)(2 * j)     * width;
            const ivf32* R2 = src + (size_t)(2 * j + 1) * width;
            ivf32* d = dst + (size_t)j * ow;
            int i;
            for (i = i_start; i < i_neon_end; i += 8) {
                __m256 LL0 = _mm256_loadu_ps(R0 + (2 * i - 1));
                __m256 MM0 = _mm256_loadu_ps(R0 + (2 * i + 7));
                __m256 whole0 = _mm256_set_m128(MM0, LL0);
                __m256 ol0 = _mm256_permutevar8x32_ps(whole0, I_L);
                __m256 ec0 = _mm256_permutevar8x32_ps(whole0, I_E);
                __m256 or0 = _mm256_permutevar8x32_ps(whole0, I_R);
                or0 = _mm256_blend_ps(or0, _mm256_set1_ps(R0[2 * i + 15]), 0x80);

                __m256 LL1 = _mm256_loadu_ps(R1 + (2 * i - 1));
                __m256 MM1 = _mm256_loadu_ps(R1 + (2 * i + 7));
                __m256 whole1 = _mm256_set_m128(MM1, LL1);
                __m256 ol1 = _mm256_permutevar8x32_ps(whole1, I_L);
                __m256 ec1 = _mm256_permutevar8x32_ps(whole1, I_E);
                __m256 or1 = _mm256_permutevar8x32_ps(whole1, I_R);
                or1 = _mm256_blend_ps(or1, _mm256_set1_ps(R1[2 * i + 15]), 0x80);

                __m256 LL2 = _mm256_loadu_ps(R2 + (2 * i - 1));
                __m256 MM2 = _mm256_loadu_ps(R2 + (2 * i + 7));
                __m256 whole2 = _mm256_set_m128(MM2, LL2);
                __m256 ol2 = _mm256_permutevar8x32_ps(whole2, I_L);
                __m256 ec2 = _mm256_permutevar8x32_ps(whole2, I_E);
                __m256 or2 = _mm256_permutevar8x32_ps(whole2, I_R);
                or2 = _mm256_blend_ps(or2, _mm256_set1_ps(R2[2 * i + 15]), 0x80);

                __m256 r0 = _mm256_fmadd_ps(ec0, _mm256_set1_ps(c1),
                          _mm256_fmadd_ps(or0, _mm256_set1_ps(c2),
                            _mm256_mul_ps(ol0, _mm256_set1_ps(c0))));
                __m256 r1 = _mm256_fmadd_ps(ec1, _mm256_set1_ps(c4),
                          _mm256_fmadd_ps(or1, _mm256_set1_ps(c5),
                            _mm256_mul_ps(ol1, _mm256_set1_ps(c3))));
                __m256 r2 = _mm256_fmadd_ps(ec2, _mm256_set1_ps(c7),
                          _mm256_fmadd_ps(or2, _mm256_set1_ps(c8),
                            _mm256_mul_ps(ol2, _mm256_set1_ps(c6))));
                __m256 acc = _mm256_add_ps(r0, _mm256_add_ps(r1, r2));
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
        for (int j = 0; j < oh; j++)
            for (int i = 0; i < ow; i++) {
                ivf32 p = fiv_conv2d_px_pad_3x3_s2(src, width, height, width, coef, i, j, zero_pad);
                dst[(size_t)j * ow + i] = accumulate ? dst[(size_t)j * ow + i] + p : p;
            }
    }
#endif

    fiv_conv2d_plane_3x3_s2_bands(dst, ow, oh, src, width, height, coef, accumulate,
                                zero_pad, i_start, i_end, j_start, j_end);
}

/* stride-2 scalar boundary pixel for 5x5 (mirror fiv_conv2d_px_pad_3x3_s2, 5x5 window).
   5x5 exists only at stride 2 in this module, so the stride is fixed here. */
static ivf32 fiv_conv2d_px_pad_5x5_s2(const ivf32* src, int width_src, int height_src, int stride_src,
                                  const ivf32 coef[25], int ox, int oy,
                                  int pt, int pl, int zero_pad)
{
    ivf32 sum = 0.0f;
    for (int kj = 0; kj < 5; kj++) {
        int sy = oy * 2 - pt + kj;
        if (sy < 0 || sy >= height_src) {
            if (zero_pad) continue;
            if (sy < 0) sy = 0;
            if (sy >= height_src) sy = height_src - 1;
        }
        const ivf32* row = src + (size_t)sy * stride_src;
        for (int ki = 0; ki < 5; ki++) {
            int sx = ox * 2 - pl + ki;
            if (sx < 0 || sx >= width_src) {
                if (zero_pad) continue;
                if (sx < 0) sx = 0;
                if (sx >= width_src) sx = width_src - 1;
            }
            sum += row[sx] * coef[kj * 5 + ki];
        }
    }
    return sum;
}

/* boundary bands for 5x5 stride-2 plane: fill the complement of the interior
   [i_start,i_end) x [j_start,j_end) with the pad-aware scalar pixel. */
static void fiv_conv2d_plane_5x5_s2_bands(ivf32* dst, int ow, int oh,
        const ivf32* src, int width, int height, const ivf32 coef[25],
        int accumulate, int zero_pad, int pt, int pl,
        int i_start, int i_end, int j_start, int j_end)
{
    for (int j = 0; j < j_start && j < oh; j++)
        for (int i = 0; i < ow; i++) {
            ivf32 p = fiv_conv2d_px_pad_5x5_s2(src, width, height, width, coef, i, j, pt, pl, zero_pad);
            dst[(size_t)j * ow + i] = accumulate ? dst[(size_t)j * ow + i] + p : p;
        }
    for (int j = j_end; j < oh; j++)
        for (int i = 0; i < ow; i++) {
            ivf32 p = fiv_conv2d_px_pad_5x5_s2(src, width, height, width, coef, i, j, pt, pl, zero_pad);
            dst[(size_t)j * ow + i] = accumulate ? dst[(size_t)j * ow + i] + p : p;
        }
    for (int j = j_start; j < j_end; j++) {
        for (int i = 0; i < i_start && i < ow; i++) {
            ivf32 p = fiv_conv2d_px_pad_5x5_s2(src, width, height, width, coef, i, j, pt, pl, zero_pad);
            dst[(size_t)j * ow + i] = accumulate ? dst[(size_t)j * ow + i] + p : p;
        }
        for (int i = i_end; i < ow; i++) {
            ivf32 p = fiv_conv2d_px_pad_5x5_s2(src, width, height, width, coef, i, j, pt, pl, zero_pad);
            dst[(size_t)j * ow + i] = accumulate ? dst[(size_t)j * ow + i] + p : p;
        }
    }
}

/* ---- 5x5 stride-2 plane convolution (same SIMD idea as the 3x3 stride-2 plane).
   Output is half-sized; sampling is strided so the horizontal lane trick of
   stride-1 does not apply. Each output column's 5 taps split into even/odd
   source streams; vld2 (NEON) / permutevar8x32 (AVX2) deinterleave them. Interior
   is the whole plane minus the padded top row / left column ring (pt=pl=1), so the
   scalar boundary bands are tiny. zero_pad/accumulate as stride-1. coef row-major
   [ky*5 + kx]. */
static void fiv_conv2d_plane_5x5_s2(ivf32* dst, int ow, int oh, int stride_dst,
                                         const ivf32* src, int width, int height, int stride_src,
                                         const ivf32 coef[25], int accumulate, int zero_pad,
                                         int pt, int pl)
{
    ivf32 c0=coef[0],  c1=coef[1],  c2=coef[2],  c3=coef[3],  c4=coef[4];
    ivf32 c5=coef[5],  c6=coef[6],  c7=coef[7],  c8=coef[8],  c9=coef[9];
    ivf32 c10=coef[10],c11=coef[11],c12=coef[12],c13=coef[13],c14=coef[14];
    ivf32 c15=coef[15],c16=coef[16],c17=coef[17],c18=coef[18],c19=coef[19];
    ivf32 c20=coef[20],c21=coef[21],c22=coef[22],c23=coef[23],c24=coef[24];

    /* interior (no padding involved): half-open [start, end). */
    int i_start = (pl == 0) ? 0 : 1;
    int i_end   = ((width + pl - 5) / 2) + 1;  if (i_end > ow) i_end = ow;
    int j_start = (pt == 0) ? 0 : 1;
    int j_end   = ((height + pt - 5) / 2) + 1;  if (j_end > oh) j_end = oh;
    if (i_end < i_start) i_end = i_start;
    if (j_end < j_start) j_end = j_start;

#if defined(FIV_USE_ARM_NEON)
    {
        int i_neon_end = i_start + ((i_end - i_start) / 4) * 4;
        for (int j = j_start; j < j_end; j++) {
            const ivf32* R0 = src + (size_t)(2 * j - 1) * width;
            const ivf32* R1 = src + (size_t)(2 * j)     * width;
            const ivf32* R2 = src + (size_t)(2 * j + 1) * width;
            const ivf32* R3 = src + (size_t)(2 * j + 2) * width;
            const ivf32* R4 = src + (size_t)(2 * j + 3) * width;
            ivf32* d = dst + (size_t)j * ow;
            int i;
            for (i = i_start; i < i_neon_end; i += 4) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                /* ky = 0 (coef 0..4) */
                {
                    float32x4x2_t A0 = vld2q_f32(R0 + (2 * i - 1));
                    float32x4x2_t A1 = vld2q_f32(R0 + (2 * i + 7));
                    float32x4_t od0 = A0.val[0], ev0 = A0.val[1];
                    float32x4_t od1 = A1.val[0], ev1 = A1.val[1];
                    float32x4_t t0 = od0;
                    float32x4_t t1 = ev0;
                    float32x4_t t2 = vextq_f32(od0, od1, 1);
                    float32x4_t t3 = vextq_f32(ev0, ev1, 1);
                    float32x4_t t4 = vextq_f32(od0, od1, 2);
                    acc = vfmaq_n_f32(acc, t0, c0);
                    acc = vfmaq_n_f32(acc, t1, c1);
                    acc = vfmaq_n_f32(acc, t2, c2);
                    acc = vfmaq_n_f32(acc, t3, c3);
                    acc = vfmaq_n_f32(acc, t4, c4);
                }
                /* ky = 1 (coef 5..9) */
                {
                    float32x4x2_t A0 = vld2q_f32(R1 + (2 * i - 1));
                    float32x4x2_t A1 = vld2q_f32(R1 + (2 * i + 7));
                    float32x4_t od0 = A0.val[0], ev0 = A0.val[1];
                    float32x4_t od1 = A1.val[0], ev1 = A1.val[1];
                    acc = vfmaq_n_f32(acc, od0, c5);
                    acc = vfmaq_n_f32(acc, ev0, c6);
                    acc = vfmaq_n_f32(acc, vextq_f32(od0, od1, 1), c7);
                    acc = vfmaq_n_f32(acc, vextq_f32(ev0, ev1, 1), c8);
                    acc = vfmaq_n_f32(acc, vextq_f32(od0, od1, 2), c9);
                }
                /* ky = 2 (coef 10..14) */
                {
                    float32x4x2_t A0 = vld2q_f32(R2 + (2 * i - 1));
                    float32x4x2_t A1 = vld2q_f32(R2 + (2 * i + 7));
                    float32x4_t od0 = A0.val[0], ev0 = A0.val[1];
                    float32x4_t od1 = A1.val[0], ev1 = A1.val[1];
                    acc = vfmaq_n_f32(acc, od0, c10);
                    acc = vfmaq_n_f32(acc, ev0, c11);
                    acc = vfmaq_n_f32(acc, vextq_f32(od0, od1, 1), c12);
                    acc = vfmaq_n_f32(acc, vextq_f32(ev0, ev1, 1), c13);
                    acc = vfmaq_n_f32(acc, vextq_f32(od0, od1, 2), c14);
                }
                /* ky = 3 (coef 15..19) */
                {
                    float32x4x2_t A0 = vld2q_f32(R3 + (2 * i - 1));
                    float32x4x2_t A1 = vld2q_f32(R3 + (2 * i + 7));
                    float32x4_t od0 = A0.val[0], ev0 = A0.val[1];
                    float32x4_t od1 = A1.val[0], ev1 = A1.val[1];
                    acc = vfmaq_n_f32(acc, od0, c15);
                    acc = vfmaq_n_f32(acc, ev0, c16);
                    acc = vfmaq_n_f32(acc, vextq_f32(od0, od1, 1), c17);
                    acc = vfmaq_n_f32(acc, vextq_f32(ev0, ev1, 1), c18);
                    acc = vfmaq_n_f32(acc, vextq_f32(od0, od1, 2), c19);
                }
                /* ky = 4 (coef 20..24) */
                {
                    float32x4x2_t A0 = vld2q_f32(R4 + (2 * i - 1));
                    float32x4x2_t A1 = vld2q_f32(R4 + (2 * i + 7));
                    float32x4_t od0 = A0.val[0], ev0 = A0.val[1];
                    float32x4_t od1 = A1.val[0], ev1 = A1.val[1];
                    acc = vfmaq_n_f32(acc, od0, c20);
                    acc = vfmaq_n_f32(acc, ev0, c21);
                    acc = vfmaq_n_f32(acc, vextq_f32(od0, od1, 1), c22);
                    acc = vfmaq_n_f32(acc, vextq_f32(ev0, ev1, 1), c23);
                    acc = vfmaq_n_f32(acc, vextq_f32(od0, od1, 2), c24);
                }
                if (accumulate) acc = vaddq_f32(acc, vld1q_f32(d + i));
                vst1q_f32(d + i, acc);
            }
            for (; i < i_end; i++) {
                ivf32 p = fiv_conv2d_px_pad_5x5_s2(src, width, height, width, coef, i, j, pt, pl, zero_pad);
                d[i] = accumulate ? d[i] + p : p;
            }
        }
    }
#elif defined(FIV_USE_AVX2)
    {
        __m256i I0 = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
        __m256i I1 = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);
        __m256i I2 = _mm256_setr_epi32(2, 4, 6, 8, 10, 12, 14, 0);
        __m256i I3 = _mm256_setr_epi32(3, 5, 7, 9, 11, 13, 15, 1);
        __m256i I4 = _mm256_setr_epi32(4, 6, 8, 10, 12, 14, 0, 2);
        int i_simd_end = i_start + ((i_end - i_start) / 8) * 8;
        for (int j = j_start; j < j_end; j++) {
            const ivf32* R0 = src + (size_t)(2 * j - 1) * width;
            const ivf32* R1 = src + (size_t)(2 * j)     * width;
            const ivf32* R2 = src + (size_t)(2 * j + 1) * width;
            const ivf32* R3 = src + (size_t)(2 * j + 2) * width;
            const ivf32* R4 = src + (size_t)(2 * j + 3) * width;
            ivf32* d = dst + (size_t)j * ow;
            int i;
            for (i = i_start; i < i_simd_end; i += 8) {
                /* two 16-wide windows: whole0 covers c=0..6, whole1 covers c=4..7
                   (shifted by 4 output cols); blend low4 from whole0, high4 from whole1. */
                __m256 LL0 = _mm256_loadu_ps(R0 + (2 * i - 1));
                __m256 MM0 = _mm256_loadu_ps(R0 + (2 * i + 7));
                __m256 whole0_r0 = _mm256_set_m128(MM0, LL0);
                __m256 LL1 = _mm256_loadu_ps(R0 + (2 * i + 7));
                __m256 MM1 = _mm256_loadu_ps(R0 + (2 * i + 15));
                __m256 whole1_r0 = _mm256_set_m128(MM1, LL1);

                __m256 LL0b = _mm256_loadu_ps(R1 + (2 * i - 1));
                __m256 MM0b = _mm256_loadu_ps(R1 + (2 * i + 7));
                __m256 whole0_r1 = _mm256_set_m128(MM0b, LL0b);
                __m256 LL1b = _mm256_loadu_ps(R1 + (2 * i + 7));
                __m256 MM1b = _mm256_loadu_ps(R1 + (2 * i + 15));
                __m256 whole1_r1 = _mm256_set_m128(MM1b, LL1b);

                __m256 LL0c = _mm256_loadu_ps(R2 + (2 * i - 1));
                __m256 MM0c = _mm256_loadu_ps(R2 + (2 * i + 7));
                __m256 whole0_r2 = _mm256_set_m128(MM0c, LL0c);
                __m256 LL1c = _mm256_loadu_ps(R2 + (2 * i + 7));
                __m256 MM1c = _mm256_loadu_ps(R2 + (2 * i + 15));
                __m256 whole1_r2 = _mm256_set_m128(MM1c, LL1c);

                __m256 LL0d = _mm256_loadu_ps(R3 + (2 * i - 1));
                __m256 MM0d = _mm256_loadu_ps(R3 + (2 * i + 7));
                __m256 whole0_r3 = _mm256_set_m128(MM0d, LL0d);
                __m256 LL1d = _mm256_loadu_ps(R3 + (2 * i + 7));
                __m256 MM1d = _mm256_loadu_ps(R3 + (2 * i + 15));
                __m256 whole1_r3 = _mm256_set_m128(MM1d, LL1d);

                __m256 LL0e = _mm256_loadu_ps(R4 + (2 * i - 1));
                __m256 MM0e = _mm256_loadu_ps(R4 + (2 * i + 7));
                __m256 whole0_r4 = _mm256_set_m128(MM0e, LL0e);
                __m256 LL1e = _mm256_loadu_ps(R4 + (2 * i + 7));
                __m256 MM1e = _mm256_loadu_ps(R4 + (2 * i + 15));
                __m256 whole1_r4 = _mm256_set_m128(MM1e, LL1e);

                #define TAP(r, w0, w1, k) _mm256_blend_ps( \
                    _mm256_permutevar8x32_ps(w0, I##k), \
                    _mm256_permutevar8x32_ps(w1, I##k), 0xF0)

                __m256 acc = _mm256_setzero_ps();
                acc = _mm256_fmadd_ps(TAP(r0, whole0_r0, whole1_r0, 0), _mm256_set1_ps(c0), acc);
                acc = _mm256_fmadd_ps(TAP(r0, whole0_r0, whole1_r0, 1), _mm256_set1_ps(c1), acc);
                acc = _mm256_fmadd_ps(TAP(r0, whole0_r0, whole1_r0, 2), _mm256_set1_ps(c2), acc);
                acc = _mm256_fmadd_ps(TAP(r0, whole0_r0, whole1_r0, 3), _mm256_set1_ps(c3), acc);
                acc = _mm256_fmadd_ps(TAP(r0, whole0_r0, whole1_r0, 4), _mm256_set1_ps(c4), acc);

                acc = _mm256_fmadd_ps(TAP(r1, whole0_r1, whole1_r1, 0), _mm256_set1_ps(c5), acc);
                acc = _mm256_fmadd_ps(TAP(r1, whole0_r1, whole1_r1, 1), _mm256_set1_ps(c6), acc);
                acc = _mm256_fmadd_ps(TAP(r1, whole0_r1, whole1_r1, 2), _mm256_set1_ps(c7), acc);
                acc = _mm256_fmadd_ps(TAP(r1, whole0_r1, whole1_r1, 3), _mm256_set1_ps(c8), acc);
                acc = _mm256_fmadd_ps(TAP(r1, whole0_r1, whole1_r1, 4), _mm256_set1_ps(c9), acc);

                acc = _mm256_fmadd_ps(TAP(r2, whole0_r2, whole1_r2, 0), _mm256_set1_ps(c10), acc);
                acc = _mm256_fmadd_ps(TAP(r2, whole0_r2, whole1_r2, 1), _mm256_set1_ps(c11), acc);
                acc = _mm256_fmadd_ps(TAP(r2, whole0_r2, whole1_r2, 2), _mm256_set1_ps(c12), acc);
                acc = _mm256_fmadd_ps(TAP(r2, whole0_r2, whole1_r2, 3), _mm256_set1_ps(c13), acc);
                acc = _mm256_fmadd_ps(TAP(r2, whole0_r2, whole1_r2, 4), _mm256_set1_ps(c14), acc);

                acc = _mm256_fmadd_ps(TAP(r3, whole0_r3, whole1_r3, 0), _mm256_set1_ps(c15), acc);
                acc = _mm256_fmadd_ps(TAP(r3, whole0_r3, whole1_r3, 1), _mm256_set1_ps(c16), acc);
                acc = _mm256_fmadd_ps(TAP(r3, whole0_r3, whole1_r3, 2), _mm256_set1_ps(c17), acc);
                acc = _mm256_fmadd_ps(TAP(r3, whole0_r3, whole1_r3, 3), _mm256_set1_ps(c18), acc);
                acc = _mm256_fmadd_ps(TAP(r3, whole0_r3, whole1_r3, 4), _mm256_set1_ps(c19), acc);

                acc = _mm256_fmadd_ps(TAP(r4, whole0_r4, whole1_r4, 0), _mm256_set1_ps(c20), acc);
                acc = _mm256_fmadd_ps(TAP(r4, whole0_r4, whole1_r4, 1), _mm256_set1_ps(c21), acc);
                acc = _mm256_fmadd_ps(TAP(r4, whole0_r4, whole1_r4, 2), _mm256_set1_ps(c22), acc);
                acc = _mm256_fmadd_ps(TAP(r4, whole0_r4, whole1_r4, 3), _mm256_set1_ps(c23), acc);
                acc = _mm256_fmadd_ps(TAP(r4, whole0_r4, whole1_r4, 4), _mm256_set1_ps(c24), acc);
                #undef TAP

                if (accumulate) acc = _mm256_add_ps(acc, _mm256_loadu_ps(d + i));
                _mm256_storeu_ps(d + i, acc);
            }
            for (; i < i_end; i++) {
                ivf32 p = fiv_conv2d_px_pad_5x5_s2(src, width, height, width, coef, i, j, pt, pl, zero_pad);
                d[i] = accumulate ? d[i] + p : p;
            }
        }
    }
#else
    {
        for (int j = 0; j < oh; j++)
            for (int i = 0; i < ow; i++) {
                ivf32 p = fiv_conv2d_px_pad_5x5_s2(src, width, height, width, coef, i, j, pt, pl, zero_pad);
                dst[(size_t)j * ow + i] = accumulate ? dst[(size_t)j * ow + i] + p : p;
            }
    }
#endif

    fiv_conv2d_plane_5x5_s2_bands(dst, ow, oh, src, width, height, coef, accumulate,
                                    zero_pad, pt, pl, i_start, i_end, j_start, j_end);
}

/* ---- generic scalar conv (any kernel / stride / explicit pad) ----
   Mirrors the reference cnn_ops.c accumulation order so results match it
   bit-for-bit: for oc, oy, ox: for ky, kx, ic: acc += v * w. Out-of-range
   taps are skipped (zero pad) or clamped (replicate); only the start pads
   pt/pl enter the index math, exactly like same_pad() in the reference.
   weight layout [c_out, c_in, kh, kw] (permuted from the [co,kh,kw,ci] blob). */
static void fiv_conv2d_generic_std(ivf32* d, const ivf32* s, const ivf32* w,
                                   int c_in, int c_out, int height, int width, int oh, int ow,
                                   int kh, int kw, int st, int pt, int pl, int zero_pad)
{
    const size_t kplane = (size_t)c_in * kh * kw;
    for (int oc = 0; oc < c_out; oc++) {
        const ivf32* wo = w + (size_t)oc * kplane;
        ivf32* dplane = d + (size_t)oc * oh * ow;
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                float acc = 0.0f;
                for (int ky = 0; ky < kh; ky++) {
                    int sy = oy * st - pt + ky;
                    if (sy < 0 || sy >= height) {
                        if (zero_pad) continue;
                        sy = sy < 0 ? 0 : height - 1;
                    }
                    const ivf32* wrow = wo + (size_t)ky * kw;
                    for (int kx = 0; kx < kw; kx++) {
                        int sx = ox * st - pl + kx;
                        if (sx < 0 || sx >= width) {
                            if (zero_pad) continue;
                            sx = sx < 0 ? 0 : width - 1;
                        }
                        const ivf32* wcol = wrow + kx;
                        for (int ic = 0; ic < c_in; ic++) {
                            float v = s[((size_t)ic * height + sy) * width + sx];
                            acc += v * wcol[(size_t)ic * kh * kw];
                        }
                    }
                }
                dplane[(size_t)oy * ow + ox] = acc;
            }
        }
    }
}

static void fiv_conv2d_generic_dw(ivf32* d, const ivf32* s, const ivf32* w,
                                  int channels, int height, int width, int oh, int ow,
                                  int kh, int kw, int st, int pt, int pl, int zero_pad)
{
    for (int c = 0; c < channels; c++) {
        const ivf32* sc = s + (size_t)c * height * width;
        const ivf32* wc = w + (size_t)c * kh * kw;
        ivf32* dc = d + (size_t)c * oh * ow;
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                float acc = 0.0f;
                for (int ky = 0; ky < kh; ky++) {
                    int sy = oy * st - pt + ky;
                    if (sy < 0 || sy >= height) {
                        if (zero_pad) continue;
                        sy = sy < 0 ? 0 : height - 1;
                    }
                    const ivf32* wrow = wc + (size_t)ky * kw;
                    for (int kx = 0; kx < kw; kx++) {
                        int sx = ox * st - pl + kx;
                        if (sx < 0 || sx >= width) {
                            if (zero_pad) continue;
                            sx = sx < 0 ? 0 : width - 1;
                        }
                        acc += sc[(size_t)sy * width + sx] * wrow[kx];
                    }
                }
                dc[(size_t)oy * ow + ox] = acc;
            }
        }
    }
}

/* ---- 1x1 pointwise: per-pixel channel mix, no spatial neighborhood ----
   out[oc][y][x] = sum_ic in[ic][y][x] * w[oc][ic]. Padding is meaningless
   here (each output pixel only sees the input pixel at the same (y,x)), so
   this dedicated branch skips the pad / kernel-window machinery entirely.
   Loop order oc -> ic -> k (1 out channel, 2 in channels, 16 pixels inner):
   the plane is zeroed once per output channel, then each 2-channel group
   accumulates in memory over 16-px vectors (vld1q_f32_x4 -> ldp pairs), the
   same structure clang -O2 emits for the flat loop. Weights enter through
   by-element FMA (vfmaq_n_f32); AVX2 has none, so set1_ps. */
static void fiv_conv2d_pw(ivf32* d, const ivf32* s, const ivf32* w,
                          int c_in, int c_out, size_t hw)
{
#if defined(FIV_USE_ARM_NEON)
    for (int oc = 0; oc < c_out; oc++) {
        const ivf32* wo = w + (size_t)oc * c_in;
        ivf32* dplane = d + (size_t)oc * hw;
        memset(dplane, 0, hw * sizeof(ivf32));   /* zero once per plane */
        int ic = 0;
        for (; ic + 2 <= c_in; ic += 2) {        /* 2 input channels */
            const ivf32* s0 = s + (size_t)ic * hw;
            const ivf32* s1 = s + (size_t)(ic + 1) * hw;
            float w0 = wo[ic], w1 = wo[ic + 1];
            size_t k = 0;
            for (; k + 16 <= hw; k += 16) {      /* 16 px = 4 x float32x4 */
                float32x4_t d0 = vld1q_f32(dplane + k);
                float32x4_t d1 = vld1q_f32(dplane + k + 4);
                float32x4_t d2 = vld1q_f32(dplane + k + 8);
                float32x4_t d3 = vld1q_f32(dplane + k + 12);
                float32x4x4_t v0 = vld1q_f32_x4(s0 + k);
                float32x4x4_t v1 = vld1q_f32_x4(s1 + k);
                d0 = vfmaq_n_f32(d0, v0.val[0], w0); d0 = vfmaq_n_f32(d0, v1.val[0], w1);
                d1 = vfmaq_n_f32(d1, v0.val[1], w0); d1 = vfmaq_n_f32(d1, v1.val[1], w1);
                d2 = vfmaq_n_f32(d2, v0.val[2], w0); d2 = vfmaq_n_f32(d2, v1.val[2], w1);
                d3 = vfmaq_n_f32(d3, v0.val[3], w0); d3 = vfmaq_n_f32(d3, v1.val[3], w1);
                vst1q_f32(dplane + k, d0);
                vst1q_f32(dplane + k + 4, d1);
                vst1q_f32(dplane + k + 8, d2);
                vst1q_f32(dplane + k + 12, d3);
            }
            for (; k < hw; k++) dplane[k] += s0[k] * w0 + s1[k] * w1;   /* px tail */
        }
        for (; ic < c_in; ic++) {                /* in-channel tail (1) */
            const ivf32* s0 = s + (size_t)ic * hw;
            float w0 = wo[ic];
            size_t k = 0;
            for (; k + 16 <= hw; k += 16) {
                float32x4_t d0 = vld1q_f32(dplane + k);
                float32x4_t d1 = vld1q_f32(dplane + k + 4);
                float32x4_t d2 = vld1q_f32(dplane + k + 8);
                float32x4_t d3 = vld1q_f32(dplane + k + 12);
                float32x4x4_t v = vld1q_f32_x4(s0 + k);
                d0 = vfmaq_n_f32(d0, v.val[0], w0);
                d1 = vfmaq_n_f32(d1, v.val[1], w0);
                d2 = vfmaq_n_f32(d2, v.val[2], w0);
                d3 = vfmaq_n_f32(d3, v.val[3], w0);
                vst1q_f32(dplane + k, d0);
                vst1q_f32(dplane + k + 4, d1);
                vst1q_f32(dplane + k + 8, d2);
                vst1q_f32(dplane + k + 12, d3);
            }
            for (; k < hw; k++) dplane[k] += s0[k] * w0;               /* px tail */
        }
    }
#elif defined(FIV_USE_AVX2)
    for (int oc = 0; oc < c_out; oc++) {
        const ivf32* wo = w + (size_t)oc * c_in;
        ivf32* dplane = d + (size_t)oc * hw;
        memset(dplane, 0, hw * sizeof(ivf32));
        int ic = 0;
        for (; ic + 2 <= c_in; ic += 2) {        /* 2 input channels */
            const ivf32* s0 = s + (size_t)ic * hw;
            const ivf32* s1 = s + (size_t)(ic + 1) * hw;
            __m256 w0 = _mm256_set1_ps(wo[ic]), w1 = _mm256_set1_ps(wo[ic + 1]);
            size_t k = 0;
            for (; k + 16 <= hw; k += 16) {      /* 16 px = 2 x __m256 */
                __m256 d0 = _mm256_loadu_ps(dplane + k);
                __m256 d1 = _mm256_loadu_ps(dplane + k + 8);
                __m256 a0 = _mm256_loadu_ps(s0 + k), a1 = _mm256_loadu_ps(s0 + k + 8);
                __m256 b0 = _mm256_loadu_ps(s1 + k), b1 = _mm256_loadu_ps(s1 + k + 8);
                d0 = _mm256_fmadd_ps(a0, w0, d0); d0 = _mm256_fmadd_ps(b0, w1, d0);
                d1 = _mm256_fmadd_ps(a1, w0, d1); d1 = _mm256_fmadd_ps(b1, w1, d1);
                _mm256_storeu_ps(dplane + k, d0);
                _mm256_storeu_ps(dplane + k + 8, d1);
            }
            for (; k < hw; k++) dplane[k] += s0[k] * wo[ic] + s1[k] * wo[ic + 1];
        }
        for (; ic < c_in; ic++) {                /* in-channel tail (1) */
            const ivf32* s0 = s + (size_t)ic * hw;
            __m256 w0 = _mm256_set1_ps(wo[ic]);
            size_t k = 0;
            for (; k + 16 <= hw; k += 16) {
                __m256 d0 = _mm256_loadu_ps(dplane + k);
                __m256 d1 = _mm256_loadu_ps(dplane + k + 8);
                __m256 a0 = _mm256_loadu_ps(s0 + k), a1 = _mm256_loadu_ps(s0 + k + 8);
                d0 = _mm256_fmadd_ps(a0, w0, d0);
                d1 = _mm256_fmadd_ps(a1, w0, d1);
                _mm256_storeu_ps(dplane + k, d0);
                _mm256_storeu_ps(dplane + k + 8, d1);
            }
            for (; k < hw; k++) dplane[k] += s0[k] * wo[ic];
        }
    }
#else
    /* scalar fallback: same oc -> ic -> k order, flat inner loop */
    for (int oc = 0; oc < c_out; oc++) {
        const ivf32* wo = w + (size_t)oc * c_in;
        ivf32* dplane = d + (size_t)oc * hw;
        for (size_t k = 0; k < hw; k++) dplane[k] = 0.0f;
        for (int ic = 0; ic < c_in; ic++) {
            const ivf32* schan = s + (size_t)ic * hw;
            float wv = wo[ic];
            for (size_t k = 0; k < hw; k++) dplane[k] += schan[k] * wv;
        }
    }
#endif
}

/* ---- public API ---- */

fiv_ret fiv_tensor_conv2d(void* dst, void* src, void* kernel, fiv_conv2d_params* params)
{
    fiv_tensor_hdr* sh = (fiv_tensor_hdr*)src;
    fiv_tensor_hdr* dh = (fiv_tensor_hdr*)dst;
    fiv_tensor_hdr* kh = (fiv_tensor_hdr*)kernel;
    if (!dst || !src || !kernel || !params) return FIV_RET_ERR_PARA;

    if (sh->id < FIV_ID_TENSOR3D || sh->id > FIV_ID_TENSOR5D) return FIV_RET_ERR_PARA;  /* need channels, height, width */
    if (kh->id != FIV_ID_TENSOR4D) return FIV_RET_ERR_PARA;
    if (dh->id != sh->id) return FIV_RET_ERR_PARA;
    if (sh->dtype != FIV_32F1 || dh->dtype != FIV_32F1 || kh->dtype != FIV_32F1)
        return FIV_RET_ERR_NOT_SUPPORT;
    if (!sh->data_continue || !dh->data_continue || !kh->data_continue)
        return FIV_RET_ERR_PARA;

    if (params->conv2d_method != FIV_CONV2D_STD &&
        params->conv2d_method != FIV_CONV2D_DEPTHWISE &&
        params->conv2d_method != FIV_CONV2D_POINTWISE)
        return FIV_RET_ERR_NOT_SUPPORT;
    if (params->kernel_size_x < 1 || params->kernel_size_y < 1 || params->stride < 1)
        return FIV_RET_ERR_PARA;
    if (params->pad_top < 0 || params->pad_bottom < 0 || params->pad_left < 0 || params->pad_right < 0)
        return FIV_RET_ERR_PARA;
    if (params->padding_method != 0 && params->padding_method != 1) return FIV_RET_ERR_PARA;

    size_t c_in, height, width, n_batch;
    switch (sh->id) {
    case FIV_ID_TENSOR3D: {
        const fiv_tensor3d* t = (const fiv_tensor3d*)sh;
        c_in = t->channels; height = t->height; width = t->width; n_batch = 1;
        break;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)sh;
        c_in = t->channels; height = t->height; width = t->width; n_batch = t->batch;
        break;
    }
    default: {
        const fiv_tensor5d* t = (const fiv_tensor5d*)sh;
        c_in = t->channels; height = t->height; width = t->width;
        n_batch = t->batch * t->times;
        break;
    }
    }
    if (c_in == 0 || height == 0 || width == 0) return FIV_RET_ERR_PARA;

    int kx = params->kernel_size_x, ky = params->kernel_size_y;
    int st = params->stride;
    const fiv_tensor4d* k = (const fiv_tensor4d*)kernel;
    if (k->shapes[2] != (size_t)ky || k->shapes[3] != (size_t)kx) return FIV_RET_ERR_PARA;
    size_t k_cout = k->shapes[0];
    size_t k_cin  = k->shapes[1];
    if (params->input_channels != (int)c_in) return FIV_RET_ERR_PARA;

    size_t c_out;
    if (params->conv2d_method == FIV_CONV2D_DEPTHWISE) {
        if (k_cin != 1 || k_cout != c_in) return FIV_RET_ERR_PARA;
        c_out = c_in;
    } else {  /* STD / POINTWISE */
        if (k_cin != c_in) return FIV_RET_ERR_PARA;
        c_out = k_cout;
    }
    if (params->output_channels != (int)c_out) return FIV_RET_ERR_PARA;

    size_t oh = (height + (size_t)st - 1) / (size_t)st;
    size_t ow = (width + (size_t)st - 1) / (size_t)st;
    size_t d_c, d_h, d_w;
    switch (dh->id) {
    case FIV_ID_TENSOR3D: {
        const fiv_tensor3d* t = (const fiv_tensor3d*)dh;
        d_c = t->channels; d_h = t->height; d_w = t->width;
        break;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)dh;
        d_c = t->channels; d_h = t->height; d_w = t->width;
        break;
    }
    default: {
        const fiv_tensor5d* t = (const fiv_tensor5d*)dh;
        d_c = t->channels; d_h = t->height; d_w = t->width;
        break;
    }
    }
    if (d_c != c_out || d_h != oh || d_w != ow) return FIV_RET_ERR_PARA;

    const ivf32* s = sh->data.fl;
    const ivf32* w = kh->data.fl;
    ivf32* d = dh->data.fl;
    size_t hw    = height * width;
    size_t o_hw   = oh * ow;
    size_t s_chan = c_in * hw;
    size_t d_chan = c_out * o_hw;
    size_t k_chan = k_cin * (size_t)ky * kx;
    int zero_pad = (params->padding_method == 0);

    /* Legacy 3x3 stride-1 keeps its SIMD fast path (implicit p0 = p1 = 1 each
       side) whenever the requested padding is either unset (0) or exactly
       same-padding (1); any other explicit pad / stride / kernel goes through
       the generic scalar kernels above with the literal pad values. */
    int legacy_3x3_s1 = (kx == 3 && ky == 3 && st == 1 &&
                       ((params->pad_top == 0 && params->pad_bottom == 0 &&
                         params->pad_left == 0 && params->pad_right == 0) ||
                        (params->pad_top == 1 && params->pad_bottom == 1 &&
                         params->pad_left == 1 && params->pad_right == 1)));

    /* 3x3 stride-2 keeps its SIMD fast path (implicit p0=p1=1) on the same
       pad conditions as the stride-1 legacy branch. */
    int legacy_3x3_s2 = (kx == 3 && ky == 3 && st == 2 &&
                     ((params->pad_top == 0 && params->pad_bottom == 0 &&
                       params->pad_left == 0 && params->pad_right == 0) ||
                      (params->pad_top == 1 && params->pad_bottom == 1 &&
                       params->pad_left == 1 && params->pad_right == 1)));

    /* 5x5 stride-2 fast path: the BlazeFace stem (3->24) and any kernel of this
       shape. SAME padding always puts pad_top=pad_left=1, so the branchless SIMD
       interior (pt=pl=1) is exact; other pads fall through to the generic scalar. */
    int legacy_5x5_s2 = (kx == 5 && ky == 5 && st == 2);

    /* 1x1 stride-1 (all pointwise layers) has no spatial neighborhood: each
       output pixel only sees the input pixel at the same (y,x), so padding is
       meaningless and is IGNORED -- the dedicated branch never pads. */
    int is_pw = (params->conv2d_method != FIV_CONV2D_DEPTHWISE &&
                 kx == 1 && ky == 1 && st == 1);

    for (size_t b = 0; b < n_batch; b++) {
        const ivf32* sb = s + b * s_chan;
        ivf32* db = d + b * d_chan;
        if (is_pw) {
            fiv_conv2d_pw(db, sb, w, (int)c_in, (int)c_out, hw);
        } else if (legacy_3x3_s1) {
            if (params->conv2d_method == FIV_CONV2D_DEPTHWISE) {
                for (size_t c = 0; c < c_out; c++)
                    fiv_conv2d_plane_3x3_s1(db + c * o_hw, (int)width, (int)height, (int)width,
                                     sb + c * hw, (int)width, (int)height, (int)width,
                                     w + c * 9, 0, zero_pad);
            } else {
                for (size_t oc = 0; oc < c_out; oc++)
                    for (size_t ic = 0; ic < c_in; ic++)
                        fiv_conv2d_plane_3x3_s1(db + oc * o_hw, (int)width, (int)height, (int)width,
                                         sb + ic * hw, (int)width, (int)height, (int)width,
                                         w + oc * k_chan + ic * 9, ic > 0, zero_pad);
            }
        } else if (legacy_3x3_s2) {
            if (params->conv2d_method == FIV_CONV2D_DEPTHWISE) {
                for (size_t c = 0; c < c_out; c++)
                    fiv_conv2d_plane_3x3_s2(db + c * o_hw, (int)ow, (int)oh, (int)ow,
                                            sb + c * hw, (int)width, (int)height, (int)width,
                                            w + c * 9, 0, zero_pad);
            } else {
                for (size_t oc = 0; oc < c_out; oc++)
                    for (size_t ic = 0; ic < c_in; ic++)
                        fiv_conv2d_plane_3x3_s2(db + oc * o_hw, (int)ow, (int)oh, (int)ow,
                                                 sb + ic * hw, (int)width, (int)height, (int)width,
                                                 w + oc * k_chan + ic * 9, ic > 0, zero_pad);
            }
        } else if (legacy_5x5_s2) {
            if (params->conv2d_method == FIV_CONV2D_DEPTHWISE) {
                /* no 5x5 depthwise in the network; reuse the generic scalar path */
                fiv_conv2d_generic_dw(db, sb, w, (int)c_in, (int)height, (int)width,
                                      (int)oh, (int)ow, ky, kx, st,
                                      params->pad_top, params->pad_left, zero_pad);
            } else if (params->pad_top == 1 && params->pad_left == 1) {
                for (size_t oc = 0; oc < c_out; oc++)
                    for (size_t ic = 0; ic < c_in; ic++)
                        fiv_conv2d_plane_5x5_s2(db + oc * o_hw, (int)ow, (int)oh, (int)ow,
                                                     sb + ic * hw, (int)width, (int)height, (int)width,
                                                     w + oc * k_chan + ic * 25, ic > 0, zero_pad,
                                                     params->pad_top, params->pad_left);
            } else {
                fiv_conv2d_generic_std(db, sb, w, (int)c_in, (int)c_out, (int)height, (int)width,
                                       (int)oh, (int)ow, ky, kx, st,
                                       params->pad_top, params->pad_left, zero_pad);
            }
        } else {
            int pt = params->pad_top, pl = params->pad_left;
            if (params->conv2d_method == FIV_CONV2D_DEPTHWISE)
                fiv_conv2d_generic_dw(db, sb, w, (int)c_in, (int)height, (int)width,
                                      (int)oh, (int)ow, ky, kx, st, pt, pl, zero_pad);
            else
                fiv_conv2d_generic_std(db, sb, w, (int)c_in, (int)c_out, (int)height, (int)width,
                                       (int)oh, (int)ow, ky, kx, st, pt, pl, zero_pad);
        }
    }
    return FIV_RET_OK;
}

/* ---- CONV2D_STD network node ---- */

void* fiv_conv2d_node_create(void* params)
{
    const fiv_conv2d_params* p = (const fiv_conv2d_params*)params;
    if (!p) return NULL;
    if (p->conv2d_method != FIV_CONV2D_STD &&
        p->conv2d_method != FIV_CONV2D_DEPTHWISE &&
        p->conv2d_method != FIV_CONV2D_POINTWISE)
        return NULL;
    if (p->kernel_size_x < 1 || p->kernel_size_y < 1 || p->stride < 1) return NULL;
    if (p->padding_method != 0 && p->padding_method != 1) return NULL;
    if (p->bias != 0 && p->bias != 1) return NULL;
    if (p->input_channels <= 0 || p->output_channels <= 0) return NULL;
    if (p->pad_top < 0 || p->pad_bottom < 0 || p->pad_left < 0 || p->pad_right < 0) return NULL;
    if (p->conv2d_method == FIV_CONV2D_DEPTHWISE &&
        (p->kernel_size_x != 3 || p->kernel_size_y != 3)) return NULL;
    if (p->conv2d_method == FIV_CONV2D_POINTWISE &&
        (p->kernel_size_x != 1 || p->kernel_size_y != 1 || p->stride != 1)) return NULL;

    int kcin = (p->conv2d_method == FIV_CONV2D_DEPTHWISE) ? 1 : p->input_channels;

    fiv_conv2d_node* n = (fiv_conv2d_node*)fiv_malloc(sizeof(fiv_conv2d_node));
    if (!n) return NULL;
    memset(n, 0, sizeof(fiv_conv2d_node));
    n->base.create_fn    = fiv_conv2d_node_create;
    n->base.release_fn   = fiv_conv2d_node_release;
    n->base.forward_fn   = fiv_conv2d_node_forward;
    n->base.backward_fn  = fiv_conv2d_node_backward;
    n->base.inference_fn = fiv_conv2d_node_inference;
    n->base.alloc_out_fn = fiv_conv2d_node_alloc_out;
    n->params = *p;

    size_t wsh[4] = { (size_t)p->output_channels, (size_t)kcin,
                      (size_t)p->kernel_size_y, (size_t)p->kernel_size_x };
    n->weight = fiv_create_tensor4d(wsh, FIV_32F1);
    if (!n->weight) { fiv_free(n); return NULL; }
    {
        ivf32* w = n->weight->data.fl;
        size_t nw = (size_t)p->output_channels * (size_t)kcin *
                    (size_t)p->kernel_size_y * (size_t)p->kernel_size_x;
        float s = 1.0f / sqrtf((float)(p->input_channels *
                                       p->kernel_size_y * p->kernel_size_x));
        for (size_t k = 0; k < nw; k++) w[k] = fiv_nn_rand() * s;
    }

    n->grad_weight = fiv_create_tensor4d(wsh, FIV_32F1);
    if (!n->grad_weight) { fiv_release_tensor4d(&n->weight); fiv_free(n); return NULL; }
    memset(n->grad_weight->data.ptr, 0, n->grad_weight->total_bytes);

    if (p->bias) {
        n->bias = fiv_create_tensor1d((size_t)p->output_channels, FIV_32F1);
        if (!n->bias) {
            fiv_release_tensor4d(&n->grad_weight);
            fiv_release_tensor4d(&n->weight);
            fiv_free(n);
            return NULL;
        }
        memset(n->bias->data.ptr, 0, n->bias->total_bytes);
        n->grad_bias = fiv_create_tensor1d((size_t)p->output_channels, FIV_32F1);
        if (!n->grad_bias) {
            fiv_release_tensor1d(&n->bias);
            fiv_release_tensor4d(&n->grad_weight);
            fiv_release_tensor4d(&n->weight);
            fiv_free(n);
            return NULL;
        }
        memset(n->grad_bias->data.ptr, 0, n->grad_bias->total_bytes);
    }
    return n;
}

void fiv_conv2d_node_release(void* op_state)
{
    fiv_conv2d_node* n = (fiv_conv2d_node*)op_state;
    if (!n) return;
    if (n->weight)      fiv_release_tensor4d(&n->weight);
    if (n->grad_weight) fiv_release_tensor4d(&n->grad_weight);
    if (n->bias)        fiv_release_tensor1d(&n->bias);
    if (n->grad_bias)   fiv_release_tensor1d(&n->grad_bias);
    fiv_free(n);
}

/* Output keeps the input's dims; channels = output_channels, spatial unchanged. */
void* fiv_conv2d_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{
    *out_ret = FIV_RET_OK;
    const fiv_conv2d_node* n = (const fiv_conv2d_node*)op_state;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!n || !in) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }
    if (in->id < FIV_ID_TENSOR3D || in->id > FIV_ID_TENSOR5D) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }
    if (in->dtype != FIV_32F1 || in->data_continue == 0) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }

    size_t c_out = (size_t)n->params.output_channels;
    size_t batch   = 1;
    size_t height  = 0;
    size_t width   = 0;
    switch (in->id) {
    case FIV_ID_TENSOR3D:
        height = ((const fiv_tensor3d*)in)->height;
        width = ((const fiv_tensor3d*)in)->width;
        break;
    case FIV_ID_TENSOR4D:
        batch = ((const fiv_tensor4d*)in)->batch;
        height = ((const fiv_tensor4d*)in)->height;
        width = ((const fiv_tensor4d*)in)->width;
        break;
    default:
        batch = ((const fiv_tensor5d*)in)->batch * ((const fiv_tensor5d*)in)->times;
        height = ((const fiv_tensor5d*)in)->height;
        width = ((const fiv_tensor5d*)in)->width;
        break;
    }
    /* output spatial = ceil(in / stride), same as the reference oh = (n+s-1)/s */
    size_t st = (size_t)n->params.stride;
    size_t oh = (height + st - 1) / st;
    size_t ow = (width + st - 1) / st;

    fiv_tensor_hdr* out = (fiv_tensor_hdr*)existing_output;
    if (out && out->id == in->id && out->dtype == FIV_32F1 && out->data_continue == 1) {
        size_t o_b  = 1;
        size_t o_c  = 0;
        size_t o_hh = 0;
        size_t o_ww = 0;
        switch (out->id) {
        case FIV_ID_TENSOR3D:
            o_c = ((fiv_tensor3d*)out)->channels;
            o_hh = ((fiv_tensor3d*)out)->height;
            o_ww = ((fiv_tensor3d*)out)->width;
            break;
        case FIV_ID_TENSOR4D:
            o_b = ((fiv_tensor4d*)out)->batch;
            o_c = ((fiv_tensor4d*)out)->channels;
            o_hh = ((fiv_tensor4d*)out)->height;
            o_ww = ((fiv_tensor4d*)out)->width;
            break;
        default:
            o_b = ((fiv_tensor5d*)out)->batch * ((fiv_tensor5d*)out)->times;
            o_c = ((fiv_tensor5d*)out)->channels;
            o_hh = ((fiv_tensor5d*)out)->height;
            o_ww = ((fiv_tensor5d*)out)->width;
            break;
        }
        if (o_b == batch && o_c == c_out && o_hh == oh && o_ww == ow) return out;
    }
    if (out) fiv_release_tensor((void**)&out);

    switch (in->id) {
    case FIV_ID_TENSOR3D: {
        size_t sh[3] = { c_out, oh, ow };
        out = (fiv_tensor_hdr*)fiv_create_tensor3d(sh, FIV_32F1);
        break;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)in;
        size_t sh[4] = { t->batch, c_out, oh, ow };
        out = (fiv_tensor_hdr*)fiv_create_tensor4d(sh, FIV_32F1);
        break;
    }
    default: {
        const fiv_tensor5d* t = (const fiv_tensor5d*)in;
        size_t sh[5] = { t->batch, t->times, c_out, oh, ow };
        out = (fiv_tensor_hdr*)fiv_create_tensor5d(sh, FIV_32F1);
        break;
    }
    }
    if (!out) { *out_ret = FIV_RET_ERR_MEM; return NULL; }
    return out;
}

/* out = conv(in, weight) then out[b][oc][y][x] += bias[oc] when bias enabled. */
static fiv_ret fiv_conv2d_compute(fiv_conv2d_node* n, fiv_tensor_hdr* out, fiv_tensor_hdr* in)
{
    fiv_ret r = fiv_tensor_conv2d(out, in, (void*)n->weight, &n->params);
    if (r != FIV_RET_OK) return r;
    if (!n->bias) return FIV_RET_OK;

    size_t batch   = 1;
    size_t height  = 0;
    size_t width   = 0;
    switch (in->id) {
    case FIV_ID_TENSOR3D:
        batch = 1;
        height = ((fiv_tensor3d*)in)->height;
        width = ((fiv_tensor3d*)in)->width;
        break;
    case FIV_ID_TENSOR4D:
        batch = ((fiv_tensor4d*)in)->batch;
        height = ((fiv_tensor4d*)in)->height;
        width = ((fiv_tensor4d*)in)->width;
        break;
    default:
        batch = ((fiv_tensor5d*)in)->batch * ((fiv_tensor5d*)in)->times;
        height = ((fiv_tensor5d*)in)->height;
        width = ((fiv_tensor5d*)in)->width;
        break;
    }
    size_t c_out = (size_t)n->params.output_channels;
    size_t st = (size_t)n->params.stride;
    size_t o_hw = ((height + st - 1) / st) * ((width + st - 1) / st);
    ivf32* d = out->data.fl;
    const ivf32* b = n->bias->data.fl;
    for (size_t bb = 0; bb < batch; bb++)
        for (size_t oc = 0; oc < c_out; oc++) {
            ivf32 bv = b[oc];
            ivf32* p = d + (bb * c_out + oc) * o_hw;
            for (size_t k = 0; k < o_hw; k++) p[k] += bv;
        }
    return FIV_RET_OK;
}

fiv_ret fiv_conv2d_node_forward(void* op_state, void* output, void* input)
{
    fiv_conv2d_node* n = (fiv_conv2d_node*)op_state;
    return fiv_conv2d_compute(n, (fiv_tensor_hdr*)output, (fiv_tensor_hdr*)input);
}

fiv_ret fiv_conv2d_node_inference(void* op_state, void* output, void* input)
{
    return fiv_conv2d_compute((fiv_conv2d_node*)op_state, (fiv_tensor_hdr*)output, (fiv_tensor_hdr*)input);
}

/* Conv backward (all accumulations, engine resets grads per step), generalized
   to any kernel / stride / explicit pad:
   d_w[oc][ic][ky][kx] += sum_{b,oy,ox} go[b][oc][oy][ox] * x[b][ic][sy][sx]
   db[oc]             += sum_{b,oy,ox} go[b][oc][oy][ox]
   dIn[b][ic][y][x]   += sum_{oc,ky,kx} go[b][oc][oy][ox] * w[oc][ic][ky][kx]
   where sy = oy*stride - pad_top + ky, sx = ox*stride - pad_left + kx and the
   forward padding rule applies: zero pad skips out-of-range taps, replicate
   clamps them (so a clamped tap's gradient lands on the border pixel). */
fiv_ret fiv_conv2d_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    fiv_conv2d_node* n = (fiv_conv2d_node*)op_state;
    const fiv_tensor_hdr* go = (const fiv_tensor_hdr*)grad_output;
    const fiv_tensor_hdr* x  = (const fiv_tensor_hdr*)input;
    fiv_tensor_hdr* gi = (fiv_tensor_hdr*)grad_input;
    if (!n || !go || !x) return FIV_RET_ERR_PARA;

    size_t batch  = 1;
    size_t c_in   = 0;
    size_t height = 0;
    size_t width  = 0;
    switch (x->id) {
    case FIV_ID_TENSOR3D:
        batch = 1;
        c_in = ((const fiv_tensor3d*)x)->channels;
        height = ((const fiv_tensor3d*)x)->height;
        width = ((const fiv_tensor3d*)x)->width;
        break;
    case FIV_ID_TENSOR4D:
        batch = ((const fiv_tensor4d*)x)->batch;
        c_in = ((const fiv_tensor4d*)x)->channels;
        height = ((const fiv_tensor4d*)x)->height;
        width = ((const fiv_tensor4d*)x)->width;
        break;
    default:
        batch = ((const fiv_tensor5d*)x)->batch * ((const fiv_tensor5d*)x)->times;
        c_in = ((const fiv_tensor5d*)x)->channels;
        height = ((const fiv_tensor5d*)x)->height;
        width = ((const fiv_tensor5d*)x)->width;
        break;
    }
    size_t c_out = (size_t)n->params.output_channels;
    int kh = n->params.kernel_size_y;
    int kw = n->params.kernel_size_x;
    int st = n->params.stride;
    int pt = n->params.pad_top;
    int pl = n->params.pad_left;
    /* Match the forward's legacy rule: a 3x3 stride-1 node with all pads unset
       (0) or all 1 is the historical same-padding with start pad 1. */
    if (kw == 3 && kh == 3 && st == 1 &&
        ((pt == 0 && n->params.pad_bottom == 0 && pl == 0 && n->params.pad_right == 0) ||
         (pt == 1 && n->params.pad_bottom == 1 && pl == 1 && n->params.pad_right == 1))) {
        pt = 1;
        pl = 1;
    }
    int zero_pad = (n->params.padding_method == 0);
    size_t oh = (height + (size_t)st - 1) / (size_t)st;
    size_t ow = (width + (size_t)st - 1) / (size_t)st;
    int kcin = (n->params.conv2d_method == FIV_CONV2D_DEPTHWISE) ? 1 : (int)c_in;
    int is_dw = (n->params.conv2d_method == FIV_CONV2D_DEPTHWISE);

    const ivf32* xp  = x->data.fl;
    const ivf32* gp  = go->data.fl;
    const ivf32* wp  = n->weight->data.fl;
    ivf32*       dw  = n->grad_weight->data.fl;
    ivf32*       gip = gi ? gi->data.fl : NULL;
    const size_t o_hw = oh * ow;
    const size_t ihw  = height * width;

    for (size_t b = 0; b < batch; b++) {
        const ivf32* xb = xp + b * c_in * ihw;
        const ivf32* gb = gp + b * c_out * o_hw;
        for (size_t oc = 0; oc < c_out; oc++) {
            const ivf32* goc = gb + oc * o_hw;
            if (n->grad_bias) {
                float db = 0.0f;
                for (size_t k = 0; k < o_hw; k++) db += goc[k];
                n->grad_bias->data.fl[oc] += db;
            }
            for (size_t ic = 0; ic < (size_t)kcin; ic++) {
                size_t xc = is_dw ? oc : ic;   /* depthwise: out[oc] reads in[oc] */
                const ivf32* xic = xb + xc * ihw;
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        float acc = 0.0f;
                        for (size_t oy = 0; oy < oh; oy++) {
                            int sy = (int)oy * st - pt + ky;
                            int syc;
                            if (sy < 0 || sy >= (int)height) {
                                if (zero_pad) continue;
                                syc = sy < 0 ? 0 : (int)height - 1;
                            } else {
                                syc = sy;
                            }
                            for (size_t xx = 0; xx < ow; xx++) {
                                int sx = (int)xx * st - pl + kx;
                                int sxc;
                                if (sx < 0 || sx >= (int)width) {
                                    if (zero_pad) continue;
                                    sxc = sx < 0 ? 0 : (int)width - 1;
                                } else {
                                    sxc = sx;
                                }
                                acc += goc[oy * ow + xx] * xic[(size_t)syc * width + (size_t)sxc];
                            }
                        }
                        dw[((oc * (size_t)kcin + ic) * (size_t)kh + (size_t)ky) * (size_t)kw + (size_t)kx] += acc;
                    }
                }
            }
        }
    }

    if (gip) {
        for (size_t b = 0; b < batch; b++) {
            const ivf32* gb = gp + b * c_out * o_hw;
            ivf32* gib = gip + b * c_in * ihw;
            for (size_t ic = 0; ic < c_in; ic++) {
                ivf32* giic = gib + ic * ihw;
                for (size_t oc = 0; oc < c_out; oc++) {
                    if (is_dw && oc != ic) continue;   /* depthwise: in[ic] only feeds out[ic] */
                    const ivf32* goc = gb + oc * o_hw;
                    const ivf32* wocic = wp + (oc * (size_t)kcin + ic) * (size_t)kh * kw;
                    for (size_t oy = 0; oy < oh; oy++) {
                        for (int ky = 0; ky < kh; ky++) {
                            int sy = (int)oy * st - pt + ky;
                            int y;
                            if (sy < 0 || sy >= (int)height) {
                                if (zero_pad) continue;
                                y = sy < 0 ? 0 : (int)height - 1;
                            } else {
                                y = sy;
                            }
                            for (size_t xx = 0; xx < ow; xx++) {
                                for (int kx = 0; kx < kw; kx++) {
                                    int sx = (int)xx * st - pl + kx;
                                    int xpos;
                                    if (sx < 0 || sx >= (int)width) {
                                        if (zero_pad) continue;
                                        xpos = sx < 0 ? 0 : (int)width - 1;
                                    } else {
                                        xpos = sx;
                                    }
                                    giic[(size_t)y * width + (size_t)xpos] +=
                                        goc[oy * ow + xx] * wocic[(size_t)ky * kw + (size_t)kx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return FIV_RET_OK;
}
