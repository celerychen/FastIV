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

static ivf32 conv2d_px_pad(const ivf32* src, int Ws, int Hs, int ss,
                           const ivf32 coef[9], int i, int j, int zero_pad)
{
    ivf32 sum = 0.0f;
    for (int kj = 0; kj < 3; kj++) {
        for (int ki = 0; ki < 3; ki++) {
            int sx = i + ki - 1;
            int sy = j + kj - 1;
            if (sx < 0 || sx >= Ws || sy < 0 || sy >= Hs) {
                if (zero_pad) continue;  /* zero padding: out-of-range contributes 0 */
                if (sx < 0) sx = 0;
                if (sx >= Ws) sx = Ws - 1;
                if (sy < 0) sy = 0;
                if (sy >= Hs) sy = Hs - 1;
            }
            sum += src[(size_t)sy * ss + sx] * coef[kj * 3 + ki];
        }
    }
    return sum;
}

static void fiv_conv2d_plane(ivf32* dst, int width_dst, int height_dst, int stride_dst,
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
            ivf32 p = conv2d_px_pad(src, width_src, height_src, stride_src, coef, i, j, zero_pad);
            dst[(size_t)j * stride_dst + i] =
                accumulate ? dst[(size_t)j * stride_dst + i] + p : p;
        }

    for (int j = j_end; j < height_dst; j++)
        for (int i = 0; i < width_dst; i++) {
            ivf32 p = conv2d_px_pad(src, width_src, height_src, stride_src, coef, i, j, zero_pad);
            dst[(size_t)j * stride_dst + i] =
                accumulate ? dst[(size_t)j * stride_dst + i] + p : p;
        }

    for (int j = j_start; j < j_end; j++) {
        for (int i = 0; i < i_start && i < width_dst; i++) {
            ivf32 p = conv2d_px_pad(src, width_src, height_src, stride_src, coef, i, j, zero_pad);
            dst[(size_t)j * stride_dst + i] =
                accumulate ? dst[(size_t)j * stride_dst + i] + p : p;
        }
        for (int i = i_end; i < width_dst; i++) {
            ivf32 p = conv2d_px_pad(src, width_src, height_src, stride_src, coef, i, j, zero_pad);
            dst[(size_t)j * stride_dst + i] =
                accumulate ? dst[(size_t)j * stride_dst + i] + p : p;
        }
    }
}

/* ---- public API ---- */

fiv_ret fiv_tensor_conv2d(void* dst, void* src, void* kernel, fiv_conv2d_params* params)
{
    fiv_tensor_hdr* sh = (fiv_tensor_hdr*)src;
    fiv_tensor_hdr* dh = (fiv_tensor_hdr*)dst;
    fiv_tensor_hdr* kh = (fiv_tensor_hdr*)kernel;
    if (!dst || !src || !kernel || !params) return FIV_RET_ERR_PARA;

    if (sh->id < FIV_ID_TENSOR3D || sh->id > FIV_ID_TENSOR5D) return FIV_RET_ERR_PARA;  /* need C,H,W */
    if (kh->id != FIV_ID_TENSOR4D) return FIV_RET_ERR_PARA;
    if (dh->id != sh->id) return FIV_RET_ERR_PARA;
    if (sh->dtype != FIV_32F1 || dh->dtype != FIV_32F1 || kh->dtype != FIV_32F1)
        return FIV_RET_ERR_NOT_SUPPORT;
    if (!sh->data_continue || !dh->data_continue || !kh->data_continue)
        return FIV_RET_ERR_PARA;

    if (params->conv2d_method != FIV_CONV2D_STD && params->conv2d_method != FIV_CONV2D_DEPTHWISE)
        return FIV_RET_ERR_NOT_SUPPORT;
    if (params->kernel_size_x != 3 || params->kernel_size_y != 3) return FIV_RET_ERR_NOT_SUPPORT;
    if (params->stride != 1) return FIV_RET_ERR_NOT_SUPPORT;
    if (params->padding_method != 0 && params->padding_method != 1) return FIV_RET_ERR_PARA;

    size_t C_in, H, W, n_batch;
    switch (sh->id) {
    case FIV_ID_TENSOR3D: {
        const fiv_tensor3d* t = (const fiv_tensor3d*)sh;
        C_in = t->channels; H = t->height; W = t->width; n_batch = 1;
        break;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)sh;
        C_in = t->channels; H = t->height; W = t->width; n_batch = t->batch;
        break;
    }
    default: {
        const fiv_tensor5d* t = (const fiv_tensor5d*)sh;
        C_in = t->channels; H = t->height; W = t->width;
        n_batch = t->batch * t->times;
        break;
    }
    }
    if (C_in == 0 || H == 0 || W == 0) return FIV_RET_ERR_PARA;

    const fiv_tensor4d* k = (const fiv_tensor4d*)kernel;
    if (k->shapes[2] != 3 || k->shapes[3] != 3) return FIV_RET_ERR_PARA;
    size_t kCout = k->shapes[0];
    size_t kCin  = k->shapes[1];
    if (params->input_channels != (int)C_in) return FIV_RET_ERR_PARA;

    size_t C_out;
    if (params->conv2d_method == FIV_CONV2D_STD) {
        if (kCin != C_in) return FIV_RET_ERR_PARA;
        C_out = kCout;
    } else {  /* DEPTHWISE: one kernel per channel, output channels == input channels */
        if (kCin != 1) return FIV_RET_ERR_PARA;
        if (kCout != C_in) return FIV_RET_ERR_PARA;
        C_out = C_in;
    }
    if (params->output_channels != (int)C_out) return FIV_RET_ERR_PARA;

    size_t dC, dH, dW;
    switch (dh->id) {
    case FIV_ID_TENSOR3D: {
        const fiv_tensor3d* t = (const fiv_tensor3d*)dh;
        dC = t->channels; dH = t->height; dW = t->width;
        break;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)dh;
        dC = t->channels; dH = t->height; dW = t->width;
        break;
    }
    default: {
        const fiv_tensor5d* t = (const fiv_tensor5d*)dh;
        dC = t->channels; dH = t->height; dW = t->width;
        break;
    }
    }
    if (dC != C_out || dH != H || dW != W) return FIV_RET_ERR_PARA;

    const ivf32* s = sh->data.fl;
    const ivf32* w = kh->data.fl;
    ivf32* d = dh->data.fl;
    size_t HW    = H * W;
    size_t sChan = C_in * HW;
    size_t dChan = C_out * HW;
    size_t kChan = kCin * 9;
    int zero_pad = (params->padding_method == 0);

    for (size_t b = 0; b < n_batch; b++) {
        const ivf32* sb = s + b * sChan;
        ivf32* db = d + b * dChan;
        if (params->conv2d_method == FIV_CONV2D_STD) {
            for (size_t oc = 0; oc < C_out; oc++)
                for (size_t ic = 0; ic < C_in; ic++)
                    fiv_conv2d_plane(db + oc * HW, (int)W, (int)H, (int)W,
                                     sb + ic * HW, (int)W, (int)H, (int)W,
                                     w + oc * kChan + ic * 9, ic > 0, zero_pad);
        } else {  /* DEPTHWISE: out[c] = conv(src[c], w[c]) */
            for (size_t c = 0; c < C_out; c++)
                fiv_conv2d_plane(db + c * HW, (int)W, (int)H, (int)W,
                                 sb + c * HW, (int)W, (int)H, (int)W,
                                 w + c * 9, 0, zero_pad);
        }
    }
    return FIV_RET_OK;
}

/* ---- CONV2D_STD network node ---- */

void* fiv_conv2d_node_create(void* params)
{
    const fiv_conv2d_params* p = (const fiv_conv2d_params*)params;
    if (!p) return NULL;
    if (p->conv2d_method != FIV_CONV2D_STD) return NULL;
    if (p->kernel_size_x != 3 || p->kernel_size_y != 3) return NULL;
    if (p->stride != 1) return NULL;
    if (p->padding_method != 0 && p->padding_method != 1) return NULL;
    if (p->bias != 0 && p->bias != 1) return NULL;
    if (p->input_channels <= 0 || p->output_channels <= 0) return NULL;

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

    size_t wsh[4] = { (size_t)p->output_channels, (size_t)p->input_channels, 3, 3 };
    n->weight = fiv_create_tensor4d(wsh, FIV_32F1);
    if (!n->weight) { fiv_free(n); return NULL; }
    {
        ivf32* w = n->weight->data.fl;
        size_t nw = (size_t)p->output_channels * (size_t)p->input_channels * 9;
        float s = 1.0f / sqrtf((float)(p->input_channels * 9));
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

    size_t C_out = (size_t)n->params.output_channels;
    size_t B = 1;
    size_t H = 0;
    size_t W = 0;
    switch (in->id) {
    case FIV_ID_TENSOR3D:
        H = ((const fiv_tensor3d*)in)->height;
        W = ((const fiv_tensor3d*)in)->width;
        break;
    case FIV_ID_TENSOR4D:
        B = ((const fiv_tensor4d*)in)->batch;
        H = ((const fiv_tensor4d*)in)->height;
        W = ((const fiv_tensor4d*)in)->width;
        break;
    default:
        B = ((const fiv_tensor5d*)in)->batch * ((const fiv_tensor5d*)in)->times;
        H = ((const fiv_tensor5d*)in)->height;
        W = ((const fiv_tensor5d*)in)->width;
        break;
    }

    fiv_tensor_hdr* out = (fiv_tensor_hdr*)existing_output;
    if (out && out->id == in->id && out->dtype == FIV_32F1 && out->data_continue == 1) {
        size_t oB = 1;
        size_t oC = 0;
        size_t oH = 0;
        size_t oW = 0;
        switch (out->id) {
        case FIV_ID_TENSOR3D:
            oC = ((fiv_tensor3d*)out)->channels;
            oH = ((fiv_tensor3d*)out)->height;
            oW = ((fiv_tensor3d*)out)->width;
            break;
        case FIV_ID_TENSOR4D:
            oB = ((fiv_tensor4d*)out)->batch;
            oC = ((fiv_tensor4d*)out)->channels;
            oH = ((fiv_tensor4d*)out)->height;
            oW = ((fiv_tensor4d*)out)->width;
            break;
        default:
            oB = ((fiv_tensor5d*)out)->batch * ((fiv_tensor5d*)out)->times;
            oC = ((fiv_tensor5d*)out)->channels;
            oH = ((fiv_tensor5d*)out)->height;
            oW = ((fiv_tensor5d*)out)->width;
            break;
        }
        if (oB == B && oC == C_out && oH == H && oW == W) return out;
    }
    if (out) fiv_release_tensor((void**)&out);

    switch (in->id) {
    case FIV_ID_TENSOR3D: {
        size_t sh[3] = { C_out, H, W };
        out = (fiv_tensor_hdr*)fiv_create_tensor3d(sh, FIV_32F1);
        break;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)in;
        size_t sh[4] = { t->batch, C_out, H, W };
        out = (fiv_tensor_hdr*)fiv_create_tensor4d(sh, FIV_32F1);
        break;
    }
    default: {
        const fiv_tensor5d* t = (const fiv_tensor5d*)in;
        size_t sh[5] = { t->batch, t->times, C_out, H, W };
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

    size_t B = 1;
    size_t H = 0;
    size_t W = 0;
    switch (in->id) {
    case FIV_ID_TENSOR3D:
        B = 1;
        H = ((fiv_tensor3d*)in)->height;
        W = ((fiv_tensor3d*)in)->width;
        break;
    case FIV_ID_TENSOR4D:
        B = ((fiv_tensor4d*)in)->batch;
        H = ((fiv_tensor4d*)in)->height;
        W = ((fiv_tensor4d*)in)->width;
        break;
    default:
        B = ((fiv_tensor5d*)in)->batch * ((fiv_tensor5d*)in)->times;
        H = ((fiv_tensor5d*)in)->height;
        W = ((fiv_tensor5d*)in)->width;
        break;
    }
    size_t C_out = (size_t)n->params.output_channels;
    size_t HW = H * W;
    ivf32* d = out->data.fl;
    const ivf32* b = n->bias->data.fl;
    for (size_t bb = 0; bb < B; bb++)
        for (size_t oc = 0; oc < C_out; oc++) {
            ivf32 bv = b[oc];
            ivf32* p = d + (bb * C_out + oc) * HW;
            for (size_t k = 0; k < HW; k++) p[k] += bv;
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

/* Conv backward (all accumulations, engine resets grads per step):
   dW[oc][ic][ky][kx] += sum_{b,y,x} go[b][oc][y][x] * x[b][ic][y+ky-1][x+kx-1]
   db[oc]             += sum_{b,y,x} go[b][oc][y][x]
   gi[b][ic][y][x]    += sum_{oc,ky,kx} go[b][oc][y-ky+1][x-kx+1] * w[oc][ic][ky][kx]
   Out-of-range src taps follow the forward padding rule (zero: skip, edge: clamp);
   out-of-range grad-output taps contribute 0. */
fiv_ret fiv_conv2d_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    fiv_conv2d_node* n = (fiv_conv2d_node*)op_state;
    const fiv_tensor_hdr* go = (const fiv_tensor_hdr*)grad_output;
    const fiv_tensor_hdr* x  = (const fiv_tensor_hdr*)input;
    fiv_tensor_hdr* gi = (fiv_tensor_hdr*)grad_input;
    if (!n || !go || !x) return FIV_RET_ERR_PARA;

    size_t B = 1;
    size_t C_in = 0;
    size_t H = 0;
    size_t W = 0;
    switch (x->id) {
    case FIV_ID_TENSOR3D:
        B = 1;
        C_in = ((const fiv_tensor3d*)x)->channels;
        H = ((const fiv_tensor3d*)x)->height;
        W = ((const fiv_tensor3d*)x)->width;
        break;
    case FIV_ID_TENSOR4D:
        B = ((const fiv_tensor4d*)x)->batch;
        C_in = ((const fiv_tensor4d*)x)->channels;
        H = ((const fiv_tensor4d*)x)->height;
        W = ((const fiv_tensor4d*)x)->width;
        break;
    default:
        B = ((const fiv_tensor5d*)x)->batch * ((const fiv_tensor5d*)x)->times;
        C_in = ((const fiv_tensor5d*)x)->channels;
        H = ((const fiv_tensor5d*)x)->height;
        W = ((const fiv_tensor5d*)x)->width;
        break;
    }
    size_t C_out = (size_t)n->params.output_channels;
    int zero_pad = (n->params.padding_method == 0);
    size_t HW = H * W;
    const ivf32* xp = x->data.fl;
    const ivf32* gp = go->data.fl;
    const ivf32* wp = n->weight->data.fl;
    ivf32* dw = n->grad_weight->data.fl;
    ivf32* gip = gi ? gi->data.fl : NULL;

    for (size_t b = 0; b < B; b++) {
        const ivf32* xb = xp + b * C_in * HW;
        const ivf32* gb = gp + b * C_out * HW;
        for (size_t oc = 0; oc < C_out; oc++) {
            const ivf32* goc = gb + oc * HW;
            if (n->grad_bias) {
                float db = 0.0f;
                for (size_t k = 0; k < HW; k++) db += goc[k];
                n->grad_bias->data.fl[oc] += db;
            }
            for (size_t ic = 0; ic < C_in; ic++) {
                const ivf32* xic = xb + ic * HW;
                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        float acc = 0.0f;
                        for (size_t y = 0; y < H; y++) {
                            int sy = (int)y + ky - 1;
                            for (size_t xx = 0; xx < W; xx++) {
                                int sx = (int)xx + kx - 1;
                                ivf32 xv;
                                if (sy < 0 || sy >= (int)H || sx < 0 || sx >= (int)W) {
                                    if (zero_pad) continue;
                                    int cy = sy < 0 ? 0 : (sy >= (int)H ? (int)H - 1 : sy);
                                    int cx = sx < 0 ? 0 : (sx >= (int)W ? (int)W - 1 : sx);
                                    xv = xic[(size_t)cy * W + (size_t)cx];
                                } else {
                                    xv = xic[(size_t)sy * W + (size_t)sx];
                                }
                                acc += goc[y * W + xx] * xv;
                            }
                        }
                        dw[(oc * C_in + ic) * 9 + (size_t)ky * 3 + (size_t)kx] += acc;
                    }
                }
            }
        }
    }

    if (gip) {
        for (size_t b = 0; b < B; b++) {
            const ivf32* gb = gp + b * C_out * HW;
            ivf32* gib = gip + b * C_in * HW;
            for (size_t ic = 0; ic < C_in; ic++) {
                ivf32* giic = gib + ic * HW;
                for (size_t y = 0; y < H; y++) {
                    for (size_t xx = 0; xx < W; xx++) {
                        float acc = 0.0f;
                        for (size_t oc = 0; oc < C_out; oc++) {
                            const ivf32* goc = gb + oc * HW;
                            const ivf32* wocic = wp + (oc * C_in + ic) * 9;
                            for (int ky = 0; ky < 3; ky++) {
                                int gy = (int)y - ky + 1;
                                if (gy < 0 || gy >= (int)H) continue;
                                for (int kx = 0; kx < 3; kx++) {
                                    int gx = (int)xx - kx + 1;
                                    if (gx < 0 || gx >= (int)W) continue;
                                    acc += goc[(size_t)gy * W + (size_t)gx]
                                         * wocic[(size_t)ky * 3 + (size_t)kx];
                                }
                            }
                        }
                        giic[y * W + xx] += acc;
                    }
                }
            }
        }
    }
    return FIV_RET_OK;
}
