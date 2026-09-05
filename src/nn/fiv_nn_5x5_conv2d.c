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

/* 5x5 kernel plane convolution (STD fast path, stride 2). Split out of fiv_nn_conv2d.c. */

#include "fiv_nn_5x5_conv2d.h"
#include <stddef.h>

/* 5x5 exists only at stride 2 in this module, so the stride is fixed here. */
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
void fiv_conv2d_plane_5x5_s2(ivf32* dst, int ow, int oh, int stride_dst,
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
        /* AVX2-native 5x5 stride-2. Output col o's taps sample source cols
           2o-1..2o+3, so 8 output cols i..i+7 touch source cols 2i-1..2i+17.
           For each source row load the 16-wide window [2i-1..2i+14] once and
           derive all five horizontal taps from its even/odd picks plus cheap
           in-register lane shifts (the same even/odd split the validated 3x3
           stride-2 kernel uses). Only the last 1-2 lanes of the shifted taps
           need scalar tail values (2i+15..2i+17), blended in place. The rightmost
           few columns again fall back to the scalar tail; SIMD blocks are capped
           so every load (incl. the tail scalars) stays within its source row. */
        __m256i SHIFT1 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7);
        __m256i SHIFT2 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 7, 7);
        int p_safe = (width - 18) / 2;             /* last block start with all reads in-row */
        int hi_end = (p_safe + 1 < i_end) ? (p_safe + 1) : i_end;
        if (hi_end < i_start) hi_end = i_start;
        int i_simd_end = i_start + ((hi_end - i_start) / 8) * 8;
        for (int j = j_start; j < j_end; j++) {
            const ivf32* R0 = src + (size_t)(2 * j - 1) * width;
            const ivf32* R1 = src + (size_t)(2 * j)     * width;
            const ivf32* R2 = src + (size_t)(2 * j + 1) * width;
            const ivf32* R3 = src + (size_t)(2 * j + 2) * width;
            const ivf32* R4 = src + (size_t)(2 * j + 3) * width;
            ivf32* d = dst + (size_t)j * ow;
            int i;
            for (i = i_start; i < i_simd_end; i += 8) {
#define LM5ROW(R, c0v,c1v,c2v,c3v,c4v, acc) do {                             \
                    __m256 lo  = _mm256_loadu_ps(R + (2 * i - 1));          \
                    __m256 hi  = _mm256_loadu_ps(R + (2 * i + 7));          \
                    __m256 s15 = _mm256_set1_ps((R)[2 * i + 15]);           \
                    __m256 s16 = _mm256_set1_ps((R)[2 * i + 16]);           \
                    __m256 s17 = _mm256_set1_ps((R)[2 * i + 17]);           \
                    __m256 t0 = fiv_s2_tap_even(lo, hi);                    \
                    __m256 t1 = fiv_s2_tap_odd (lo, hi);                    \
                    __m256 t2 = _mm256_permutevar8x32_ps(t0, SHIFT1);       \
                    t2 = _mm256_blend_ps(t2, s15, 0x80);                    \
                    __m256 t3 = _mm256_permutevar8x32_ps(t1, SHIFT1);       \
                    t3 = _mm256_blend_ps(t3, s16, 0x80);                    \
                    __m256 t4 = _mm256_permutevar8x32_ps(t0, SHIFT2);       \
                    t4 = _mm256_blend_ps(t4, s15, 0x40);                    \
                    t4 = _mm256_blend_ps(t4, s17, 0x80);                    \
                    acc = _mm256_fmadd_ps(t0, _mm256_set1_ps(c0v), acc);    \
                    acc = _mm256_fmadd_ps(t1, _mm256_set1_ps(c1v), acc);    \
                    acc = _mm256_fmadd_ps(t2, _mm256_set1_ps(c2v), acc);    \
                    acc = _mm256_fmadd_ps(t3, _mm256_set1_ps(c3v), acc);    \
                    acc = _mm256_fmadd_ps(t4, _mm256_set1_ps(c4v), acc);    \
                } while (0)
                __m256 acc = _mm256_setzero_ps();
                LM5ROW(R0, c0, c1, c2, c3, c4, acc);
                LM5ROW(R1, c5, c6, c7, c8, c9, acc);
                LM5ROW(R2, c10, c11, c12, c13, c14, acc);
                LM5ROW(R3, c15, c16, c17, c18, c19, acc);
                LM5ROW(R4, c20, c21, c22, c23, c24, acc);
#undef LM5ROW
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

/* ---- multi-channel STD wrapper (owns the oc / ic loops) ----
   BlazeFace stem 5x5 stride-2. Adds each input plane's contribution into the
   output plane (accumulate = ic > 0) so the sum over ic matches the scalar
   reference bit-for-bit. pt / pl: explicit start pads (SAME -> 1/1). */
void fiv_conv2d_std_5x5_s2(ivf32* d, const ivf32* s, const ivf32* w,
                           int c_in, int c_out, int width, int height,
                           int oh, int ow, int zero_pad, int pt, int pl)
{
    const size_t o_hw = (size_t)oh * ow;
    const size_t hw   = (size_t)height * width;
    for (int oc = 0; oc < c_out; oc++)
        for (int ic = 0; ic < c_in; ic++)
            fiv_conv2d_plane_5x5_s2(d + (size_t)oc * o_hw, ow, oh, ow,
                                    s + (size_t)ic * hw, width, height, width,
                                    w + ((size_t)oc * c_in + ic) * 25, ic > 0, zero_pad, pt, pl);
}
