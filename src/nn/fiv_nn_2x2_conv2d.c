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

/* 2x2 kernel plane convolution (STD fast path, stride 2). Split out of fiv_nn_conv2d.c. */

#include "fiv_nn_2x2_conv2d.h"
#include <stddef.h>

static ivf32 fiv_conv2d_px_2x2_s2(const ivf32* src, int width, int height,
                                  int ox, int oy, int pt, int pl,
                                  const ivf32 coef[4], int zero_pad)
{
    ivf32 acc = 0.0f;
    for (int ky = 0; ky < 2; ky++) {
        int sy = oy * 2 - pt + ky;
        if (sy < 0 || sy >= height) { if (zero_pad) continue; sy = sy < 0 ? 0 : height - 1; }
        for (int kx = 0; kx < 2; kx++) {
            int sx = ox * 2 - pl + kx;
            if (sx < 0 || sx >= width) { if (zero_pad) continue; sx = sx < 0 ? 0 : width - 1; }
            acc += src[(size_t)sy * width + sx] * coef[ky * 2 + kx];
        }
    }
    return acc;
}

/* ---- 2x2 stride-2 plane convolution (the landmark downsample layers) ----
   One src plane x one 2x2 coef -> one dst plane. Mirrors the 3x3 stride-2 plane:
   the interior region (all four taps in bounds) is filled with an AVX2 pass over
   8 output columns using even/odd source-column picks (row's kx=0 tap lands on
   the even columns 2*ox-pl, kx=1 on the odd columns 2*ox-pl+1 of the same row),
   and the boundary rows/cols + scalar tail via fiv_conv2d_px_2x2_s2. Called once
   per (oc, ic) with accumulate = (ic>0), matching fiv_conv2d_generic_std's loop
   order so the sum over ic is bit-identical to the scalar reference. */
void fiv_conv2d_plane_2x2_s2(ivf32* dst, int ow, int oh, int stride_dst,
                             const ivf32* src, int width, int height, int stride_src,
                             const ivf32 coef[4], int accumulate, int zero_pad,
                             int pt, int pl)
{
    /* interior output pixels whose whole 2x2 window stays in bounds */
    int j0 = (pt + 1) / 2,          j1 = (height + pt) / 2;   if (j1 > oh) j1 = oh;
    int i0 = (pl + 1) / 2,          i1 = (width + pl) / 2;    if (i1 > ow) i1 = ow;
    if (j1 < j0) j1 = j0;
    if (i1 < i0) i1 = i0;

#if defined(FIV_USE_AVX2)
    __m256 v00 = _mm256_set1_ps(coef[0]), v01 = _mm256_set1_ps(coef[1]);
    __m256 v10 = _mm256_set1_ps(coef[2]), v11 = _mm256_set1_ps(coef[3]);
    for (int j = 0; j < oh; j++) {
        ivf32* d = dst + (size_t)j * stride_dst;
        if (j >= j0 && j < j1) {
            const ivf32* R0 = src + (size_t)(2 * j - pt) * stride_src;
            const ivf32* R1 = src + (size_t)(2 * j - pt + 1) * stride_src;
            int i_vec_end = i0 + ((i1 - i0) / 8) * 8;
            int i = i0;
            for (; i < i_vec_end; i += 8) {
                const ivf32* q0 = R0 + (2 * i - pl);
                __m256 e0 = fiv_s2_tap_even(_mm256_loadu_ps(q0), _mm256_loadu_ps(q0 + 8));
                __m256 o0 = fiv_s2_tap_odd (_mm256_loadu_ps(q0), _mm256_loadu_ps(q0 + 8));
                const ivf32* q1 = R1 + (2 * i - pl);
                __m256 e1 = fiv_s2_tap_even(_mm256_loadu_ps(q1), _mm256_loadu_ps(q1 + 8));
                __m256 o1 = fiv_s2_tap_odd (_mm256_loadu_ps(q1), _mm256_loadu_ps(q1 + 8));
                __m256 acc = _mm256_mul_ps(e0, v00);
                acc = _mm256_fmadd_ps(o0, v01, acc);
                acc = _mm256_fmadd_ps(e1, v10, acc);
                acc = _mm256_fmadd_ps(o1, v11, acc);
                if (accumulate) acc = _mm256_add_ps(acc, _mm256_loadu_ps(d + i));
                _mm256_storeu_ps(d + i, acc);
            }
            /* scalar tail + boundary cols inside the interior row band */
            for (int x = 0; x < ow; x++) {
                if (x >= i0 && x < i_vec_end) { x = i_vec_end - 1; continue; }
                ivf32 p = fiv_conv2d_px_2x2_s2(src, width, height, x, j, pt, pl, coef, zero_pad);
                d[x] = accumulate ? d[x] + p : p;
            }
        } else {   /* boundary row: scalar all cols */
            for (int x = 0; x < ow; x++) {
                ivf32 p = fiv_conv2d_px_2x2_s2(src, width, height, x, j, pt, pl, coef, zero_pad);
                d[x] = accumulate ? d[x] + p : p;
            }
        }
    }
#elif defined(FIV_USE_ARM_NEON)
    /* one vld2q per source row deinterleaves 4 output taps: val[0] = kx=0
       (even cols 2*i-pl+2k), val[1] = kx=1 (odd cols). Same tap order as the
       AVX2 pass, so the sum over ic stays bit-identical to the reference. */
    float32x4_t v00 = vdupq_n_f32(coef[0]), v01 = vdupq_n_f32(coef[1]);
    float32x4_t v10 = vdupq_n_f32(coef[2]), v11 = vdupq_n_f32(coef[3]);
    for (int j = 0; j < oh; j++) {
        ivf32* d = dst + (size_t)j * stride_dst;
        if (j >= j0 && j < j1) {
            const ivf32* R0 = src + (size_t)(2 * j - pt) * stride_src;
            const ivf32* R1 = src + (size_t)(2 * j - pt + 1) * stride_src;
            int i_vec_end = i0 + ((i1 - i0) / 4) * 4;
            int i = i0;
            for (; i < i_vec_end; i += 4) {
                float32x4x2_t t0 = vld2q_f32(R0 + (2 * i - pl));
                float32x4x2_t t1 = vld2q_f32(R1 + (2 * i - pl));
                float32x4_t acc = vfmaq_f32(vmulq_f32(t0.val[0], v00), t0.val[1], v01);
                acc = vfmaq_f32(acc, t1.val[0], v10);
                acc = vfmaq_f32(acc, t1.val[1], v11);
                if (accumulate) acc = vaddq_f32(acc, vld1q_f32(d + i));
                vst1q_f32(d + i, acc);
            }
            /* scalar tail + boundary cols inside the interior row band */
            for (int x = 0; x < ow; x++) {
                if (x >= i0 && x < i_vec_end) { x = i_vec_end - 1; continue; }
                ivf32 p = fiv_conv2d_px_2x2_s2(src, width, height, x, j, pt, pl, coef, zero_pad);
                d[x] = accumulate ? d[x] + p : p;
            }
        } else {   /* boundary row: scalar all cols */
            for (int x = 0; x < ow; x++) {
                ivf32 p = fiv_conv2d_px_2x2_s2(src, width, height, x, j, pt, pl, coef, zero_pad);
                d[x] = accumulate ? d[x] + p : p;
            }
        }
    }
#else
    for (int oy = 0; oy < oh; oy++)
        for (int ox = 0; ox < ow; ox++) {
            ivf32 p = fiv_conv2d_px_2x2_s2(src, width, height, ox, oy, pt, pl, coef, zero_pad);
            if (accumulate) dst[(size_t)oy * stride_dst + ox] += p;
            else dst[(size_t)oy * stride_dst + ox] = p;
        }
#endif
}

/* ---- multi-channel STD wrapper (owns the oc / ic loops) ----
   Landmark downsample 2x2 stride-2. Adds each input plane's contribution into
   the output plane (accumulate = ic > 0) so the sum over ic matches the scalar
   reference bit-for-bit. pt / pl: explicit start pads (0 for even-dim layers). */
void fiv_conv2d_std_2x2_s2(ivf32* d, const ivf32* s, const ivf32* w,
                           int c_in, int c_out, int width, int height,
                           int oh, int ow, int zero_pad, int pt, int pl)
{
    const size_t o_hw = (size_t)oh * ow;
    const size_t hw   = (size_t)height * width;
    for (int oc = 0; oc < c_out; oc++)
        for (int ic = 0; ic < c_in; ic++)
            fiv_conv2d_plane_2x2_s2(d + (size_t)oc * o_hw, ow, oh, ow,
                                    s + (size_t)ic * hw, width, height, width,
                                    w + ((size_t)oc * c_in + ic) * 4, ic > 0, zero_pad, pt, pl);
}
