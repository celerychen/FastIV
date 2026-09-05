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

/* 1x1 kernel plane convolution (pointwise fast path). Split out of fiv_nn_conv2d.c. */

#include "fiv_nn_1x1_conv2d.h"
#include <string.h>

void fiv_conv2d_pw(ivf32* d, const ivf32* s, const ivf32* w,
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