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

#include "fiv_image.h"
#include "fiv_image_gaussion_blur.h"
#include "fiv_ctensor.h"
#include "fiv_common.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


//#define FIV_HIGH_PRECISE_GAUSS_BLUR

#define FIV_MAX_GAUSS_COEF_WIDTH (64)

static void get_gaussion_kernel(ivf64* kernel, int coef_count, ivf64 sigma)
{
#define SMALL_GAUSSION_KERNAL_SIZE (7)
    static const ivf32 small_gaussion_tab[][SMALL_GAUSSION_KERNAL_SIZE] = {
        { 1.f },
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        { 0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f }
    };

    const ivf32* fixed_kernel = coef_count % 2 == 1 && coef_count <= SMALL_GAUSSION_KERNAL_SIZE && sigma <= 0 ? small_gaussion_tab[coef_count >> 1] : 0;

    ivf64* coef_data = kernel;
    ivf64 eff_sigma = sigma > 0 ? sigma : ((coef_count - 1) * 0.5 - 1) * 0.3 + 0.8;
    ivf64 inv_two_sigma_sq = -0.5 / (eff_sigma * eff_sigma);
    ivf64 sum = 0;

    for (int i = 0; i < coef_count; i++) {
        ivf64 coord = i - (coef_count - 1) * 0.5;
        ivf64 val = fixed_kernel ? fixed_kernel[i] : expl(inv_two_sigma_sq * coord * coord);
        coef_data[i] = val;
        sum += coef_data[i];
    }

    sum = 1. / sum;

    for (int i = 0; i < coef_count; i++)
        coef_data[i] *= sum;
}

static fiv_ret fiv_compute_gaussion_filter_coef(ivf32* gauss_coef, int* gauss_coef_width, ivf32 sigma, int real_type_flag, int size)
{
    fiv_ret ret = FIV_RET_OK;
    int i, coef_count;
    ivf64 kernel_f64[FIV_MAX_GAUSS_COEF_WIDTH];

    if (real_type_flag)    coef_count = 4;
    else                   coef_count = 3;

    if (size <= 0) {
        int k = (int)(sigma * (ivf32)coef_count * 2.0f + 1.0f + 0.5f);
        if ((k & 1) == 0) k++;
        *gauss_coef_width = k;
    } else {
        *gauss_coef_width = (size & 1) ? size : 2 * size + 1;
    }
    int gauss_radius = (*gauss_coef_width - 1) / 2;
    (void)gauss_radius;

    if (*gauss_coef_width > FIV_MAX_GAUSS_COEF_WIDTH)
        return FIV_RET_ERR_NOT_SUPPORT;

    get_gaussion_kernel(kernel_f64, *gauss_coef_width, sigma);

    for (i = 0; i < *gauss_coef_width; i++)
        gauss_coef[i] = (ivf32)kernel_f64[i];

    return ret;
}


static void row_filter_u8_q8(iv8u* buf_row, const iv8u* src_row, int width, int ch_count, const iv16u* coef, int ksize)
{
    int radius = (ksize - 1) / 2;
    const int row_elems = width * ch_count;
    int col = 0;

    if (width <= ksize) {
        for (; col < width; col++) {
            for (int c = 0; c < ch_count; c++) {
                int acc = (int)coef[0] * src_row[(col - radius) * ch_count + c];
                for (int k = 1; k < ksize; k++) {
                    int px = col - radius + k;
                    if (px < 0) px = 0; else if (px >= width) px = width - 1;
                    acc += (int)coef[k] * src_row[px * ch_count + c];
                }
                int v = (acc + 128) >> 8;
                if (v < 0) v = 0; else if (v > 255) v = 255;
                buf_row[col * ch_count + c] = (iv8u)v;
            }
        }
        return;
    }

    for (; col < radius; col++) {
        for (int c = 0; c < ch_count; c++) {
            int acc = (int)coef[0] * src_row[(col - radius) * ch_count + c];
            for (int k = 1; k < ksize; k++) {
                int px = col - radius + k;
                if (px < 0) px = 0;
                acc += (int)coef[k] * src_row[px * ch_count + c];
            }
            int v = (acc + 128) >> 8;
            if (v < 0) v = 0; else if (v > 255) v = 255;
            buf_row[col * ch_count + c] = (iv8u)v;
        }
    }

    {
        int i = radius * ch_count;
        int i_end = (width - radius) * ch_count;

#if defined(FIV_USE_AVX2)
        __m256i zero_vec = _mm256_setzero_si256();
        __m256i round128 = _mm256_set1_epi16(128);
        for (; i + 32 <= i_end; i += 32) {
            __m256i acc_lo = zero_vec, acc_hi = zero_vec;
            const iv8u* src_ptr = src_row + i;
            for (int k = 0; k < ksize; k++) {
                const iv8u* p = src_ptr + (k - radius) * ch_count;
                __m256i weight = _mm256_set1_epi16((short)coef[k]);
                __m256i pix_vec = _mm256_loadu_si256((const __m256i*)p);
                __m256i pix_hi = _mm256_unpackhi_epi8(pix_vec, zero_vec);
                __m256i pix_lo = _mm256_unpacklo_epi8(pix_vec, zero_vec);
                pix_lo = _mm256_mullo_epi16(pix_lo, weight);
                pix_hi = _mm256_mullo_epi16(pix_hi, weight);
                acc_lo = _mm256_adds_epu16(acc_lo, pix_lo);
                acc_hi = _mm256_adds_epu16(acc_hi, pix_hi);
            }
            acc_lo = _mm256_adds_epu16(acc_lo, round128);
            acc_hi = _mm256_adds_epu16(acc_hi, round128);
            acc_lo = _mm256_srli_epi16(acc_lo, 8);
            acc_hi = _mm256_srli_epi16(acc_hi, 8);
            acc_lo = _mm256_packus_epi16(acc_lo, acc_hi);
            _mm256_storeu_si256((__m256i*)(buf_row + i), acc_lo);
        }
#endif

#if defined(FIV_USE_AVX2) || defined(FIV_USE_AVX)
        __m128i zero_vec_sse = _mm_setzero_si128();
        __m128i round128_sse = _mm_set1_epi16(128);
        for (; i + 16 <= i_end; i += 16) {
            __m128i acc_lo = zero_vec_sse, acc_hi = zero_vec_sse;
            const iv8u* src_ptr = src_row + i;
            for (int k = 0; k < ksize; k++) {
                const iv8u* p = src_ptr + (k - radius) * ch_count;
                __m128i weight = _mm_set1_epi16((short)coef[k]);
                __m128i pix_vec = _mm_loadu_si128((const __m128i*)p);
                __m128i pix_hi = _mm_unpackhi_epi8(pix_vec, zero_vec_sse);
                __m128i pix_lo = _mm_unpacklo_epi8(pix_vec, zero_vec_sse);
                pix_lo = _mm_mullo_epi16(pix_lo, weight);
                pix_hi = _mm_mullo_epi16(pix_hi, weight);
                acc_lo = _mm_adds_epu16(acc_lo, pix_lo);
                acc_hi = _mm_adds_epu16(acc_hi, pix_hi);
            }
            acc_lo = _mm_adds_epu16(acc_lo, round128_sse);
            acc_hi = _mm_adds_epu16(acc_hi, round128_sse);
            acc_lo = _mm_srli_epi16(acc_lo, 8);
            acc_hi = _mm_srli_epi16(acc_hi, 8);
            acc_lo = _mm_packus_epi16(acc_lo, acc_hi);
            _mm_storeu_si128((__m128i*)(buf_row + i), acc_lo);
        }
#endif

        for (; i < i_end; i++) {
            int acc = (int)coef[0] * src_row[i - radius * ch_count];
            for (int k = 1; k < ksize; k++)
                acc += (int)coef[k] * src_row[i + (k - radius) * ch_count];
            int v = (acc + 128) >> 8;
            if (v < 0) v = 0; else if (v > 255) v = 255;
            buf_row[i] = (iv8u)v;
        }
    }

    for (col = width - radius; col < width; col++) {
        for (int c = 0; c < ch_count; c++) {
            int acc = (int)coef[0] * src_row[(col - radius) * ch_count + c];
            for (int k = 1; k < ksize; k++) {
                int px = col - radius + k;
                if (px >= width) px = width - 1;
                acc += (int)coef[k] * src_row[px * ch_count + c];
            }
            int v = (acc + 128) >> 8;
            if (v < 0) v = 0; else if (v > 255) v = 255;
            buf_row[col * ch_count + c] = (iv8u)v;
        }
    }
}

static void row_filter_f32(ivf32* buf_row, const ivf32* src_row, int width, int ch_count, const ivf32* coef, int ksize)
{
    int radius = (ksize - 1) / 2;
    int col = 0;

    if (width <= ksize) {
        for (; col < width; col++) {
            for (int c = 0; c < ch_count; c++) {
                ivf32 acc = 0;
                for (int k = 0; k < ksize; k++) {
                    int px = col - radius + k;
                    if (px < 0) px = 0; else if (px >= width) px = width - 1;
                    acc += coef[k] * src_row[(size_t)px * ch_count + c];
                }
                buf_row[(size_t)col * ch_count + c] = acc;
            }
        }
        return;
    }

    for (; col < radius; col++) {
        for (int c = 0; c < ch_count; c++) {
            ivf32 acc = 0;
            for (int k = 0; k < ksize; k++) {
                int px = col - radius + k;
                if (px < 0) px = 0;
                acc += coef[k] * src_row[(size_t)px * ch_count + c];
            }
            buf_row[(size_t)col * ch_count + c] = acc;
        }
    }

    {
        int i = radius * ch_count;
        int i_end = (width - radius) * ch_count;

#if defined(FIV_USE_AVX2)
        for (; i + 8 <= i_end; i += 8) {
            const ivf32* base = src_row + i;
            __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(base - radius * ch_count), _mm256_set1_ps(coef[0]));
            for (int k = 1; k < ksize; k++)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(base + (k - radius) * ch_count),
                                      _mm256_set1_ps(coef[k]), acc);
            _mm256_storeu_ps(buf_row + i, acc);
        }
#endif

        for (; i < i_end; i++) {
            ivf32 acc = coef[0] * src_row[i - radius * ch_count];
            for (int k = 1; k < ksize; k++)
                acc += coef[k] * src_row[i + (k - radius) * ch_count];
            buf_row[i] = acc;
        }
    }

    for (col = width - radius; col < width; col++) {
        for (int c = 0; c < ch_count; c++) {
            ivf32 acc = 0;
            for (int k = 0; k < ksize; k++) {
                int px = col - radius + k;
                if (px >= width) px = width - 1;
                acc += coef[k] * src_row[(size_t)px * ch_count + c];
            }
            buf_row[(size_t)col * ch_count + c] = acc;
        }
    }
}


static void col_filter_u8_q8(iv8u* dst_row, const iv8u* row_ptrs[], const iv16u* coef, int ksize, int row_elems)
{
    int radius = (ksize - 1) / 2;
    const iv16u* coef_center = coef + radius;
    const iv8u** centered_rows = row_ptrs + radius;
    int i = 0, k;

#if defined(FIV_USE_AVX2)
    __m256i zero_vec = _mm256_setzero_si256();
    __m256i round128 = _mm256_set1_epi16(128);
    for (; i + 32 <= row_elems; i += 32) {
        __m256i weight = _mm256_set1_epi16((short)coef_center[0]);
        __m256i acc_lo, acc_hi;
        __m256i src_vec = _mm256_loadu_si256((const __m256i*)(centered_rows[0] + i));
        __m256i src_lo = _mm256_unpacklo_epi8(src_vec, zero_vec);
        __m256i src_hi = _mm256_unpackhi_epi8(src_vec, zero_vec);
        acc_lo = _mm256_mullo_epi16(src_lo, weight);
        acc_hi = _mm256_mullo_epi16(src_hi, weight);
        for (k = 1; k <= radius; k++) {
            __m256i pos_vec = _mm256_loadu_si256((const __m256i*)(centered_rows[k] + i));
            __m256i neg_vec = _mm256_loadu_si256((const __m256i*)(centered_rows[-k] + i));
            __m256i pos_lo = _mm256_unpacklo_epi8(pos_vec, zero_vec);
            __m256i pos_hi = _mm256_unpackhi_epi8(pos_vec, zero_vec);
            __m256i neg_lo = _mm256_unpacklo_epi8(neg_vec, zero_vec);
            __m256i neg_hi = _mm256_unpackhi_epi8(neg_vec, zero_vec);
            pos_lo = _mm256_adds_epu16(pos_lo, neg_lo);
            pos_hi = _mm256_adds_epu16(pos_hi, neg_hi);
            weight = _mm256_set1_epi16((short)coef_center[k]);
            pos_lo = _mm256_mullo_epi16(pos_lo, weight);
            pos_hi = _mm256_mullo_epi16(pos_hi, weight);
            acc_lo = _mm256_adds_epu16(acc_lo, pos_lo);
            acc_hi = _mm256_adds_epu16(acc_hi, pos_hi);
        }
        acc_lo = _mm256_adds_epu16(acc_lo, round128);
        acc_hi = _mm256_adds_epu16(acc_hi, round128);
        acc_lo = _mm256_srli_epi16(acc_lo, 8);
        acc_hi = _mm256_srli_epi16(acc_hi, 8);
        acc_lo = _mm256_packus_epi16(acc_lo, acc_hi);
        _mm256_storeu_si256((__m256i*)(dst_row + i), acc_lo);
    }
#endif

#if defined(FIV_USE_AVX2) || defined(FIV_USE_AVX)
    __m128i zero_vec_sse = _mm_setzero_si128();
    __m128i round128_sse = _mm_set1_epi16(128);
    for (; i + 16 <= row_elems; i += 16) {
        __m128i weight = _mm_set1_epi16((short)coef_center[0]);
        __m128i acc_lo, acc_hi;
        __m128i src_vec = _mm_loadu_si128((const __m128i*)(centered_rows[0] + i));
        __m128i src_lo = _mm_unpacklo_epi8(src_vec, zero_vec_sse);
        __m128i src_hi = _mm_unpackhi_epi8(src_vec, zero_vec_sse);
        acc_lo = _mm_mullo_epi16(src_lo, weight);
        acc_hi = _mm_mullo_epi16(src_hi, weight);
        for (k = 1; k <= radius; k++) {
            __m128i pos_vec = _mm_loadu_si128((const __m128i*)(centered_rows[k] + i));
            __m128i neg_vec = _mm_loadu_si128((const __m128i*)(centered_rows[-k] + i));
            __m128i pos_lo = _mm_unpacklo_epi8(pos_vec, zero_vec_sse);
            __m128i pos_hi = _mm_unpackhi_epi8(pos_vec, zero_vec_sse);
            __m128i neg_lo = _mm_unpacklo_epi8(neg_vec, zero_vec_sse);
            __m128i neg_hi = _mm_unpackhi_epi8(neg_vec, zero_vec_sse);
            pos_lo = _mm_adds_epu16(pos_lo, neg_lo);
            pos_hi = _mm_adds_epu16(pos_hi, neg_hi);
            weight = _mm_set1_epi16((short)coef_center[k]);
            pos_lo = _mm_mullo_epi16(pos_lo, weight);
            pos_hi = _mm_mullo_epi16(pos_hi, weight);
            acc_lo = _mm_adds_epu16(acc_lo, pos_lo);
            acc_hi = _mm_adds_epu16(acc_hi, pos_hi);
        }
        acc_lo = _mm_adds_epu16(acc_lo, round128_sse);
        acc_hi = _mm_adds_epu16(acc_hi, round128_sse);
        acc_lo = _mm_srli_epi16(acc_lo, 8);
        acc_hi = _mm_srli_epi16(acc_hi, 8);
        acc_lo = _mm_packus_epi16(acc_lo, acc_hi);
        _mm_storeu_si128((__m128i*)(dst_row + i), acc_lo);
    }
#endif

#if defined(FIV_USE_AVX2) || defined(FIV_USE_AVX)
    for (; i + 4 <= row_elems; i += 4) {
        __m128i weight = _mm_set1_epi16((short)coef_center[0]);
        __m128i acc_lo = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*(const int*)(centered_rows[0] + i)));
        acc_lo = _mm_mullo_epi16(acc_lo, weight);
        for (k = 1; k <= radius; k++) {
            __m128i pos_hi = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*(const int*)(centered_rows[k] + i)));
            __m128i neg_hi = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*(const int*)(centered_rows[-k] + i)));
            pos_hi = _mm_adds_epu16(pos_hi, neg_hi);
            weight = _mm_set1_epi16((short)coef_center[k]);
            pos_hi = _mm_mullo_epi16(pos_hi, weight);
            acc_lo = _mm_adds_epu16(acc_lo, pos_hi);
        }
        acc_lo = _mm_adds_epu16(acc_lo, round128_sse);
        acc_lo = _mm_srli_epi16(acc_lo, 8);
        acc_lo = _mm_packus_epi16(acc_lo, acc_lo);
        *(int*)(dst_row + i) = _mm_cvtsi128_si32(acc_lo);
    }
#endif

    for (; i < row_elems; i++) {
        int acc = (int)coef_center[0] * centered_rows[0][i];
        for (k = 1; k <= radius; k++)
            acc += (int)coef_center[k] * (centered_rows[k][i] + centered_rows[-k][i]);
        dst_row[i] = (iv8u)((acc + 128) >> 8);
    }
}

static void col_filter_f32(ivf32* dst_row, const ivf32* row_ptrs[], const ivf32* coef, int ksize, int row_elems)
{
    int i = 0, k;

#if defined(FIV_USE_AVX2)
    for (; i + 8 <= row_elems; i += 8) {
        __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(row_ptrs[0] + i), _mm256_set1_ps(coef[0]));
        for (k = 1; k < ksize; k++)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row_ptrs[k] + i), _mm256_set1_ps(coef[k]), acc);
        _mm256_storeu_ps(dst_row + i, acc);
    }
#endif

    for (; i < row_elems; i++) {
        ivf32 acc = row_ptrs[0][i] * coef[0];
        for (k = 1; k < ksize; k++)
            acc += row_ptrs[k][i] * coef[k];
        dst_row[i] = acc;
    }
}


static inline const ivf32* ring_row(const ivf32* buf, int last_slot, int last_y, int ring_size, int row_elems, int src_row_idx)
{
    return buf + (size_t)((last_slot - (last_y - src_row_idx) + ring_size) % ring_size) * row_elems;
}


static inline void gather_rows(const ivf32** rows, const ivf32* buf, int last_slot, int last_y,
                               int ring_size, int row_elems, int out_row, int radius, int ksize, int height)
{
    int base = out_row - radius;
    if (height <= ksize) {
        for (int k = 0; k < ksize; k++) {
            int idx = base + k;
            if (idx < 0) idx = 0; else if (idx >= height) idx = height - 1;
            rows[k] = ring_row(buf, last_slot, last_y, ring_size, row_elems, idx);
        }
    } else if (out_row < radius) {
        for (int k = 0; k < ksize; k++) {
            int idx = base + k;
            if (idx < 0) idx = 0;
            rows[k] = ring_row(buf, last_slot, last_y, ring_size, row_elems, idx);
        }
    } else if (out_row >= height - radius) {
        for (int k = 0; k < ksize; k++) {
            int idx = base + k;
            if (idx >= height) idx = height - 1;
            rows[k] = ring_row(buf, last_slot, last_y, ring_size, row_elems, idx);
        }
    } else {
        for (int k = 0; k < ksize; k++)
            rows[k] = ring_row(buf, last_slot, last_y, ring_size, row_elems, base + k);
    }
}

static inline const iv8u* ring_row_u8(const iv8u* buf, int last_slot, int last_y, int ring_size, int row_elems, int src_row_idx)
{
    return buf + (size_t)((last_slot - (last_y - src_row_idx) + ring_size) % ring_size) * row_elems;
}

static inline void gather_rows_u8(const iv8u** rows, const iv8u* buf, int last_slot, int last_y,
                                  int ring_size, int row_elems, int out_row, int radius, int ksize, int height)
{
    int base = out_row - radius;
    if (height <= ksize) {
        for (int k = 0; k < ksize; k++) {
            int idx = base + k;
            if (idx < 0) idx = 0; else if (idx >= height) idx = height - 1;
            rows[k] = ring_row_u8(buf, last_slot, last_y, ring_size, row_elems, idx);
        }
    } else if (out_row < radius) {
        for (int k = 0; k < ksize; k++) {
            int idx = base + k;
            if (idx < 0) idx = 0;
            rows[k] = ring_row_u8(buf, last_slot, last_y, ring_size, row_elems, idx);
        }
    } else if (out_row >= height - radius) {
        for (int k = 0; k < ksize; k++) {
            int idx = base + k;
            if (idx >= height) idx = height - 1;
            rows[k] = ring_row_u8(buf, last_slot, last_y, ring_size, row_elems, idx);
        }
    } else {
        for (int k = 0; k < ksize; k++)
            rows[k] = ring_row_u8(buf, last_slot, last_y, ring_size, row_elems, base + k);
    }
}


static fiv_ret fiv_image_gaussian_blur_raw_f32(iv8u* dst, int width, int height, int stride,
                                               int channels, const iv8u* src, ivf32 sigma, int size)
{
    ivf32 coef[FIV_MAX_GAUSS_COEF_WIDTH];
    int ksize;
    fiv_ret ret = fiv_compute_gaussion_filter_coef(coef, &ksize, sigma, 1, size);
    if (ret != FIV_RET_OK) return ret;
    if (ksize <= 0 || ksize > FIV_MAX_GAUSS_COEF_WIDTH) return FIV_RET_ERR_NOT_SUPPORT;

    int ch_count  = channels;
    int row_elems = width * ch_count;
    int ring_size = ksize + 3;
    int radius    = (ksize - 1) / 2;

    ivf32* buf = (ivf32*)fiv_malloc((size_t)ring_size * row_elems * sizeof(ivf32));
    if (!buf) return FIV_RET_ERR_MEM;

    int write_pos = 0, last_y = -1, next_out = 0, last_slot = 0;
    for (int y = 0; y < height; y += 4) {
        for (int dy = 0; dy < 4; dy++) {
            int cur_y = y + dy;
            if (cur_y >= height) break;
            int slot = write_pos;
            const iv8u* src_row = src + (size_t)cur_y * stride;
            ivf32* buf_row = buf + (size_t)slot * row_elems;
            row_filter_f32(buf_row, (const ivf32*)src_row, width, ch_count, coef, ksize);
            write_pos = (slot + 1) % ring_size;
            last_y = cur_y;
        }
        last_slot = (write_pos - 1 + ring_size) % ring_size;
        int ready = last_y - radius;
        if (ready > height - 1) ready = height - 1;
        if (ready < 0) ready = -1;
        int avail = height - next_out;
        if (avail > 4) avail = 4;
        if (avail > ready - next_out + 1) avail = ready - next_out + 1;
        for (int r = 0; r < avail; r++) {
            int out_row = next_out + r;
            const ivf32* rows[FIV_MAX_GAUSS_COEF_WIDTH];
            gather_rows(rows, buf, last_slot, last_y, ring_size, row_elems, out_row, radius, ksize, height);
            iv8u* dst_row = dst + (size_t)out_row * stride;
            col_filter_f32((ivf32*)dst_row, rows, coef, ksize, row_elems);
        }
        next_out += avail;
    }
    last_slot = (write_pos - 1 + ring_size) % ring_size;
    while (next_out < height) {
        int out_row = next_out;
        const ivf32* rows[FIV_MAX_GAUSS_COEF_WIDTH];
        gather_rows(rows, buf, last_slot, last_y, ring_size, row_elems, out_row, radius, ksize, height);
        iv8u* dst_row = dst + (size_t)out_row * stride;
        col_filter_f32((ivf32*)dst_row, rows, coef, ksize, row_elems);
        next_out++;
    }
    fiv_free(buf);
    return FIV_RET_OK;
}


static fiv_ret fiv_image_gaussian_blur_raw_u8(iv8u* dst, int width, int height, int stride,
                                              int channels, const iv8u* src, ivf32 sigma, int size)
{
    ivf32 coef[FIV_MAX_GAUSS_COEF_WIDTH];
    iv16u coef_q8[FIV_MAX_GAUSS_COEF_WIDTH];
    int ksize, i;
    fiv_ret ret = fiv_compute_gaussion_filter_coef(coef, &ksize, sigma, 0, size);
    if (ret != FIV_RET_OK) return ret;
    if (ksize <= 0 || ksize > FIV_MAX_GAUSS_COEF_WIDTH) return FIV_RET_ERR_NOT_SUPPORT;
    for (i = 0; i < ksize; i++) coef_q8[i] = (iv16u)(coef[i] * 256.f + 0.5f);

    int ch_count  = channels;
    int row_elems = width * ch_count;
    int ring_size = ksize + 3;
    int radius    = (ksize - 1) / 2;

    iv8u* buf = (iv8u*)fiv_malloc((size_t)ring_size * row_elems * sizeof(iv8u));
    if (!buf) return FIV_RET_ERR_MEM;

    int write_pos = 0, last_y = -1, next_out = 0, last_slot = 0;
    for (int y = 0; y < height; y += 4) {
        for (int dy = 0; dy < 4; dy++) {
            int cur_y = y + dy;
            if (cur_y >= height) break;
            int slot = write_pos;
            const iv8u* src_row = src + (size_t)cur_y * stride;
            iv8u* buf_row = buf + (size_t)slot * row_elems;
            row_filter_u8_q8(buf_row, src_row, width, ch_count, coef_q8, ksize);
            write_pos = (slot + 1) % ring_size;
            last_y = cur_y;
        }
        last_slot = (write_pos - 1 + ring_size) % ring_size;
        int ready = last_y - radius;
        if (ready > height - 1) ready = height - 1;
        if (ready < 0) ready = -1;
        int avail = height - next_out;
        if (avail > 4) avail = 4;
        if (avail > ready - next_out + 1) avail = ready - next_out + 1;
        for (int r = 0; r < avail; r++) {
            int out_row = next_out + r;
            const iv8u* rows[FIV_MAX_GAUSS_COEF_WIDTH];
            gather_rows_u8(rows, buf, last_slot, last_y, ring_size, row_elems, out_row, radius, ksize, height);
            iv8u* dst_row = dst + (size_t)out_row * stride;
            col_filter_u8_q8(dst_row, rows, coef_q8, ksize, row_elems);
        }
        next_out += avail;
    }
    last_slot = (write_pos - 1 + ring_size) % ring_size;
    while (next_out < height) {
        int out_row = next_out;
        const iv8u* rows[FIV_MAX_GAUSS_COEF_WIDTH];
        gather_rows_u8(rows, buf, last_slot, last_y, ring_size, row_elems, out_row, radius, ksize, height);
        iv8u* dst_row = dst + (size_t)out_row * stride;
        col_filter_u8_q8(dst_row, rows, coef_q8, ksize, row_elems);
        next_out++;
    }
    fiv_free(buf);
    return FIV_RET_OK;
}


static fiv_ret fiv_image_gaussian_blur_raw(iv8u* dst, int width, int height, int stride,
                                           int channels, const iv8u* src, ivf32 sigma,
                                           int is_float, int size)
{
    if (is_float)
        return fiv_image_gaussian_blur_raw_f32(dst, width, height, stride, channels, src, sigma, size);
    return fiv_image_gaussian_blur_raw_u8(dst, width, height, stride, channels, src, sigma, size);
}


#if defined(FIV_HIGH_PRECISE_GAUSS_BLUR)
/* ===========================================================================
 * Precise 8U path: 16-bit intermediate, single round (OpenCV-aligned)
 *
 * The standard 8U path stores the row-filter result back into 8-bit and then
 * rounds a second time in the column filter (double round), which diverges
 * from OpenCV's 8U GaussianBlur by up to 3 on full-noise inputs.  This path
 * keeps the row-filter result in a 16-bit intermediate buffer (range ~[0,255],
 * already divided back by 256 with rounding) and rounds only ONCE at the very
 * end of the column filter -- matching OpenCV's single-round 8U separable
 * blur, so the max pixel difference vs OpenCV drops to <=1.
 *
 * The 16-bit intermediate ring buffer is exactly 2x the 8U path's size.
 * Row-filter SIMD: u8 in -> 16-bit accumulators -> store 16-bit (no pack to
 * 8U), step 16 i16 elements per 256-bit lane.  Column-filter SIMD: i16 in ->
 * 32-bit accumulators (cvtepu16_epi32 + mullo_epi32) -> round + packus to 8U.
 * =========================================================================== */

static inline const iv16u* ring_row_u16(const iv16u* buf, int last_slot, int last_y, int ring_size, int row_elems, int src_row_idx)
{
    return buf + (size_t)((last_slot - (last_y - src_row_idx) + ring_size) % ring_size) * row_elems;
}

static inline void gather_rows_u16(const iv16u** rows, const iv16u* buf, int last_slot, int last_y,
                                   int ring_size, int row_elems, int out_row, int radius, int ksize, int height)
{
    int base = out_row - radius;
    if (height <= ksize) {
        for (int k = 0; k < ksize; k++) {
            int idx = base + k;
            if (idx < 0) idx = 0; else if (idx >= height) idx = height - 1;
            rows[k] = ring_row_u16(buf, last_slot, last_y, ring_size, row_elems, idx);
        }
    } else if (out_row < radius) {
        for (int k = 0; k < ksize; k++) {
            int idx = base + k;
            if (idx < 0) idx = 0;
            rows[k] = ring_row_u16(buf, last_slot, last_y, ring_size, row_elems, idx);
        }
    } else if (out_row >= height - radius) {
        for (int k = 0; k < ksize; k++) {
            int idx = base + k;
            if (idx >= height) idx = height - 1;
            rows[k] = ring_row_u16(buf, last_slot, last_y, ring_size, row_elems, idx);
        }
    } else {
        for (int k = 0; k < ksize; k++)
            rows[k] = ring_row_u16(buf, last_slot, last_y, ring_size, row_elems, base + k);
    }
}

/* Row filter: 8U in, 16-bit intermediate out (round back to [0,255] scale,
 * NO final 8U pack -- the single rounding happens in the column filter). */
static void row_filter_u8_to_u16(iv16u* buf_row, const iv8u* src_row, int width, int ch_count, const iv16u* coef, int ksize)
{
    int radius = (ksize - 1) / 2;
    const int row_elems = width * ch_count;
    int col = 0;

    if (width <= ksize) {
        for (; col < width; col++) {
            for (int c = 0; c < ch_count; c++) {
                int acc = (int)coef[0] * src_row[(col - radius) * ch_count + c];
                for (int k = 1; k < ksize; k++) {
                    int px = col - radius + k;
                    if (px < 0) px = 0; else if (px >= width) px = width - 1;
                    acc += (int)coef[k] * src_row[px * ch_count + c];
                }
                buf_row[col * ch_count + c] = (iv16u)acc;
            }
        }
        return;
    }

    for (; col < radius; col++) {
        for (int c = 0; c < ch_count; c++) {
            int acc = (int)coef[0] * src_row[(col - radius) * ch_count + c];
            for (int k = 1; k < ksize; k++) {
                int px = col - radius + k;
                if (px < 0) px = 0;
                acc += (int)coef[k] * src_row[px * ch_count + c];
            }
                buf_row[col * ch_count + c] = (iv16u)acc;
        }
    }

    {
        int i = radius * ch_count;
        int i_end = (width - radius) * ch_count;

#if defined(FIV_USE_AVX2)
        __m256i zero_vec = _mm256_setzero_si256();
        /* 256-bit lane holds 16 u8 -> 16 i16 accumulators.  Store the
         * un-rounded Q7 fixed-point row sum directly (the single rounding
         * happens only in the column filter). */
        for (; i + 32 <= i_end; i += 32) {
            __m256i acc_lo = zero_vec, acc_hi = zero_vec;
            const iv8u* src_ptr = src_row + i;
            for (int k = 0; k < ksize; k++) {
                const iv8u* p = src_ptr + (k - radius) * ch_count;
                __m256i weight = _mm256_set1_epi16((short)coef[k]);
                __m256i pix_vec = _mm256_loadu_si256((const __m256i*)p);
                __m256i pix_hi = _mm256_unpackhi_epi8(pix_vec, zero_vec);
                __m256i pix_lo = _mm256_unpacklo_epi8(pix_vec, zero_vec);
                pix_lo = _mm256_mullo_epi16(pix_lo, weight);
                pix_hi = _mm256_mullo_epi16(pix_hi, weight);
                acc_lo = _mm256_adds_epu16(acc_lo, pix_lo);
                acc_hi = _mm256_adds_epu16(acc_hi, pix_hi);
            }
            /* acc_lo = [b0..b7, b16..b23], acc_hi = [b8..b15, b24..b31] across
             * the two 128-bit lanes; write in pixel order as four 128-bit
             * stores (do NOT packus to bytes -- that would merge pairs of
             * pixels into single 16-bit slots in the i16 buffer). */
            _mm_storeu_si128((__m128i*)(buf_row + i),       _mm256_castsi256_si128(acc_lo));
            _mm_storeu_si128((__m128i*)(buf_row + i + 8),   _mm256_castsi256_si128(acc_hi));
            _mm_storeu_si128((__m128i*)(buf_row + i + 16),  _mm256_extracti128_si256(acc_lo, 1));
            _mm_storeu_si128((__m128i*)(buf_row + i + 24),  _mm256_extracti128_si256(acc_hi, 1));
        }
#endif

#if defined(FIV_USE_AVX2) || defined(FIV_USE_AVX)
        __m128i zero_vec_sse = _mm_setzero_si128();
        for (; i + 16 <= i_end; i += 16) {
            __m128i acc_lo = zero_vec_sse, acc_hi = zero_vec_sse;
            const iv8u* src_ptr = src_row + i;
            for (int k = 0; k < ksize; k++) {
                const iv8u* p = src_ptr + (k - radius) * ch_count;
                __m128i weight = _mm_set1_epi16((short)coef[k]);
                __m128i pix_vec = _mm_loadu_si128((const __m128i*)p);
                __m128i pix_hi = _mm_unpackhi_epi8(pix_vec, zero_vec_sse);
                __m128i pix_lo = _mm_unpacklo_epi8(pix_vec, zero_vec_sse);
                pix_lo = _mm_mullo_epi16(pix_lo, weight);
                pix_hi = _mm_mullo_epi16(pix_hi, weight);
                acc_lo = _mm_adds_epu16(acc_lo, pix_lo);
                acc_hi = _mm_adds_epu16(acc_hi, pix_hi);
            }
            _mm_storeu_si128((__m128i*)(buf_row + i),     acc_lo);
            _mm_storeu_si128((__m128i*)(buf_row + i + 8), acc_hi);
        }
#endif

        for (; i < i_end; i++) {
            int acc = (int)coef[0] * src_row[i - radius * ch_count];
            for (int k = 1; k < ksize; k++)
                acc += (int)coef[k] * src_row[i + (k - radius) * ch_count];
            buf_row[i] = (iv16u)acc;
        }
    }

    for (col = width - radius; col < width; col++) {
        for (int c = 0; c < ch_count; c++) {
            int acc = (int)coef[0] * src_row[(col - radius) * ch_count + c];
            for (int k = 1; k < ksize; k++) {
                int px = col - radius + k;
                if (px >= width) px = width - 1;
                acc += (int)coef[k] * src_row[px * ch_count + c];
            }
                buf_row[col * ch_count + c] = (iv16u)acc;
        }
    }
}

/* Column filter: 16-bit intermediate in, 8U out.  Single round at the end. */
static void col_filter_u16_to_u8(iv8u* dst_row, const iv16u* row_ptrs[], const iv16u* coef, int ksize, int row_elems)
{
    int radius = (ksize - 1) / 2;
    const iv16u* coef_center = coef + radius;
    const iv16u** centered_rows = row_ptrs + radius;
    int i = 0, k;

#if defined(FIV_USE_AVX2)
    __m256i round128 = _mm256_set1_epi32(32768);
    for (; i + 16 <= row_elems; i += 16) {
        __m256i weight = _mm256_set1_epi32((int)coef_center[0]);
        __m256i src_vec = _mm256_loadu_si256((const __m256i*)(centered_rows[0] + i));
        __m256i src_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src_vec));
        __m256i src_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src_vec, 1));
        __m256i acc_lo = _mm256_mullo_epi32(src_lo, weight);
        __m256i acc_hi = _mm256_mullo_epi32(src_hi, weight);
        for (k = 1; k <= radius; k++) {
            __m256i pos_vec = _mm256_loadu_si256((const __m256i*)(centered_rows[k] + i));
            __m256i neg_vec = _mm256_loadu_si256((const __m256i*)(centered_rows[-k] + i));
            __m256i pos_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(pos_vec));
            __m256i pos_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(pos_vec, 1));
            __m256i neg_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(neg_vec));
            __m256i neg_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(neg_vec, 1));
            pos_lo = _mm256_add_epi32(pos_lo, neg_lo);
            pos_hi = _mm256_add_epi32(pos_hi, neg_hi);
            weight = _mm256_set1_epi32((int)coef_center[k]);
            pos_lo = _mm256_mullo_epi32(pos_lo, weight);
            pos_hi = _mm256_mullo_epi32(pos_hi, weight);
            acc_lo = _mm256_add_epi32(acc_lo, pos_lo);
            acc_hi = _mm256_add_epi32(acc_hi, pos_hi);
        }
        acc_lo = _mm256_add_epi32(acc_lo, round128);
        acc_hi = _mm256_add_epi32(acc_hi, round128);
        acc_lo = _mm256_srai_epi32(acc_lo, 16);
        acc_hi = _mm256_srai_epi32(acc_hi, 16);
        /* packs_epi32 interleaves to [lo0..3,hi0..3,lo4..7,hi4..7] (as 16 i16).
         * permute4x64 at 64-bit granularity reorders the 4 qwords
         * [lo0..3, hi0..3, lo4..7, hi4..7] -> [lo0..7, hi0..7] so each 128-bit
         * half then packs cleanly to 8 u8 in source order. */
        __m256i packed16 = _mm256_packs_epi32(acc_lo, acc_hi);
        packed16 = _mm256_permute4x64_epi64(packed16, _MM_SHUFFLE(3, 1, 2, 0));
        __m128i lo16 = _mm256_castsi256_si128(packed16);
        __m128i hi16 = _mm256_extracti128_si256(packed16, 1);
        __m128i out8 = _mm_packus_epi16(lo16, hi16);
        _mm_storeu_si128((__m128i*)(dst_row + i), out8);
    }
#endif

#if defined(FIV_USE_AVX2) || defined(FIV_USE_AVX)
    __m128i round128_sse = _mm_set1_epi32(32768);
    for (; i + 8 <= row_elems; i += 8) {
        __m128i weight = _mm_set1_epi32((int)coef_center[0]);
        __m128i src_vec = _mm_loadu_si128((const __m128i*)(centered_rows[0] + i));
        __m128i src_lo = _mm_cvtepu16_epi32(src_vec);
        __m128i src_hi = _mm_cvtepu16_epi32(_mm_srli_si128(src_vec, 8));
        __m128i acc_lo = _mm_mullo_epi32(src_lo, weight);
        __m128i acc_hi = _mm_mullo_epi32(src_hi, weight);
        for (k = 1; k <= radius; k++) {
            __m128i pos_vec = _mm_loadu_si128((const __m128i*)(centered_rows[k] + i));
            __m128i neg_vec = _mm_loadu_si128((const __m128i*)(centered_rows[-k] + i));
            __m128i pos_lo = _mm_cvtepu16_epi32(pos_vec);
            __m128i pos_hi = _mm_cvtepu16_epi32(_mm_srli_si128(pos_vec, 8));
            __m128i neg_lo = _mm_cvtepu16_epi32(neg_vec);
            __m128i neg_hi = _mm_cvtepu16_epi32(_mm_srli_si128(neg_vec, 8));
            pos_lo = _mm_add_epi32(pos_lo, neg_lo);
            pos_hi = _mm_add_epi32(pos_hi, neg_hi);
            weight = _mm_set1_epi32((int)coef_center[k]);
            pos_lo = _mm_mullo_epi32(pos_lo, weight);
            pos_hi = _mm_mullo_epi32(pos_hi, weight);
            acc_lo = _mm_add_epi32(acc_lo, pos_lo);
            acc_hi = _mm_add_epi32(acc_hi, pos_hi);
        }
        acc_lo = _mm_add_epi32(acc_lo, round128_sse);
        acc_hi = _mm_add_epi32(acc_hi, round128_sse);
        acc_lo = _mm_srai_epi32(acc_lo, 16);
        acc_hi = _mm_srai_epi32(acc_hi, 16);
        __m128i packed16 = _mm_packs_epi32(acc_lo, acc_hi);
        __m128i packed8 = _mm_packus_epi16(packed16, packed16);
        _mm_storel_epi64((__m128i*)(dst_row + i), packed8);
    }
    for (; i + 4 <= row_elems; i += 4) {
        __m128i weight = _mm_set1_epi32((int)coef_center[0]);
        __m128i s0 = _mm_loadu_si64((const void*)(centered_rows[0] + i));
        __m128i acc = _mm_cvtepu16_epi32(s0);   /* 4 i16 -> 4 i32 */
        acc = _mm_mullo_epi32(acc, weight);
        for (k = 1; k <= radius; k++) {
            __m128i p = _mm_loadu_si64((const void*)(centered_rows[k] + i));
            __m128i n = _mm_loadu_si64((const void*)(centered_rows[-k] + i));
            __m128i pl = _mm_cvtepu16_epi32(p);
            __m128i nl = _mm_cvtepu16_epi32(n);
            pl = _mm_add_epi32(pl, nl);
            weight = _mm_set1_epi32((int)coef_center[k]);
            pl = _mm_mullo_epi32(pl, weight);
            acc = _mm_add_epi32(acc, pl);
        }
        acc = _mm_add_epi32(acc, round128_sse);
        acc = _mm_srai_epi32(acc, 16);
        __m128i packed16 = _mm_packs_epi32(acc, acc);
        __m128i packed8 = _mm_packus_epi16(packed16, packed16);
        *(int*)(dst_row + i) = _mm_cvtsi128_si32(packed8);  /* store 4 bytes */
    }
#endif

    for (; i < row_elems; i++) {
        int acc = (int)coef_center[0] * centered_rows[0][i];
        for (k = 1; k <= radius; k++)
            acc += (int)coef_center[k] * (centered_rows[k][i] + centered_rows[-k][i]);
        dst_row[i] = (iv8u)((acc + 32768) >> 16);
    }
}

static fiv_ret fiv_image_gaussian_blur_raw_u8_precise(iv8u* dst, int width, int height, int stride,
                                                     int channels, const iv8u* src, ivf32 sigma, int size)
{
    ivf32 coef[FIV_MAX_GAUSS_COEF_WIDTH];
    iv16u coef_q8[FIV_MAX_GAUSS_COEF_WIDTH];
    int ksize, i;
    fiv_ret ret = fiv_compute_gaussion_filter_coef(coef, &ksize, sigma, 0, size);
    if (ret != FIV_RET_OK) return ret;
    if (ksize <= 0 || ksize > FIV_MAX_GAUSS_COEF_WIDTH) return FIV_RET_ERR_NOT_SUPPORT;
    for (i = 0; i < ksize; i++) coef_q8[i] = (iv16u)(coef[i] * 256.f + 0.5f);
    int radius    = (ksize - 1) / 2;
    /* Renormalize the Q8 coefficients so they sum to EXACTLY 256.  The
     * separable fixed-point product scales by (sum)^2; if sum != 256 the
     * result drifts (e.g. sum=255 -> *0.9922 -> ~2 darker at value 255).
     * Absorb the rounding residual into the center coefficient. */
    {
        int sum_q8 = 0;
        for (i = 0; i < ksize; i++) sum_q8 += coef_q8[i];
        coef_q8[radius] += (iv16u)(256 - sum_q8);
    }

    int ch_count  = channels;
    int row_elems = width * ch_count;
    int ring_size = ksize + 3;

    /* Un-rounded 16-bit (unsigned) intermediate ring buffer: 2x the 8U
       path's byte size.  Must be UNSIGNED -- the un-rounded Q8 row sum can
       reach ~65280, which overflows a signed 16-bit. */
    iv16u* buf = (iv16u*)fiv_malloc((size_t)ring_size * row_elems * sizeof(iv16u));
    if (!buf) return FIV_RET_ERR_MEM;

    int write_pos = 0, last_y = -1, next_out = 0, last_slot = 0;
    for (int y = 0; y < height; y += 4) {
        for (int dy = 0; dy < 4; dy++) {
            int cur_y = y + dy;
            if (cur_y >= height) break;
            int slot = write_pos;
            const iv8u* src_row = src + (size_t)cur_y * stride;
            iv16u* buf_row = buf + (size_t)slot * row_elems;
            row_filter_u8_to_u16(buf_row, src_row, width, ch_count, coef_q8, ksize);
            write_pos = (slot + 1) % ring_size;
            last_y = cur_y;
        }
        last_slot = (write_pos - 1 + ring_size) % ring_size;
        int ready = last_y - radius;
        if (ready > height - 1) ready = height - 1;
        if (ready < 0) ready = -1;
        int avail = height - next_out;
        if (avail > 4) avail = 4;
        if (avail > ready - next_out + 1) avail = ready - next_out + 1;
        for (int r = 0; r < avail; r++) {
            int out_row = next_out + r;
            const iv16u* rows[FIV_MAX_GAUSS_COEF_WIDTH];
            gather_rows_u16(rows, buf, last_slot, last_y, ring_size, row_elems, out_row, radius, ksize, height);
            iv8u* dst_row = dst + (size_t)out_row * stride;
            col_filter_u16_to_u8(dst_row, rows, coef_q8, ksize, row_elems);
        }
        next_out += avail;
    }
    last_slot = (write_pos - 1 + ring_size) % ring_size;
    while (next_out < height) {
        int out_row = next_out;
        const iv16u* rows[FIV_MAX_GAUSS_COEF_WIDTH];
        gather_rows_u16(rows, buf, last_slot, last_y, ring_size, row_elems, out_row, radius, ksize, height);
        iv8u* dst_row = dst + (size_t)out_row * stride;
        col_filter_u16_to_u8(dst_row, rows, coef_q8, ksize, row_elems);
        next_out++;
    }
    fiv_free(buf);
    return FIV_RET_OK;
}
#endif

fiv_ret fiv_image_gaussian_blur_precise(fiv_mat* dst, fiv_mat* src, ivf32 sigma, int size)
{
    if (!dst || !src) return FIV_RET_ERR_PARA;
    if (dst->dtype != src->dtype) return FIV_RET_ERR_PARA;
    if (dst->width != src->width || dst->height != src->height) return FIV_RET_ERR_PARA;
    if (dst->strides[0] != src->strides[0]) return FIV_RET_ERR_PARA;
    if (sigma < 0.1f) return FIV_RET_ERR_PARA;

    int channels = (int)((iv32u)src->dtype) / 16 + 1;
    if (channels != 1 && channels != 3 && channels != 4) return FIV_RET_ERR_NOT_SUPPORT;

    int is_float = (((iv32u)src->dtype) % 16 == 8) ? 1 : 0;
    int width    = (int)src->width;
    int height   = (int)src->height;
    int stride   = (int)src->strides[0];

    /* 32F is already single-round (float-exact); only the 8U path benefits
       from the 16-bit intermediate. Route 32F to the standard path. */
    if (is_float)
        return fiv_image_gaussian_blur_raw_f32(dst->data.ptr8u, width, height, stride, channels, src->data.ptr8u, sigma, size);
#if defined(FIV_HIGH_PRECISE_GAUSS_BLUR)
    return fiv_image_gaussian_blur_raw_u8_precise(dst->data.ptr8u, width, height, stride, channels, src->data.ptr8u, sigma, size);
#else
    return fiv_image_gaussian_blur_raw_u8(dst->data.ptr8u, width, height, stride, channels, src->data.ptr8u, sigma, size);
#endif
}


fiv_ret fiv_image_gaussian_blur(fiv_mat* dst, fiv_mat* src, ivf32 sigma, int size)
{
    if (!dst || !src) return FIV_RET_ERR_PARA;
    if (dst->dtype != src->dtype) return FIV_RET_ERR_PARA;
    if (dst->width != src->width || dst->height != src->height) return FIV_RET_ERR_PARA;
    if (dst->strides[0] != src->strides[0]) return FIV_RET_ERR_PARA;
    if (sigma < 0.1f) return FIV_RET_ERR_PARA;

    int channels = (int)((iv32u)src->dtype) / 16 + 1;
    if (channels != 1 && channels != 3 && channels != 4) return FIV_RET_ERR_NOT_SUPPORT;

    int is_float = (((iv32u)src->dtype) % 16 == 8) ? 1 : 0;
    int width    = (int)src->width;
    int height   = (int)src->height;
    int stride   = (int)src->strides[0];

    return fiv_image_gaussian_blur_raw(dst->data.ptr8u, width, height, stride, channels,
                                       src->data.ptr8u, sigma, is_float, size);
}

fiv_ret fiv_image_filter(fiv_mat* dst, fiv_mat* src, fiv_image_filter_type filter_type, void* filter_params)
{
    if (filter_type == FIV_GAUSSION_BLUR) {
        if (!filter_params) return FIV_RET_ERR_PARA;
        fiv_gaussion_blur_params* params = (fiv_gaussion_blur_params*)filter_params;
        return fiv_image_gaussian_blur(dst, src, params->sigma, 0);
    }
    return FIV_RET_ERR_NOT_SUPPORT;
}
