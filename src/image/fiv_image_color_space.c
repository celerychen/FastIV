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

#include "fiv_image_color_space.h"


/* Y = (wr*R + wg*G + wb*B) >> 8, exact integer contract; sum = 256. */
#define FIV_CS_GRAY_WR  77
#define FIV_CS_GRAY_WG  150
#define FIV_CS_GRAY_WB  29


static fiv_ret fiv_cs_check_common(const fiv_mat* image_dst, const fiv_mat* image_src) {
    if (image_dst == NULL || image_src == NULL) return FIV_RET_ERR_PARA;
    if (image_dst->height != image_src->height ||
        image_dst->width  != image_src->width)  return FIV_RET_ERR_PARA;
    if (image_dst->data.ptr8u == NULL)          return FIV_RET_ERR_DATA_UNINITED;
    if (image_src->data.ptr8u == NULL)          return FIV_RET_ERR_DATA_UNINITED;
    return FIV_RET_OK;
}


static fiv_ret fiv_cs_swap_rb(fiv_mat* image_dst, const fiv_mat* image_src) {
    int    height        = (int)image_src->height;
    int    width         = (int)image_src->width;
    int    src_stride    = (int)image_src->strides[0];
    int    dst_stride    = (int)image_dst->strides[0];
    iv8u*  src_data      = image_src->data.ptr8u;
    iv8u*  dst_data      = image_dst->data.ptr8u;
    int    row_index;
    int    col_index;

    for (row_index = 0; row_index < height; row_index++) {
        const iv8u* src_row = src_data + (size_t)row_index * src_stride;
        iv8u*       dst_row = dst_data + (size_t)row_index * dst_stride;
        for (col_index = 0; col_index < width; col_index++) {
            size_t      off    = (size_t)col_index * 3;
            const iv8u  red    = src_row[off + 0];
            const iv8u  blue   = src_row[off + 2];
            dst_row[off + 0]   = blue;
            dst_row[off + 1]   = src_row[off + 1];
            dst_row[off + 2]   = red;
        }
    }
    return FIV_RET_OK;
}


static void fiv_cs_to_gray_scalar_core(iv8u* dst_row, const iv8u* src_row,
                                       int width, int red_index, int green_index,
                                       int blue_index) {
    int col_index;
    for (col_index = 0; col_index < width; col_index++) {
        size_t off   = (size_t)col_index * 3;
        int    red   = src_row[off + red_index];
        int    green = src_row[off + green_index];
        int    blue  = src_row[off + blue_index];
        int    gray  = (FIV_CS_GRAY_WR * red + FIV_CS_GRAY_WG * green +
                        FIV_CS_GRAY_WB * blue) >> 8;
        dst_row[col_index] = (iv8u)(gray & 0xff);
    }
}


fiv_ret fiv_cs_to_gray_scalar(fiv_mat* image_dst, const fiv_mat* image_src,
                              int red_index, int green_index, int blue_index) {
    int   height     = (int)image_src->height;
    int   src_stride = (int)image_src->strides[0];
    int   dst_stride = (int)image_dst->strides[0];
    iv8u* src_data   = image_src->data.ptr8u;
    iv8u* dst_data   = image_dst->data.ptr8u;
    int   row_index;

    for (row_index = 0; row_index < height; row_index++) {
        const iv8u* src_row = src_data + (size_t)row_index * src_stride;
        iv8u*       dst_row = dst_data + (size_t)row_index * dst_stride;
        fiv_cs_to_gray_scalar_core(dst_row, src_row, (int)image_src->width,
                                   red_index, green_index, blue_index);
    }
    return FIV_RET_OK;
}


/* x86 SIMD deinterleave + tail store adapted from komrad36/RGB2Y.h; weights
   replaced by the exact (wr*R+wg*G+wb*B)>>8 to match the scalar core. */
#if defined(FIV_USE_AVX2) || defined(FIV_USE_X86_SIMD)

#define FIV_CS_BMASK _mm256_setr_epi8( \
    0, 3, 6, -1, -1, -1, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, \
   -1, -1, -1,  1,  4,  7, -1, -1,  9, 12, -1, -1, -1, -1, -1, -1)
#define FIV_CS_GMASK _mm256_setr_epi8( \
    1, 4, 7, -1, -1,  9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, \
   -1, -1, -1,  2,  5, -1, -1, -1, 10, 13, -1, -1, -1, -1, -1, -1)
#define FIV_CS_RMASK _mm256_setr_epi8( \
    2, 5, -1, -1, -1, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, \
   -1, -1,  0,  3,  6, -1, -1,  8, 11, 14, -1, -1, -1, -1, -1, -1)

#define FIV_CS_WB  _mm256_set1_epi16((iv16s)FIV_CS_GRAY_WB)
#define FIV_CS_WG  _mm256_set1_epi16((iv16s)FIV_CS_GRAY_WG)
#define FIV_CS_WR  _mm256_set1_epi16((iv16s)FIV_CS_GRAY_WR)
#define FIV_CS_WB_SWAPPED _mm256_set1_epi16((iv16s)FIV_CS_GRAY_WR)
#define FIV_CS_WR_SWAPPED _mm256_set1_epi16((iv16s)FIV_CS_GRAY_WB)

static void fiv_cs_process_10px(const iv8u* pt, int cols_minus_j,
                                iv8u* out, int last_col, int swap_rb) {
    const __m256i w_r = swap_rb ? FIV_CS_WR_SWAPPED : FIV_CS_WR;
    const __m256i w_b = swap_rb ? FIV_CS_WB_SWAPPED : FIV_CS_WB;
    __m256i in1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(pt)));
    __m256i in2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(pt + 15)));
    __m256i raw = _mm256_packus_epi16(in1, in2);

    __m256i b1 = _mm256_shuffle_epi8(raw, FIV_CS_BMASK);
    __m256i g1 = _mm256_shuffle_epi8(raw, FIV_CS_GMASK);
    __m256i r1 = _mm256_shuffle_epi8(raw, FIV_CS_RMASK);

    /* Widen each 128-bit half to 16-bit, weighted multiply-add, >>8, pack. */
    __m256i b_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b1));
    __m256i b_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b1, 1));
    __m256i g_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g1));
    __m256i g_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(g1, 1));
    __m256i r_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r1));
    __m256i r_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r1, 1));

    __m256i s_lo = _mm256_srli_epi16(_mm256_add_epi16(
                       _mm256_add_epi16(_mm256_mullo_epi16(r_lo, w_r),
                                        _mm256_mullo_epi16(g_lo, FIV_CS_WG)),
                       _mm256_mullo_epi16(b_lo, w_b)), 8);
    __m256i s_hi = _mm256_srli_epi16(_mm256_add_epi16(
                       _mm256_add_epi16(_mm256_mullo_epi16(r_hi, w_r),
                                        _mm256_mullo_epi16(g_hi, FIV_CS_WG)),
                       _mm256_mullo_epi16(b_hi, w_b)), 8);

    __m256i gray = _mm256_packus_epi16(s_lo, s_hi);
    __m128i h3   = _mm_adds_epu8(_mm256_castsi256_si128(gray),
                                 _mm256_extracti128_si256(gray, 1));

    if (last_col) {
        switch (cols_minus_j) {
        case 15: out[14] = (iv8u)_mm_extract_epi8(h3, 14);
        case 14: out[13] = (iv8u)_mm_extract_epi8(h3, 13);
        case 13: out[12] = (iv8u)_mm_extract_epi8(h3, 12);
        case 12: out[11] = (iv8u)_mm_extract_epi8(h3, 11);
        case 11: out[10] = (iv8u)_mm_extract_epi8(h3, 10);
        case 10: out[9]  = (iv8u)_mm_extract_epi8(h3, 9);
        case 9:  out[8]  = (iv8u)_mm_extract_epi8(h3, 8);
        case 8:  out[7]  = (iv8u)_mm_extract_epi8(h3, 7);
        case 7:  out[6]  = (iv8u)_mm_extract_epi8(h3, 6);
        case 6:  out[5]  = (iv8u)_mm_extract_epi8(h3, 5);
        case 5:  out[4]  = (iv8u)_mm_extract_epi8(h3, 4);
        case 4:  out[3]  = (iv8u)_mm_extract_epi8(h3, 3);
        case 3:  out[2]  = (iv8u)_mm_extract_epi8(h3, 2);
        case 2:  out[1]  = (iv8u)_mm_extract_epi8(h3, 1);
        case 1:  out[0]  = (iv8u)_mm_extract_epi8(h3, 0);
        }
    } else {
        _mm_storeu_si128((__m128i*)out, h3);
    }
}

#endif /* FIV_USE_AVX2 || FIV_USE_X86_SIMD */


#ifdef FIV_USE_ARM_NEON

/* 16-bit multiply lanes (vmull_u8/vmlal_u8), >>8 folded into vaddhn_u16. */
static inline void fiv_cs_gray_16px_neon(iv8u* dst, const iv8u* src, int swap_rb) {
    uint8x16x3_t  vec      = vld3q_u8(src);
    uint8x16_t    red_vec   = vec.val[0];
    uint8x16_t    green_vec = vec.val[1];
    uint8x16_t    blue_vec  = vec.val[2];
    uint8x16_t    weight_r  = vdupq_n_u8((iv8u)(swap_rb ? FIV_CS_GRAY_WB : FIV_CS_GRAY_WR));
    uint8x16_t    weight_g  = vdupq_n_u8((iv8u)FIV_CS_GRAY_WG);
    uint8x16_t    weight_b  = vdupq_n_u8((iv8u)(swap_rb ? FIV_CS_GRAY_WR : FIV_CS_GRAY_WB));

    uint16x8_t acc_low = vmull_u8(vget_low_u8(red_vec),   vget_low_u8(weight_r));
    acc_low = vmlal_u8(acc_low, vget_low_u8(green_vec), vget_low_u8(weight_g));
    uint16x8_t bw_low  = vmull_u8(vget_low_u8(blue_vec),  vget_low_u8(weight_b));
    uint8x8_t  y_low   = vaddhn_u16(acc_low, bw_low);

    uint16x8_t acc_high = vmull_u8(vget_high_u8(red_vec),   vget_high_u8(weight_r));
    acc_high = vmlal_u8(acc_high, vget_high_u8(green_vec), vget_high_u8(weight_g));
    uint16x8_t bw_high  = vmull_u8(vget_high_u8(blue_vec),  vget_high_u8(weight_b));
    uint8x8_t  y_high   = vaddhn_u16(acc_high, bw_high);

    vst1q_u8(dst, vcombine_u8(y_low, y_high));
}

static inline void fiv_cs_gray_48px_neon(iv8u* dst, const iv8u* src, int swap_rb) {
    fiv_cs_gray_16px_neon(dst + 0,  src + 0  * 3, swap_rb);
    fiv_cs_gray_16px_neon(dst + 16, src + 16 * 3, swap_rb);
    fiv_cs_gray_16px_neon(dst + 32, src + 32 * 3, swap_rb);
}

#endif /* FIV_USE_ARM_NEON */


static fiv_ret fiv_cs_to_gray_simd(fiv_mat* image_dst, const fiv_mat* image_src,
                                   int red_index, int green_index, int blue_index) {
    int swap_rb = (red_index == 2) ? 1 : 0;
#if defined(FIV_USE_AVX2) || defined(FIV_USE_X86_SIMD)
    int    height     = (int)image_src->height;
    int    width      = (int)image_src->width;
    int    src_stride = (int)image_src->strides[0];
    int    dst_stride = (int)image_dst->strides[0];
    iv8u*  src_data   = image_src->data.ptr8u;
    iv8u*  dst_data   = image_dst->data.ptr8u;
    int    row_index;

    for (row_index = 0; row_index < height; row_index++) {
        const iv8u* src_row = src_data + (size_t)row_index * src_stride;
        iv8u*       dst_row = dst_data + (size_t)row_index * dst_stride;
        int         j = 0;
        for (; j + 10 <= width; j += 10) {
            fiv_cs_process_10px(src_row + j * 3, width - j, dst_row + j, 0, swap_rb);
        }
        if (j < width) {
            fiv_cs_process_10px(src_row + j * 3, width - j, dst_row + j, 1, swap_rb);
        }
    }
    return FIV_RET_OK;
#elif defined(FIV_USE_ARM_NEON)
    int    height     = (int)image_src->height;
    int    width      = (int)image_src->width;
    int    src_stride = (int)image_src->strides[0];
    int    dst_stride = (int)image_dst->strides[0];
    iv8u*  src_data   = image_src->data.ptr8u;
    iv8u*  dst_data   = image_dst->data.ptr8u;
    int    row_index;

    for (row_index = 0; row_index < height; row_index++) {
        const iv8u* src_row = src_data + (size_t)row_index * src_stride;
        iv8u*       dst_row = dst_data + (size_t)row_index * dst_stride;
        int         col_index = 0;
        for (; col_index + 48 <= width; col_index += 48) {
            fiv_cs_gray_48px_neon(dst_row + col_index, src_row + col_index * 3, swap_rb);
        }
        for (; col_index + 16 <= width; col_index += 16) {
            fiv_cs_gray_16px_neon(dst_row + col_index, src_row + col_index * 3, swap_rb);
        }
        if (col_index < width) {
            fiv_cs_to_gray_scalar_core(dst_row + col_index, src_row + col_index * 3,
                                       width - col_index, red_index, green_index, blue_index);
        }
    }
    return FIV_RET_OK;
#else
    return fiv_cs_to_gray_scalar(image_dst, image_src, red_index, green_index, blue_index);
#endif
}


fiv_ret fiv_image_color_space_convertor(fiv_mat* image_dst, fiv_mat* image_src,
                                        fiv_cs_convertor_type type) {
    fiv_ret result = fiv_cs_check_common(image_dst, image_src);
    if (result != FIV_RET_OK) return result;

    switch (type) {
    case FIV_CS_RGB2BGR:
    case FIV_CS_BGR2RGB:
        if (image_src->dtype != FIV_8U3 || image_dst->dtype != FIV_8U3)
            return FIV_RET_ERR_PARA;
        fiv_cs_swap_rb(image_dst, image_src);
        image_dst->color_space_type = (type == FIV_CS_RGB2BGR) ? FIV_BGR24_CS : FIV_RGB24_CS;
        break;

    case FIV_CS_RGB2GRAY:
        if (image_src->dtype != FIV_8U3 || image_dst->dtype != FIV_8U1)
            return FIV_RET_ERR_PARA;
        fiv_cs_to_gray_simd(image_dst, image_src, 0, 1, 2);
        image_dst->color_space_type = FIV_GRAY8_CS;
        break;

    case FIV_CS_BGR2GRAY:
        if (image_src->dtype != FIV_8U3 || image_dst->dtype != FIV_8U1)
            return FIV_RET_ERR_PARA;
        fiv_cs_to_gray_simd(image_dst, image_src, 2, 1, 0);
        image_dst->color_space_type = FIV_GRAY8_CS;
        break;

    default:
        return FIV_RET_ERR_NOT_SUPPORT;
    }

    return FIV_RET_OK;
}
