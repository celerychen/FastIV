/*
 * FastIV - Fast image and vision
 * Copyright (C) 2026 Celery Chen
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * See LICENSE file in project root for full license text.
 *
 * Image resize: nearest-neighbor (precomputed integer index table) and
 * bilinear (Q11 fixed-point with a precomputed horizontal-fraction table),
 * supporting 8-bit single-channel (FIV_8U1) and three-channel interleaved
 * (FIV_8U3) images. dst must already be sized to the target dimensions.
 * The structure mirrors fastiv0's src/image/fiv_image_resizer.c.
 */

#include <math.h>

#include "fiv_image_resizer.h"
#include "fiv_common.h"
#include "fiv_data_typedefs.h"

#define LOC_LINEAR_Q (11)

#ifndef FIV_MIN
#define FIV_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif


static fiv_ret fiv_resizer_nn_raw(iv8u* dst, int dst_width, int dst_height, int dst_stride,
                                  iv8u* src, int src_width, int src_height, int src_stride,
                                  int channels);
static fiv_ret fiv_resizer_bilinear_one_channel(iv8u* dst, int dst_width, int dst_height, int stride_dst,
                                                iv8u* src, int src_width, int src_height, int stride_src);
static fiv_ret fiv_resizer_bilinear_three_channel(iv8u* dst, int dst_width, int dst_height, int stride_dst,
                                                  iv8u* src, int src_width, int src_height, int stride_src);


static fiv_ret fiv_resizer_check_common(fiv_mat* dst, fiv_mat* src)
{
    if (dst == NULL || src == NULL)                       return FIV_RET_ERR_PARA;
    if (dst->data.ptr8u == NULL || src->data.ptr8u == NULL)
        return FIV_RET_ERR_DATA_UNINITED;
    if (dst->dtype != src->dtype)                         return FIV_RET_ERR_PARA;
    if (dst->dtype != FIV_8U1 && dst->dtype != FIV_8U3)   return FIV_RET_ERR_PARA;
    if (dst->width == 0 || dst->height == 0)              return FIV_RET_ERR_PARA;
    if (src->width == 0 || src->height == 0)              return FIV_RET_ERR_PARA;
    return FIV_RET_OK;
}


/* ---- nearest-neighbor: precompute the integer source-x table once --------- */
static fiv_ret fiv_resizer_nn_raw(iv8u* dst, int dst_width, int dst_height, int dst_stride,
                                  iv8u* src, int src_width, int src_height, int src_stride,
                                  int channels)
{
    ivf32  fx = (ivf32)src_width / dst_width;
    ivf32  fy = (ivf32)src_height / dst_height;
    iv32s* tab;
    int    x, y;

    tab = (iv32s*)fiv_malloc(sizeof(iv32s) * dst_width * channels);
    if (tab == NULL) {
        return FIV_RET_ERR_MEM;
    }

    for (x = 0; x < dst_width; x++) {
        int index_x = (int)(x * fx);
        tab[x] = FIV_MIN(index_x, src_width - 1) * channels;
    }

    if (channels == 1) {
        for (y = 0; y < dst_height; y++) {
            iv8u* ptr_dst = dst + y * dst_stride;
            iv8u* ptr_src;
            int index_y = FIV_MIN((int)(y * fy), src_height - 1);
            ptr_src = src + src_stride * index_y;
            for (x = 0; x <= dst_width - 4; x += 4) {
                ptr_dst[x + 0] = ptr_src[tab[x + 0]];
                ptr_dst[x + 1] = ptr_src[tab[x + 1]];
                ptr_dst[x + 2] = ptr_src[tab[x + 2]];
                ptr_dst[x + 3] = ptr_src[tab[x + 3]];
            }
            for (; x < dst_width; x++) {
                ptr_dst[x] = ptr_src[tab[x]];
            }
        }
    } else {
        for (y = 0; y < dst_height; y++) {
            iv8u* ptr_dst = dst + y * dst_stride;
            iv8u* ptr_src;
            int index_y = FIV_MIN((int)(y * fy), src_height - 1);
            ptr_src = src + src_stride * index_y;
            for (x = 0; x <= dst_width - 2; x += 2) {
                iv8u* t1 = ptr_src + tab[x];
                iv8u* t2 = ptr_src + tab[x + 1];
                ptr_dst[0] = t1[0];
                ptr_dst[1] = t1[1];
                ptr_dst[2] = t1[2];
                ptr_dst[3] = t2[0];
                ptr_dst[4] = t2[1];
                ptr_dst[5] = t2[2];
                ptr_dst += 6;
            }
            for (; x < dst_width; x++) {
                iv8u* t = ptr_src + tab[x];
                ptr_dst[0] = t[0];
                ptr_dst[1] = t[1];
                ptr_dst[2] = t[2];
                ptr_dst += 3;
            }
        }
    }

    fiv_free(tab);
    return FIV_RET_OK;
}


/* ---- Q11 fixed-point bilinear ---------------------------------------------- */
static fiv_ret fiv_resizer_bilinear_one_channel(iv8u* dst, int dst_width, int dst_height, int stride_dst,
                                                iv8u* src, int src_width, int src_height, int stride_src)
{
    ivf32  fy = (ivf32)src_height / dst_height;
    ivf32  fx = (ivf32)src_width / dst_width;
    iv16s* tab;
    int    im, jm, i, j;

    tab = (iv16s*)fiv_malloc(sizeof(iv16s) * dst_width * 2);
    if (tab == NULL) {
        return FIV_RET_ERR_MEM;
    }

    for (j = 0; j < dst_width; j++) {
        ivf32 x_frac = (j + 0.5f) * fx - 0.5f;
        jm = (int)floorf(x_frac);
        x_frac = x_frac - jm;
        if (jm < 0) {
            jm = 0;
            x_frac = 0;
        }
        if (jm >= src_width - 1) {
            x_frac = 0;
            jm = src_width - 1;
        }
        tab[2 * j + 0] = (iv16s)(x_frac * (1 << LOC_LINEAR_Q));
        tab[2 * j + 1] = (iv16s)jm;
    }

    for (i = 0; i < dst_height; i++) {
        ivf32 y_frac = (i + 0.5f) * fy - 0.5f;
        iv8u* src_line;
        iv8u* src_line2;
        iv8u* dst_line;
        iv16s y_frac_fixed;
        int   im_1;

        im = (int)floorf(y_frac);
        y_frac = y_frac - im;
        if (im < 0) {
            im = 0;
            y_frac = 0;
        }
        if (im >= src_height - 1) {
            y_frac = 0;
            im = src_height - 1;
        }
        y_frac_fixed = (iv16s)(y_frac * (1 << LOC_LINEAR_Q));
        src_line = &src[im * stride_src];
        im_1 = im + 1;
        im_1 = (im_1 > src_height - 1) ? src_height - 1 : im_1;
        src_line2 = &src[im_1 * stride_src];
        dst_line = &dst[i * stride_dst];

        for (j = 0; j < dst_width; j++) {
            int  x_frac = tab[2 * j + 0];
            int  index  = tab[2 * j + 1];
            iv8u* ptr_src_t1 = &src_line[index];
            iv8u* ptr_src_t2 = &src_line2[index];
            int t1 = x_frac * (ptr_src_t1[1] - ptr_src_t1[0]) + (ptr_src_t1[0] << LOC_LINEAR_Q);
            int t2 = x_frac * (ptr_src_t2[1] - ptr_src_t2[0]) + (ptr_src_t2[0] << LOC_LINEAR_Q);

            t1 += 1 << (LOC_LINEAR_Q - 1);
            t2 += 1 << (LOC_LINEAR_Q - 1);
            t1 >>= LOC_LINEAR_Q;
            t2 >>= LOC_LINEAR_Q;

            t2          = y_frac_fixed * (t2 - t1) + (t1 << LOC_LINEAR_Q);
            dst_line[j] = (iv8u)((t2 + (1 << (LOC_LINEAR_Q - 1))) >> LOC_LINEAR_Q);
        }
    }

    fiv_free(tab);
    return FIV_RET_OK;
}


static fiv_ret fiv_resizer_bilinear_three_channel(iv8u* dst, int dst_width, int dst_height, int stride_dst,
                                                  iv8u* src, int src_width, int src_height, int stride_src)
{
    fiv_ret ret = FIV_RET_OK;
    ivf32   fy = (ivf32)src_height / dst_height;
    ivf32   fx = (ivf32)src_width / dst_width;
    iv16s*  tab;
    int     i, j;

    tab = (iv16s*)fiv_malloc(sizeof(iv16s) * dst_width * 2);
    if (tab == NULL) {
        return FIV_RET_ERR_MEM;
    }

    for (j = 0; j < dst_width; j++) {
        ivf32 x_frac = (j + 0.5f) * fx - 0.5f;
        int   jm = (int)floorf(x_frac);
        x_frac = x_frac - jm;
        if (jm < 0) {
            jm = 0;
            x_frac = 0;
        }
        if (jm >= src_width - 1) {
            x_frac = 0;
            jm = src_width - 1;
        }
        tab[2 * j + 0] = (iv16s)(x_frac * (1 << LOC_LINEAR_Q));
        tab[2 * j + 1] = (iv16s)jm;
    }

    for (i = 0; i < dst_height; i++) {
        ivf32 y_frac = (i + 0.5f) * fy - 0.5f;
        iv8u* src_line1;
        iv8u* src_line2;
        iv8u* dst_line;
        iv16s y_frac_fixed;
        int   im, im_1;

        im = (int)floorf(y_frac);
        y_frac = y_frac - im;
        if (im < 0) {
            im = 0;
            y_frac = 0;
        }
        if (im >= src_height - 1) {
            y_frac = 0;
            im = src_height - 1;
        }
        y_frac_fixed = (iv16s)(y_frac * (1 << LOC_LINEAR_Q));

        src_line1 = &src[im * stride_src];
        im_1 = im + 1;
        im_1 = (im_1 > src_height - 1) ? src_height - 1 : im_1;
        src_line2 = &src[im_1 * stride_src];
        dst_line = &dst[i * stride_dst];

        for (j = 0; j < dst_width; j++) {
            int  x_frac = tab[2 * j + 0];
            int  index  = tab[2 * j + 1];
            index += index << 1;
            iv8u* ptr_src_t1 = &src_line1[index];
            iv8u* ptr_src_t2 = &src_line2[index];

            int t1 = x_frac * (ptr_src_t1[3] - ptr_src_t1[0]) + (ptr_src_t1[0] << LOC_LINEAR_Q);
            int t2 = x_frac * (ptr_src_t2[3] - ptr_src_t2[0]) + (ptr_src_t2[0] << LOC_LINEAR_Q);
            int t3 = x_frac * (ptr_src_t1[4] - ptr_src_t1[1]) + (ptr_src_t1[1] << LOC_LINEAR_Q);
            int t4 = x_frac * (ptr_src_t2[4] - ptr_src_t2[1]) + (ptr_src_t2[1] << LOC_LINEAR_Q);
            int t5 = x_frac * (ptr_src_t1[5] - ptr_src_t1[2]) + (ptr_src_t1[2] << LOC_LINEAR_Q);
            int t6 = x_frac * (ptr_src_t2[5] - ptr_src_t2[2]) + (ptr_src_t2[2] << LOC_LINEAR_Q);

            t1 += 1 << (LOC_LINEAR_Q - 1);
            t2 += 1 << (LOC_LINEAR_Q - 1);
            t1 >>= LOC_LINEAR_Q;
            t2 >>= LOC_LINEAR_Q;
            t2 = y_frac_fixed * (t2 - t1) + (t1 << LOC_LINEAR_Q);
            t2 += 1 << (LOC_LINEAR_Q - 1);
            t2 >>= LOC_LINEAR_Q;

            t3 += 1 << (LOC_LINEAR_Q - 1);
            t4 += 1 << (LOC_LINEAR_Q - 1);
            t3 >>= LOC_LINEAR_Q;
            t4 >>= LOC_LINEAR_Q;
            t4 = y_frac_fixed * (t4 - t3) + (t3 << LOC_LINEAR_Q);
            t4 += 1 << (LOC_LINEAR_Q - 1);
            t4 >>= LOC_LINEAR_Q;

            t5 += 1 << (LOC_LINEAR_Q - 1);
            t6 += 1 << (LOC_LINEAR_Q - 1);
            t5 >>= LOC_LINEAR_Q;
            t6 >>= LOC_LINEAR_Q;
            t6 = y_frac_fixed * (t6 - t5) + (t5 << LOC_LINEAR_Q);
            t6 += 1 << (LOC_LINEAR_Q - 1);
            t6 >>= LOC_LINEAR_Q;

            dst_line[3 * j + 0] = (iv8u)t2;
            dst_line[3 * j + 1] = (iv8u)t4;
            dst_line[3 * j + 2] = (iv8u)t6;
        }
    }

    fiv_free(tab);
    return ret;
}


/* ---- public entry points -------------------------------------------------- */
fiv_ret fiv_image_resize_nn(fiv_mat* dst, fiv_mat* src)
{
    fiv_ret ret = fiv_resizer_check_common(dst, src);
    if (ret != FIV_RET_OK) return ret;
    return fiv_resizer_nn_raw(dst->data.ptr8u, (int)dst->width, (int)dst->height, (int)dst->strides[0],
                              src->data.ptr8u, (int)src->width, (int)src->height, (int)src->strides[0],
                              (dst->dtype == FIV_8U3) ? 3 : 1);
}


fiv_ret fiv_image_resize_bilinear(fiv_mat* dst, fiv_mat* src)
{
    fiv_ret ret = fiv_resizer_check_common(dst, src);
    if (ret != FIV_RET_OK) return ret;
    if (dst->dtype == FIV_8U1)
        return fiv_resizer_bilinear_one_channel(dst->data.ptr8u, (int)dst->width, (int)dst->height, (int)dst->strides[0],
                                                src->data.ptr8u, (int)src->width, (int)src->height, (int)src->strides[0]);
    return fiv_resizer_bilinear_three_channel(dst->data.ptr8u, (int)dst->width, (int)dst->height, (int)dst->strides[0],
                                              src->data.ptr8u, (int)src->width, (int)src->height, (int)src->strides[0]);
}


/* ---- public dispatcher (declared in fiv_image.h) -------------------------- */
fiv_ret fiv_image_resize(fiv_mat* dst, fiv_mat* src, fiv_resizer_type type)
{
    switch (type) {
    case FIV_NN_RESIZER:     return fiv_image_resize_nn(dst, src);
    case FIV_BILEAR_RESIZER: return fiv_image_resize_bilinear(dst, src);
    default:                 return FIV_RET_ERR_NOT_SUPPORT;
    }
}