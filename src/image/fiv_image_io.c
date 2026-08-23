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

#include "fiv_image_io.h"
#include "fiv_ctensor.h"
#include "fiv_common.h"

#define STBI_MALLOC(sz)        fiv_malloc(sz)
#define STBI_FREE(p)           fiv_free(p)
#define STBI_REALLOC(p, newsz) fiv_realloc(p, newsz)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


static int fiv_image_io_ext_is(const char* file_path, const char* ext) {
    size_t path_len = 0, ext_len = 0;
    size_t scan_index, cmp_index;
    while (file_path[path_len]) path_len++;
    while (ext[ext_len])        ext_len++;
    if (path_len < ext_len) return 0;
    scan_index = path_len - ext_len;
    for (cmp_index = 0; cmp_index < ext_len; cmp_index++) {
        char path_ch = file_path[scan_index + cmp_index];
        char ext_ch  = ext[cmp_index];
        if (path_ch >= 'A' && path_ch <= 'Z') path_ch = (char)(path_ch + 32);
        if (ext_ch  >= 'A' && ext_ch  <= 'Z') ext_ch  = (char)(ext_ch + 32);
        if (path_ch != ext_ch) return 0;
    }
    return 1;
}


fiv_mat* fiv_create_image_from_file(char* file_name, int color_flag) {
    int            desired_channels;
    fiv_data_type  dtype;
    iv8u           color_space;
    int            width, height, channels;
    stbi_uc*       buf;
    size_t         shape[2];
    fiv_tensor2d*  tensor;

    if (file_name == NULL) return NULL;

    switch (color_flag) {
    case FIV_GRAY8_CS:  desired_channels = 1; dtype = FIV_8U1; color_space = FIV_GRAY8_CS;  break;
    case FIV_RGB24_CS:  desired_channels = 3; dtype = FIV_8U3; color_space = FIV_RGB24_CS;  break;
    case FIV_BGR24_CS:  desired_channels = 3; dtype = FIV_8U3; color_space = FIV_BGR24_CS;  break;
    case FIV_RGBA32_CS: desired_channels = 4; dtype = FIV_8U4; color_space = FIV_RGBA32_CS; break;
    case FIV_BGRA32_CS: desired_channels = 4; dtype = FIV_8U4; color_space = FIV_BGRA32_CS; break;
    default:            desired_channels = 3; dtype = FIV_8U3; color_space = FIV_RGB24_CS;  break;
    }

    buf = stbi_load(file_name, &width, &height, &channels, desired_channels);
    if (buf == NULL) return NULL;

    shape[0] = (size_t)height;
    shape[1] = (size_t)width;
    tensor = fiv_create_tensor2d_header(shape, dtype);
    if (tensor == NULL) {
        stbi_image_free(buf);
        return NULL;
    }
    if (fiv_tensor2d_set_data(tensor, buf, (size_t)width * height * (size_t)desired_channels) != FIV_RET_OK) {
        stbi_image_free(buf);
        fiv_release_tensor2d(&tensor);
        return NULL;
    }
    tensor->reference        = 1;
    tensor->color_space_type = color_space;

    return (fiv_mat*)tensor;
}


fiv_ret fiv_image_write(char* file_name, fiv_mat* image) {
    int         width;
    int         height;
    int         channel_count;
    int         src_stride;
    iv8u*       pixel_data;
    iv8u*       out_data;
    int         out_stride;
    iv8u*       tight_buffer;
    int         r, c;
    size_t      row_bytes;
    fiv_ret     result;

    if (file_name == NULL || image == NULL) return FIV_RET_ERR_PARA;

    switch (image->dtype) {
    case FIV_8U1: channel_count = 1; break;
    case FIV_8U3: channel_count = 3; break;
    case FIV_8U4: channel_count = 4; break;
    default:      return FIV_RET_ERR_PARA;
    }

    height       = (int)image->height;
    width        = (int)image->width;
    src_stride   = (int)image->strides[0];
    pixel_data   = image->data.ptr8u;

    out_data    = pixel_data;
    out_stride  = src_stride;
    tight_buffer = NULL;
    row_bytes   = (size_t)width * channel_count;
    if (src_stride != (int)row_bytes) {
        tight_buffer = (iv8u*)fiv_malloc((size_t)height * row_bytes);
        if (tight_buffer == NULL) return FIV_RET_ERR_MEM;
        for (r = 0; r < height; r++) {
            const iv8u* src_row = pixel_data  + (size_t)r * src_stride;
            iv8u*       dst_row = tight_buffer + (size_t)r * row_bytes;
            for (c = 0; c < width; c++) {
                size_t channel_index;
                size_t off = (size_t)c * channel_count;
                for (channel_index = 0; channel_index < (size_t)channel_count; channel_index++)
                    dst_row[off + channel_index] = src_row[off + channel_index];
            }
        }
        out_data   = tight_buffer;
        out_stride = (int)row_bytes;
    }

    result = FIV_RET_OK;
    if (fiv_image_io_ext_is(file_name, ".png")) {
        if (!stbi_write_png(file_name, width, height, channel_count, out_data, out_stride))
            result = FIV_RET_ERR_OPEN_FILE;
    } else if (fiv_image_io_ext_is(file_name, ".jpg") ||
               fiv_image_io_ext_is(file_name, ".jpeg")) {
        if (!stbi_write_jpg(file_name, width, height, channel_count, out_data, 90))
            result = FIV_RET_ERR_OPEN_FILE;
    } else if (fiv_image_io_ext_is(file_name, ".bmp")) {
        if (!stbi_write_bmp(file_name, width, height, channel_count, out_data))
            result = FIV_RET_ERR_OPEN_FILE;
    } else if (fiv_image_io_ext_is(file_name, ".tga")) {
        if (!stbi_write_tga(file_name, width, height, channel_count, out_data))
            result = FIV_RET_ERR_OPEN_FILE;
    } else {
        if (!stbi_write_png(file_name, width, height, channel_count, out_data, out_stride))
            result = FIV_RET_ERR_OPEN_FILE;
    }

    if (tight_buffer != NULL) fiv_free(tight_buffer);
    return result;
}
