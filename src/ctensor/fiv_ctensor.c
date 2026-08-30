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



#include "fiv_ctensor.h"
#include "fiv_common.h"
#include "fiv_binary_op.h"
#include <stdint.h>
#include <string.h>

// dtype -> bytes: static lookup table.
// Enum is organized into 6 groups (multiplier x1/x2/x3/x4/x8/x16), each with
// 12 defined types + 4 FIV_UNDEF placeholders, 16 enum values per group
// (indices 0..91). UNDEF entries are 0; a 0 element_bytes means return NULL.
// Defined entry bytes = base width (x1 group) * group multiplier:
//   8U/8S=1, 16U/16S=2, 32U/32S=4, 64U/64S=8, 32F=4, 64F=8, 16F/16BF=2
static const iv8u fiv_dtype_size_table[] = {
    // x1  (indices 0..15)
    1, 1, 2, 2, 4, 4, 8, 8, 4, 8, 2, 2, 0, 0, 0, 0,
    // x2  (indices 16..31)
    2, 2, 4, 4, 8, 8, 16, 16, 8, 16, 4, 4, 0, 0, 0, 0,
    // x3  (indices 32..47)
    3, 3, 6, 6, 12, 12, 24, 24, 12, 24, 6, 6, 0, 0, 0, 0,
    // x4  (indices 48..63)
    4, 4, 8, 8, 16, 16, 32, 32, 16, 32, 8, 8, 0, 0, 0, 0,
    // x8  (indices 64..79)
    8, 8, 16, 16, 32, 32, 64, 64, 32, 64, 16, 16, 0, 0, 0, 0,
    // x16 (indices 80..91)
    16, 16, 32, 32, 64, 64, 128, 128, 64, 128, 32, 32,
};




fiv_tensor1d* fiv_create_tensor1d(size_t size, fiv_data_type data_type) {
	size_t        total_bytes;
	iv8u          element_bytes;
	fiv_tensor1d* t;

	if(data_type < FIV_8U1 || data_type > FIV_16BF16) {
		return NULL;
	}
	element_bytes = fiv_dtype_size_table[(int)data_type];
	if (element_bytes == 0) return NULL;
	if (size != 0 && element_bytes > SIZE_MAX / size) return NULL; // overflow (size 0 ok: empty tensor)
	total_bytes   = (size == 0) ? 0 : element_bytes * size;

	t = (fiv_tensor1d*)fiv_malloc(sizeof(fiv_tensor1d));
	if (t == NULL) return NULL;

	t->id             = FIV_ID_TENSOR1D;
	t->dtype          = data_type;
	t->shapes[0]      = size;
	t->data_continue  = 1;
	t->reference      = 1;
	t->element_bytes  = element_bytes;
	t->meta_info[3]   = 0;
	t->total_bytes    = total_bytes;
	t->strides[0]     = element_bytes;

	t->data.ptr = (size == 0) ? NULL : fiv_malloc(total_bytes);
	if (size != 0 && t->data.ptr == NULL) {
		fiv_free(t);
		return NULL;
	}
    return t;
}



fiv_ret fiv_release_tensor1d(fiv_tensor1d** pp) {
    if (!pp || !*pp) return FIV_RET_ERR_PARA;
    fiv_tensor1d* t = *pp;
    if (t->reference && t->data.ptr) fiv_free(t->data.ptr);
    fiv_free(t);
    *pp = NULL;
    return FIV_RET_OK;
}


fiv_tensor2d* fiv_create_tensor2d(size_t size[2], fiv_data_type data_type) {
	size_t        total_bytes;
	iv8u          element_bytes;
	fiv_tensor2d* t;

	if(size == NULL || size[0] == 0 || size[1] == 0 ||
	    data_type < FIV_8U1 || data_type > FIV_16BF16) {
		return NULL;
	}
	element_bytes = fiv_dtype_size_table[(int)data_type];
	if (element_bytes == 0) return NULL;
	total_bytes = element_bytes * size[0] * size[1];
	t = (fiv_tensor2d*)fiv_malloc(sizeof(fiv_tensor2d));
	if (t == NULL) return NULL;

	t->id             = FIV_ID_TENSOR2D;
	t->dtype          = data_type;
	t->shapes[0]      = size[0];                 // rows / height
	t->shapes[1]      = size[1];                 // cols / width
	t->data_continue  = 1;
	t->reference      = 1;
	t->element_bytes  = element_bytes;
	t->meta_info[3]   = 0;
	t->total_bytes    = total_bytes;
	t->strides[1]     = element_bytes;
	t->strides[0]     = t->strides[1] * size[1];

	t->data.ptr = fiv_malloc(total_bytes);
	if(t->data.ptr == NULL){
		fiv_free(t);
		return NULL;
	}
    return t;
}


fiv_ret fiv_release_tensor2d(fiv_tensor2d** pp) {
    if (!pp || !*pp) return FIV_RET_ERR_PARA;
    fiv_tensor2d* t = *pp;
    if (t->reference && t->data.ptr) fiv_free(t->data.ptr);
    fiv_free(t);
    *pp = NULL;
    return FIV_RET_OK;
}



// ---------------------------------------------------------------------------
// 3D / 4D / 5D implementations
// Convention: row-major, contiguous data, strides in bytes,
//   strides[dim-1] == element_bytes, strides[i] = strides[i+1] * shapes[i+1]
// ---------------------------------------------------------------------------

fiv_tensor3d* fiv_create_tensor3d(size_t size[3], fiv_data_type data_type)
{
    size_t        total_bytes;
    iv8u          element_bytes;
    fiv_tensor3d* t;

    if (size == NULL || size[0] == 0 || size[1] == 0 || size[2] == 0 ||
        data_type < FIV_8U1 || data_type > FIV_16BF16) {
        return NULL;
    }
    element_bytes = fiv_dtype_size_table[(int)data_type];
    if (element_bytes == 0) return NULL;
    total_bytes   = element_bytes * size[0] * size[1] * size[2];

    t = (fiv_tensor3d*)fiv_malloc(sizeof(fiv_tensor3d));
    if (t == NULL) return NULL;

    t->id            = FIV_ID_TENSOR3D;
    t->dtype         = data_type;

    t->shapes[0]     = size[0];                      // depth  / channels
    t->shapes[1]     = size[1];                      // rows   / height
    t->shapes[2]     = size[2];                      // cols   / width

    t->reference     = 1;                            // data owned by this struct, freed on release
    t->data_continue = 1;
    t->element_bytes = element_bytes;
    t->meta_info[3]  = 0;                            // clear reserved bytes

    t->strides[2]    = element_bytes;
    t->strides[1]    = t->strides[2] * size[2];
    t->strides[0]    = t->strides[1] * size[1];
    t->total_bytes   = total_bytes;

    t->data.ptr = fiv_malloc(total_bytes);
    if (t->data.ptr == NULL) {
        fiv_free(t);
        return NULL;
    }
    return t;
}


fiv_ret fiv_release_tensor3d(fiv_tensor3d** pp)
{
    if (!pp || !*pp) return FIV_RET_ERR_PARA;
    fiv_tensor3d* t = *pp;
    if (t->reference && t->data.ptr) fiv_free(t->data.ptr);
    fiv_free(t);
    *pp = NULL;
    return FIV_RET_OK;
}


fiv_tensor4d* fiv_create_tensor4d(size_t size[4], fiv_data_type data_type)
{
    size_t        total_bytes;
    iv8u          element_bytes;
    fiv_tensor4d* t;

    if (size == NULL || size[0] == 0 || size[1] == 0 || size[2] == 0 || size[3] == 0 ||
        data_type < FIV_8U1 || data_type > FIV_16BF16) {
        return NULL;
    }
    element_bytes = fiv_dtype_size_table[(int)data_type];
    if (element_bytes == 0) return NULL;
    total_bytes   = element_bytes * size[0] * size[1] * size[2] * size[3];

    t = (fiv_tensor4d*)fiv_malloc(sizeof(fiv_tensor4d));
    if (t == NULL) return NULL;

    t->id            = FIV_ID_TENSOR4D;
    t->dtype         = data_type;

    t->shapes[0]     = size[0];                      // batch  / cubes
    t->shapes[1]     = size[1];                      // depth  / channels
    t->shapes[2]     = size[2];                      // rows   / height
    t->shapes[3]     = size[3];                      // cols   / width

    t->reference     = 1;
    t->data_continue = 1;
    t->element_bytes = element_bytes;
    t->meta_info[3]  = 0;

    t->strides[3]    = element_bytes;
    t->strides[2]    = t->strides[3] * size[3];
    t->strides[1]    = t->strides[2] * size[2];
    t->strides[0]    = t->strides[1] * size[1];
    t->total_bytes   = total_bytes;

    t->data.ptr = fiv_malloc(total_bytes);
    if (t->data.ptr == NULL) {
        fiv_free(t);
        return NULL;
    }
    return t;
}


fiv_ret fiv_release_tensor4d(fiv_tensor4d** pp)
{
    if (!pp || !*pp) return FIV_RET_ERR_PARA;
    fiv_tensor4d* t = *pp;
    if (t->reference && t->data.ptr) fiv_free(t->data.ptr);
    fiv_free(t);
    *pp = NULL;
    return FIV_RET_OK;
}


fiv_tensor5d* fiv_create_tensor5d(size_t size[5], fiv_data_type data_type)
{
    size_t        total_bytes;
    iv8u          element_bytes;
    fiv_tensor5d* t;

    if (size == NULL || size[0] == 0 || size[1] == 0 || size[2] == 0 ||
        size[3] == 0 || size[4] == 0 ||
        data_type < FIV_8U1 || data_type > FIV_16BF16) {
        return NULL;
    }
    element_bytes = fiv_dtype_size_table[(int)data_type];
    if (element_bytes == 0) return NULL;
    total_bytes   = element_bytes * size[0] * size[1] * size[2] * size[3] * size[4];

    t = (fiv_tensor5d*)fiv_malloc(sizeof(fiv_tensor5d));
    if (t == NULL) return NULL;

    t->id            = FIV_ID_TENSOR5D;
    t->dtype         = data_type;

    t->shapes[0]     = size[0];                      // batch  / blocks
    t->shapes[1]     = size[1];                      // times  / cubes
    t->shapes[2]     = size[2];                      // depth  / channels
    t->shapes[3]     = size[3];                      // rows   / height
    t->shapes[4]     = size[4];                      // cols   / width

    t->reference     = 1;
    t->data_continue = 1;
    t->element_bytes = element_bytes;
    t->meta_info[3]  = 0;

    t->strides[4]    = element_bytes;
    t->strides[3]    = t->strides[4] * size[4];
    t->strides[2]    = t->strides[3] * size[3];
    t->strides[1]    = t->strides[2] * size[2];
    t->strides[0]    = t->strides[1] * size[1];
    t->total_bytes   = total_bytes;

    t->data.ptr = fiv_malloc(total_bytes);
    if (t->data.ptr == NULL) {
        fiv_free(t);
        return NULL;
    }
    return t;
}


fiv_ret fiv_release_tensor5d(fiv_tensor5d** pp)
{
    if (!pp || !*pp) return FIV_RET_ERR_PARA;
    fiv_tensor5d* t = *pp;
    if (t->reference && t->data.ptr) fiv_free(t->data.ptr);
    fiv_free(t);
    *pp = NULL;
    return FIV_RET_OK;
}



// ---------------------------------------------------------------------------
// set interfaces: bind an existing tensor struct to an external data buffer,
// no memory allocation anywhere.
//   - set_header : fill metadata only (id/dtype/shapes/strides/total_bytes/
//                  element_bytes); does not own the data.
//   - set_data   : attach the external data pointer; require the external
//                  buffer size >= total_bytes, otherwise return parameter error.
// Since the struct does not own the data, a later release only frees the
// struct itself and never touches the external buffer.
// ---------------------------------------------------------------------------

fiv_ret fiv_tensor1d_set_header(fiv_tensor1d* t, size_t size, fiv_data_type data_type)
{
    iv8u element_bytes;
    if (t == NULL) return FIV_RET_ERR_PARA;
    if (data_type < FIV_8U1 || data_type > FIV_16BF16) return FIV_RET_ERR_PARA;
    element_bytes = fiv_dtype_size_table[(int)data_type];
    if (element_bytes == 0) return FIV_RET_ERR_PARA;
    if (size == 0 || element_bytes > SIZE_MAX / size) return FIV_RET_ERR_PARA;  // empty dim / overflow

    t->id             = FIV_ID_TENSOR1D;
    t->dtype          = data_type;
    t->shapes[0]      = size;
    t->data_continue  = 1;
    t->reference      = 0;            // does not own data
    t->element_bytes  = element_bytes;
    t->meta_info[3]   = 0;
    t->total_bytes    = (size_t)element_bytes * size;
    t->strides[0]     = element_bytes;
    t->data.ptr       = NULL;
    return FIV_RET_OK;
}


fiv_ret fiv_tensor1d_set_data(fiv_tensor1d* t, void* data, size_t size)
{
    if (t == NULL || data == NULL) return FIV_RET_ERR_PARA;
    if (t->total_bytes == 0 || size < t->total_bytes) return FIV_RET_ERR_PARA;  // external buffer too small
    t->data.ptr   = data;
    t->reference  = 0;            // external data, not freed on release
    return FIV_RET_OK;
}


fiv_ret fiv_tensor2d_set_header(fiv_tensor2d* t, size_t size[2], fiv_data_type data_type)
{
    iv8u element_bytes;
    if (t == NULL || size == NULL) return FIV_RET_ERR_PARA;
    if (size[0] == 0 || size[1] == 0) return FIV_RET_ERR_PARA;
    if (data_type < FIV_8U1 || data_type > FIV_16BF16) return FIV_RET_ERR_PARA;
    element_bytes = fiv_dtype_size_table[(int)data_type];
    if (element_bytes == 0) return FIV_RET_ERR_PARA;

    t->id             = FIV_ID_TENSOR2D;
    t->dtype          = data_type;
    t->shapes[0]      = size[0];
    t->shapes[1]      = size[1];
    t->data_continue  = 1;
    t->reference      = 0;
    t->element_bytes  = element_bytes;
    t->meta_info[3]   = 0;
    t->total_bytes    = (size_t)element_bytes * size[0] * size[1];
    t->strides[1]     = element_bytes;
    t->strides[0]     = t->strides[1] * size[1];
    t->data.ptr       = NULL;
    return FIV_RET_OK;
}


fiv_ret fiv_tensor2d_set_data(fiv_tensor2d* t, void* data, size_t size)
{
    if (t == NULL || data == NULL) return FIV_RET_ERR_PARA;
    if (t->total_bytes == 0 || size < t->total_bytes) return FIV_RET_ERR_PARA;
    t->data.ptr   = data;
    t->reference  = 0;
    return FIV_RET_OK;
}


fiv_ret fiv_tensor3d_set_header(fiv_tensor3d* t, size_t size[3], fiv_data_type data_type)
{
    iv8u element_bytes;
    if (t == NULL || size == NULL) return FIV_RET_ERR_PARA;
    if (size[0] == 0 || size[1] == 0 || size[2] == 0) return FIV_RET_ERR_PARA;
    if (data_type < FIV_8U1 || data_type > FIV_16BF16) return FIV_RET_ERR_PARA;
    element_bytes = fiv_dtype_size_table[(int)data_type];
    if (element_bytes == 0) return FIV_RET_ERR_PARA;

    t->id            = FIV_ID_TENSOR3D;
    t->dtype         = data_type;
    t->shapes[0]     = size[0];
    t->shapes[1]     = size[1];
    t->shapes[2]     = size[2];
    t->reference     = 0;
    t->data_continue = 1;
    t->element_bytes = element_bytes;
    t->meta_info[3]  = 0;
    t->strides[2]    = element_bytes;
    t->strides[1]    = t->strides[2] * size[2];
    t->strides[0]    = t->strides[1] * size[1];
    t->total_bytes   = (size_t)element_bytes * size[0] * size[1] * size[2];
    t->data.ptr      = NULL;
    return FIV_RET_OK;
}


fiv_ret fiv_tensor3d_set_data(fiv_tensor3d* t, void* data, size_t size)
{
    if (t == NULL || data == NULL) return FIV_RET_ERR_PARA;
    if (t->total_bytes == 0 || size < t->total_bytes) return FIV_RET_ERR_PARA;
    t->data.ptr   = data;
    t->reference  = 0;
    return FIV_RET_OK;
}


fiv_ret fiv_tensor4d_set_header(fiv_tensor4d* t, size_t size[4], fiv_data_type data_type)
{
    iv8u element_bytes;
    if (t == NULL || size == NULL) return FIV_RET_ERR_PARA;
    if (size[0] == 0 || size[1] == 0 || size[2] == 0 || size[3] == 0) return FIV_RET_ERR_PARA;
    if (data_type < FIV_8U1 || data_type > FIV_16BF16) return FIV_RET_ERR_PARA;
    element_bytes = fiv_dtype_size_table[(int)data_type];
    if (element_bytes == 0) return FIV_RET_ERR_PARA;

    t->id            = FIV_ID_TENSOR4D;
    t->dtype         = data_type;
    t->shapes[0]     = size[0];
    t->shapes[1]     = size[1];
    t->shapes[2]     = size[2];
    t->shapes[3]     = size[3];
    t->reference     = 0;
    t->data_continue = 1;
    t->element_bytes = element_bytes;
    t->meta_info[3]  = 0;
    t->strides[3]    = element_bytes;
    t->strides[2]    = t->strides[3] * size[3];
    t->strides[1]    = t->strides[2] * size[2];
    t->strides[0]    = t->strides[1] * size[1];
    t->total_bytes   = (size_t)element_bytes * size[0] * size[1] * size[2] * size[3];
    t->data.ptr      = NULL;
    return FIV_RET_OK;
}


fiv_ret fiv_tensor4d_set_data(fiv_tensor4d* t, void* data, size_t size)
{
    if (t == NULL || data == NULL) return FIV_RET_ERR_PARA;
    if (t->total_bytes == 0 || size < t->total_bytes) return FIV_RET_ERR_PARA;
    t->data.ptr   = data;
    t->reference  = 0;
    return FIV_RET_OK;
}


fiv_ret fiv_tensor5d_set_header(fiv_tensor5d* t, size_t size[5], fiv_data_type data_type)
{
    iv8u element_bytes;
    if (t == NULL || size == NULL) return FIV_RET_ERR_PARA;
    if (size[0] == 0 || size[1] == 0 || size[2] == 0 || size[3] == 0 || size[4] == 0)
        return FIV_RET_ERR_PARA;
    if (data_type < FIV_8U1 || data_type > FIV_16BF16) return FIV_RET_ERR_PARA;
    element_bytes = fiv_dtype_size_table[(int)data_type];
    if (element_bytes == 0) return FIV_RET_ERR_PARA;

    t->id            = FIV_ID_TENSOR5D;
    t->dtype         = data_type;
    t->shapes[0]     = size[0];
    t->shapes[1]     = size[1];
    t->shapes[2]     = size[2];
    t->shapes[3]     = size[3];
    t->shapes[4]     = size[4];
    t->reference     = 0;
    t->data_continue = 1;
    t->element_bytes = element_bytes;
    t->meta_info[3]  = 0;
    t->strides[4]    = element_bytes;
    t->strides[3]    = t->strides[4] * size[4];
    t->strides[2]    = t->strides[3] * size[3];
    t->strides[1]    = t->strides[2] * size[2];
    t->strides[0]    = t->strides[1] * size[1];
    t->total_bytes   = (size_t)element_bytes * size[0] * size[1] * size[2] * size[3] * size[4];
    t->data.ptr      = NULL;
    return FIV_RET_OK;
}


fiv_ret fiv_tensor5d_set_data(fiv_tensor5d* t, void* data, size_t size)
{
    if (t == NULL || data == NULL) return FIV_RET_ERR_PARA;
    if (t->total_bytes == 0 || size < t->total_bytes) return FIV_RET_ERR_PARA;
    t->data.ptr   = data;
    t->reference  = 0;
    return FIV_RET_OK;
}


// ---------------------------------------------------------------------------
// create_header interfaces: allocate the struct and fill its metadata header
// only; do NOT allocate data.
//   - data.ptr is NULL, struct does not own the data.
//   - a later release only frees the struct itself, never the data.
// Typical usage: create_header -> set_data(external buffer) -> ... -> release
// ---------------------------------------------------------------------------

fiv_tensor1d* fiv_create_tensor1d_header(size_t size, fiv_data_type data_type)
{
    iv8u           element_bytes;
    fiv_tensor1d*  t;

    if (data_type < FIV_8U1 || data_type > FIV_16BF16) return NULL;
    element_bytes = fiv_dtype_size_table[(int)data_type];
    if (element_bytes == 0) return NULL;
    if (size == 0 || element_bytes > SIZE_MAX / size) return NULL;  // empty dim / overflow

    t = (fiv_tensor1d*)fiv_malloc(sizeof(fiv_tensor1d));
    if (t == NULL) return NULL;

    t->id             = FIV_ID_TENSOR1D;
    t->dtype          = data_type;
    t->shapes[0]      = size;
    t->data_continue  = 1;
    t->reference      = 0;            // does not own data, not freed on release
    t->element_bytes  = element_bytes;
    t->meta_info[3]   = 0;
    t->total_bytes    = (size_t)element_bytes * size;
    t->strides[0]     = element_bytes;
    t->data.ptr       = NULL;
    return t;
}


fiv_tensor2d* fiv_create_tensor2d_header(size_t size[2], fiv_data_type data_type)
{
    iv8u           element_bytes;
    fiv_tensor2d*  t;

    if (size == NULL || size[0] == 0 || size[1] == 0 ||
        data_type < FIV_8U1 || data_type > FIV_16BF16) return NULL;
    element_bytes = fiv_dtype_size_table[(int)data_type];
    if (element_bytes == 0) return NULL;

    t = (fiv_tensor2d*)fiv_malloc(sizeof(fiv_tensor2d));
    if (t == NULL) return NULL;

    t->id             = FIV_ID_TENSOR2D;
    t->dtype          = data_type;
    t->shapes[0]      = size[0];
    t->shapes[1]      = size[1];
    t->data_continue  = 1;
    t->reference      = 0;
    t->element_bytes  = element_bytes;
    t->meta_info[3]   = 0;
    t->total_bytes    = (size_t)element_bytes * size[0] * size[1];
    t->strides[1]     = element_bytes;
    t->strides[0]     = t->strides[1] * size[1];
    t->data.ptr       = NULL;
    return t;
}


fiv_tensor3d* fiv_create_tensor3d_header(size_t size[3], fiv_data_type data_type)
{
    iv8u           element_bytes;
    fiv_tensor3d*  t;

    if (size == NULL || size[0] == 0 || size[1] == 0 || size[2] == 0 ||
        data_type < FIV_8U1 || data_type > FIV_16BF16) return NULL;
    element_bytes = fiv_dtype_size_table[(int)data_type];
    if (element_bytes == 0) return NULL;

    t = (fiv_tensor3d*)fiv_malloc(sizeof(fiv_tensor3d));
    if (t == NULL) return NULL;

    t->id            = FIV_ID_TENSOR3D;
    t->dtype         = data_type;
    t->shapes[0]     = size[0];
    t->shapes[1]     = size[1];
    t->shapes[2]     = size[2];
    t->reference     = 0;
    t->data_continue = 1;
    t->element_bytes = element_bytes;
    t->meta_info[3]  = 0;
    t->strides[2]    = element_bytes;
    t->strides[1]    = t->strides[2] * size[2];
    t->strides[0]    = t->strides[1] * size[1];
    t->total_bytes   = (size_t)element_bytes * size[0] * size[1] * size[2];
    t->data.ptr      = NULL;
    return t;
}


fiv_tensor4d* fiv_create_tensor4d_header(size_t size[4], fiv_data_type data_type)
{
    iv8u           element_bytes;
    fiv_tensor4d*  t;

    if (size == NULL || size[0] == 0 || size[1] == 0 || size[2] == 0 || size[3] == 0 ||
        data_type < FIV_8U1 || data_type > FIV_16BF16) return NULL;
    element_bytes = fiv_dtype_size_table[(int)data_type];
    if (element_bytes == 0) return NULL;

    t = (fiv_tensor4d*)fiv_malloc(sizeof(fiv_tensor4d));
    if (t == NULL) return NULL;

    t->id            = FIV_ID_TENSOR4D;
    t->dtype         = data_type;
    t->shapes[0]     = size[0];
    t->shapes[1]     = size[1];
    t->shapes[2]     = size[2];
    t->shapes[3]     = size[3];
    t->reference     = 0;
    t->data_continue = 1;
    t->element_bytes = element_bytes;
    t->meta_info[3]  = 0;
    t->strides[3]    = element_bytes;
    t->strides[2]    = t->strides[3] * size[3];
    t->strides[1]    = t->strides[2] * size[2];
    t->strides[0]    = t->strides[1] * size[1];
    t->total_bytes   = (size_t)element_bytes * size[0] * size[1] * size[2] * size[3];
    t->data.ptr      = NULL;
    return t;
}


fiv_tensor5d* fiv_create_tensor5d_header(size_t size[5], fiv_data_type data_type)
{
    iv8u           element_bytes;
    fiv_tensor5d*  t;

    if (size == NULL || size[0] == 0 || size[1] == 0 || size[2] == 0 ||
        size[3] == 0 || size[4] == 0 ||
        data_type < FIV_8U1 || data_type > FIV_16BF16) return NULL;
    element_bytes = fiv_dtype_size_table[(int)data_type];
    if (element_bytes == 0) return NULL;

    t = (fiv_tensor5d*)fiv_malloc(sizeof(fiv_tensor5d));
    if (t == NULL) return NULL;

    t->id            = FIV_ID_TENSOR5D;
    t->dtype         = data_type;
    t->shapes[0]     = size[0];
    t->shapes[1]     = size[1];
    t->shapes[2]     = size[2];
    t->shapes[3]     = size[3];
    t->shapes[4]     = size[4];
    t->reference     = 0;
    t->data_continue = 1;
    t->element_bytes = element_bytes;
    t->meta_info[3]  = 0;
    t->strides[4]    = element_bytes;
    t->strides[3]    = t->strides[4] * size[4];
    t->strides[2]    = t->strides[3] * size[3];
    t->strides[1]    = t->strides[2] * size[2];
    t->strides[0]    = t->strides[1] * size[1];
    t->total_bytes   = (size_t)element_bytes * size[0] * size[1] * size[2] * size[3] * size[4];
    t->data.ptr      = NULL;
    return t;
}


// ---------------------------------------------------------------------------
// Zero-copy view: wrap a sub-region of an existing tensor as a new tensor.
// Inherits src's strides, offsets data.ptr to the sub-region, sets reference=0
// (releasing the view does NOT free src's buffer), and data_continue truthfully.
// offset[k]+size[k] must not exceed src's shape; src must already hold data.
// ---------------------------------------------------------------------------

fiv_ret fiv_tensor_view(void* dst, const void* src, const size_t offset[], const size_t size[])
{
    fiv_data_id id;

    if (dst == NULL || src == NULL || offset == NULL || size == NULL) return FIV_RET_ERR_PARA;
    id = *(const fiv_data_id*)src;
    if (id < FIV_ID_TENSOR1D || id > FIV_ID_TENSOR5D) return FIV_RET_ERR_PARA;

    switch (id) {
    case FIV_ID_TENSOR1D: {
        const fiv_tensor1d* s = (const fiv_tensor1d*)src;
        fiv_tensor1d*       d = (fiv_tensor1d*)dst;
        if (size[0] == 0 || offset[0] + size[0] > s->shapes[0]) return FIV_RET_ERR_PARA;
        if (s->data.ptr == NULL) return FIV_RET_ERR_PARA;
        memcpy(d, s, sizeof(fiv_tensor1d));
        d->shapes[0]     = size[0];
        d->reference     = 0;
        d->data_continue = (size[0] == s->shapes[0]) ? 1 : 0;
        d->total_bytes   = (size_t)d->element_bytes * size[0];
        d->data.ptr      = (iv8u*)d->data.ptr + offset[0] * d->strides[0];
        return FIV_RET_OK;
    }
    case FIV_ID_TENSOR2D: {
        const fiv_tensor2d* s = (const fiv_tensor2d*)src;
        fiv_tensor2d*       d = (fiv_tensor2d*)dst;
        if (size[0] == 0 || offset[0] + size[0] > s->shapes[0]) return FIV_RET_ERR_PARA;
        if (size[1] == 0 || offset[1] + size[1] > s->shapes[1]) return FIV_RET_ERR_PARA;
        if (s->data.ptr == NULL) return FIV_RET_ERR_PARA;
        memcpy(d, s, sizeof(fiv_tensor2d));
        d->shapes[0]     = size[0];
        d->shapes[1]     = size[1];
        d->reference     = 0;
        d->data_continue = (d->strides[1] == (size_t)d->element_bytes &&
                            d->strides[0] == (size_t)d->element_bytes * size[1]) ? 1 : 0;
        d->total_bytes   = (size_t)d->element_bytes * size[0] * size[1];
        d->data.ptr      = (iv8u*)d->data.ptr + offset[0] * d->strides[0] + offset[1] * d->strides[1];
        return FIV_RET_OK;
    }
    case FIV_ID_TENSOR3D: {
        const fiv_tensor3d* s = (const fiv_tensor3d*)src;
        fiv_tensor3d*       d = (fiv_tensor3d*)dst;
        if (size[0] == 0 || offset[0] + size[0] > s->shapes[0]) return FIV_RET_ERR_PARA;
        if (size[1] == 0 || offset[1] + size[1] > s->shapes[1]) return FIV_RET_ERR_PARA;
        if (size[2] == 0 || offset[2] + size[2] > s->shapes[2]) return FIV_RET_ERR_PARA;
        if (s->data.ptr == NULL) return FIV_RET_ERR_PARA;
        memcpy(d, s, sizeof(fiv_tensor3d));
        d->shapes[0]     = size[0];
        d->shapes[1]     = size[1];
        d->shapes[2]     = size[2];
        d->reference     = 0;
        d->data_continue = (d->strides[2] == (size_t)d->element_bytes &&
                            d->strides[1] == (size_t)d->element_bytes * size[2] &&
                            d->strides[0] == (size_t)d->element_bytes * size[2] * size[1]) ? 1 : 0;
        d->total_bytes   = (size_t)d->element_bytes * size[0] * size[1] * size[2];
        d->data.ptr      = (iv8u*)d->data.ptr
                         + offset[0] * d->strides[0] + offset[1] * d->strides[1] + offset[2] * d->strides[2];
        return FIV_RET_OK;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* s = (const fiv_tensor4d*)src;
        fiv_tensor4d*       d = (fiv_tensor4d*)dst;
        if (size[0] == 0 || offset[0] + size[0] > s->shapes[0]) return FIV_RET_ERR_PARA;
        if (size[1] == 0 || offset[1] + size[1] > s->shapes[1]) return FIV_RET_ERR_PARA;
        if (size[2] == 0 || offset[2] + size[2] > s->shapes[2]) return FIV_RET_ERR_PARA;
        if (size[3] == 0 || offset[3] + size[3] > s->shapes[3]) return FIV_RET_ERR_PARA;
        if (s->data.ptr == NULL) return FIV_RET_ERR_PARA;
        memcpy(d, s, sizeof(fiv_tensor4d));
        d->shapes[0]     = size[0];
        d->shapes[1]     = size[1];
        d->shapes[2]     = size[2];
        d->shapes[3]     = size[3];
        d->reference     = 0;
        d->data_continue = (d->strides[3] == (size_t)d->element_bytes &&
                            d->strides[2] == (size_t)d->element_bytes * size[3] &&
                            d->strides[1] == (size_t)d->element_bytes * size[3] * size[2] &&
                            d->strides[0] == (size_t)d->element_bytes * size[3] * size[2] * size[1]) ? 1 : 0;
        d->total_bytes   = (size_t)d->element_bytes * size[0] * size[1] * size[2] * size[3];
        d->data.ptr      = (iv8u*)d->data.ptr
                         + offset[0] * d->strides[0] + offset[1] * d->strides[1]
                         + offset[2] * d->strides[2] + offset[3] * d->strides[3];
        return FIV_RET_OK;
    }
    case FIV_ID_TENSOR5D: {
        const fiv_tensor5d* s = (const fiv_tensor5d*)src;
        fiv_tensor5d*       d = (fiv_tensor5d*)dst;
        if (size[0] == 0 || offset[0] + size[0] > s->shapes[0]) return FIV_RET_ERR_PARA;
        if (size[1] == 0 || offset[1] + size[1] > s->shapes[1]) return FIV_RET_ERR_PARA;
        if (size[2] == 0 || offset[2] + size[2] > s->shapes[2]) return FIV_RET_ERR_PARA;
        if (size[3] == 0 || offset[3] + size[3] > s->shapes[3]) return FIV_RET_ERR_PARA;
        if (size[4] == 0 || offset[4] + size[4] > s->shapes[4]) return FIV_RET_ERR_PARA;
        if (s->data.ptr == NULL) return FIV_RET_ERR_PARA;
        memcpy(d, s, sizeof(fiv_tensor5d));
        d->shapes[0]     = size[0];
        d->shapes[1]     = size[1];
        d->shapes[2]     = size[2];
        d->shapes[3]     = size[3];
        d->shapes[4]     = size[4];
        d->reference     = 0;
        d->data_continue = (d->strides[4] == (size_t)d->element_bytes &&
                            d->strides[3] == (size_t)d->element_bytes * size[4] &&
                            d->strides[2] == (size_t)d->element_bytes * size[4] * size[3] &&
                            d->strides[1] == (size_t)d->element_bytes * size[4] * size[3] * size[2] &&
                            d->strides[0] == (size_t)d->element_bytes * size[4] * size[3] * size[2] * size[1]) ? 1 : 0;
        d->total_bytes   = (size_t)d->element_bytes * size[0] * size[1] * size[2] * size[3] * size[4];
        d->data.ptr      = (iv8u*)d->data.ptr
                         + offset[0] * d->strides[0] + offset[1] * d->strides[1]
                         + offset[2] * d->strides[2] + offset[3] * d->strides[3] + offset[4] * d->strides[4];
        return FIV_RET_OK;
    }
    default: return FIV_RET_ERR_PARA;
    }
}


// ---------------------------------------------------------------------------
// Zero-copy reshape: reinterpret an existing tensor's contiguous buffer under a
// new shape. Shares src's buffer (no allocation), total element count must match
// (else FIV_RET_ERR_PARA), and src must be contiguous (data_continue == 1); a
// strided/non-contiguous src cannot be reshaped as a shared view, so it is rejected.
// The new tensor reads elements in row-major order, just like NumPy's reshape.
// ---------------------------------------------------------------------------

fiv_ret fiv_tensor_reshape(void* dst, const void* src, int dim, const size_t size[])
{
    fiv_data_id   id;
    iv8u          eb;
    size_t        total;

    if (dst == NULL || src == NULL || size == NULL) return FIV_RET_ERR_PARA;
    if (dim < 1 || dim > 5) return FIV_RET_ERR_PARA;
    id = *(const fiv_data_id*)src;
    if (id < FIV_ID_TENSOR1D || id > FIV_ID_TENSOR5D) return FIV_RET_ERR_PARA;

    /* After the struct reorder the common prefix (id, dtype, meta, data.ptr,
       total_bytes) has identical offsets in every dim, so a 1D pointer can read
       the fields reshape keeps and a prefix-only memcpy carries them safely even
       when src and dst have different dimensions. */
    const fiv_tensor1d* s0 = (const fiv_tensor1d*)src;
    if (s0->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (s0->data_continue == 0) return FIV_RET_ERR_PARA;
    eb = s0->element_bytes;
    total = s0->total_bytes;
    memcpy(dst, src, offsetof(fiv_tensor1d, shapes));

    /* total element count must stay the same as src */
    switch (dim) {
    case 1:
        if (size[0] == 0) return FIV_RET_ERR_PARA;
        if ((size_t)eb * size[0] != total) return FIV_RET_ERR_PARA;
        break;
    case 2:
        if (size[0] == 0 || size[1] == 0) return FIV_RET_ERR_PARA;
        if ((size_t)eb * size[0] * size[1] != total) return FIV_RET_ERR_PARA;
        break;
    case 3:
        if (size[0] == 0 || size[1] == 0 || size[2] == 0) return FIV_RET_ERR_PARA;
        if ((size_t)eb * size[0] * size[1] * size[2] != total) return FIV_RET_ERR_PARA;
        break;
    case 4:
        if (size[0] == 0 || size[1] == 0 || size[2] == 0 || size[3] == 0) return FIV_RET_ERR_PARA;
        if ((size_t)eb * size[0] * size[1] * size[2] * size[3] != total) return FIV_RET_ERR_PARA;
        break;
    case 5:
        if (size[0] == 0 || size[1] == 0 || size[2] == 0 || size[3] == 0 || size[4] == 0) return FIV_RET_ERR_PARA;
        if ((size_t)eb * size[0] * size[1] * size[2] * size[3] * size[4] != total) return FIV_RET_ERR_PARA;
        break;
    default: return FIV_RET_ERR_PARA;
    }

    /* fill dst with C-order strides (hand-expanded, no loop); reference=0,
       contiguous. dtype/element_bytes/total_bytes/data.ptr were carried by the
       prefix memcpy. */
    switch (dim) {
    case 1: {
        fiv_tensor1d* d = (fiv_tensor1d*)dst;
        d->id            = FIV_ID_TENSOR1D;
        d->reference     = 0;
        d->data_continue = 1;
        d->shapes[0]     = size[0];
        d->strides[0]    = eb;
        return FIV_RET_OK;
    }
    case 2: {
        fiv_tensor2d* d = (fiv_tensor2d*)dst;
        d->id            = FIV_ID_TENSOR2D;
        d->reference     = 0;
        d->data_continue = 1;
        d->shapes[0]     = size[0];
        d->shapes[1]     = size[1];
        d->strides[1]    = eb;
        d->strides[0]    = (size_t)eb * size[1];
        return FIV_RET_OK;
    }
    case 3: {
        fiv_tensor3d* d = (fiv_tensor3d*)dst;
        d->id            = FIV_ID_TENSOR3D;
        d->reference     = 0;
        d->data_continue = 1;
        d->shapes[0]     = size[0];
        d->shapes[1]     = size[1];
        d->shapes[2]     = size[2];
        d->strides[2]    = eb;
        d->strides[1]    = (size_t)eb * size[2];
        d->strides[0]    = (size_t)eb * size[2] * size[1];
        return FIV_RET_OK;
    }
    case 4: {
        fiv_tensor4d* d = (fiv_tensor4d*)dst;
        d->id            = FIV_ID_TENSOR4D;
        d->reference     = 0;
        d->data_continue = 1;
        d->shapes[0]     = size[0];
        d->shapes[1]     = size[1];
        d->shapes[2]     = size[2];
        d->shapes[3]     = size[3];
        d->strides[3]    = eb;
        d->strides[2]    = (size_t)eb * size[3];
        d->strides[1]    = (size_t)eb * size[3] * size[2];
        d->strides[0]    = (size_t)eb * size[3] * size[2] * size[1];
        return FIV_RET_OK;
    }
    case 5: {
        fiv_tensor5d* d = (fiv_tensor5d*)dst;
        d->id            = FIV_ID_TENSOR5D;
        d->reference     = 0;
        d->data_continue = 1;
        d->shapes[0]     = size[0];
        d->shapes[1]     = size[1];
        d->shapes[2]     = size[2];
        d->shapes[3]     = size[3];
        d->shapes[4]     = size[4];
        d->strides[4]    = eb;
        d->strides[3]    = (size_t)eb * size[4];
        d->strides[2]    = (size_t)eb * size[4] * size[3];
        d->strides[1]    = (size_t)eb * size[4] * size[3] * size[2];
        d->strides[0]    = (size_t)eb * size[4] * size[3] * size[2] * size[1];
        return FIV_RET_OK;
    }
    default: return FIV_RET_ERR_PARA;
    }
}


// ---------------------------------------------------------------------------
// Deep copy: allocate a new tensor and copy src's data into a fresh buffer.
// The copy is independent of src (reference = 1, so releasing it frees its own
// buffer). Strides and data_continue are preserved as-is, including for a
// non-contiguous src, so the new tensor is a faithful byte-for-byte copy.
// Returns NULL if src is NULL, not a tensor, or holds no data.
// ---------------------------------------------------------------------------

void* fiv_create_tensor_from_tensor(void* tensor)
{
    fiv_data_id id;

    if (tensor == NULL) return NULL;
    id = *(fiv_data_id*)tensor;
    if (id < FIV_ID_TENSOR1D || id > FIV_ID_TENSOR5D) return NULL;

    switch (id) {
    case FIV_ID_TENSOR1D: {
        fiv_tensor1d* s = (fiv_tensor1d*)tensor;
        fiv_tensor1d* t;
        size_t        span;
        if (s->data.ptr == NULL) return NULL;
        t = (fiv_tensor1d*)fiv_malloc(sizeof(fiv_tensor1d));
        if (t == NULL) return NULL;
        memcpy(t, s, sizeof(fiv_tensor1d));                 /* copy all header fields (incl. strides) in one shot */
        span = (size_t)s->element_bytes + (s->shapes[0] - 1) * s->strides[0];
        t->data.ptr = fiv_malloc(span);                     /* re-point to a fresh, independent buffer */
        if (t->data.ptr == NULL) { fiv_free(t); return NULL; }
        memcpy(t->data.ptr, s->data.ptr, span);
        t->reference = 1;                                   /* now owns its own buffer */
        return t;
    }
    case FIV_ID_TENSOR2D: {
        fiv_tensor2d* s = (fiv_tensor2d*)tensor;
        fiv_tensor2d* t;
        size_t        span;
        if (s->data.ptr == NULL) return NULL;
        t = (fiv_tensor2d*)fiv_malloc(sizeof(fiv_tensor2d));
        if (t == NULL) return NULL;
        memcpy(t, s, sizeof(fiv_tensor2d));
        span = (size_t)s->element_bytes
             + (s->shapes[0] - 1) * s->strides[0]
             + (s->shapes[1] - 1) * s->strides[1];
        t->data.ptr = fiv_malloc(span);
        if (t->data.ptr == NULL) { fiv_free(t); return NULL; }
        memcpy(t->data.ptr, s->data.ptr, span);
        t->reference = 1;
        return t;
    }
    case FIV_ID_TENSOR3D: {
        fiv_tensor3d* s = (fiv_tensor3d*)tensor;
        fiv_tensor3d* t;
        size_t        span;
        if (s->data.ptr == NULL) return NULL;
        t = (fiv_tensor3d*)fiv_malloc(sizeof(fiv_tensor3d));
        if (t == NULL) return NULL;
        memcpy(t, s, sizeof(fiv_tensor3d));
        span = (size_t)s->element_bytes
             + (s->shapes[0] - 1) * s->strides[0]
             + (s->shapes[1] - 1) * s->strides[1]
             + (s->shapes[2] - 1) * s->strides[2];
        t->data.ptr = fiv_malloc(span);
        if (t->data.ptr == NULL) { fiv_free(t); return NULL; }
        memcpy(t->data.ptr, s->data.ptr, span);
        t->reference = 1;
        return t;
    }
    case FIV_ID_TENSOR4D: {
        fiv_tensor4d* s = (fiv_tensor4d*)tensor;
        fiv_tensor4d* t;
        size_t        span;
        if (s->data.ptr == NULL) return NULL;
        t = (fiv_tensor4d*)fiv_malloc(sizeof(fiv_tensor4d));
        if (t == NULL) return NULL;
        memcpy(t, s, sizeof(fiv_tensor4d));
        span = (size_t)s->element_bytes
             + (s->shapes[0] - 1) * s->strides[0]
             + (s->shapes[1] - 1) * s->strides[1]
             + (s->shapes[2] - 1) * s->strides[2]
             + (s->shapes[3] - 1) * s->strides[3];
        t->data.ptr = fiv_malloc(span);
        if (t->data.ptr == NULL) { fiv_free(t); return NULL; }
        memcpy(t->data.ptr, s->data.ptr, span);
        t->reference = 1;
        return t;
    }
    case FIV_ID_TENSOR5D: {
        fiv_tensor5d* s = (fiv_tensor5d*)tensor;
        fiv_tensor5d* t;
        size_t        span;
        if (s->data.ptr == NULL) return NULL;
        t = (fiv_tensor5d*)fiv_malloc(sizeof(fiv_tensor5d));
        if (t == NULL) return NULL;
        memcpy(t, s, sizeof(fiv_tensor5d));
        span = (size_t)s->element_bytes
             + (s->shapes[0] - 1) * s->strides[0]
             + (s->shapes[1] - 1) * s->strides[1]
             + (s->shapes[2] - 1) * s->strides[2]
             + (s->shapes[3] - 1) * s->strides[3]
             + (s->shapes[4] - 1) * s->strides[4];
        t->data.ptr = fiv_malloc(span);
        if (t->data.ptr == NULL) { fiv_free(t); return NULL; }
        memcpy(t->data.ptr, s->data.ptr, span);
        t->reference = 1;
        return t;
    }
    default: return NULL;
    }
}


// ---------------------------------------------------------------------------
// Header copy: allocate a new tensor struct and copy src's metadata, but leave
// data.ptr = NULL and reference = 0. The new tensor carries no data of its own;
// releasing it does not free anything. Use this to clone shape/dtype/strides
// without owning or copying the underlying buffer.
// Returns NULL if src is NULL or not a tensor.
// ---------------------------------------------------------------------------

void* fiv_create_tensor_header_from_tensor(void* tensor)
{
    fiv_data_id id;

    if (tensor == NULL) return NULL;
    id = *(fiv_data_id*)tensor;
    if (id < FIV_ID_TENSOR1D || id > FIV_ID_TENSOR5D) return NULL;

    switch (id) {
    case FIV_ID_TENSOR1D: {
        fiv_tensor1d* s = (fiv_tensor1d*)tensor;
        fiv_tensor1d* t = (fiv_tensor1d*)fiv_malloc(sizeof(fiv_tensor1d));
        if (t == NULL) return NULL;
        memcpy(t, s, sizeof(fiv_tensor1d));   /* copy all header fields, then drop data ownership */
        t->reference = 0;
        t->data.ptr  = NULL;
        return t;
    }
    case FIV_ID_TENSOR2D: {
        fiv_tensor2d* s = (fiv_tensor2d*)tensor;
        fiv_tensor2d* t = (fiv_tensor2d*)fiv_malloc(sizeof(fiv_tensor2d));
        if (t == NULL) return NULL;
        memcpy(t, s, sizeof(fiv_tensor2d));
        t->reference = 0;
        t->data.ptr  = NULL;
        return t;
    }
    case FIV_ID_TENSOR3D: {
        fiv_tensor3d* s = (fiv_tensor3d*)tensor;
        fiv_tensor3d* t = (fiv_tensor3d*)fiv_malloc(sizeof(fiv_tensor3d));
        if (t == NULL) return NULL;
        memcpy(t, s, sizeof(fiv_tensor3d));
        t->reference = 0;
        t->data.ptr  = NULL;
        return t;
    }
    case FIV_ID_TENSOR4D: {
        fiv_tensor4d* s = (fiv_tensor4d*)tensor;
        fiv_tensor4d* t = (fiv_tensor4d*)fiv_malloc(sizeof(fiv_tensor4d));
        if (t == NULL) return NULL;
        memcpy(t, s, sizeof(fiv_tensor4d));
        t->reference = 0;
        t->data.ptr  = NULL;
        return t;
    }
    case FIV_ID_TENSOR5D: {
        fiv_tensor5d* s = (fiv_tensor5d*)tensor;
        fiv_tensor5d* t = (fiv_tensor5d*)fiv_malloc(sizeof(fiv_tensor5d));
        if (t == NULL) return NULL;
        memcpy(t, s, sizeof(fiv_tensor5d));
        t->reference = 0;
        t->data.ptr  = NULL;
        return t;
    }
    default: return NULL;
    }
}


// ---------------------------------------------------------------------------
// Alloc-like: same shape and dtype as src, but a fresh (uninitialized) data
// buffer. Mirror fiv_create_tensor_header_from_tensor for the header, then
// allocate the buffer via fiv_create_tensorN, so the result owns its buffer and
// is contiguous regardless of whether src is. src must be a tensor (1D~5D).
// ---------------------------------------------------------------------------

void* fiv_create_tensor_like_tensor(void* tensor)
{
    fiv_data_id id;

    if (tensor == NULL) return NULL;
    id = *(fiv_data_id*)tensor;
    if (id < FIV_ID_TENSOR1D || id > FIV_ID_TENSOR5D) return NULL;

    switch (id) {
    case FIV_ID_TENSOR1D: {
        fiv_tensor1d* s = (fiv_tensor1d*)tensor;
        return fiv_create_tensor1d(s->shapes[0], s->dtype);
    }
    case FIV_ID_TENSOR2D: {
        fiv_tensor2d* s = (fiv_tensor2d*)tensor;
        size_t sh[2] = { s->shapes[0], s->shapes[1] };
        return fiv_create_tensor2d(sh, s->dtype);
    }
    case FIV_ID_TENSOR3D: {
        fiv_tensor3d* s = (fiv_tensor3d*)tensor;
        size_t sh[3] = { s->shapes[0], s->shapes[1], s->shapes[2] };
        return fiv_create_tensor3d(sh, s->dtype);
    }
    case FIV_ID_TENSOR4D: {
        fiv_tensor4d* s = (fiv_tensor4d*)tensor;
        size_t sh[4] = { s->shapes[0], s->shapes[1], s->shapes[2], s->shapes[3] };
        return fiv_create_tensor4d(sh, s->dtype);
    }
    case FIV_ID_TENSOR5D: {
        fiv_tensor5d* s = (fiv_tensor5d*)tensor;
        size_t sh[5] = { s->shapes[0], s->shapes[1], s->shapes[2], s->shapes[3], s->shapes[4] };
        return fiv_create_tensor5d(sh, s->dtype);
    }
    default: return NULL;
    }
}


// ---------------------------------------------------------------------------
// Element-wise binary ops: c = a <op> b, per scalar component.
// float32 (32F), float64 (64F) and signed int32 (32S) dtypes are supported;
// anything else returns FIV_RET_ERR_NOT_SUPPORT. a, b and c must share the same
// dtype, the same dimension, and the same total element count (same shape; there
// is NO broadcasting), and each must be contiguous (data_continue == 1) and hold
// data. The underlying kernels in fiv_binary_op.c operate on flat, contiguous
// element arrays, which is why non-contiguous (strided) tensors are rejected.
// In-place is allowed (c may alias a or b) because the op is purely element-wise.
// ---------------------------------------------------------------------------

typedef enum : iv32u {
    FIV_BINOP_ADD = 0,
    FIV_BINOP_SUB,
    FIV_BINOP_MUL,
    FIV_BINOP_DIV
} fiv_binop;

static fiv_ret fiv_tensor_binary_op(void* c, const void* a, const void* b, fiv_binop op)
{
    fiv_data_id id;

    if (c == NULL || a == NULL || b == NULL) return FIV_RET_ERR_PARA;

    id = *(const fiv_data_id*)a;
    if (id < FIV_ID_TENSOR1D || id > FIV_ID_TENSOR5D) return FIV_RET_ERR_PARA;
    if (*(const fiv_data_id*)b != id) return FIV_RET_ERR_PARA;   /* same dimension */
    if (*(const fiv_data_id*)c != id) return FIV_RET_ERR_PARA;

    /* After the struct reorder the common prefix (id, dtype, meta, data, total_bytes)
       is byte-identical across dims, so a 1D view reads every field the op needs. */
    const fiv_tensor1d* a0 = (const fiv_tensor1d*)a;
    const fiv_tensor1d* b0 = (const fiv_tensor1d*)b;
    fiv_tensor1d*       c0 = (fiv_tensor1d*)c;

    if (a0->data.ptr == NULL || b0->data.ptr == NULL || c0->data.ptr == NULL)
        return FIV_RET_ERR_PARA;
    /* flat element-wise kernels assume contiguous storage */
    if (a0->data_continue == 0 || b0->data_continue == 0 || c0->data_continue == 0)
        return FIV_RET_ERR_PARA;
    /* same dtype across all three (no mixed-type promotion) */
    if (a0->dtype != b0->dtype || b0->dtype != c0->dtype)
        return FIV_RET_ERR_PARA;
    /* same element count (same shape; no broadcasting) */
    if (a0->total_bytes != b0->total_bytes || b0->total_bytes != c0->total_bytes)
        return FIV_RET_ERR_PARA;

    if ((int)a0->dtype < FIV_8U1 || (int)a0->dtype > FIV_16BF16) return FIV_RET_ERR_PARA;
    /* dtype groups are 16 values wide; slot 5 = signed int32 (32S), slot 8 = float32 (32F) */
    int    slot = (int)a0->dtype % 16;
    size_t n;

    if (slot == 8) {            /* float32 (32F family) */
        n = a0->total_bytes / 4;
        switch (op) {
        case FIV_BINOP_ADD: fiv_add_ivf32(c0->data.fl, a0->data.fl, b0->data.fl, n); break;
        case FIV_BINOP_SUB: fiv_sub_ivf32(c0->data.fl, a0->data.fl, b0->data.fl, n); break;
        case FIV_BINOP_MUL: fiv_mul_ivf32(c0->data.fl, a0->data.fl, b0->data.fl, n); break;
        case FIV_BINOP_DIV: fiv_div_ivf32(c0->data.fl, a0->data.fl, b0->data.fl, n); break;
        }
    } else if (slot == 5) {     /* signed int32 (32S family) */
        n = a0->total_bytes / 4;
        switch (op) {
        case FIV_BINOP_ADD: fiv_add_iv32s(c0->data.ptr32s, a0->data.ptr32s, b0->data.ptr32s, n); break;
        case FIV_BINOP_SUB: fiv_sub_iv32s(c0->data.ptr32s, a0->data.ptr32s, b0->data.ptr32s, n); break;
        case FIV_BINOP_MUL: fiv_mul_iv32s(c0->data.ptr32s, a0->data.ptr32s, b0->data.ptr32s, n); break;
        case FIV_BINOP_DIV: fiv_div_iv32s(c0->data.ptr32s, a0->data.ptr32s, b0->data.ptr32s, n); break;
        }
    } else if (slot == 9) {     /* float64 (64F family) */
        n = a0->total_bytes / 8;
        switch (op) {
        case FIV_BINOP_ADD: fiv_add_ivf64(c0->data.db, a0->data.db, b0->data.db, n); break;
        case FIV_BINOP_SUB: fiv_sub_ivf64(c0->data.db, a0->data.db, b0->data.db, n); break;
        case FIV_BINOP_MUL: fiv_mul_ivf64(c0->data.db, a0->data.db, b0->data.db, n); break;
        case FIV_BINOP_DIV: fiv_div_ivf64(c0->data.db, a0->data.db, b0->data.db, n); break;
        }
    } else {
        return FIV_RET_ERR_NOT_SUPPORT;   /* only int32 / float32 / float64 supported */
    }
    return FIV_RET_OK;
}

fiv_ret fiv_tensor_add(void* c, const void* a, const void* b) { return fiv_tensor_binary_op(c, a, b, FIV_BINOP_ADD); }
fiv_ret fiv_tensor_sub(void* c, const void* a, const void* b) { return fiv_tensor_binary_op(c, a, b, FIV_BINOP_SUB); }
fiv_ret fiv_tensor_mul(void* c, const void* a, const void* b) { return fiv_tensor_binary_op(c, a, b, FIV_BINOP_MUL); }
fiv_ret fiv_tensor_div(void* c, const void* a, const void* b) { return fiv_tensor_binary_op(c, a, b, FIV_BINOP_DIV); }


// ---------------------------------------------------------------------------
// Generic create: dispatch by dimension
// ---------------------------------------------------------------------------

void* fiv_create_tensor(int dim, size_t size[], fiv_data_type data_type) {
    if (size == NULL) return NULL;
    switch (dim) {
    case 1: return fiv_create_tensor1d(size[0], data_type);
    case 2: return fiv_create_tensor2d(size, data_type);
    case 3: return fiv_create_tensor3d(size, data_type);
    case 4: return fiv_create_tensor4d(size, data_type);
    case 5: return fiv_create_tensor5d(size, data_type);
    default: return NULL;
    }
}


// Each tensor struct's first field is fiv_data_id, so the real type can be
// read from the header and dispatched accordingly.
fiv_ret fiv_release_tensor(void** tensor)
{
	fiv_data_id id;

	if (tensor == NULL || *tensor == NULL){
		return FIV_RET_ERR_PARA;
	}
	id = *(const fiv_data_id*)(*tensor);
	switch (id) {
	case FIV_ID_TENSOR1D: return fiv_release_tensor1d((fiv_tensor1d**)tensor);
	case FIV_ID_TENSOR2D: return fiv_release_tensor2d((fiv_tensor2d**)tensor);
	case FIV_ID_TENSOR3D: return fiv_release_tensor3d((fiv_tensor3d**)tensor);
	case FIV_ID_TENSOR4D: return fiv_release_tensor4d((fiv_tensor4d**)tensor);
	case FIV_ID_TENSOR5D: return fiv_release_tensor5d((fiv_tensor5d**)tensor);
	default: return FIV_RET_ERR_PARA;
	}
}
