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

#ifndef _FIV_CTENSOR_H_
#define _FIV_CTENSOR_H_

#include <stddef.h>
#include "fiv_data_typedefs.h"

#ifdef __cplusplus
extern "C" {
#endif


/* ============================ Tensor IDs ============================ */
typedef enum {
    FIV_ID_START  = 0,
    FIV_ID_TENSOR1D,
    FIV_ID_TENSOR2D,
    FIV_ID_TENSOR3D,
    FIV_ID_TENSOR4D,
    FIV_ID_TENSOR5D,
    FIV_ID_IMAGE,
} fiv_data_id;


/* ===================== Unions shared by all tensor dims ===================== */
/* reference / data_continue / element_bytes share memory with meta_info[4] */
#define FIV_META_UNION \
    union { \
        struct { iv8u reference, data_continue, element_bytes; }; \
        struct { iv8u meta_info[4]; }; \
    }

/* data pointer: same memory interpreted as different element types */
#define FIV_DATA_UNION \
    union { \
        void*  ptr;   \
        iv8u*  ptr8u; \
        iv8s*  ptr8s; \
        iv16u* ptr16u; \
        iv16s* ptr16s; \
        iv32u* ptr32u; \
        iv32s* ptr32s; \
        iv64u* ptr64u; \
        iv64s* ptr64s; \
        ivf32* fl;    \
        ivf64* db;    \
    } data


/* ============================== 1D tensor ============================== */
typedef struct {
    fiv_data_id   id;
    fiv_data_type dtype;
    FIV_META_UNION;
    FIV_DATA_UNION;
    size_t total_bytes;
    union {
        struct { size_t length; };
        struct { size_t width; };
        struct { size_t shapes[1]; };
    };
    size_t strides[1];
} fiv_tensor1d;


/* ======================= 2D tensor (matrix) ======================= */
typedef struct {
    fiv_data_id   id;
    fiv_data_type dtype;
    FIV_META_UNION;
    FIV_DATA_UNION;
    size_t total_bytes;
    union {
        struct { size_t rows, cols; };
        struct { size_t height, width; };
        struct { size_t shapes[2]; };
    };
    size_t strides[2];
} fiv_tensor2d;


/* ============================== 3D tensor ============================== */
typedef struct {
    fiv_data_id   id;
    fiv_data_type dtype;
    FIV_META_UNION;
    FIV_DATA_UNION;
    size_t total_bytes;
    union {
        struct { size_t depth, rows, cols; };
        struct { size_t channels, height, width; };
        struct { size_t shapes[3]; };
    };
    size_t strides[3];
} fiv_tensor3d;


/* ============================== 4D tensor ============================== */
typedef struct {
    fiv_data_id   id;
    fiv_data_type dtype;
    FIV_META_UNION;
    FIV_DATA_UNION;
    size_t total_bytes;
    union {
        struct { size_t batch, depth, rows, cols; };
        struct { size_t cubes, channels, height, width; };
        struct { size_t shapes[4]; };
    };
    size_t strides[4];
} fiv_tensor4d;


/* ============================== 5D tensor ============================== */
typedef struct {
    fiv_data_id   id;
    fiv_data_type dtype;
    FIV_META_UNION;
    FIV_DATA_UNION;
    size_t total_bytes;
    union {
        struct { size_t batch, times, depth, rows, cols; };
        struct { size_t blocks, cubes, channels, height, width; };
        struct { size_t shapes[5]; };
    };
    size_t strides[5];
} fiv_tensor5d;


/* ============================== Aliases ============================== */
typedef fiv_tensor2d mat;
typedef fiv_tensor1d vec;


/* ===================== Create / release (1D~5D) ===================== */

/* Create a tensor and allocate its data buffer; the returned tensor must be released by the caller */
fiv_tensor1d* fiv_create_tensor1d(size_t size, fiv_data_type data_type);
fiv_tensor2d* fiv_create_tensor2d(size_t size[2], fiv_data_type data_type);
fiv_tensor3d* fiv_create_tensor3d(size_t size[3], fiv_data_type data_type);
fiv_tensor4d* fiv_create_tensor4d(size_t size[4], fiv_data_type data_type);
fiv_tensor5d* fiv_create_tensor5d(size_t size[5], fiv_data_type data_type);

/* Release a tensor. A tensor created by fiv_create_tensorXd also frees its data buffer;
   a tensor wrapping an external buffer via set_header/set_data only frees the struct,
   leaving the external buffer untouched (caller still owns it) */
fiv_ret fiv_release_tensor1d(fiv_tensor1d** tensor1d);
fiv_ret fiv_release_tensor2d(fiv_tensor2d** tensor2d);
fiv_ret fiv_release_tensor3d(fiv_tensor3d** tensor3d);
fiv_ret fiv_release_tensor4d(fiv_tensor4d** tensor4d);
fiv_ret fiv_release_tensor5d(fiv_tensor5d** tensor5d);

/* Attach an existing external buffer to the tensor (no allocation); the buffer size
   must be >= total_bytes, otherwise FIV_RET_ERR_PARA is returned; the data remains caller-owned */
fiv_ret fiv_tensor1d_set_data(fiv_tensor1d* tensor1d, void* data, size_t size);
fiv_ret fiv_tensor2d_set_data(fiv_tensor2d* tensor2d, void* data, size_t size);
fiv_ret fiv_tensor3d_set_data(fiv_tensor3d* tensor3d, void* data, size_t size);
fiv_ret fiv_tensor4d_set_data(fiv_tensor4d* tensor4d, void* data, size_t size);
fiv_ret fiv_tensor5d_set_data(fiv_tensor5d* tensor5d, void* data, size_t size);

/* Fill the tensor header from dtype and shape only (no allocation); data pointer is left
   NULL and must be set afterwards via set_data */
fiv_ret fiv_tensor1d_set_header(fiv_tensor1d* tensor1d, size_t size, fiv_data_type data_type);
fiv_ret fiv_tensor2d_set_header(fiv_tensor2d* tensor2d, size_t size[2], fiv_data_type data_type);
fiv_ret fiv_tensor3d_set_header(fiv_tensor3d* tensor3d, size_t size[3], fiv_data_type data_type);
fiv_ret fiv_tensor4d_set_header(fiv_tensor4d* tensor4d, size_t size[4], fiv_data_type data_type);
fiv_ret fiv_tensor5d_set_header(fiv_tensor5d* tensor5d, size_t size[5], fiv_data_type data_type);

/* Allocate a tensor header and fill its metadata, but do NOT allocate the data buffer
   (data left NULL); call set_data afterwards to attach an external buffer */
fiv_tensor1d* fiv_create_tensor1d_header(size_t size, fiv_data_type data_type);
fiv_tensor2d* fiv_create_tensor2d_header(size_t size[2], fiv_data_type data_type);
fiv_tensor3d* fiv_create_tensor3d_header(size_t size[3], fiv_data_type data_type);
fiv_tensor4d* fiv_create_tensor4d_header(size_t size[4], fiv_data_type data_type);
fiv_tensor5d* fiv_create_tensor5d_header(size_t size[5], fiv_data_type data_type);


/* ===================== Generic dispatch (by dim / id) ===================== */

/* Create a tensor by dim (1~5); returns void*, caller must cast to the concrete type */
void* fiv_create_tensor(int dim, size_t size[], fiv_data_type data_type);
/* Release a tensor by its type; pass the address of the concrete-type pointer (void**) */
fiv_ret fiv_release_tensor(void** tensor);

/* Create a zero-copy view of an existing tensor. The view shares src's buffer (no
   allocation), points data at the sub-region given by offset[]/size[] (one entry per
   dimension), and inherits src's strides so element access stays correct. data_continue
   is set to reflect the view's actual contiguity. The view does NOT own the buffer:
   releasing it will not free src's memory, so the caller must keep src alive while the
   view is used. offset[k]+size[k] must not exceed src's shape, and src must hold data. */
fiv_ret fiv_tensor_view(void* dst, const void* src, const size_t offset[], const size_t size[]);

/* Reinterpret an existing tensor's contiguous buffer under a new shape (no allocation,
   shared memory, like NumPy's reshape). The total element count must match src's, and
   src must be contiguous (data_continue == 1); a strided view cannot be reshaped as a
   shared view and is rejected. Elements are read in row-major order. dst must be a struct
   of the requested dim, allocated by the caller (stack or *_header). The reshaped tensor
   does NOT own the buffer: releasing it will not free src's memory. */
fiv_ret fiv_tensor_reshape(void* dst, const void* src, int dim, const size_t size[]);

/* Create a deep copy of an existing tensor: a new struct plus a fresh buffer that
   holds src's data. The copy is independent of src and owns its buffer (reference = 1),
   so releasing it frees that buffer. Strides and data_continue are copied as-is, so a
   non-contiguous src yields a faithful, also non-contiguous copy. Returns NULL if src is
   NULL, not a tensor, or holds no data. */
void* fiv_create_tensor_from_tensor(void* tensor);

/* Create a copy of an existing tensor's header only: shape, dtype, strides and other
   metadata are copied, but data.ptr is left NULL and the new tensor owns nothing
   (reference = 0). Releasing it frees only the struct. Use this to clone a tensor's
   description without copying or owning its data. Returns NULL if src is NULL or not a
   tensor. */
void* fiv_create_tensor_header_from_tensor(void* tensor);


/* ===================== Element-wise binary ops (float32 / int32) ===================== */

/* Compute c = a <op> b component-wise. Supports float32 (32F family) and signed int32
   (32S family) dtypes only; other dtypes return FIV_RET_ERR_NOT_SUPPORT. a, b and c must
   share the same dtype, the same dimension and the same total element count (same shape;
   no broadcasting), and each must be contiguous (data_continue == 1) and hold data; on any
   mismatch FIV_RET_ERR_PARA is returned. c may alias a or b (in-place) since the op is
   purely element-wise. The underlying kernels live in fiv_binary_op.c. */
fiv_ret fiv_tensor_add(void* c, const void* a, const void* b);
fiv_ret fiv_tensor_sub(void* c, const void* a, const void* b);
fiv_ret fiv_tensor_mul(void* c, const void* a, const void* b);
fiv_ret fiv_tensor_div(void* c, const void* a, const void* b);


#ifdef __cplusplus
}
#endif

#endif  /* _FIV_CTENSOR_H_ */
