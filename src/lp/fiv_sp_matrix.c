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

#include "fiv_sp_matrix.h"
#include "fiv_common.h"   /* fiv_malloc / fiv_free */

#include <stdlib.h>
#include <string.h>
#include <math.h>


/* ------------------------------------------------------------------ */
/* Runtime SIMD probe (process-wide singleton)                        */
/* ------------------------------------------------------------------ */
static fiv_sparse_runtime s_runtime_state = { 1, 0 };

fiv_ret fiv_sparse_runtime_init(void)
{
#if defined(FIV_USE_AVX512)
    s_runtime_state.simd_lanes = 8;
#elif defined(FIV_USE_AVX2)
    s_runtime_state.simd_lanes = 4;
#elif defined(FIV_USE_ARM_NEON)
    s_runtime_state.simd_lanes = 2;
#else
    s_runtime_state.simd_lanes = 1;
#endif
    s_runtime_state.use_intrinsics = 0;   /* primary path is auto-vectorized standard C */
    return FIV_RET_OK;
}

fiv_ret fiv_sparse_runtime_get(const fiv_sparse_runtime **runtime_info)
{
    if (runtime_info == NULL) return FIV_RET_ERR_PARA;
    *runtime_info = &s_runtime_state;
    return FIV_RET_OK;
}


/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

/* Element byte size for a supported dtype (mirrors how ctensor reports
 * element_bytes from fiv_data_type). Only FIV_64F1 is wired into the kernels. */
static size_t fiv_sparse_elem_bytes(fiv_data_type dtype)
{
    switch (dtype) {
        case FIV_64F1:  return sizeof(ivf64);
        case FIV_32F1:  return sizeof(ivf32);
        case FIV_64S1:  return sizeof(iv64s);
        case FIV_64U1:  return sizeof(iv64u);
        case FIV_32S1:  return sizeof(iv32s);
        case FIV_32U1:  return sizeof(iv32u);
        case FIV_16F1:  return sizeof(ivf16);
        case FIV_16BF1: return sizeof(ivbf16);
        case FIV_16S1:  return sizeof(iv16s);
        case FIV_16U1:  return sizeof(iv16u);
        case FIV_8S1:   return sizeof(iv8s);
        case FIV_8U1:   return sizeof(iv8u);
        default:        return 0;
    }
}

/* Fill the shared generic header prefix for a freshly built sparse matrix.
 * Note: it does NOT touch hdr.data.ptr / total_bytes (those are set by the
 * caller once the value buffer is allocated). */
static void fiv_sparse_set_header(fiv_sparse_mat *sparse_matrix,
                                  fiv_sparse_fmt format, fiv_data_type dtype)
{
    sparse_matrix->hdr.id = FIV_ID_TENSOR2D;
    sparse_matrix->hdr.dtype = dtype;
    sparse_matrix->hdr.reference = 0;
    sparse_matrix->hdr.data_continue = 1;
    sparse_matrix->hdr.element_bytes = (iv8u)fiv_sparse_elem_bytes(dtype);
    sparse_matrix->hdr.color_space_type = (iv8u)format;
    sparse_matrix->packed = NULL;
}

fiv_sparse_fmt fiv_sparse_get_fmt(const fiv_sparse_mat *sparse_matrix)
{
    if (sparse_matrix == NULL) return FIV_SPARSE_CSR;
    return (fiv_sparse_fmt)sparse_matrix->hdr.color_space_type;
}


/* ------------------------------------------------------------------ */
/* Build: allocate an empty matrix                                    */
/* ------------------------------------------------------------------ */

fiv_sparse_mat *fiv_create_sp_matrix(fiv_sparse_fmt format, size_t num_rows,
                                     size_t num_cols, size_t num_nonzeros,
                                     fiv_data_type value_dtype)
{
    if (num_rows == 0 || num_cols == 0) return NULL;
    if (fiv_sparse_elem_bytes(value_dtype) == 0) return NULL;

    fiv_sparse_mat *sparse_matrix = (fiv_sparse_mat *)fiv_malloc(sizeof(fiv_sparse_mat));
    if (sparse_matrix == NULL) return NULL;
    memset(sparse_matrix, 0, sizeof(*sparse_matrix));
    sparse_matrix->rows = num_rows;
    sparse_matrix->cols = num_cols;
    sparse_matrix->nnz = num_nonzeros;

    size_t ptr_count = 0;    /* length of indptr buffer (0 for COO) */
    size_t index_count = (num_nonzeros > 0) ? num_nonzeros : 0;
    switch (format) {
        case FIV_SPARSE_CSR: ptr_count = num_rows + 1; break;
        case FIV_SPARSE_CSC: ptr_count = num_cols + 1; break;
        case FIV_SPARSE_COO: ptr_count = 0;            break;
        default: fiv_free(sparse_matrix); return NULL;
    }

    int *rowptr = (ptr_count > 0) ? (int *)fiv_malloc(ptr_count * sizeof(int)) : NULL;
    int *colind = (index_count > 0) ? (int *)fiv_malloc(index_count * sizeof(int)) : NULL;
    int *rowidx = (format == FIV_SPARSE_COO && num_nonzeros > 0)
                      ? (int *)fiv_malloc(num_nonzeros * sizeof(int)) : NULL;
    void *value_buffer = (num_nonzeros > 0)
                      ? fiv_malloc(num_nonzeros * fiv_sparse_elem_bytes(value_dtype)) : NULL;

    if ((ptr_count > 0 && rowptr == NULL) ||
        (index_count > 0 && colind == NULL) ||
        (format == FIV_SPARSE_COO && num_nonzeros > 0 && rowidx == NULL) ||
        (num_nonzeros > 0 && value_buffer == NULL)) {
        fiv_free(rowptr); fiv_free(colind); fiv_free(rowidx); fiv_free(value_buffer);
        fiv_free(sparse_matrix);
        return NULL;
    }

    sparse_matrix->indptr = rowptr;
    sparse_matrix->indices = colind;
    sparse_matrix->rowidx = (format == FIV_SPARSE_COO) ? rowidx : NULL;
    sparse_matrix->hdr.data.ptr = value_buffer;
    sparse_matrix->hdr.total_bytes = num_nonzeros * fiv_sparse_elem_bytes(value_dtype);
    fiv_sparse_set_header(sparse_matrix, format, value_dtype);
    return sparse_matrix;
}


/* ------------------------------------------------------------------ */
/* Build: COO -> CSR                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    int   row_index;
    int   col_index;
    ivf64 value;
} fiv_coo_entry;

static int fiv_coo_cmp(const void *lhs, const void *rhs)
{
    const fiv_coo_entry *entry_a = (const fiv_coo_entry *)lhs;
    const fiv_coo_entry *entry_b = (const fiv_coo_entry *)rhs;
    if (entry_a->row_index != entry_b->row_index)
        return (entry_a->row_index < entry_b->row_index) ? -1 : 1;
    if (entry_a->col_index != entry_b->col_index)
        return (entry_a->col_index < entry_b->col_index) ? -1 : 1;
    return 0;
}

fiv_sparse_mat *fiv_create_sp_matrix_from_coo(const int *row_indices, const int *col_indices,
                                              const void *values, fiv_data_type value_dtype,
                                              size_t num_nonzeros, size_t num_rows, size_t num_cols)
{
    if (row_indices == NULL || col_indices == NULL || values == NULL) return NULL;
    if (num_rows == 0 || num_cols == 0) return NULL;
    if (value_dtype != FIV_64F1) return NULL;   /* only f64 today */

    const ivf64 *typed_values = (const ivf64 *)values;

    /* Bounds check. */
    for (size_t entry = 0; entry < num_nonzeros; entry++) {
        if (row_indices[entry] < 0 || (size_t)row_indices[entry] >= num_rows ||
            col_indices[entry] < 0 || (size_t)col_indices[entry] >= num_cols)
            return NULL;
    }

    /* Copy + sort triplets by (row, col). */
    fiv_coo_entry *coo_entries = (fiv_coo_entry *)fiv_malloc((num_nonzeros ? num_nonzeros : 1) * sizeof(fiv_coo_entry));
    if (coo_entries == NULL) return NULL;
    for (size_t entry = 0; entry < num_nonzeros; entry++) {
        coo_entries[entry].row_index = row_indices[entry];
        coo_entries[entry].col_index = col_indices[entry];
        coo_entries[entry].value = typed_values[entry];
    }
    if (num_nonzeros > 0) qsort(coo_entries, num_nonzeros, sizeof(fiv_coo_entry), fiv_coo_cmp);

    /* Collapse duplicates (accumulate) to get the CSR nnz. */
    size_t csr_nnz = 0;
    for (size_t entry = 0; entry < num_nonzeros; ) {
        size_t next_entry = entry + 1;
        while (next_entry < num_nonzeros &&
               coo_entries[next_entry].row_index == coo_entries[entry].row_index &&
               coo_entries[next_entry].col_index == coo_entries[entry].col_index)
            next_entry++;
        csr_nnz++;
        entry = next_entry;
    }

    fiv_sparse_mat *sparse_matrix = fiv_create_sp_matrix(FIV_SPARSE_CSR, num_rows, num_cols,
                                                        csr_nnz, FIV_64F1);
    if (sparse_matrix == NULL) { fiv_free(coo_entries); return NULL; }

    int  *rowptr = sparse_matrix->indptr;
    int  *colind = sparse_matrix->indices;
    ivf64 *value_buffer = (ivf64 *)sparse_matrix->hdr.data.ptr;

    /* Fill CSR: collapse duplicates, close empty rows with correct indptr.
     * rowptr[row] is the start of row `row`; the inline code sets rowptr[row+1]
     * at the END of each run, so the tail loop only needs to fill rows strictly
     * after the last populated row (current_row + 1 .. num_rows). */
    size_t write_index = 0;       /* write index into colind / value_buffer */
    size_t current_row = 0;       /* last row whose boundary has been set */
    rowptr[0] = 0;
    for (size_t entry = 0; entry < num_nonzeros; ) {
        size_t next_entry = entry + 1;
        ivf64 accumulator = coo_entries[entry].value;
        while (next_entry < num_nonzeros &&
               coo_entries[next_entry].row_index == coo_entries[entry].row_index &&
               coo_entries[next_entry].col_index == coo_entries[entry].col_index) {
            accumulator += coo_entries[next_entry].value;
            next_entry++;
        }
        /* Close any empty rows strictly before this entry's row. */
        while (current_row < (size_t)coo_entries[entry].row_index) {
            current_row++;
            rowptr[current_row] = (int)write_index;
        }
        /* current_row == this entry's row now; write the collapsed entry. */
        colind[write_index] = coo_entries[entry].col_index;
        value_buffer[write_index] = accumulator;
        write_index++;
        rowptr[current_row + 1] = (int)write_index;
        current_row = (size_t)coo_entries[entry].row_index;
        entry = next_entry;
    }
    /* Close remaining rows (current_row+1 .. num_rows) with the final write index. */
    while (current_row < num_rows) {
        current_row++;
        rowptr[current_row] = (int)write_index;
    }

    fiv_free(coo_entries);
    return sparse_matrix;
}


/* ------------------------------------------------------------------ */
/* Build: dense (fiv_mat) <-> sparse                                   */
/* ------------------------------------------------------------------ */

/* Read one element of a dense matrix (row-major) as f64. NOTE: fiv_tensor2d
 * strides are BYTE strides, so divide by element_bytes to get the element index. */
static ivf64 fiv_dense_get_f64(const fiv_mat *dense_matrix, size_t row_index, size_t col_index)
{
    size_t byte_offset = row_index * dense_matrix->strides[0] + col_index * dense_matrix->strides[1];
    size_t elem_index = byte_offset / dense_matrix->element_bytes;
    if (dense_matrix->dtype == FIV_64F1) return dense_matrix->data.db[elem_index];
    return (ivf64)dense_matrix->data.fl[elem_index];   /* FIV_32F1 widened */
}

/* Write one element of a dense matrix (row-major). See fiv_dense_get_f64 re: strides. */
static void fiv_dense_set_f64(fiv_mat *dense_matrix, size_t row_index, size_t col_index, ivf64 value)
{
    size_t byte_offset = row_index * dense_matrix->strides[0] + col_index * dense_matrix->strides[1];
    size_t elem_index = byte_offset / dense_matrix->element_bytes;
    if (dense_matrix->dtype == FIV_64F1) dense_matrix->data.db[elem_index] = value;
    else dense_matrix->data.fl[elem_index] = (ivf32)value;
}

fiv_sparse_mat *fiv_create_sp_matrix_from_dense(const fiv_mat *dense_matrix)
{
    if (dense_matrix == NULL) return NULL;
    if (dense_matrix->dtype != FIV_64F1 && dense_matrix->dtype != FIV_32F1) return NULL;

    const size_t num_rows = dense_matrix->rows;
    const size_t num_cols = dense_matrix->cols;

    /* Count exact nonzeros (zeros are dropped). */
    size_t num_nonzeros = 0;
    for (size_t row_index = 0; row_index < num_rows; row_index++)
        for (size_t col_index = 0; col_index < num_cols; col_index++)
            if (fiv_dense_get_f64(dense_matrix, row_index, col_index) != 0.0) num_nonzeros++;

    fiv_sparse_mat *sparse_matrix = fiv_create_sp_matrix(FIV_SPARSE_CSR, num_rows, num_cols,
                                                        num_nonzeros, FIV_64F1);
    if (sparse_matrix == NULL) return NULL;

    int  *rowptr = sparse_matrix->indptr;
    int  *colind = sparse_matrix->indices;
    ivf64 *value_buffer = (ivf64 *)sparse_matrix->hdr.data.ptr;

    size_t write_index = 0;
    rowptr[0] = 0;
    for (size_t row_index = 0; row_index < num_rows; row_index++) {
        for (size_t col_index = 0; col_index < num_cols; col_index++) {
            ivf64 value = fiv_dense_get_f64(dense_matrix, row_index, col_index);
            if (value != 0.0) {
                colind[write_index] = (int)col_index;
                value_buffer[write_index] = value;
                write_index++;
            }
        }
        rowptr[row_index + 1] = (int)write_index;
    }
    return sparse_matrix;
}

fiv_mat *fiv_create_dense_matrix_from_sp_matrix(const fiv_sparse_mat *sparse_matrix)
{
    if (sparse_matrix == NULL) return NULL;

    const fiv_sparse_fmt format = fiv_sparse_get_fmt(sparse_matrix);
    if (format != FIV_SPARSE_CSR && format != FIV_SPARSE_CSC && format != FIV_SPARSE_COO)
        return NULL;

    const fiv_data_type value_dtype = sparse_matrix->hdr.dtype;   /* FIV_64F1 today */
    const size_t num_rows = sparse_matrix->rows;
    const size_t num_cols = sparse_matrix->cols;

    size_t shape[2] = { num_rows, num_cols };
    fiv_mat *dense_matrix = fiv_create_tensor2d(shape, value_dtype);
    if (dense_matrix == NULL) return NULL;

    memset(dense_matrix->data.ptr, 0, dense_matrix->total_bytes);

    if (format == FIV_SPARSE_CSR) {
        for (size_t row_index = 0; row_index < num_rows; row_index++)
            for (int entry = sparse_matrix->indptr[row_index];
                 entry < sparse_matrix->indptr[row_index + 1]; entry++)
                fiv_dense_set_f64(dense_matrix, row_index,
                                  (size_t)sparse_matrix->indices[entry],
                                  ((const ivf64 *)sparse_matrix->hdr.data.ptr)[entry]);
    } else if (format == FIV_SPARSE_CSC) {
        for (size_t col_index = 0; col_index < num_cols; col_index++)
            for (int entry = sparse_matrix->indptr[col_index];
                 entry < sparse_matrix->indptr[col_index + 1]; entry++)
                fiv_dense_set_f64(dense_matrix,
                                  (size_t)sparse_matrix->indices[entry], col_index,
                                  ((const ivf64 *)sparse_matrix->hdr.data.ptr)[entry]);
    } else {   /* COO */
        for (size_t entry = 0; entry < sparse_matrix->nnz; entry++)
            fiv_dense_set_f64(dense_matrix,
                              (size_t)sparse_matrix->rowidx[entry],
                              (size_t)sparse_matrix->indices[entry],
                              ((const ivf64 *)sparse_matrix->hdr.data.ptr)[entry]);
    }
    return dense_matrix;
}


/* ------------------------------------------------------------------ */
/* Build: CSR -> CSC (transpose)                                      */
/* ------------------------------------------------------------------ */

fiv_ret fiv_sparse_transpose(fiv_sparse_mat **result_mat, const fiv_sparse_mat *source_mat)
{
    if (result_mat == NULL) return FIV_RET_ERR_PARA;
    *result_mat = NULL;
    if (source_mat == NULL) return FIV_RET_ERR_PARA;
    if (fiv_sparse_get_fmt(source_mat) != FIV_SPARSE_CSR) return FIV_RET_ERR_PARA;

    const size_t num_rows = source_mat->rows;
    const size_t num_cols = source_mat->cols;
    const size_t num_nonzeros = source_mat->nnz;

    fiv_sparse_mat *transpose_mat = fiv_create_sp_matrix(FIV_SPARSE_CSC, num_rows, num_cols,
                                                         num_nonzeros, source_mat->hdr.dtype);
    if (transpose_mat == NULL) return FIV_RET_ERR_MEM;

    int  *colptr = transpose_mat->indptr;
    int  *rowind = transpose_mat->indices;
    ivf64 *value_buffer = (ivf64 *)transpose_mat->hdr.data.ptr;

    /* Column counts -> prefix sum into colptr. */
    for (size_t col = 0; col < num_cols; col++) colptr[col] = 0;
    for (size_t entry = 0; entry < num_nonzeros; entry++) colptr[source_mat->indices[entry]]++;
    int col_accumulator = 0;
    for (size_t col = 0; col < num_cols; col++) {
        int col_count = colptr[col];
        colptr[col] = col_accumulator;
        col_accumulator += col_count;
    }
    colptr[num_cols] = col_accumulator;

    /* next_col tracks the next free slot per column while scattering. */
    int *next_col = (int *)fiv_malloc((num_cols ? num_cols : 1) * sizeof(int));
    if (next_col == NULL) { fiv_release_sp_matrix(&transpose_mat); return FIV_RET_ERR_MEM; }
    for (size_t col = 0; col < num_cols; col++) next_col[col] = colptr[col];

    const ivf64 *source_values = (const ivf64 *)source_mat->hdr.data.ptr;
    for (size_t row = 0; row < num_rows; row++) {
        for (int entry = source_mat->indptr[row]; entry < source_mat->indptr[row + 1]; entry++) {
            int col = source_mat->indices[entry];
            int dest_index = next_col[col]++;
            rowind[dest_index] = (int)row;
            value_buffer[dest_index] = source_values[entry];
        }
    }
    fiv_free(next_col);
    *result_mat = transpose_mat;
    return FIV_RET_OK;
}


/* ------------------------------------------------------------------ */
/* Release                                                            */
/* ------------------------------------------------------------------ */

fiv_ret fiv_release_sp_matrix_packed(fiv_sparse_mat *source_mat)
{
    if (source_mat == NULL || source_mat->packed == NULL) return FIV_RET_OK;
    fiv_sparse_unpack_csrl(&source_mat->packed->csrl);
    fiv_sparse_unpack_csrl(&source_mat->packed->csrlT);
    fiv_free(source_mat->packed);
    source_mat->packed = NULL;
    return FIV_RET_OK;
}

fiv_ret fiv_release_sp_matrix(fiv_sparse_mat **sparse_matrix)
{
    if (sparse_matrix == NULL || *sparse_matrix == NULL) return FIV_RET_OK;
    fiv_sparse_mat *matrix = *sparse_matrix;
    fiv_release_sp_matrix_packed(matrix);
    fiv_free(matrix->hdr.data.ptr);   /* values (may be NULL when nnz==0) */
    fiv_free(matrix->indptr);
    fiv_free(matrix->indices);
    fiv_free(matrix->rowidx);
    fiv_free(matrix);
    *sparse_matrix = NULL;
    return FIV_RET_OK;
}


/* ------------------------------------------------------------------ */
/* SpMV (scalar / auto-vectorized standard C)                         */
/* ------------------------------------------------------------------ */

fiv_ret fiv_sparse_matmul_vec(fiv_vec *output_vec, const fiv_sparse_mat *sparse_matrix,
                              const fiv_vec *input_vec, int transpose_flag)
{
    if (output_vec == NULL || sparse_matrix == NULL || input_vec == NULL) return FIV_RET_ERR_PARA;
    if (output_vec->dtype != FIV_64F1 || input_vec->dtype != FIV_64F1) return FIV_RET_ERR_PARA;
    if (sparse_matrix->hdr.dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;

    const fiv_sparse_fmt format = fiv_sparse_get_fmt(sparse_matrix);
    const ivf64 *input_data = (const ivf64 *)input_vec->data.ptr;
    ivf64 *output_data = (ivf64 *)output_vec->data.ptr;

    if (transpose_flag == 0) {
        /* output = A * input ; A in CSR or COO ; out len = rows, in len = cols */
        if (format != FIV_SPARSE_CSR && format != FIV_SPARSE_COO) return FIV_RET_ERR_PARA;
        if (output_vec->length != sparse_matrix->rows || input_vec->length != sparse_matrix->cols)
            return FIV_RET_ERR_PARA;
        for (size_t row = 0; row < sparse_matrix->rows; row++) output_data[row] = 0.0;
        if (format == FIV_SPARSE_CSR) {
            const ivf64 *matrix_values = (const ivf64 *)sparse_matrix->hdr.data.ptr;
            for (size_t row = 0; row < sparse_matrix->rows; row++) {
                ivf64 accumulator = 0.0;
                for (int entry = sparse_matrix->indptr[row]; entry < sparse_matrix->indptr[row + 1]; entry++)
                    accumulator += matrix_values[entry] * input_data[sparse_matrix->indices[entry]];
                output_data[row] = accumulator;
            }
        } else { /* COO: scatter-accumulate */
            for (size_t entry = 0; entry < sparse_matrix->nnz; entry++)
                output_data[sparse_matrix->rowidx[entry]] +=
                    sparse_matrix->hdr.data.db[entry] * input_data[sparse_matrix->indices[entry]];
        }
    } else {
        /* output = Aᵀ * input ; A in CSC or COO ; out len = cols, in len = rows */
        if (format != FIV_SPARSE_CSC && format != FIV_SPARSE_COO) return FIV_RET_ERR_PARA;
        if (output_vec->length != sparse_matrix->cols || input_vec->length != sparse_matrix->rows)
            return FIV_RET_ERR_PARA;
        for (size_t col = 0; col < sparse_matrix->cols; col++) output_data[col] = 0.0;
        if (format == FIV_SPARSE_CSC) {
            const ivf64 *matrix_values = (const ivf64 *)sparse_matrix->hdr.data.ptr;
            for (size_t col = 0; col < sparse_matrix->cols; col++) {
                ivf64 accumulator = 0.0;
                for (int entry = sparse_matrix->indptr[col]; entry < sparse_matrix->indptr[col + 1]; entry++)
                    accumulator += matrix_values[entry] * input_data[sparse_matrix->indices[entry]];
                output_data[col] = accumulator;
            }
        } else { /* COO: out indexed by colidx, in by rowidx */
            for (size_t entry = 0; entry < sparse_matrix->nnz; entry++)
                output_data[sparse_matrix->indices[entry]] +=
                    sparse_matrix->hdr.data.db[entry] * input_data[sparse_matrix->rowidx[entry]];
        }
    }
    return FIV_RET_OK;
}


/* ------------------------------------------------------------------ */
/* Reductions                                                         */
/* ------------------------------------------------------------------ */

fiv_ret fiv_sparse_reduce_abs_max(const fiv_sparse_mat *sparse_matrix, int dim, fiv_vec *output_vec)
{
    if (sparse_matrix == NULL || output_vec == NULL) return FIV_RET_ERR_PARA;
    if (output_vec->dtype != FIV_64F1) return FIV_RET_ERR_PARA;
    if (sparse_matrix->hdr.dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;
    const fiv_sparse_fmt format = fiv_sparse_get_fmt(sparse_matrix);

    if (dim == 0) {
        if (format != FIV_SPARSE_CSR) return FIV_RET_ERR_PARA;
        if (output_vec->length != sparse_matrix->rows) return FIV_RET_ERR_PARA;
        const ivf64 *matrix_values = (const ivf64 *)sparse_matrix->hdr.data.ptr;
        for (size_t row = 0; row < sparse_matrix->rows; row++) {
            ivf64 max_abs = 0.0;
            for (int entry = sparse_matrix->indptr[row]; entry < sparse_matrix->indptr[row + 1]; entry++) {
                ivf64 abs_value = fabs(matrix_values[entry]);
                if (abs_value > max_abs) max_abs = abs_value;
            }
            output_vec->data.db[row] = max_abs;
        }
    } else if (dim == 1) {
        if (format != FIV_SPARSE_CSC) return FIV_RET_ERR_PARA;
        if (output_vec->length != sparse_matrix->cols) return FIV_RET_ERR_PARA;
        const ivf64 *matrix_values = (const ivf64 *)sparse_matrix->hdr.data.ptr;
        for (size_t col = 0; col < sparse_matrix->cols; col++) {
            ivf64 max_abs = 0.0;
            for (int entry = sparse_matrix->indptr[col]; entry < sparse_matrix->indptr[col + 1]; entry++) {
                ivf64 abs_value = fabs(matrix_values[entry]);
                if (abs_value > max_abs) max_abs = abs_value;
            }
            output_vec->data.db[col] = max_abs;
        }
    } else {
        return FIV_RET_ERR_PARA;
    }
    return FIV_RET_OK;
}

fiv_ret fiv_sparse_reduce_pow_abs_sum(const fiv_sparse_mat *sparse_matrix, int dim,
                                      ivf64 exponent, fiv_vec *output_vec)
{
    if (sparse_matrix == NULL || output_vec == NULL) return FIV_RET_ERR_PARA;
    if (output_vec->dtype != FIV_64F1) return FIV_RET_ERR_PARA;
    if (sparse_matrix->hdr.dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;
    const fiv_sparse_fmt format = fiv_sparse_get_fmt(sparse_matrix);

    if (dim == 0) {
        if (format != FIV_SPARSE_CSR) return FIV_RET_ERR_PARA;
        if (output_vec->length != sparse_matrix->rows) return FIV_RET_ERR_PARA;
        const ivf64 *matrix_values = (const ivf64 *)sparse_matrix->hdr.data.ptr;
        for (size_t row = 0; row < sparse_matrix->rows; row++) {
            ivf64 sum_value = 0.0;
            for (int entry = sparse_matrix->indptr[row]; entry < sparse_matrix->indptr[row + 1]; entry++)
                sum_value += pow(fabs(matrix_values[entry]), exponent);
            output_vec->data.db[row] = sum_value;
        }
    } else if (dim == 1) {
        if (format != FIV_SPARSE_CSC) return FIV_RET_ERR_PARA;
        if (output_vec->length != sparse_matrix->cols) return FIV_RET_ERR_PARA;
        const ivf64 *matrix_values = (const ivf64 *)sparse_matrix->hdr.data.ptr;
        for (size_t col = 0; col < sparse_matrix->cols; col++) {
            ivf64 sum_value = 0.0;
            for (int entry = sparse_matrix->indptr[col]; entry < sparse_matrix->indptr[col + 1]; entry++)
                sum_value += pow(fabs(matrix_values[entry]), exponent);
            output_vec->data.db[col] = sum_value;
        }
    } else {
        return FIV_RET_ERR_PARA;
    }
    return FIV_RET_OK;
}


/* ------------------------------------------------------------------ */
/* CSRL packing                                                       */
/* ------------------------------------------------------------------ */

/* Generic pack: given (num_rows_dim, indptr[len+1], indices[len nnz]) where indices
 * are sorted and segments are runs of consecutive columns, fill a CSRL view.
 * ordered_values points at the value buffer in the SAME traversal order (CSR/CSC). */
static fiv_ret fiv_sparse_pack_generic(fiv_sparse_csrl *packed_view, size_t num_rows_dim,
                                       const int *rowptr, const int *colind,
                                       const ivf64 *ordered_values)
{
    size_t num_nonzeros = (rowptr[num_rows_dim] > 0) ? (size_t)rowptr[num_rows_dim] : 0;
    packed_view->rows = num_rows_dim;
    packed_view->cols = 0;       /* filled by caller (cols of original matrix) */
    packed_view->nz = num_nonzeros;
    packed_view->simd_lanes = s_runtime_state.simd_lanes;

    /* Pass 1: count segments. */
    size_t num_segments = 0;
    for (size_t row = 0; row < num_rows_dim; row++) {
        int row_start = rowptr[row], row_end = rowptr[row + 1];
        if (row_start == row_end) continue;
        num_segments++;
        for (int entry = row_start + 1; entry < row_end; entry++)
            if (colind[entry] != colind[entry - 1] + 1) num_segments++;
    }

    packed_view->nzseg = num_segments;
    packed_view->jas = (int *)fiv_malloc((num_segments ? num_segments : 1) * sizeof(int));
    packed_view->jan = (int *)fiv_malloc((num_segments ? num_segments : 1) * sizeof(int));
    packed_view->ptr = (int *)fiv_malloc((num_rows_dim + 1) * sizeof(int));
    ivf64 *packed_values = (ivf64 *)fiv_malloc((num_nonzeros ? num_nonzeros : 1) * sizeof(ivf64));
    if (packed_view->jas == NULL || packed_view->jan == NULL ||
        packed_view->ptr == NULL || packed_values == NULL) {
        fiv_free(packed_view->jas); fiv_free(packed_view->jan);
        fiv_free(packed_view->ptr); fiv_free(packed_values);
        packed_view->jas = packed_view->jan = packed_view->ptr = NULL;
        packed_view->hdr.data.ptr = NULL;
        return FIV_RET_ERR_MEM;
    }

    /* Pass 2: fill. */
    size_t segment = 0;
    for (size_t row = 0; row < num_rows_dim; row++) {
        packed_view->ptr[row] = (int)segment;
        int row_start = rowptr[row], row_end = rowptr[row + 1];
        if (row_start == row_end) continue;
        packed_view->jas[segment] = colind[row_start];
        packed_view->jan[segment] = 1;
        segment++;
        for (int entry = row_start + 1; entry < row_end; entry++) {
            if (colind[entry] == colind[entry - 1] + 1) {
                packed_view->jan[segment - 1]++;
            } else {
                packed_view->jas[segment] = colind[entry];
                packed_view->jan[segment] = 1;
                segment++;
            }
        }
    }
    packed_view->ptr[num_rows_dim] = (int)segment;

    /* Copy values in CSR/CSC traversal order. */
    for (size_t entry = 0; entry < num_nonzeros; entry++) packed_values[entry] = ordered_values[entry];

    packed_view->hdr.id = FIV_ID_TENSOR2D;
    packed_view->hdr.dtype = FIV_64F1;
    packed_view->hdr.data_continue = 1;
    packed_view->hdr.element_bytes = (iv8u)sizeof(ivf64);
    packed_view->hdr.data.ptr = packed_values;
    packed_view->hdr.total_bytes = num_nonzeros * sizeof(ivf64);
    return FIV_RET_OK;
}

fiv_ret fiv_sparse_pack_csrl(fiv_sparse_csrl *packed_view, const fiv_sparse_mat *source_mat)
{
    if (packed_view == NULL || source_mat == NULL) return FIV_RET_ERR_PARA;
    memset(packed_view, 0, sizeof(*packed_view));
    if (fiv_sparse_get_fmt(source_mat) != FIV_SPARSE_CSR) return FIV_RET_ERR_PARA;
    fiv_ret ret_status = fiv_sparse_pack_generic(packed_view, source_mat->rows,
                                                 source_mat->indptr, source_mat->indices,
                                                 (const ivf64 *)source_mat->hdr.data.ptr);
    if (ret_status == FIV_RET_OK) packed_view->cols = source_mat->cols;
    return ret_status;
}

fiv_ret fiv_sparse_pack_csccsrl(fiv_sparse_csccsrl *packed_view, const fiv_sparse_mat *source_mat)
{
    if (packed_view == NULL || source_mat == NULL) return FIV_RET_ERR_PARA;
    memset(packed_view, 0, sizeof(*packed_view));
    if (fiv_sparse_get_fmt(source_mat) != FIV_SPARSE_CSC) return FIV_RET_ERR_PARA;
    /* Treat CSC as "num_cols columns, colptr/rowind": dim = cols, indices = rowind. */
    fiv_ret ret_status = fiv_sparse_pack_generic(packed_view, source_mat->cols,
                                                 source_mat->indptr, source_mat->indices,
                                                 (const ivf64 *)source_mat->hdr.data.ptr);
    if (ret_status == FIV_RET_OK) packed_view->cols = source_mat->rows;   /* original row count */
    return ret_status;
}

fiv_ret fiv_sparse_unpack_csrl(fiv_sparse_csrl *packed_view)
{
    if (packed_view == NULL) return FIV_RET_OK;
    fiv_free(packed_view->jas);
    fiv_free(packed_view->jan);
    fiv_free(packed_view->ptr);
    fiv_free(packed_view->hdr.data.ptr);
    packed_view->jas = packed_view->jan = packed_view->ptr = NULL;
    packed_view->hdr.data.ptr = NULL;
    packed_view->nz = packed_view->nzseg = 0;
    packed_view->rows = packed_view->cols = 0;
    return FIV_RET_OK;
}

fiv_ret fiv_sparse_build_packed(fiv_sparse_mat *source_mat, const fiv_sparse_mat *transpose_mat)
{
    if (source_mat == NULL) return FIV_RET_ERR_PARA;
    fiv_release_sp_matrix_packed(source_mat);   /* idempotent */

    fiv_sparse_packed *packed_view = (fiv_sparse_packed *)fiv_malloc(sizeof(fiv_sparse_packed));
    if (packed_view == NULL) return FIV_RET_ERR_MEM;
    memset(packed_view, 0, sizeof(*packed_view));

    if (fiv_sparse_get_fmt(source_mat) == FIV_SPARSE_CSR) {
        fiv_ret ret_status = fiv_sparse_pack_csrl(&packed_view->csrl, source_mat);
        if (ret_status != FIV_RET_OK) { fiv_free(packed_view); return ret_status; }
        /* Degeneracy guard: poor locality -> skip SIMD view entirely. */
        ivf64 segment_avg = (packed_view->csrl.nzseg > 0)
            ? (ivf64)packed_view->csrl.nz / (ivf64)packed_view->csrl.nzseg : 0.0;
        if (segment_avg < FIV_SPARSE_DEGENERATE_MIN_SEG_AVG) {
            fiv_sparse_unpack_csrl(&packed_view->csrl);
            fiv_sparse_unpack_csrl(&packed_view->csrlT);
            fiv_free(packed_view);
            return FIV_RET_OK;   /* keep packed == NULL */
        }
    }

    if (transpose_mat != NULL && fiv_sparse_get_fmt(transpose_mat) == FIV_SPARSE_CSC) {
        fiv_ret ret_status = fiv_sparse_pack_csccsrl(&packed_view->csrlT, transpose_mat);
        if (ret_status != FIV_RET_OK) { fiv_release_sp_matrix_packed(source_mat); return ret_status; }
    }

    source_mat->packed = packed_view;
    return FIV_RET_OK;
}


/* ------------------------------------------------------------------ */
/* Packed SpMV (auto-vectorized standard C; segment-inner x is contiguous) */
/* ------------------------------------------------------------------ */

fiv_ret fiv_sparse_matmul_vec_packed(fiv_vec *output_vec, const fiv_sparse_packed *packed_view,
                                     const fiv_vec *input_vec, int transpose_flag)
{
    if (output_vec == NULL || packed_view == NULL || input_vec == NULL) return FIV_RET_ERR_PARA;
    if (output_vec->dtype != FIV_64F1 || input_vec->dtype != FIV_64F1) return FIV_RET_ERR_PARA;

    const ivf64 *input_data = (const ivf64 *)input_vec->data.ptr;
    ivf64 *output_data = (ivf64 *)output_vec->data.ptr;

    if (transpose_flag == 0) {
        const fiv_sparse_csrl *csrl_view = &packed_view->csrl;
        if (output_vec->length != csrl_view->rows || input_vec->length != csrl_view->cols)
            return FIV_RET_ERR_PARA;
        const ivf64 *matrix_values = (const ivf64 *)csrl_view->hdr.data.ptr;
        size_t value_base = 0;
        for (size_t row = 0; row < csrl_view->rows; row++) {
            ivf64 accumulator = 0.0;
            for (int segment = csrl_view->ptr[row]; segment < csrl_view->ptr[row + 1]; segment++) {
                int start_col = csrl_view->jas[segment];
                int segment_length = csrl_view->jan[segment];
                /* consecutive x load -> auto-vectorizable */
                for (int offset = 0; offset < segment_length; offset++)
                    accumulator += matrix_values[value_base + offset] * input_data[start_col + offset];
                value_base += (size_t)segment_length;
            }
            output_data[row] = accumulator;
        }
    } else {
        const fiv_sparse_csrl *csrl_view = &packed_view->csrlT;
        if (csrl_view->hdr.data.ptr == NULL) return FIV_RET_ERR_PARA;   /* csrlT not packed */
        if (output_vec->length != csrl_view->rows || input_vec->length != csrl_view->cols)
            return FIV_RET_ERR_PARA;
        const ivf64 *matrix_values = (const ivf64 *)csrl_view->hdr.data.ptr;
        size_t value_base = 0;
        for (size_t row = 0; row < csrl_view->rows; row++) {
            ivf64 accumulator = 0.0;
            for (int segment = csrl_view->ptr[row]; segment < csrl_view->ptr[row + 1]; segment++) {
                int start_col = csrl_view->jas[segment];
                int segment_length = csrl_view->jan[segment];
                for (int offset = 0; offset < segment_length; offset++)
                    accumulator += matrix_values[value_base + offset] * input_data[start_col + offset];
                value_base += (size_t)segment_length;
            }
            output_data[row] = accumulator;
        }
    }
    return FIV_RET_OK;
}
