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

#ifndef _FIV_SP_MATRIX_H_
#define _FIV_SP_MATRIX_H_

#include "fiv_ctensor.h"   /* fiv_tensor_hdr, fiv_vec, fiv_ret, fiv_data_type, fiv_scalar */

#ifdef __cplusplus
extern "C" {
#endif


/* =========================================================================
 *  Sparse matrix for PDLP (and general LP / SpMV).
 *
 *  Design mirrors FastIV's dense tensors (see fiv_ctensor.h):
 *    - The matrix embeds fiv_tensor_hdr as its FIRST member, so it can be cast
 *      to fiv_tensor_hdr* to read id / dtype / element_bytes / data.ptr /
 *      total_bytes. The numerical values live in hdr.data.ptr (a generic void*),
 *      interpreted according to hdr.dtype and hdr.element_bytes -- exactly like a
 *      dense tensor. The element TYPE is therefore NOT fixed in the struct; only
 *      FIV_64F1 is implemented today (kernels dispatch on hdr.dtype and reject
 *      other dtypes with FIV_RET_ERR_NOT_SUPPORT), but the storage definition is
 *      dtype-agnostic.
 *    - Shape (rows/cols/nnz) is declared locally because the generic header
 *      carries no shape. The matrix is its OWN structure, NOT a fiv_tensor2d
 *      (no strides, no "is-a" relation).
 *
 *  Storage layout (canonical):
 *    CSR  - row-compressed  (indptr = rowptr, indices = colind)   -> A x
 *    CSC  - col-compressed  (indptr = colptr, indices = rowind)   -> Aᵀ x
 *    COO  - triplet         (indptr = NULL, indices = colidx, rowidx = row)
 *
 *  The SIMD-friendly CSRL packed view (Liu Fangfang et al. 2014) is an OPTIONAL
 *  acceleration layer hung off fiv_sparse_mat->packed; correctness never depends
 *  on it (a pure standard-C scalar / auto-vectorized path is always available).
 *
 *  Lifecycle follows FastIV's create/release convention: every fiv_create_sp_matrix*
 *  is paired with fiv_release_sp_matrix, which frees all owned buffers and the
 *  struct itself.
 * ========================================================================= */

typedef FIV_ENUM(iv8u) {
    FIV_SPARSE_CSR = 0,
    FIV_SPARSE_CSC,
    FIV_SPARSE_COO
} fiv_sparse_fmt;


/* CSRL packed view (Compressed Sparse Row with Local information).
 * Within a row, consecutive column indices form a "segment"; the segment's
 * columns are contiguous so x[c..c+L) is a contiguous load -> SIMD friendly.
 * Values are stored in hdr.data.ptr in CSR traversal order (no column array is
 * needed; col = jas + k is implied). Like the matrix, the element type is carried
 * by hdr.dtype / hdr.element_bytes rather than baked into the struct. */
typedef struct {
    fiv_tensor_hdr hdr;     /* shared prefix; hdr.data.ptr -> values (void*);
                              hdr.dtype / hdr.element_bytes carry the element type */
    size_t rows;
    size_t cols;
    size_t nz;              /* #nonzeros (== fiv_sparse_mat.nnz)                */
    size_t nzseg;          /* #segments (total across all rows)                */
    int    simd_lanes;      /* detected f64 SIMD lanes (1 / 2 / 4 / 8)          */
    int   *jas;             /* [nzseg] segment start column index               */
    int   *jan;             /* [nzseg] segment length (>=1)                     */
    int   *ptr;             /* [rows+1] ptr[row] = first segment of row; ptr[rows]=nzseg */
} fiv_sparse_csrl;

/* CSC-CSRL: the same struct, used for Kᵀ y (columns carry row-index segments). */
typedef fiv_sparse_csrl fiv_sparse_csccsrl;

/* Optional SIMD view attached to a fiv_sparse_mat. */
typedef struct {
    fiv_sparse_csrl    csrl;    /* K  (CSR) packed view   */
    fiv_sparse_csccsrl csrlT;   /* Kᵀ (CSC) packed view   */
} fiv_sparse_packed;


/* The sparse matrix. Shares ONLY fiv_tensor_hdr as its first member. */
typedef struct {
    fiv_tensor_hdr hdr;        /* id = FIV_ID_TENSOR2D;
                                  dtype = element type (FIV_64F1 today);
                                  data.ptr -> values (void*);
                                  element_bytes / total_bytes carry size */
    size_t rows;               /* matrix row count (header has none) */
    size_t cols;               /* matrix column count */
    size_t nnz;                /* #nonzeros: value buffer length = nnz */
    int   *indptr;             /* CSR: rowptr; CSC: colptr; COO: NULL */
    int   *indices;            /* CSR: colind; CSC: rowind; COO: colidx */
    int   *rowidx;             /* COO only (row index); other formats NULL */
    fiv_sparse_packed *packed; /* optional CSRL view (default NULL) */
} fiv_sparse_mat;


/* Runtime SIMD probe result. */
typedef struct {
    int simd_lanes;       /* f64 SIMD lanes: NEON=2 / AVX2=4 / AVX512=8 / scalar=1 */
    int use_intrinsics;   /* 0 = auto-vectorized standard-C path (default)        */
} fiv_sparse_runtime;


/* ============================ Query ============================ */

/* Read the storage format from the generic header's color_space_type byte. */
fiv_sparse_fmt fiv_sparse_get_fmt(const fiv_sparse_mat *sparse_matrix);


/* ============================ Build / release ============================ */

/* Allocate an EMPTY sparse matrix with the given format, shape, nnz and dtype.
 * All owned buffers (indptr / indices / rowidx / values) are allocated and
 * zeroed; values live in hdr.data.ptr. This is the bare allocator that mirrors
 * fiv_create_tensor1d -- the caller fills the buffers, or uses one of the
 * factories below. Returns NULL on allocation failure or unsupported dtype. */
fiv_sparse_mat *fiv_create_sp_matrix(fiv_sparse_fmt format, size_t num_rows,
                                     size_t num_cols, size_t num_nonzeros,
                                     fiv_data_type value_dtype);

/* Build a CSR matrix from COO triplets. Entries with identical (row,col) are
 * ACCUMULATED (so repeated contributions, e.g. from MPS sections, merge).
 * values points at num_nonzeros elements of value_dtype; only FIV_64F1 is
 * supported today (others return NULL). Returns the new matrix, or NULL on
 * error; release it with fiv_release_sp_matrix. */
fiv_sparse_mat *fiv_create_sp_matrix_from_coo(const int *row_indices, const int *col_indices,
                                              const void *values, fiv_data_type value_dtype,
                                              size_t num_nonzeros, size_t num_rows, size_t num_cols);

/* Convert a dense matrix (fiv_mat, row-major, honoring its strides) into a CSR
 * sparse matrix. Exact zeros are dropped. The result always carries FIV_64F1
 * (PDLP's working precision); a FIV_32F1 dense input is widened. Unsupported
 * dense dtypes return NULL. Release the result with fiv_release_sp_matrix. */
fiv_sparse_mat *fiv_create_sp_matrix_from_dense(const fiv_mat *dense_matrix);

/* Convert a sparse matrix (CSR / CSC / COO) into a dense fiv_mat (row-major,
 * FIV_64F1), zero-initialized then scattered. Release the result with
 * fiv_release_tensor2d. */
fiv_mat *fiv_create_dense_matrix_from_sp_matrix(const fiv_sparse_mat *sparse_matrix);

/* Materialize the transpose of a CSR matrix into an independent CSC (own
 * buffers, values reordered). Used once to get Kᵀ for Kᵀ y. *result_mat is set
 * to the new matrix on FIV_RET_OK (release with fiv_release_sp_matrix); on
 * error it is set to NULL. */
fiv_ret fiv_sparse_transpose(fiv_sparse_mat **result_mat, const fiv_sparse_mat *source_mat);

/* Release all buffers owned by *sparse_matrix (values / indptr / indices /
 * rowidx / packed) and the struct itself; *sparse_matrix is set to NULL.
 * Safe when *sparse_matrix == NULL. Pairs with fiv_create_sp_matrix* . */
fiv_ret fiv_release_sp_matrix(fiv_sparse_mat **sparse_matrix);


/* ============================ SpMV ============================ */

/* output_vec = A * input_vec (transpose_flag==0, A must be CSR/COO)  |
 * output_vec = Aᵀ * input_vec (transpose_flag!=0, A must be CSC/COO).
 * All FIV_64F1; output/input lengths are validated against the matrix shape,
 * and format/transpose mismatch returns FIV_RET_ERR_PARA. */
fiv_ret fiv_sparse_matmul_vec(fiv_vec *output_vec, const fiv_sparse_mat *sparse_matrix,
                              const fiv_vec *input_vec, int transpose_flag);


/* ============================ Reductions ============================ */

/* output_vec = max over entries of |value| along dim. dim==0 -> per row (CSR only);
 * dim==1 -> per column (CSC only). Other format/dim combos return FIV_RET_ERR_PARA. */
fiv_ret fiv_sparse_reduce_abs_max(const fiv_sparse_mat *sparse_matrix, int dim, fiv_vec *output_vec);

/* output_vec = sum over entries of |value|^exponent along dim. Same dim/format rule
 * as reduce_abs_max. Used by Ruiz (exponent=1) and Pock-Chambolle (exponent=2-alpha). */
fiv_ret fiv_sparse_reduce_pow_abs_sum(const fiv_sparse_mat *sparse_matrix, int dim,
                                      ivf64 exponent, fiv_vec *output_vec);


/* ============================ CSRL packed view ============================ */

/* Pack a CSR matrix into its CSRL view (values copied in CSR order). */
fiv_ret fiv_sparse_pack_csrl(fiv_sparse_csrl *packed_view, const fiv_sparse_mat *source_mat);

/* Pack a CSC matrix into its CSC-CSRL view (values copied in CSC order). */
fiv_ret fiv_sparse_pack_csccsrl(fiv_sparse_csccsrl *packed_view, const fiv_sparse_mat *source_mat);

/* Free the buffers inside a single CSRL view (not the view struct itself). */
fiv_ret fiv_sparse_unpack_csrl(fiv_sparse_csrl *packed_view);

/* Build K->packed from K (CSR) and optional KT (CSC). If the matrix has poor
 * locality (avg segment length nz/nzseg < FIV_SPARSE_DEGENERATE_MIN_SEG_AVG)
 * packed is left NULL and the scalar path is used. Idempotent. */
fiv_ret fiv_sparse_build_packed(fiv_sparse_mat *source_mat, const fiv_sparse_mat *transpose_mat);

/* Free K->packed (both views) and set it NULL. Pairs with fiv_sparse_build_packed. */
fiv_ret fiv_release_sp_matrix_packed(fiv_sparse_mat *source_mat);

/* Preferred SpMV: uses the packed CSRL view when present (transpose_flag selects
 * csrl / csrlT), otherwise falls back to fiv_sparse_matmul_vec scalar path. */
fiv_ret fiv_sparse_matmul_vec_packed(fiv_vec *output_vec, const fiv_sparse_packed *packed_view,
                                     const fiv_vec *input_vec, int transpose_flag);


/* ============================ Runtime ============================ */

/* Detect SIMD lane count from compile-time macros; call once at startup. */
fiv_ret fiv_sparse_runtime_init(void);

/* Read the runtime probe (valid after fiv_sparse_runtime_init). *runtime_info is
 * set to the process-wide probe; returns FIV_RET_ERR_PARA if runtime_info is NULL. */
fiv_ret fiv_sparse_runtime_get(const fiv_sparse_runtime **runtime_info);


/* Below this avg segment length, CSRL buys nothing (metadata overhead dominates)
   and we keep packed == NULL to use the scalar CSR/CSC path. */
#define FIV_SPARSE_DEGENERATE_MIN_SEG_AVG 1.5


#ifdef __cplusplus
}
#endif

#endif  /* _FIV_SP_MATRIX_H_ */
