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

/* 64-bit (ivf64 / double) matrix multiplication. Logic is copied verbatim from
   fiv_mat_mul.c (the float32 path): the same four transpose variants, the
   small (non-blocked) vs blocked dispatch, the cache-blocked panel packing,
   and the public-API contract (error checks, beta/alpha handling, in-place
   prohibition). The float32 path relies on width-specific SIMD micro-kernels;
   here the micro-kernels are scalar loops, so the algorithmic logic is fully
   preserved while the element type is ivf64. */

#include "fiv_mat_mul_db.h"
#include "fiv_common.h"

#include <string.h>   /* memset */

/* ============================================================================
   Small (non-blocked) matrix kernels. C = alpha * op(A) * op(B) + beta * C.
   ivf64 throughout; strides are in elements.
   ========================================================================== */

/* C = A * B, A: rows_a x cols_a, B: cols_a x cols_b */
static void fiv_small_matrix_mul_matrix_real64(
    ivf64* data_a, int rows_a, int cols_a, int stride_a,
    ivf64* data_b, int cols_b, int stride_b,
    ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta)
{
    for (int i = 0; i < rows_a; i++) {
        ivf64* ptr_a = &data_a[i * stride_a];
        ivf64* ptr_c = &data_c[i * stride_c];
        if (beta == 0.0) {
            memset(ptr_c, 0, sizeof(ivf64) * cols_b);
        } else if (beta != 1.0) {
            for (int l = 0; l < cols_b; l++) ptr_c[l] *= beta;
        }
        for (int k = 0; k < cols_a; k++) {
            ivf64 t_a_d = ptr_a[k] * alpha;
            ivf64* ptr_b = &data_b[k * stride_b];
            for (int j = 0; j < cols_b; j++) ptr_c[j] += t_a_d * ptr_b[j];
        }
    }
}

/* C = A * B^T, A: rows_a x cols_a, B: rows_b x cols_a, result rows_a x rows_b */
static void fiv_small_matrix_mul_matrix_t_real64(
    ivf64* data_a, int rows_a, int cols_a, int stride_a,
    ivf64* data_b, int rows_b, int stride_b,
    ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta)
{
    for (int i = 0; i < rows_a; i++) {
        ivf64* ptr_a = &data_a[i * stride_a];
        ivf64* ptr_c = &data_c[i * stride_c];
        if (beta == 0.0) {
            memset(ptr_c, 0, sizeof(ivf64) * rows_b);
        } else if (beta != 1.0) {
            for (int l = 0; l < rows_b; l++) ptr_c[l] *= beta;
        }
        for (int k = 0; k < cols_a; k++) {
            ivf64 t_a_d = ptr_a[k] * alpha;
            for (int j = 0; j < rows_b; j++) ptr_c[j] += t_a_d * data_b[j * stride_b + k];
        }
    }
}

/* C = A^T * B, A: rows_a x cols_a, B: rows_a x cols_b, result cols_a x cols_b */
static void fiv_small_matrix_t_mul_matrix_real64(
    ivf64* data_a, int rows_a, int cols_a, int stride_a,
    ivf64* data_b, int cols_b, int stride_b,
    ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta)
{
    for (int i = 0; i < cols_a; i++) {
        ivf64* ptr_c = &data_c[i * stride_c];
        if (beta == 0.0) {
            memset(ptr_c, 0, sizeof(ivf64) * cols_b);
        } else if (beta != 1.0) {
            for (int l = 0; l < cols_b; l++) ptr_c[l] *= beta;
        }
        for (int k = 0; k < rows_a; k++) {
            ivf64 t_a_d = data_a[k * stride_a + i] * alpha;
            ivf64* ptr_b = &data_b[k * stride_b];
            for (int j = 0; j < cols_b; j++) ptr_c[j] += t_a_d * ptr_b[j];
        }
    }
}

/* C = A^T * B^T, A: rows_a x cols_a, B: rows_b x cols_a, result cols_a x rows_b */
static void fiv_small_matrix_t_mul_matrix_t_real64(
    ivf64* data_a, int rows_a, int cols_a, int stride_a,
    ivf64* data_b, int rows_b, int stride_b,
    ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta)
{
    for (int i = 0; i < cols_a; i++) {
        ivf64* ptr_c = &data_c[i * stride_c];
        if (beta == 0.0) {
            memset(ptr_c, 0, sizeof(ivf64) * rows_b);
        } else if (beta != 1.0) {
            for (int l = 0; l < rows_b; l++) ptr_c[l] *= beta;
        }
        for (int k = 0; k < rows_a; k++) {
            ivf64 t_a_d = data_a[k * stride_a + i] * alpha;
            for (int j = 0; j < rows_b; j++) ptr_c[j] += t_a_d * data_b[j * stride_b + k];
        }
    }
}

/* ============================================================================
   Blocked matrix multiplication (ivf64). Panel packing and the 8x8 micro-kernel
   mirror the float32 blocked path; the micro-kernel is a scalar loop.
   ========================================================================== */

typedef void(*ptr_func_mat_mul_mxkxn_kernel_db)
    (int kc, ivf64 alpha, ivf64* a, ivf64* b, ivf64 beta, ivf64* c, int inc_row_c, int inc_col_c);

/* Blocking derived from the L3 cache size (see float32 path); the K budget
   divisor is 8 bytes/element for ivf64 instead of 4. */
#ifndef FIV_L3_CACHE_BYTES
#define FIV_L3_CACHE_BYTES (8 * 1024 * 1024)
#endif
#if _DEBUG
#define FIV_M_BLOCK_DB 32
#define FIV_K_BLOCK_DB 32
#define FIV_N_BLOCK_DB 32
#else
#define FIV_M_BLOCK_DB 512
#define FIV_N_BLOCK_DB 480
#define FIV_BLOCK_K_CALC_DB() \
    ((((FIV_L3_CACHE_BYTES / 2) / (8 * (FIV_M_BLOCK_DB + FIV_N_BLOCK_DB))) / 128) * 128)
#define FIV_K_BLOCK_DB (FIV_BLOCK_K_CALC_DB() < 64 ? 64 : FIV_BLOCK_K_CALC_DB())
#endif

static void copy_mrxk_blocked_db(
    int k, int kernel_m, ivf64* a, int inc_row_a, int inc_col_a, ivf64* buffer)
{
    int i, j;
    for (j = 0; j < k; j++) {
        for (i = 0; i < kernel_m; i++) buffer[i] = a[i * inc_row_a];
        buffer += kernel_m;
        a += inc_col_a;
    }
}

static void copy_a_blocked_db(int mc, int kc, int kernel_m, ivf64* a, int inc_row_a, int inc_col_a, ivf64* buffer)
{
    int mp = mc / kernel_m;
    int mr = mc % kernel_m;
    int i, j;
    for (i = 0; i < mp; i++) {
        copy_mrxk_blocked_db(kc, kernel_m, a, inc_row_a, inc_col_a, buffer);
        buffer += kc * kernel_m;
        a += kernel_m * inc_row_a;
    }
    if (mr > 0) {
        for (j = 0; j < kc; j++) {
            for (i = 0; i < mr; i++) buffer[i] = a[i * inc_row_a];
            for (i = mr; i < kernel_m; i++) buffer[i] = 0.0;
            buffer += kernel_m;
            a += inc_col_a;
        }
    }
}

static void copy_nrxk_blocked_db(int k, int kernel_n, ivf64* b, int inc_row_b, int inc_col_b, ivf64* buffer)
{
    int i, j;
    for (i = 0; i < k; i++) {
        for (j = 0; j < kernel_n; j++) buffer[j] = b[j * inc_col_b];
        buffer += kernel_n;
        b += inc_row_b;
    }
}

static void copy_b_blocked_db(int kc, int nc, int kernel_n, ivf64* b, int inc_row_b, int inc_col_b, ivf64* buffer)
{
    int np = nc / kernel_n;
    int nr = nc % kernel_n;
    int i, j;
    for (j = 0; j < np; j++) {
        copy_nrxk_blocked_db(kc, kernel_n, b, inc_row_b, inc_col_b, buffer);
        buffer += kc * kernel_n;
        b += kernel_n * inc_col_b;
    }
    if (nr > 0) {
        for (i = 0; i < kc; i++) {
            for (j = 0; j < nr; j++) buffer[j] = b[j * inc_col_b];
            for (j = nr; j < kernel_n; j++) buffer[j] = 0.0;
            buffer += kernel_n;
            b += inc_row_b;
        }
    }
}

#define FIV_KERNEL_M_8_DB  (8)
#define FIV_KERNEL_N_8_DB  (8)

/* 8x8 scalar micro-kernel: C_tile += alpha * A_panel(8xkc) * B_panel(kc x 8) + beta * C_tile.
   c is addressed with (row=j, col=i) in the tile via c[j*inc_row_c + i*inc_col_c],
   matching the float32 micro-kernel so the packing/layout stays identical. */
static void mat_mul_8xkx8_kernel_db(
    int kc, ivf64 alpha, ivf64* a, ivf64* b, ivf64 beta,
    ivf64* c, int inc_row_c, int inc_col_c)
{
    ivf64 ab[FIV_KERNEL_M_8_DB * FIV_KERNEL_N_8_DB];
    int i, j, l;
    memset(ab, 0, sizeof(ivf64) * FIV_KERNEL_M_8_DB * FIV_KERNEL_N_8_DB);
    for (l = 0; l < kc; l++) {
        for (j = 0; j < FIV_KERNEL_N_8_DB; j++) {
            for (i = 0; i < FIV_KERNEL_M_8_DB; i++) {
                ab[i * FIV_KERNEL_M_8_DB + j] += a[i] * b[j];
            }
        }
        a += FIV_KERNEL_M_8_DB;
        b += FIV_KERNEL_N_8_DB;
    }

    if (beta == 0.0) {
        for (j = 0; j < FIV_KERNEL_N_8_DB; j++)
            for (i = 0; i < FIV_KERNEL_M_8_DB; i++)
                c[j * inc_row_c + i * inc_col_c] = 0.0;
    } else if (beta != 1.0) {
        for (j = 0; j < FIV_KERNEL_N_8_DB; j++)
            for (i = 0; i < FIV_KERNEL_M_8_DB; i++)
                c[j * inc_row_c + i * inc_col_c] *= beta;
    }

    if (alpha == 1.0) {
        for (j = 0; j < FIV_KERNEL_N_8_DB; j++)
            for (i = 0; i < FIV_KERNEL_M_8_DB; i++)
                c[j * inc_row_c + i * inc_col_c] += ab[i + j * FIV_KERNEL_M_8_DB];
    } else {
        for (j = 0; j < FIV_KERNEL_N_8_DB; j++)
            for (i = 0; i < FIV_KERNEL_M_8_DB; i++)
                c[j * inc_row_c + i * inc_col_c] += alpha * ab[i + j * FIV_KERNEL_M_8_DB];
    }
}

#undef FIV_KERNEL_M_8_DB
#undef FIV_KERNEL_N_8_DB

#define FIV_MAX_KERNEL_SIZE_DB 96   /* >= 8*8, holds the remainder tile */

static void dgeaxpy_row_major_db(
    int m, int n, ivf64 alpha,
    ivf64* x, int inc_row_x, int inc_col_x,
    ivf64* y, int inc_row_y, int inc_col_y)
{
    int i, j;
    if (alpha != 1.0) {
        for (i = 0; i < m; i++)
            for (j = 0; j < n; j++)
                y[i * inc_row_y + j * inc_col_y] += alpha * x[i * inc_row_x + j * inc_col_x];
    } else {
        for (i = 0; i < m; i++)
            for (j = 0; j < n; j++)
                y[i * inc_row_y + j * inc_col_y] += x[i * inc_row_x + j * inc_col_x];
    }
}

static void dgescal_row_major_db(
    int m, int n, ivf64 alpha,
    ivf64* x, int inc_row_x, int inc_col_x)
{
    int i, j;
    if (alpha != 1.0) {
        for (i = 0; i < m; i++)
            for (j = 0; j < n; j++)
                x[i * inc_row_x + j * inc_col_x] *= alpha;
    }
}

static void mat_mul_kernel_row_major_db(
    int mc, int nc, int kc,
    ivf64 alpha, ivf64 beta,
    ivf64* c, int inc_row_c, int inc_col_c,
    ivf64* blocked_a, ivf64* blocked_b,
    int kernel_m_size, int kernel_n_size,
    ptr_func_mat_mul_mxkxn_kernel_db kernel)
{
    int mp = (mc + kernel_m_size - 1) / kernel_m_size;
    int np = (nc + kernel_n_size - 1) / kernel_n_size;
    int _mr = mc % kernel_m_size;
    int _nr = nc % kernel_n_size;
    int i;
    for (i = 0; i < mp; i++) {
        int j, mr, nr;
        ivf64* ptr_a = &blocked_a[i * kc * kernel_m_size];
        ivf64* ptr_c = &c[i * kernel_m_size * inc_row_c];
        mr = (i != mp - 1 || _mr == 0) ? kernel_m_size : _mr;
        for (j = 0; j < np; j++) {
            ivf64* ptr_b_j = &blocked_b[j * kc * kernel_n_size];
            ivf64* ptr_c_j = &ptr_c[j * kernel_n_size * inc_col_c];
            nr = (j != np - 1 || _nr == 0) ? kernel_n_size : _nr;
            if (mr == kernel_m_size && nr == kernel_n_size) {
                kernel(kc, alpha, ptr_a, ptr_b_j, beta, ptr_c_j, inc_row_c, inc_col_c);
            } else {
                ivf64 blocked_c[FIV_MAX_KERNEL_SIZE_DB];
                kernel(kc, alpha, ptr_a, ptr_b_j, 0.0, blocked_c, 1, kernel_m_size);
                dgescal_row_major_db(mr, nr, beta, ptr_c_j, inc_row_c, inc_col_c);
                dgeaxpy_row_major_db(mr, nr, 1.0, blocked_c, 1, kernel_m_size, ptr_c_j, inc_row_c, inc_col_c);
            }
        }
    }
}

static void blocked_mat_mul_row_major_real64(
    int m, int n, int k,
    ivf64 alpha,
    ivf64* a, int inc_row_a, int inc_col_a,
    ivf64* b, int inc_row_b, int inc_col_b,
    ivf64 beta,
    ivf64* c, int inc_row_c, int inc_col_c)
{
    int mb = (m + FIV_M_BLOCK_DB - 1) / FIV_M_BLOCK_DB;
    int nb = (n + FIV_N_BLOCK_DB - 1) / FIV_N_BLOCK_DB;
    int kb = (k + FIV_K_BLOCK_DB - 1) / FIV_K_BLOCK_DB;
    int _mc = m % FIV_M_BLOCK_DB;
    int _nc = n % FIV_N_BLOCK_DB;
    int _kc = k % FIV_K_BLOCK_DB;

    if (alpha == 0.0 || k == 0) {
        dgescal_row_major_db(m, n, beta, c, inc_row_c, inc_col_c);
        return;
    }

    int mc, nc, kc, i, j, l;
    int kernel_m_size = 8, kernel_n_size = 8;
    ptr_func_mat_mul_mxkxn_kernel_db kernel_func = mat_mul_8xkx8_kernel_db;

    int a_zero_rest = FIV_M_BLOCK_DB % kernel_m_size == 0 ? 0 : kernel_m_size - FIV_M_BLOCK_DB % kernel_m_size;
    int b_zero_rest = FIV_N_BLOCK_DB % kernel_n_size == 0 ? 0 : kernel_n_size - FIV_N_BLOCK_DB % kernel_n_size;

    ivf64* blocked_a = (ivf64*)fiv_malloc(sizeof(ivf64) * (FIV_M_BLOCK_DB + a_zero_rest) * FIV_K_BLOCK_DB);
    ivf64* blocked_b = (ivf64*)fiv_malloc(sizeof(ivf64) * (FIV_N_BLOCK_DB + b_zero_rest) * FIV_K_BLOCK_DB);
    if (blocked_a && blocked_b) {
        for (i = 0; i < mb; i++) {
            ivf64* ptr_a = &a[i * FIV_M_BLOCK_DB * inc_row_a];
            ivf64* ptr_c = &c[i * FIV_M_BLOCK_DB * inc_row_c];
            mc = (i != mb - 1 || _mc == 0) ? FIV_M_BLOCK_DB : _mc;
            for (l = 0; l < kb; l++) {
                ivf64* ptr_a_l = &ptr_a[l * FIV_K_BLOCK_DB * inc_col_a];
                ivf64* ptr_b   = &b[l * FIV_K_BLOCK_DB * inc_row_b];
                kc = (l != kb - 1 || _kc == 0) ? FIV_K_BLOCK_DB : _kc;
                ivf64 _beta = (l == 0) ? beta : 1.0;
                copy_a_blocked_db(mc, kc, kernel_m_size, ptr_a_l, inc_row_a, inc_col_a, blocked_a);
                for (j = 0; j < nb; j++) {
                    ivf64* ptr_c_j = &ptr_c[j * FIV_N_BLOCK_DB * inc_col_c];
                    ivf64* ptr_b_j = &ptr_b[j * FIV_N_BLOCK_DB * inc_col_b];
                    nc = (j != nb - 1 || _nc == 0) ? FIV_N_BLOCK_DB : _nc;
                    copy_b_blocked_db(kc, nc, kernel_n_size, ptr_b_j, inc_row_b, inc_col_b, blocked_b);
                    mat_mul_kernel_row_major_db(
                        mc, nc, kc, alpha, _beta,
                        ptr_c_j, inc_row_c, inc_col_c,
                        blocked_a, blocked_b,
                        kernel_m_size, kernel_n_size,
                        kernel_func);
                }
            }
        }
    }

    fiv_free(blocked_b);
    fiv_free(blocked_a);
}

/* Blocked GEMM entry (ivf64). a_t/b_t: 1 means the operand is used transposed.
   m/n/k are the effective dims of op(A)/op(B)/op(C). */
static void fiv_matrix_mul_blocked_real64(
    int a_t, int b_t, int m, int n, int k,
    ivf64 alpha,
    ivf64* a, int lda,
    ivf64* b, int ldb,
    ivf64 beta,
    ivf64* c, int ldc)
{
    int i, j;
    if (m <= 0 || n <= 0 || ((alpha == 0.0 || k <= 0) && (beta == 1.0))) {
        return;
    }
    if (alpha == 0.0) {
        if (beta == 0.0) {
            for (j = 0; j < n; j++)
                for (i = 0; i < m; i++)
                    c[j * ldc + i] = 0.0;
        } else {
            for (j = 0; j < n; j++)
                for (i = 0; i < m; i++)
                    c[j * ldc + i] *= beta;
        }
        return;
    }
    if (a_t == 0 && b_t == 0) {
        blocked_mat_mul_row_major_real64(m, n, k, alpha, a, lda, 1, b, ldb, 1, beta, c, ldc, 1);
    } else if (a_t && b_t == 0) {
        blocked_mat_mul_row_major_real64(m, n, k, alpha, a, 1, lda, b, ldb, 1, beta, c, ldc, 1);
    } else if (a_t == 0 && b_t) {
        blocked_mat_mul_row_major_real64(m, n, k, alpha, a, lda, 1, b, 1, ldb, beta, c, ldc, 1);
    } else {
        blocked_mat_mul_row_major_real64(m, n, k, alpha, a, 1, lda, b, 1, ldb, beta, c, ldc, 1);
    }
}

/* ============================================================================
   64-bit (ivf64 / double) full API: dst = alpha * op(A) * op(B) + beta * dst.
   This is the dtype-specific backend invoked by the generic fiv_matrix_mul
   (api/fiv_matrix.h) when the operands are FIV_64F1; it is not a standalone
   public interface. Dispatched to the small (non-blocked) path when A+B+C fit
   in the L3-cache budget, otherwise to the blocked path.
   ========================================================================== */

fiv_ret fiv_matrix_mul_real64(fiv_mat* dst, const fiv_mat* A, const fiv_mat* B,
                              int a_transpose, int b_transpose, fiv_scalar alpha, fiv_scalar beta)
{
    if (dst == NULL || A == NULL || B == NULL) return FIV_RET_ERR_PARA;
    if (dst->data.ptr == NULL || A->data.ptr == NULL || B->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (dst->data_continue == 0 || A->data_continue == 0 || B->data_continue == 0) return FIV_RET_ERR_PARA;
    if (A->dtype != FIV_64F1 || B->dtype != FIV_64F1 || dst->dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (alpha.id != FIV_ID_SCALAR || alpha.dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (beta.id != FIV_ID_SCALAR || beta.dtype != FIV_64F1) return FIV_RET_ERR_NOT_SUPPORT;
    ivf64 alpha_f = alpha.data.value_fp64;
    ivf64 beta_f  = beta.data.value_fp64;
    if (dst->data.ptr == A->data.ptr || dst->data.ptr == B->data.ptr) return FIV_RET_ERR_PARA;

    const int ra = (int)A->shapes[0];
    const int ca = (int)A->shapes[1];
    const int rb = (int)B->shapes[0];
    const int cb = (int)B->shapes[1];
    if (ra <= 0 || ca <= 0 || rb <= 0 || cb <= 0) return FIV_RET_ERR_PARA;

    const int M  = a_transpose ? ca : ra;
    const int N  = b_transpose ? rb : cb;
    const int K  = a_transpose ? ra : ca;    /* cols of op(A) */
    const int Kb = b_transpose ? cb : rb;    /* rows of op(B) */
    if (K != Kb) return FIV_RET_ERR_PARA;

    if ((int)dst->shapes[0] != M || (int)dst->shapes[1] != N) return FIV_RET_ERR_PARA;
    if (dst->total_bytes < (size_t)M * (size_t)N * (size_t)dst->element_bytes) return FIV_RET_ERR_PARA;

    ivf64* a = (ivf64*)A->data.ptr;
    ivf64* b = (ivf64*)B->data.ptr;
    ivf64* c = (ivf64*)dst->data.ptr;

    const size_t ws_bytes = ((size_t)ra * (size_t)ca + (size_t)rb * (size_t)cb +
                             (size_t)M * (size_t)N) * (size_t)A->element_bytes;

    if (ws_bytes <= FIV_MAT_MUL_DB_L3_LIMIT_BYTES) {
        if (!a_transpose && !b_transpose)
            fiv_small_matrix_mul_matrix_real64(a, ra, ca, ca, b, cb, cb, c, N, alpha_f, beta_f);
        else if (!a_transpose && b_transpose)
            fiv_small_matrix_mul_matrix_t_real64(a, ra, ca, ca, b, rb, cb, c, N, alpha_f, beta_f);
        else if (a_transpose && !b_transpose)
            fiv_small_matrix_t_mul_matrix_real64(a, ra, ca, ca, b, cb, cb, c, N, alpha_f, beta_f);
        else
            fiv_small_matrix_t_mul_matrix_t_real64(a, ra, ca, ca, b, rb, cb, c, N, alpha_f, beta_f);
    } else {
        fiv_matrix_mul_blocked_real64(a_transpose, b_transpose, M, N, K, alpha_f, a, ca, b, cb, beta_f, c, N);
    }

    dst->shapes[0]   = (size_t)M;
    dst->shapes[1]   = (size_t)N;
    dst->strides[0]  = (size_t)N * (size_t)dst->element_bytes;
    dst->strides[1]  = (size_t)dst->element_bytes;
    dst->total_bytes = (size_t)M * (size_t)N * (size_t)dst->element_bytes;
    dst->data_continue = 1;
    return FIV_RET_OK;
}
