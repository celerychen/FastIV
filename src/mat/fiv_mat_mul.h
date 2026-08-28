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

#ifndef _FIV_MAT_MUL_H_
#define _FIV_MAT_MUL_H_

#include "fiv_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Working set (A + B + C in bytes) below which the non-blocked small-matrix
   path is used; above it the blocked path takes over. The default is chosen so
   the three matrices fit in a typical CPU L3 cache. Override with
   -DFIV_MAT_MUL_L3_LIMIT_BYTES=<n> to tune for a specific CPU. */
#ifndef FIV_MAT_MUL_L3_LIMIT_BYTES
#define FIV_MAT_MUL_L3_LIMIT_BYTES (8u * 1024u * 1024u)
#endif

/* Blocked GEMM engine (real32). a_t/b_t: 1 means the operand is used
   transposed. m/n/k are the effective dims of op(A)/op(B)/op(C). Non-static on
   purpose: the blocked factorization drivers (fiv_mat_cholesky.c,
   fiv_mat_lu.c) call it directly on sub-block anchors. */
void fiv_matrix_mul_real32(int a_t, int b_t, int m, int n, int k,
                           ivf32 alpha, ivf32* a, int lda,
                           ivf32* b, int ldb, ivf32 beta,
                           ivf32* c, int ldc);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MAT_MUL_H_ */
