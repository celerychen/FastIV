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

/* In-place blocked Cholesky factorization (float32, native row-major). The
   public fiv_matrix_cholesky (api/fiv_matrix.h) is routed through the
   blocked GEMM engine (fiv_mat_mul.h). */
#ifndef _FIV_MAT_CHOLESKY_H_
#define _FIV_MAT_CHOLESKY_H_

#include "fiv_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* In-place blocked Cholesky of a square, contiguous float32 SPD matrix A_io.
   lower!=0: A = L*L^T, L in the lower triangle; lower==0: A = U^T*U, U = L^T
   in the upper. Only the referenced triangle is read; the other is scratch.
   NOT_POS_DEF if not positive definite. */
fiv_ret fiv_matrix_cholesky(fiv_mat* mat_a, int lower);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MAT_CHOLESKY_H_ */