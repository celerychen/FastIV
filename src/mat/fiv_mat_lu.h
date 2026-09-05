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

/* In-place blocked LU factorization with partial pivoting (float32, native
   row-major). The public fiv_matrix_lu (api/fiv_matrix.h) is routed through
   the blocked GEMM engine (fiv_mat_mul.h). */
#ifndef _FIV_MAT_LU_H_
#define _FIV_MAT_LU_H_

#include "fiv_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* In-place blocked LU with partial pivoting of a contiguous float32 matrix
   A_io. piv[min(rows,cols)] records row swaps, P*A = L*U; A_io packs L (unit
   diagonal implicit) + U. SINGULAR on a zero pivot (factors still written). */
fiv_ret fiv_matrix_lu(fiv_mat* mat_a, int* piv);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MAT_LU_H_ */