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

/* In-place symmetric eigen decomposition (float32, native row-major). The
   public fiv_matrix_eig_sym (api/fiv_matrix.h) uses Householder
   tridiagonalization + implicit-shift QL. */
#ifndef _FIV_MAT_EIG_SYM_H_
#define _FIV_MAT_EIG_SYM_H_

#include "fiv_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* In-place symmetric eigen decomposition (Householder tridiagonalization +
   implicit-shift QL) of a square, contiguous float32 symmetric matrix A_io
   (destroyed). evals[dim] gets the ascending eigenvalues; optional mat_evec
   (dim x dim, must not alias A_io) gets orthonormal eigenvectors in its
   columns. Only the triangle selected by upper is read; the other is
   scratch. UNKNOWN if the QL iteration fails to converge. */
fiv_ret fiv_matrix_eig_sym(fiv_mat* mat_a, ivf32* evals, fiv_mat* mat_evec, int upper);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MAT_EIG_SYM_H_ */