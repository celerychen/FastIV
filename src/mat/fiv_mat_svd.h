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

/* Header for fiv_mat_svd.c: declares the two singular value decomposition
   backends implemented in that module. NOT part of the public API: public
   users call fiv_matrix_svd (api/fiv_matrix.h), which dispatches to these.
   Both share the caller-visible input/output format documented on
   fiv_matrix_svd and preserve the input matrix.

   fiv_matrix_svd_bcd     - blocked Householder bidiagonalization +
                            Golub-Kahan QR (LAPACK dgesdd lineage); V is
                            produced column-major, matching the public
                            contract natively. Fastest for large matrices.
   fiv_matrix_svd_jacobi  - one-sided Jacobi sweeps (OpenCV SVD::compute
                            lineage); output convention follows OpenCV:
                            mat_u (rows x k) holds the left singular vectors
                            in its COLUMNS, mat_vt (k x cols) holds V^T with
                            the right singular vectors in its ROWS. Highest
                            accuracy; prefer for small / ill-conditioned
                            inputs. */

#ifndef _FIV_MAT_SVD_H_
#define _FIV_MAT_SVD_H_

#include "fiv_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

fiv_ret fiv_matrix_svd_bcd(fiv_mat* mat_a, ivf32* sing_vals, fiv_mat* mat_u, fiv_mat* mat_v);

fiv_ret fiv_matrix_svd_jacobi(fiv_mat* mat_a, ivf32* sing_vals, fiv_mat* mat_u, fiv_mat* mat_vt);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MAT_SVD_H_ */