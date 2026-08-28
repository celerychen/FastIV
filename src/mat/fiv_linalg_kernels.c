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

/* See fiv_linalg_kernels.h for the contracts. Everything here stays in ivf32
   (float): dot products accumulate in a float acc and the only math helper is
   sqrtf. No ivf64 / double is involved. */

#include "fiv_linalg_kernels.h"

#include <math.h>
#include <float.h>

int fiv_potrf_lower_block_real32(ivf32* p11, int nb, int row_stride)
{
    for (int i = 0; i < nb; i++) {
        ivf32* row_i = p11 + (size_t)i * row_stride;

        for (int j = 0; j < i; j++) {
            const ivf32* row_j = p11 + (size_t)j * row_stride;
            ivf32 acc = 0.0f;
            for (int t = 0; t < j; t++) {
                acc += row_i[t] * row_j[t];
            }
            row_i[j] = (row_i[j] - acc) / row_j[j];
        }

        ivf32 acc = 0.0f;
        for (int t = 0; t < i; t++) {
            acc += row_i[t] * row_i[t];
        }
        ivf32 rem = row_i[i] - acc;
        if (rem <= 0.0f) {
            return -1;
        }
        row_i[i] = sqrtf(rem);
    }
    return 0;
}

void fiv_cholesky_strip_solve_real32(const ivf32* p11, int kb, int ldm11,
                                     ivf32* p21, int mb, int ldm21)
{
    for (int r = 0; r < mb; r++) {
        ivf32* row_r = p21 + (size_t)r * ldm21;
        for (int c = 0; c < kb; c++) {
            const ivf32* row_c = p11 + (size_t)c * ldm11;
            ivf32 acc = 0.0f;
            for (int t = 0; t < c; t++) {
                acc += row_r[t] * row_c[t];
            }
            row_r[c] = (row_r[c] - acc) / row_c[c];
        }
    }
}

int fiv_getrf_panel_real32(ivf32* p11, int kb, int mrows, int row_stride,
                           int* piv, int row0)
{
    int first_zero = 0;

    /* Panel scale used for the singularity test below. A column is treated as
       singular when its partial-pivot magnitude drops below a small fraction of
       the panel's largest magnitude (relative threshold), so we never test a
       float for exact equality with zero. */
    ivf32 amax = 0.0f;
    for (int r = 0; r < mrows; r++) {
        const ivf32* row = p11 + (size_t)r * row_stride;
        for (int j = 0; j < kb; j++) {
            ivf32 mag = fabsf(row[j]);
            if (mag > amax) amax = mag;
        }
    }
    const ivf32 singular_tol = amax * FLT_EPSILON;

    for (int c = 0; c < kb; c++) {
        int pivot_row = c;
        ivf32 best = fabsf(p11[(size_t)c * row_stride + c]);
        for (int r = c + 1; r < mrows; r++) {
            ivf32 mag = fabsf(p11[(size_t)r * row_stride + c]);
            if (mag > best) { best = mag; pivot_row = r; }
        }
        piv[c] = row0 + pivot_row;

        if (pivot_row != c) {
            ivf32* row_c = p11 + (size_t)c * row_stride;
            ivf32* row_p = p11 + (size_t)pivot_row * row_stride;
            for (int j = 0; j < kb; j++) {
                ivf32 swap_tmp = row_c[j]; row_c[j] = row_p[j]; row_p[j] = swap_tmp;
            }
        }

        if (best <= singular_tol) {
            if (first_zero == 0) first_zero = c + 1;
            continue;
        }

        const ivf32 pivot = p11[(size_t)c * row_stride + c];
        for (int r = c + 1; r < mrows; r++) {
            p11[(size_t)r * row_stride + c] = p11[(size_t)r * row_stride + c] / pivot;
        }

        const ivf32* row_c = p11 + (size_t)c * row_stride;
        for (int r = c + 1; r < mrows; r++) {
            ivf32* row_r = p11 + (size_t)r * row_stride;
            const ivf32 factor = row_r[c];
            for (int j = c + 1; j < kb; j++) {
                row_r[j] = row_r[j] - factor * row_c[j];
            }
        }
    }
    return first_zero;
}

void fiv_lu_strip_solve_real32(const ivf32* p11, int kb, int ldm11,
                               ivf32* p12, int ncols, int ld12)
{
    for (int i = 1; i < kb; i++) {
        const ivf32* li = p11 + (size_t)i * ldm11;
        ivf32* ui = p12 + (size_t)i * ld12;
        for (int t = 0; t < i; t++) {
            const ivf32 factor = li[t];
            const ivf32* ut = p12 + (size_t)t * ld12;
            for (int j = 0; j < ncols; j++) {
                ui[j] = ui[j] - factor * ut[j];
            }
        }
    }
}
