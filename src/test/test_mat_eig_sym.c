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

/* Correctness tests for fiv_matrix_eig_sym (api/fiv_matrix.h): symmetric
 * eigen decomposition. Verifies against a double-precision cyclic Jacobi
 * reference: eigenvalue agreement (spectral-radius relative), residual
 * ||A*V - V*diag(eval)||_F / ||A||_F, orthonormality of V, both input-
 * triangle modes (unreferenced triangle poisoned by sentinels), the
 * eigenvalues-only mode, degenerate/singular spectra, and the error paths.
 * Sizes beyond the Jacobi reference budget (n > 300) are checked through the
 * spectral-moment identities sum(eval) == trace(A) and sum(eval^2) ==
 * ||A||_F^2 instead. Perf smoke prints values-only (n=800) and
 * with-eigenvectors (n=300). */

#include "fiv_matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static int g_fail = 0;
static int g_pass = 0;
#define CHECK(c, msg)                                                           \
    do {                                                                        \
        if (!(c)) { printf("  [FAIL] %s @%d\n", (msg), __LINE__); g_fail++; }   \
        else       { g_pass++; }                                                \
    } while (0)

static unsigned g_seed = 20260829u;
static float rnd(void)
{
    g_seed = g_seed * 1103515245u + 12345u;
    return (float)((g_seed >> 8) & 0xffff) / 4096.f - 4.f;
}

static float fabsf_local(float val) { return val < 0 ? -val : val; }

static const float kSentinel = 1234567.f;

/* Build a symmetric double truth matrix. rank_def > 0 zeroes the trailing
   rows of the generator R before forming A = R*R^T (PSD) so A carries that
   many exact zero eigenvalues; otherwise A = (R + R^T)/2 with a mixed-sign
   spectrum. */
static void gen_sym(int dim, int rank_def, double* a_d)
{
    float* gen = (float*)malloc(sizeof(float) * (size_t)dim * dim);
    for (int idx = 0; idx < dim * dim; idx++) gen[idx] = rnd();
    for (int row = dim - (rank_def > 0 ? rank_def : 0); row < dim; row++) {
        for (int col = 0; col < dim; col++) gen[(size_t)row * dim + col] = 0.0f;
    }
    for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
            double acc = 0.0;
            for (int mid = 0; mid < dim; mid++) {
                acc += (double)gen[(size_t)row * dim + mid] *
                       (double)gen[(size_t)col * dim + mid];
            }
            a_d[(size_t)row * dim + col] = rank_def > 0
                ? acc
                : acc * 0.25 + 0.5 * ((double)gen[(size_t)row * dim + col] +
                                      (double)gen[(size_t)col * dim + row]);
        }
    }
    free(gen);
}

/* Reference cyclic Jacobi in double: evals ascending, evecs column idx is
   the eigenvector for evals[idx]. Converges quadratically; 80 sweeps is
   far beyond the ~10 a dense random symmetric matrix needs. */
static void ref_jacobi_eig(const double* a_in, int dim,
                           double* evals, double* evecs)
{
    double* work = (double*)malloc(sizeof(double) * (size_t)dim * dim);
    double fro = 0.0;
    for (int idx = 0; idx < dim * dim; idx++) {
        work[idx] = a_in[idx];
        fro += a_in[idx] * a_in[idx];
    }
    for (int row = 0; row < dim; row++)
        for (int col = 0; col < dim; col++)
            evecs[(size_t)row * dim + col] = (row == col) ? 1.0 : 0.0;

    const double stop = 1e-14 * (fro > 0.0 ? sqrt(fro) : 1.0);
    for (int sweep = 0; sweep < 80; sweep++) {
        double off = 0.0;
        for (int row = 0; row < dim; row++)
            for (int col = row + 1; col < dim; col++)
                off += work[(size_t)row * dim + col] *
                       work[(size_t)row * dim + col];
        if (sqrt(off) <= stop) break;

        for (int piv = 0; piv < dim - 1; piv++) {
            for (int qiv = piv + 1; qiv < dim; qiv++) {
                const double apq = work[(size_t)piv * dim + qiv];
                if (fabs(apq) <= 1e-300) continue;
                const double theta =
                    (work[(size_t)qiv * dim + qiv] - work[(size_t)piv * dim + piv]) /
                    (2.0 * apq);
                const double tang =
                    (theta >= 0.0 ? 1.0 : -1.0) /
                    (fabs(theta) + sqrt(theta * theta + 1.0));
                const double cosr = 1.0 / sqrt(tang * tang + 1.0);
                const double sinr = tang * cosr;

                for (int idx = 0; idx < dim; idx++) {   /* A <- A * J */
                    const double wpv = work[(size_t)piv * dim + idx];
                    const double wqv = work[(size_t)qiv * dim + idx];
                    work[(size_t)piv * dim + idx] = cosr * wpv - sinr * wqv;
                    work[(size_t)qiv * dim + idx] = sinr * wpv + cosr * wqv;
                }
                for (int idx = 0; idx < dim; idx++) {   /* A <- J^T * A */
                    const double wpv = work[(size_t)idx * dim + piv];
                    const double wqv = work[(size_t)idx * dim + qiv];
                    work[(size_t)idx * dim + piv] = cosr * wpv - sinr * wqv;
                    work[(size_t)idx * dim + qiv] = sinr * wpv + cosr * wqv;
                }
                for (int idx = 0; idx < dim; idx++) {   /* E <- E * J */
                    const double vpv = evecs[(size_t)idx * dim + piv];
                    const double vqv = evecs[(size_t)idx * dim + qiv];
                    evecs[(size_t)idx * dim + piv] = cosr * vpv - sinr * vqv;
                    evecs[(size_t)idx * dim + qiv] = sinr * vpv + cosr * vqv;
                }
            }
        }
    }

    for (int idx = 0; idx < dim; idx++) {
        evals[idx] = work[(size_t)idx * dim + idx];
    }
    /* ascending sort with matching column permutation (selection swaps) */
    for (int slot = 0; slot < dim; slot++) {
        int best = slot;
        for (int probe = slot + 1; probe < dim; probe++) {
            if (evals[probe] < evals[best]) best = probe;
        }
        if (best != slot) {
            const double swap_val = evals[slot]; evals[slot] = evals[best]; evals[best] = swap_val;
            for (int row = 0; row < dim; row++) {
                const double swap_vec = evecs[(size_t)row * dim + slot];
                evecs[(size_t)row * dim + slot] = evecs[(size_t)row * dim + best];
                evecs[(size_t)row * dim + best] = swap_vec;
            }
        }
    }
    free(work);
}

/* Load the referenced triangle from the double truth; the unreferenced one
   gets a sentinel garbage so the mirror step is actually exercised. */
static void load_matrix(fiv_mat* mat, const double* a_d, int upper_mode)
{
    const int dim = (int)mat->shapes[0];
    ivf32* dat = mat->data.fl;
    for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
            const int referenced = upper_mode ? (col >= row) : (col <= row);
            dat[(size_t)row * dim + col] = referenced
                ? (ivf32)a_d[(size_t)row * dim + col]
                : kSentinel;
        }
    }
}

/* one case: mode / vectors / size; eigenvalues, residual, orthonormality */
static void run_case(int upper_mode, int with_vecs, int dim,
                     int rank_def, int do_timing)
{
    char name[96];
    snprintf(name, sizeof(name), "n=%4d %s %s%s", dim,
             upper_mode ? "upper" : "lower",
             with_vecs ? "vecs" : "vals ",
             rank_def > 0 ? " rank-def" : "");

    size_t sh[2] = { (size_t)dim, (size_t)dim };
    fiv_mat* mat_a = fiv_create_tensor2d(sh, FIV_32F1);
    fiv_mat* mat_v = with_vecs ? fiv_create_tensor2d(sh, FIV_32F1) : NULL;
    CHECK(mat_a != NULL && (!with_vecs || mat_v != NULL), name);
    if (mat_a == NULL || (with_vecs && mat_v == NULL)) goto out;

    double* a_d = (double*)malloc(sizeof(double) * (size_t)dim * dim);
    double* ref_evals = (double*)malloc(sizeof(double) * dim);
    double* ref_evecs = (double*)malloc(sizeof(double) * (size_t)dim * dim);
    ivf32* evals = (ivf32*)malloc(sizeof(ivf32) * dim);

    gen_sym(dim, rank_def, a_d);
    load_matrix(mat_a, a_d, upper_mode);

    clock_t time0 = clock();
    const fiv_ret ret = fiv_matrix_eig_sym(mat_a, evals, mat_v, upper_mode);
    clock_t time1 = clock();
    CHECK(ret == FIV_RET_OK, name);
    if (ret != FIV_RET_OK) goto out_free;

    if (do_timing) {
        printf("  [perf] %s: %.3fs\n", name, (double)(time1 - time0) / CLOCKS_PER_SEC);
    }

    if (dim <= 300) {
        ref_jacobi_eig(a_d, dim, ref_evals, ref_evecs);

        /* eigenvalues vs the double reference, relative to the spectral
           radius (absolute error is norm-scaled for backward-stable eig) */
        double scale = 1.0;
        for (int idx = 0; idx < dim; idx++) {
            const double mag = fabs(ref_evals[idx]);
            if (mag > scale) scale = mag;
        }
        double worst = 0.0;
        for (int idx = 0; idx < dim; idx++) {
            const double diff = fabs((double)evals[idx] - ref_evals[idx]);
            if (diff / scale > worst) worst = diff / scale;
        }
        CHECK(worst < 1e-3, "eigenvalues match reference");
        if (worst > 1e-5)
            printf("    (%s worst eval dev %.2e)\n", name, worst);
    } else {
        /* spectral moments: sum(eval) == trace(A), sum(eval^2) == ||A||_F^2 */
        double eval_sum = 0.0, eval_sq = 0.0, trace = 0.0, fro_sq = 0.0;
        for (int idx = 0; idx < dim; idx++) {
            eval_sum += (double)evals[idx];
            eval_sq += (double)evals[idx] * (double)evals[idx];
            trace += a_d[(size_t)idx * dim + idx];
        }
        for (int idx = 0; idx < dim * dim; idx++) fro_sq += a_d[idx] * a_d[idx];
        const double moment_scale = fro_sq > 0.0 ? sqrt(fro_sq) : 1.0;
        CHECK(fabs(eval_sum - trace) < 1e-3 * moment_scale &&
              fabs(eval_sq - fro_sq) < 1e-3 * fro_sq,
              "spectral moments match");
    }

    if (with_vecs) {
        const ivf32* vdat = mat_v->data.fl;
        double fro_a = 0.0;
        for (int idx = 0; idx < dim * dim; idx++) fro_a += a_d[idx] * a_d[idx];
        const double fro_scale = fro_a > 0.0 ? sqrt(fro_a) : 1.0;

        /* residual ||A*V - V*diag(evals)||_F / ||A||_F against the double
           truth (column-wise: R[:,j] = A*v_j - eval_j*v_j) */
        {
            double resid = 0.0;
            for (int col = 0; col < dim; col++) {
                for (int row = 0; row < dim; row++) {
                    double acc = 0.0;
                    for (int mid = 0; mid < dim; mid++) {
                        acc += a_d[(size_t)row * dim + mid] *
                               (double)vdat[(size_t)mid * dim + col];
                    }
                    acc -= (double)evals[col] * (double)vdat[(size_t)row * dim + col];
                    resid += acc * acc;
                }
            }
            const double rel = sqrt(resid) / fro_scale;
            CHECK(rel < 1e-3, "eigen residual small");
            if (rel > 1e-5)
                printf("    (%s residual %.2e)\n", name, rel);
        }

        /* orthonormality max |V^T V - I| */
        {
            double worst = 0.0;
            for (int col_a = 0; col_a < dim; col_a++) {
                for (int col_b = col_a; col_b < dim; col_b++) {
                    double acc = 0.0;
                    for (int row = 0; row < dim; row++) {
                        acc += (double)vdat[(size_t)row * dim + col_a] *
                               (double)vdat[(size_t)row * dim + col_b];
                    }
                    const double dev = fabs(acc - (col_a == col_b ? 1.0 : 0.0));
                    if (dev > worst) worst = dev;
                }
            }
            CHECK(worst < 1e-3, "eigenvectors orthonormal");
            if (worst > 1e-5)
                printf("    (%s orthonormality dev %.2e)\n", name, worst);
        }
    }

out_free:
    free(evals);
    free(ref_evecs);
    free(ref_evals);
    free(a_d);
out:
    fiv_release_tensor2d(&mat_a);
    if (mat_v != NULL) fiv_release_tensor2d(&mat_v);
}

static void test_error_paths(void)
{
    printf("  error paths:\n");

    size_t sh2[2] = { 2, 2 };
    fiv_mat* m22 = fiv_create_tensor2d(sh2, FIV_32F1);
    ivf32* evals2 = (ivf32*)malloc(sizeof(ivf32) * 2);
    CHECK(m22 != NULL && evals2 != NULL, "alloc ok");

    CHECK(fiv_matrix_eig_sym(NULL, evals2, NULL, 0) == FIV_RET_ERR_PARA, "null tensor");
    CHECK(fiv_matrix_eig_sym(m22, NULL, NULL, 0) == FIV_RET_ERR_PARA, "null evals");

    size_t sh23[2] = { 2, 3 };
    fiv_mat* m23 = fiv_create_tensor2d(sh23, FIV_32F1);
    CHECK(m23 != NULL &&
          fiv_matrix_eig_sym(m23, evals2, NULL, 0) == FIV_RET_ERR_PARA,
          "non-square rejected");
    fiv_release_tensor2d(&m23);

    fiv_mat* mi = fiv_create_tensor2d(sh2, FIV_32S1);
    CHECK(mi != NULL &&
          fiv_matrix_eig_sym(mi, evals2, NULL, 0) == FIV_RET_ERR_NOT_SUPPORT,
          "32S rejected");
    fiv_release_tensor2d(&mi);

    m22->data_continue = 0;
    CHECK(fiv_matrix_eig_sym(m22, evals2, NULL, 0) == FIV_RET_ERR_PARA,
          "non-contiguous rejected");
    m22->data_continue = 1;

    /* eigenvector output validation */
    fiv_mat* mbad = fiv_create_tensor2d(sh2, FIV_32S1);
    CHECK(mbad != NULL &&
          fiv_matrix_eig_sym(m22, evals2, mbad, 0) == FIV_RET_ERR_NOT_SUPPORT,
          "32S evec rejected");
    fiv_release_tensor2d(&mbad);

    size_t sh13[2] = { 1, 3 };
    fiv_mat* m13 = fiv_create_tensor2d(sh13, FIV_32F1);
    CHECK(m13 != NULL &&
          fiv_matrix_eig_sym(m22, evals2, m13, 0) == FIV_RET_ERR_PARA,
          "bad evec shape rejected");
    fiv_release_tensor2d(&m13);

    fiv_mat* malias = fiv_create_tensor2d(sh2, FIV_32F1);
    void* saved_ptr = malias->data.ptr;
    malias->data.ptr = m22->data.ptr;
    CHECK(fiv_matrix_eig_sym(m22, evals2, malias, 0) == FIV_RET_ERR_PARA,
          "evec aliasing A rejected");
    malias->data.ptr = saved_ptr;
    fiv_release_tensor2d(&malias);

    /* --- functional small cases --- */

    /* 1x1: eigenvalue is the entry */
    size_t sh1[2] = { 1, 1 };
    fiv_mat* m1 = fiv_create_tensor2d(sh1, FIV_32F1);
    m1->data.fl[0] = -3.5f;
    CHECK(fiv_matrix_eig_sym(m1, evals2, NULL, 0) == FIV_RET_OK, "1x1 ok");
    CHECK(evals2[0] == -3.5f, "1x1 eigenvalue exact");

    /* exact 2x2 [[4,1],[1,2]] -> eigenvalues 3 +- sqrt(2) */
    m22->data.fl[0] = 4.f; m22->data.fl[1] = 1.f;
    m22->data.fl[2] = 1.f; m22->data.fl[3] = 2.f;
    CHECK(fiv_matrix_eig_sym(m22, evals2, NULL, 0) == FIV_RET_OK, "2x2 ok");
    CHECK(fabsf_local(evals2[0] - (3.0f - (float)sqrt(2.0))) < 1e-5f &&
          fabsf_local(evals2[1] - (3.0f + (float)sqrt(2.0))) < 1e-5f,
          "2x2 eigenvalues exact");

    /* zero-diagonal [[0,1],[1,0]] -> {-1, +1} */
    m22->data.fl[0] = 0.f; m22->data.fl[1] = 1.f;
    m22->data.fl[2] = 1.f; m22->data.fl[3] = 0.f;
    CHECK(fiv_matrix_eig_sym(m22, evals2, NULL, 0) == FIV_RET_OK, "offdiag ok");
    CHECK(fabsf_local(evals2[0] + 1.f) < 1e-6f && fabsf_local(evals2[1] - 1.f) < 1e-6f,
          "offdiag eigenvalues exact");

    /* PSD singular [[1,1],[1,1]] -> {0, 2}: must SUCCEED (unlike cholesky) */
    m22->data.fl[0] = 1.f; m22->data.fl[1] = 1.f;
    m22->data.fl[2] = 1.f; m22->data.fl[3] = 1.f;
    CHECK(fiv_matrix_eig_sym(m22, evals2, NULL, 0) == FIV_RET_OK, "singular ok");
    CHECK(fabsf_local(evals2[0]) < 1e-6f && fabsf_local(evals2[1] - 2.f) < 1e-6f,
          "singular eigenvalues exact");

    /* diagonal input stays exact through both modes */
    m22->data.fl[0] = 3.f; m22->data.fl[1] = 0.f;
    m22->data.fl[2] = 0.f; m22->data.fl[3] = -1.f;
    CHECK(fiv_matrix_eig_sym(m22, evals2, NULL, 1) == FIV_RET_OK, "diag upper ok");
    CHECK(evals2[0] == -1.f && evals2[1] == 3.f, "diag eigenvalues exact");
    m22->data.fl[0] = 3.f; m22->data.fl[1] = 0.f;
    m22->data.fl[2] = 0.f; m22->data.fl[3] = -1.f;
    CHECK(fiv_matrix_eig_sym(m22, evals2, NULL, 0) == FIV_RET_OK, "diag lower ok");
    CHECK(evals2[0] == -1.f && evals2[1] == 3.f, "diag eigenvalues exact");

    fiv_release_tensor2d(&m1);
    fiv_release_tensor2d(&m22);
    free(evals2);
}

int main(void)
{
    printf("=== test_mat_eig_sym ===\n");

    static const int kSizes[] = { 1, 2, 3, 5, 7, 63, 64, 65, 66, 127, 128, 129, 200 };
    for (unsigned s = 0; s < sizeof(kSizes) / sizeof(kSizes[0]); s++) {
        run_case(0, 1, kSizes[s], 0, 0);
        run_case(1, 1, kSizes[s], 0, 0);
    }

    /* degenerate spectrum (exact zero eigenvalues), both modes */
    run_case(0, 1, 100, 30, 0);
    run_case(1, 0, 100, 30, 0);

    test_error_paths();

    /* eigenvalues-only call must agree with the reference too */
    run_case(0, 0, 200, 0, 0);

    printf("  perf smoke:\n");
    run_case(0, 0, 800, 0, 1);
    run_case(0, 1, 300, 0, 1);

    printf("  pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
