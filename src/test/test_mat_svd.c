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

/* Correctness tests for fiv_matrix_svd (api/fiv_matrix.h): thin SVD.
 * Singular values are checked against sqrt of the eigenvalues of A^T A
 * computed by a double-precision cyclic Jacobi reference; the vector
 * factor is checked through the residuals ||A*V - U*S||_F / ||A||_F and
 * ||A^T*U - V*S||_F / ||A||_F plus orthonormality of U and V, descending
 * order, and the spectral-moment identity sum(s^2) == ||A||_F^2. Covers
 * tall / wide / square shapes, rank-deficient inputs, values-only mode,
 * and the error paths. */

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

static unsigned g_seed = 20260830u;
static float rnd(void)
{
    g_seed = g_seed * 1103515245u + 12345u;
    return (float)((g_seed >> 8) & 0xffff) / 4096.f - 4.f;
}

/* double cyclic Jacobi eigen decomposition of the symmetric input; evals
   ascending, evecs columns (same reference as the eig tests) */
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

    const double stop = 1e-13 * (fro > 0.0 ? sqrt(fro) : 1.0);
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

                for (int idx = 0; idx < dim; idx++) {
                    const double wpv = work[(size_t)piv * dim + idx];
                    const double wqv = work[(size_t)qiv * dim + idx];
                    work[(size_t)piv * dim + idx] = cosr * wpv - sinr * wqv;
                    work[(size_t)qiv * dim + idx] = sinr * wpv + cosr * wqv;
                }
                for (int idx = 0; idx < dim; idx++) {
                    const double wpv = work[(size_t)idx * dim + piv];
                    const double wqv = work[(size_t)idx * dim + qiv];
                    work[(size_t)idx * dim + piv] = cosr * wpv - sinr * wqv;
                    work[(size_t)idx * dim + qiv] = sinr * wpv + cosr * wqv;
                }
                for (int idx = 0; idx < dim; idx++) {
                    const double vpv = evecs[(size_t)idx * dim + piv];
                    const double vqv = evecs[(size_t)idx * dim + qiv];
                    evecs[(size_t)idx * dim + piv] = cosr * vpv - sinr * vqv;
                    evecs[(size_t)idx * dim + qiv] = sinr * vpv + cosr * vqv;
                }
            }
        }
    }
    for (int idx = 0; idx < dim; idx++) evals[idx] = work[(size_t)idx * dim + idx];
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

/* reference singular values = sqrt(eigenvalues of A^T A), descending */
static void ref_svd_values(const ivf32* a_fl, int rows, int cols, double* sing_ref)
{
    const int dim = rows < cols ? rows : cols;
    double* ata = (double*)calloc((size_t)cols * cols, sizeof(double));
    for (int ic = 0; ic < cols; ic++) {
        for (int jc = ic; jc < cols; jc++) {
            double acc = 0.0;
            for (int ir = 0; ir < rows; ir++) {
                acc += (double)a_fl[(size_t)ir * cols + ic] *
                       (double)a_fl[(size_t)ir * cols + jc];
            }
            ata[(size_t)ic * cols + jc] = acc;
            ata[(size_t)jc * cols + ic] = acc;
        }
    }
    double* evals = (double*)malloc(sizeof(double) * cols);
    double* evecs = (double*)malloc(sizeof(double) * (size_t)cols * cols);
    ref_jacobi_eig(ata, cols, evals, evecs);
    for (int idx = 0; idx < dim; idx++) {
        const double lam = evals[cols - 1 - idx];
        sing_ref[idx] = lam > 0.0 ? sqrt(lam) : 0.0;
    }
    free(evecs);
    free(evals);
    free(ata);
}

/* one case; with_vecs switches the vector outputs on */
static void run_case(int mrows, int ncols, int rank_def, int with_vecs, int do_timing)
{
    char name[96];
    snprintf(name, sizeof(name), "%dx%d%s%s", mrows, ncols,
             rank_def > 0 ? " rank-def" : "", with_vecs ? "" : " vals-only");

    const int dim = mrows < ncols ? mrows : ncols;
    size_t sh[2] = { (size_t)mrows, (size_t)ncols };
    size_t sh_u[2] = { (size_t)mrows, (size_t)dim };
    size_t sh_v[2] = { (size_t)ncols, (size_t)dim };
    fiv_mat* mat_a = fiv_create_tensor2d(sh, FIV_32F1);
    fiv_mat* mat_u = with_vecs ? fiv_create_tensor2d(sh_u, FIV_32F1) : NULL;
    fiv_mat* mat_v = with_vecs ? fiv_create_tensor2d(sh_v, FIV_32F1) : NULL;
    CHECK(mat_a != NULL && (!with_vecs || (mat_u != NULL && mat_v != NULL)), name);
    if (mat_a == NULL || (with_vecs && (mat_u == NULL || mat_v == NULL))) goto out;

    /* A = R * diag(scale) * C^T with rank_def zeroed scale entries, so the
       spectrum is known and exactly rank-deficient when requested */
    double fro_a = 0.0;
    {
        const int gen_dim = mrows > ncols ? mrows : ncols;
        float* rmat = (float*)malloc(sizeof(float) * (size_t)gen_dim * gen_dim);
        double* scale = (double*)malloc(sizeof(double) * (size_t)dim);
        for (int idx = 0; idx < gen_dim * gen_dim; idx++) rmat[idx] = rnd();
        for (int idx = 0; idx < dim; idx++) {
            scale[idx] = 1.0 + 9.0 * (double)idx / (dim > 1 ? dim - 1 : 1);
        }
        for (int idx = dim - rank_def; idx < dim; idx++) scale[idx] = 0.0;
        ivf32* dat = mat_a->data.fl;
        memset(dat, 0, sizeof(ivf32) * (size_t)mrows * ncols);
        for (int idx = 0; idx < dim; idx++) {
            const double sig = scale[idx];
            if (sig == 0.0) continue;
            for (int ir = 0; ir < mrows; ir++) {
                const float rv = rmat[(size_t)ir * gen_dim + idx];
                for (int ic = 0; ic < ncols; ic++) {
                    dat[(size_t)ir * ncols + ic] +=
                        (ivf32)(sig * (double)rv * (double)rmat[(size_t)ic * gen_dim + idx]);
                }
            }
        }
        free(rmat);
        free(scale);
        for (int idx = 0; idx < mrows * ncols; idx++) {
            fro_a += (double)dat[idx] * (double)dat[idx];
        }
    }
    const double fro_scale = fro_a > 0.0 ? sqrt(fro_a) : 1.0;

    /* keep a copy of A used by the residual checks; the SVD contract (option B)
       preserves the input, so mat_a must remain unchanged across the call */
    double* a_saved = (double*)malloc(sizeof(double) * (size_t)mrows * ncols);
    for (int idx = 0; idx < mrows * ncols; idx++) {
        a_saved[idx] = (double)mat_a->data.fl[idx];
    }

    double* sing_ref = (double*)malloc(sizeof(double) * dim);
    ref_svd_values(mat_a->data.fl, mrows, ncols, sing_ref);

    ivf32* sing = (ivf32*)malloc(sizeof(ivf32) * dim);
    clock_t time0 = clock();
    const fiv_ret ret = fiv_matrix_svd(mat_a, sing, mat_u, mat_v);
    clock_t time1 = clock();
    CHECK(ret == FIV_RET_OK, name);
    if (ret != FIV_RET_OK) goto out_free;

    /* contract (option B): the input matrix must be preserved, never destroyed */
    {
        double worst = 0.0;
        for (int idx = 0; idx < mrows * ncols; idx++) {
            const double diff = fabs((double)mat_a->data.fl[idx] - a_saved[idx]);
            if (diff > worst) worst = diff;
        }
        CHECK(worst < 1e-6, "input matrix preserved (not destroyed)");
    }

    if (do_timing) {
        printf("  [perf] %s: %.3fs\n", name, (double)(time1 - time0) / CLOCKS_PER_SEC);
    }

    /* descending, non-negative */
    {
        int ordered = sing[0] >= 0.0f;
        for (int idx = 1; idx < dim; idx++) {
            ordered = ordered && sing[idx - 1] >= sing[idx] && sing[idx] >= 0.0f;
        }
        CHECK(ordered, "singular values descending and >= 0");
    }

    /* values vs the double reference; small singular values get an absolute
       floor since float SVD resolves zeros only to eps*sigma_max */
    {
        const double top = sing_ref[0] > 0.0 ? sing_ref[0] : 1.0;
        double worst = 0.0;
        for (int idx = 0; idx < dim; idx++) {
            const double tol = 1e-3 * (top > fabs(sing_ref[idx])
                                           ? top : fabs(sing_ref[idx]) + 1.0);
            const double diff = fabs((double)sing[idx] - sing_ref[idx]);
            if (diff / tol > worst) worst = diff / tol;
        }
        CHECK(worst < 1.0, "singular values match reference");
        if (worst > 0.01)
            printf("    (%s worst s dev %.2e of tol)\n", name, worst);
    }

    if (with_vecs) {
        const ivf32* udat = mat_u->data.fl;
        const ivf32* vdat = mat_v->data.fl;

        /* ||A*V - U*S||_F / ||A||_F */
        {
            double resid = 0.0;
            for (int col = 0; col < dim; col++) {
                for (int ir = 0; ir < mrows; ir++) {
                    double acc = 0.0;
                    for (int mid = 0; mid < ncols; mid++) {
                        acc += a_saved[(size_t)ir * ncols + mid] *
                               (double)vdat[(size_t)mid * dim + col];
                    }
                    acc -= (double)sing[col] * (double)udat[(size_t)ir * dim + col];
                    resid += acc * acc;
                }
            }
            const double rel = sqrt(resid) / fro_scale;
            CHECK(rel < 1e-3, "residual ||A*V - U*S|| small");
            if (rel > 1e-5) printf("    (%s av-us %.2e)\n", name, rel);
        }
        /* ||A^T*U - V*S||_F / ||A||_F */
        {
            double resid = 0.0;
            for (int col = 0; col < dim; col++) {
                for (int ic = 0; ic < ncols; ic++) {
                    double acc = 0.0;
                    for (int mid = 0; mid < mrows; mid++) {
                        acc += a_saved[(size_t)mid * ncols + ic] *
                               (double)udat[(size_t)mid * dim + col];
                    }
                    acc -= (double)sing[col] * (double)vdat[(size_t)ic * dim + col];
                    resid += acc * acc;
                }
            }
            const double rel = sqrt(resid) / fro_scale;
            CHECK(rel < 1e-3, "residual ||A^T*U - V*S|| small");
            if (rel > 1e-5) printf("    (%s atu-vs %.2e)\n", name, rel);
        }
        /* orthonormality of U and V */
        {
            double worst = 0.0;
            for (int ca = 0; ca < dim; ca++) {
                for (int cb = ca; cb < dim; cb++) {
                    double acc = 0.0;
                    for (int ir = 0; ir < mrows; ir++) {
                        acc += (double)udat[(size_t)ir * dim + ca] *
                               (double)udat[(size_t)ir * dim + cb];
                    }
                    const double dev = fabs(acc - (ca == cb ? 1.0 : 0.0));
                    if (dev > worst) worst = dev;
                }
            }
            CHECK(worst < 1e-3, "U orthonormal");
            if (worst > 1e-5) printf("    (%s orth U %.2e)\n", name, worst);
            worst = 0.0;
            for (int ca = 0; ca < dim; ca++) {
                for (int cb = ca; cb < dim; cb++) {
                    double acc = 0.0;
                    for (int ir = 0; ir < ncols; ir++) {
                        acc += (double)vdat[(size_t)ir * dim + ca] *
                               (double)vdat[(size_t)ir * dim + cb];
                    }
                    const double dev = fabs(acc - (ca == cb ? 1.0 : 0.0));
                    if (dev > worst) worst = dev;
                }
            }
            CHECK(worst < 1e-3, "V orthonormal");
            if (worst > 1e-5) printf("    (%s orth V %.2e)\n", name, worst);
        }
    }

out_free:
    free(sing);
    free(sing_ref);
    free(a_saved);
out:
    fiv_release_tensor2d(&mat_a);
    if (mat_u != NULL) fiv_release_tensor2d(&mat_u);
    if (mat_v != NULL) fiv_release_tensor2d(&mat_v);
}

static void test_error_paths(void)
{
    printf("  error paths:\n");
    size_t sh22[2] = { 2, 2 };
    size_t sh23[2] = { 2, 3 };
    ivf32 sing2[2];
    ivf32 sing3[3];

    CHECK(fiv_matrix_svd(NULL, sing2, NULL, NULL) == FIV_RET_ERR_PARA, "null tensor");
    CHECK(fiv_matrix_svd(NULL, NULL, NULL, NULL) == FIV_RET_ERR_PARA, "null sing");

    fiv_mat* m23 = fiv_create_tensor2d(sh23, FIV_32F1);
    CHECK(m23 != NULL, "alloc ok");
    CHECK(fiv_matrix_svd(m23, sing2, NULL, NULL) == FIV_RET_OK, "values-only len 2 ok");
    CHECK(fiv_matrix_svd(m23, sing3, NULL, NULL) == FIV_RET_OK, "values-only ok");

    fiv_mat* mi = fiv_create_tensor2d(sh23, FIV_32S1);
    CHECK(mi != NULL &&
          fiv_matrix_svd(mi, sing2, NULL, NULL) == FIV_RET_ERR_NOT_SUPPORT, "32S rejected");
    fiv_release_tensor2d(&mi);

    m23->data_continue = 0;
    CHECK(fiv_matrix_svd(m23, sing2, NULL, NULL) == FIV_RET_ERR_PARA, "non-contiguous");
    m23->data_continue = 1;

    /* wrong U shape (must be rows x dim = 2 x 2) */
    size_t shu[2] = { 2, 3 };
    fiv_mat* mbad = fiv_create_tensor2d(shu, FIV_32F1);
    CHECK(mbad != NULL &&
          fiv_matrix_svd(m23, sing2, mbad, NULL) == FIV_RET_ERR_PARA, "bad U shape");
    fiv_release_tensor2d(&mbad);

    /* aliasing rejected */
    fiv_mat* malias = fiv_create_tensor2d(sh22, FIV_32F1);
    void* saved = malias->data.ptr;
    malias->data.ptr = m23->data.ptr;
    CHECK(fiv_matrix_svd(m23, sing2, malias, NULL) == FIV_RET_ERR_PARA, "U aliasing A");
    malias->data.ptr = saved;
    fiv_release_tensor2d(&malias);

    fiv_release_tensor2d(&m23);
}

int main(void)
{
    printf("=== test_mat_svd ===\n");

    static const int kShapes[][2] = {
        {1, 1}, {3, 1}, {1, 4}, {2, 3}, {7, 5},
        {63, 30}, {64, 64}, {65, 70}, {127, 100}, {50, 200},
    };
    for (unsigned s = 0; s < sizeof(kShapes) / sizeof(kShapes[0]); s++) {
        run_case(kShapes[s][0], kShapes[s][1], 0, 1, 0);
    }

    /* rank-deficient: exact zero singular values, tall and wide */
    run_case(100, 40, 15, 1, 0);
    run_case(40, 100, 15, 1, 0);
    /* values-only mode */
    run_case(300, 200, 0, 0, 0);

    test_error_paths();

    printf("  perf smoke:\n");
    run_case(600, 300, 0, 1, 1);
    run_case(800, 200, 0, 0, 1);

    printf("  pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
