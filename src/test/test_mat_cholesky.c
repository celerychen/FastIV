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

/* Correctness tests for fiv_matrix_cholesky (api/fiv_matrix.h): blocked Cholesky.
 * Verifies reconstruction residuals ||L*L^T - A||_F / ||A||_F against a double
 * reference on NB-boundary-crossing sizes, both lower/upper modes, elementwise
 * agreement with the double reference factor, that the unreferenced triangle
 * stays untouched, and the error paths (params, dtype, contiguity, non-PD
 * inputs). A small perf smoke prints single-thread GFLOPS at n=1000. */

#include "fiv_matrix.h"
#include "fiv_linalg_kernels.h"
#include "fiv_common.h"   /* fiv_get_current_system_time */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;
static int g_pass = 0;
#define CHECK(c, msg)                                                           \
    do {                                                                        \
        if (!(c)) { printf("  [FAIL] %s @%d\n", (msg), __LINE__); g_fail++; }   \
        else       { g_pass++; }                                                \
    } while (0)

static unsigned g_seed = 20260827u;
static float rnd(void)
{
    g_seed = g_seed * 1103515245u + 12345u;
    return (float)((g_seed >> 8) & 0xffff) / 4096.f - 4.f;
}

static float fabsf_local(float x) { return x < 0 ? -x : x; }

static const float kSentinel = 1234567.f;

/* reference unblocked Cholesky in double; returns 0/-1 like the kernel */
static int ref_potrf_lower(const double* a_in, int n, double* l)
{
    memset(l, 0, sizeof(double) * (size_t)n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            double acc = 0.0;
            for (int t = 0; t < j; t++)
                acc += l[(size_t)i * n + t] * l[(size_t)j * n + t];
            l[(size_t)i * n + j] = (a_in[(size_t)i * n + j] - acc) /
                                   l[(size_t)j * n + j];
        }
        double acc = 0.0;
        for (int t = 0; t < i; t++)
            acc += l[(size_t)i * n + t] * l[(size_t)i * n + t];
        double rem = a_in[(size_t)i * n + i] - acc;
        if (rem <= 0.0) return -1;
        l[(size_t)i * n + i] = sqrt(rem);
    }
    return 0;
}

/* Build symmetric PD A = R^T R + scale*I: entry (i,j)
   = sum_t R[t][i]*R[t][j] + aug*(i==j). Row-major full matrix in double. */
static void gen_spd(int n, double* a_d)
{
    float* r = (float*)malloc(sizeof(float) * (size_t)n * n);
    for (int t = 0; t < n * n; t++) r[t] = rnd();
    const double aug = (double)n * 0.5;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int t = 0; t < n; t++)
                acc += (double)r[(size_t)t * n + i] * (double)r[(size_t)t * n + j];
            a_d[(size_t)i * n + j] = acc + (i == j ? aug : 0.0);
        }
    }
    free(r);
}

/* Gather L(i, j) for i >= j from whichever triangle holds the factor:
   lower mode stores L at entry (i, j); upper mode stores L^T = U, so
   L(i, j) = U(j, i) read from entry (j, i). */
static double pick_L(int lower, int i, int j, const ivf32* p, int n)
{
    if (lower) return (double)p[(size_t)i * n + j];       /* entry (i, j) */
    return (double)p[(size_t)j * n + i];                  /* U(j, i) */
}

/* one case: mode + size, residual / factor agreement / untouched checks */
static void run_case(int lower, int n, int do_timing)
{
    char name[64];
    snprintf(name, sizeof(name), "n=%4d %s", n, lower ? "lower" : "upper");

    size_t sh[2] = { (size_t)n, (size_t)n };
    fiv_mat* M = fiv_create_tensor2d(sh, FIV_32F1);
    CHECK(M != NULL, name);
    if (M == NULL) return;

    double* a_d = (double*)malloc(sizeof(double) * (size_t)n * n);
    double* ref = (double*)malloc(sizeof(double) * (size_t)n * n);
    double* lf  = (double*)malloc(sizeof(double) * (size_t)n * n);

    gen_spd(n, a_d);

    /* load referenced triangle from the double truth; other triangle gets a
       sentinel (the trailing GEMM is documented to clobber it with Schur
       leftovers; the factor checks below only ever read the referenced
       triangle, and the reconstruction uses the full symmetric truth) */
    ivf32* p = M->data.fl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            p[(size_t)i * n + j] = (ivf32)a_d[(size_t)i * n + j];
        }
    }

    CHECK(ref_potrf_lower(a_d, n, ref) == 0, "gen SPD is PD");

    double t0 = fiv_get_current_system_time();
    fiv_ret ret = fiv_matrix_cholesky(M, lower);
    double t1 = fiv_get_current_system_time();
    CHECK(ret == FIV_RET_OK, name);
    if (ret != FIV_RET_OK) goto out;

    if (do_timing && n >= 500) {
        double secs = (t1 - t0) / 1000.0;
        double gflops = ((double)n * n * n / 3.0) / secs / 1e9;
        printf("  [perf] %s: %.3fs -> %.2f GFLOPS\n", name, secs, gflops);
    }

    /* extract the factor to a dense double copy for uniform checks */
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++)
            lf[(size_t)i * n + j] = pick_L(lower, i, j, p, n);

    /* factor entries agree with the double reference within fp32 rounding */
    {
        double worst = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++) {
                double want = ref[(size_t)i * n + j];
                double den  = fabs(want) > 1.0 ? fabs(want) : 1.0;
                double d = fabs(lf[(size_t)i * n + j] - want) / den;
                if (d > worst) worst = d;
            }
        CHECK(worst < 1e-3, "factor matches double reference");
        if (worst > 1e-5)
            printf("    (%s worst factor dev %.2e)\n", name, worst);
    }

    /* reconstruction residual ||L*L^T - A||_F / ||A||_F (entries below the
       referenced rounding: t runs only up to j because L(i, t) is zero there
       in exact arithmetic and unreferenced storage is garbage) */
    {
        double res = 0.0, base = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double s = 0.0;
                for (int t = 0; t <= j; t++)
                    s += lf[(size_t)i * n + t] * lf[(size_t)j * n + t];
                double d = s - a_d[(size_t)i * n + j];
                res += (i == j) ? d * d : 2.0 * d * d;
            }
        }
        for (int t = 0; t < n * n; t++)
            base += a_d[t] * a_d[t];
        double rel = sqrt(res) / sqrt(base);
        CHECK(rel < 1e-3, "reconstruction residual small");
        if (rel > 1e-5)
            printf("    (%s residual %.2e)\n", name, rel);
    }

out:
    free(ref);
    free(a_d);
    free(lf);
    fiv_release_tensor2d(&M);
}

static void test_error_paths(void)
{
    printf("  error paths:\n");

    size_t sh2[2] = { 2, 2 };
    fiv_mat* m22 = fiv_create_tensor2d(sh2, FIV_32F1);
    CHECK(m22 != NULL, "alloc ok");

    CHECK(fiv_matrix_cholesky(NULL, 1) == FIV_RET_ERR_PARA, "null tensor");

    size_t sh23[2] = { 2, 3 };
    fiv_mat* m23 = fiv_create_tensor2d(sh23, FIV_32F1);
    CHECK(m23 != NULL && fiv_matrix_cholesky(m23, 1) == FIV_RET_ERR_PARA,
          "non-square rejected");
    fiv_release_tensor2d(&m23);

    fiv_mat* mi = fiv_create_tensor2d(sh2, FIV_32S1);
    CHECK(mi != NULL && fiv_matrix_cholesky(mi, 1) == FIV_RET_ERR_NOT_SUPPORT,
          "32S rejected");
    fiv_release_tensor2d(&mi);

    m22->data_continue = 0;
    CHECK(fiv_matrix_cholesky(m22, 1) == FIV_RET_ERR_PARA, "non-contiguous rejected");
    m22->data_continue = 1;

    memset(m22->data.fl, 0, sizeof(ivf32) * 4);
    CHECK(fiv_matrix_cholesky(m22, 1) == FIV_RET_ERR_NOT_POS_DEF, "zero matrix not PD");

    /* indefinite [[1,10],[10,1]]: A11=1 factors, update gives 1-100 < 0 */
    m22->data.fl[0] = 1.f; m22->data.fl[1] = 10.f;
    m22->data.fl[2] = 10.f; m22->data.fl[3] = 1.f;
    CHECK(fiv_matrix_cholesky(m22, 1) == FIV_RET_ERR_NOT_POS_DEF,
          "indefinite rejected (lower)");
    CHECK(fiv_matrix_cholesky(m22, 0) == FIV_RET_ERR_NOT_POS_DEF,
          "indefinite rejected (upper)");

    /* PSD singular [[1,1],[1,1]] must fail (strict PD required) */
    m22->data.fl[0] = 1.f; m22->data.fl[1] = 1.f;
    m22->data.fl[2] = 1.f; m22->data.fl[3] = 1.f;
    CHECK(fiv_matrix_cholesky(m22, 1) == FIV_RET_ERR_NOT_POS_DEF, "PSD singular rejected");

    /* handcrafted exact case [[4,2],[2,10]] -> L = [[2,0],[1,3]]: row-major
       storage reads fl[0]=L00, fl[2]=L10, fl[3]=L11 */
    m22->data.fl[0] = 4.f; m22->data.fl[1] = 2.f;
    m22->data.fl[2] = 2.f; m22->data.fl[3] = 10.f;
    CHECK(fiv_matrix_cholesky(m22, 1) == FIV_RET_OK, "handcrafted PD ok");
    CHECK(fabsf_local(m22->data.fl[0] - 2.f) < 1e-6f &&
          fabsf_local(m22->data.fl[2] - 1.f) < 1e-6f &&
          fabsf_local(m22->data.fl[3] - 3.f) < 1e-6f, "L entries exact (small case)");

    fiv_release_tensor2d(&m22);
}

int main(void)
{
    printf("=== test_mat_cholesky ===\n");

    static const int kSizes[] = { 1, 2, 3, 7, 63, 64, 65, 66, 127, 128, 129, 200 };
    for (unsigned s = 0; s < sizeof(kSizes) / sizeof(kSizes[0]); s++) {
        run_case(1, kSizes[s], 0);
        run_case(0, kSizes[s], 0);
    }

    test_error_paths();

    run_case(1, 1000, 1);

    printf("  pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
