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

/* Correctness tests for fiv_matrix_lu (api/fiv_matrix.h): blocked partial-
 * pivoting LU. The primary check is the defining identity P*A = L*U: the
 * returned pivot sequence is replayed on a double copy of the input and
 * compared against the packed factors read back from storage, on NB-boundary-
 * crossing square and rectangular sizes. Elementwise comparison against a
 * double reference factor is deliberately avoided: float vs double panels can
 * pick different (but equally valid) near-tie pivots, which is legal and not
 * an error. Also checks pivot-index validity, element growth, handcrafted
 * exact small cases, and the error paths (params, dtype, contiguity,
 * singular inputs). A small perf smoke prints single-thread GFLOPS at
 * n=1000. */

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

static unsigned g_seed = 20260828u;
static float rnd(void)
{
    g_seed = g_seed * 1103515245u + 12345u;
    return (float)((g_seed >> 8) & 0xffff) / 4096.f - 4.f;
}

static float fabsf_local(float x) { return x < 0 ? -x : x; }

/* || P*A - L*U ||_F / ||A||_F with P replayed from the returned pivots and
   L/U read from the packed row-major storage (unit diagonal of L implicit).
   Also reports max|storage| for the growth check. */
static double reconstruct(const ivf32* lu, const int* piv,
                          const double* a0, int m, int n,
                          double* out_growth)
{
    const int mn = m < n ? m : n;
    double* pa = (double*)malloc(sizeof(double) * (size_t)m * n);
    memcpy(pa, a0, sizeof(double) * (size_t)m * n);

    for (int k = 0; k < mn; k++) {
        const int r = piv[k];
        if (r != k) {
            for (int j = 0; j < n; j++) {
                double t = pa[(size_t)k * n + j];
                pa[(size_t)k * n + j] = pa[(size_t)r * n + j];
                pa[(size_t)r * n + j] = t;
            }
        }
    }

    double res = 0.0, base = 0.0, growth = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int t = 0; t < mn; t++) {
                double l = (i == t) ? 1.0
                          : (i > t) ? (double)lu[(size_t)i * n + t] : 0.0;
                double u = (j >= t) ? (double)lu[(size_t)t * n + j] : 0.0;
                s += l * u;
            }
            double d = s - pa[(size_t)i * n + j];
            res += d * d;
            base += a0[(size_t)i * n + j] * a0[(size_t)i * n + j];
        }
    }
    for (int t = 0; t < m * n; t++) {
        double v = fabs((double)lu[t]);
        if (v > growth) growth = v;
    }
    free(pa);
    *out_growth = growth;
    return sqrt(res) / sqrt(base);
}

/* one case: shape + timing; random data, P*A = L*U residual, pivots, growth */
static void run_case(int m, int n, int do_timing)
{
    char name[64];
    snprintf(name, sizeof(name), "%dx%d", m, n);

    size_t sh[2] = { (size_t)m, (size_t)n };
    fiv_mat* M = fiv_create_tensor2d(sh, FIV_32F1);
    CHECK(M != NULL, name);
    if (M == NULL) return;

    const int mn = m < n ? m : n;
    double* a0 = (double*)malloc(sizeof(double) * (size_t)m * n);
    int* piv = (int*)malloc(sizeof(int) * (size_t)mn);

    ivf32* p = M->data.fl;
    for (int i = 0; i < m * n; i++) {
        float v = rnd();
        p[i] = v;
        a0[i] = (double)v;
    }

    clock_t t0 = clock();
    fiv_ret ret = fiv_matrix_lu(M, piv);
    clock_t t1 = clock();
    CHECK(ret == FIV_RET_OK, name);
    if (ret != FIV_RET_OK) goto out;

    if (do_timing && m >= 500 && n >= 500) {
        double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
        double gflops = (2.0 * (double)mn * mn * (m > n ? m : n) / 3.0)
                        / secs / 1e9;
        printf("  [perf] %s: %.3fs -> %.2f GFLOPS\n", name, secs, gflops);
    }

    /* pivot indices must be legal interchanges: row k swaps with a row k..m-1 */
    {
        int ok = 1;
        for (int k = 0; k < mn; k++)
            if (piv[k] < k || piv[k] >= m) ok = 0;
        CHECK(ok, "pivot indices legal");
    }

    {
        double growth = 0.0;
        double rel = reconstruct(p, piv, a0, m, n, &growth);
        CHECK(rel < 1e-3, "P*A = L*U residual small");
        if (rel > 1e-5)
            printf("    (%s residual %.2e)\n", name, rel);

        double amax = 0.0;
        for (int t = 0; t < m * n; t++) {
            double v = fabs(a0[t]);
            if (v > amax) amax = v;
        }
        CHECK(growth <= 1e4 * (amax + 1.0), "element growth bounded");
    }

out:
    free(piv);
    free(a0);
    fiv_release_tensor2d(&M);
}

static void test_error_paths(void)
{
    printf("  error paths:\n");

    size_t sh2[2] = { 2, 2 };
    fiv_mat* m22 = fiv_create_tensor2d(sh2, FIV_32F1);
    CHECK(m22 != NULL, "alloc ok");
    int piv2[2];

    CHECK(fiv_matrix_lu(NULL, piv2) == FIV_RET_ERR_PARA, "null tensor");
    CHECK(fiv_matrix_lu(m22, NULL) == FIV_RET_ERR_PARA, "null piv");

    fiv_mat* mi = fiv_create_tensor2d(sh2, FIV_32S1);
    CHECK(mi != NULL && fiv_matrix_lu(mi, piv2) == FIV_RET_ERR_NOT_SUPPORT,
          "32S rejected");
    fiv_release_tensor2d(&mi);

    m22->data_continue = 0;
    CHECK(fiv_matrix_lu(m22, piv2) == FIV_RET_ERR_PARA, "non-contiguous rejected");
    m22->data_continue = 1;

    memset(m22->data.fl, 0, sizeof(ivf32) * 4);
    CHECK(fiv_matrix_lu(m22, piv2) == FIV_RET_ERR_SINGULAR, "zero matrix singular");

    /* rank-deficient [[1,2],[2,4]]: pivot 2, multiplier 0.5, remainder 0 */
    m22->data.fl[0] = 1.f; m22->data.fl[1] = 2.f;
    m22->data.fl[2] = 2.f; m22->data.fl[3] = 4.f;
    CHECK(fiv_matrix_lu(m22, piv2) == FIV_RET_ERR_SINGULAR, "rank-2x singular");

    /* near-singular [[2000,1000],[1000,500.0001]]: float elimination leaves a
       pivot of ~1e-4 (representable, NOT zero). amax=2000 -> relative threshold
       2000*FLT_EPSILON ~= 2.4e-4, so 1e-4 is flagged singular. A literal
       `== 0.0f` test misses this and wrongly reports OK (proven by mutation). */
    m22->data.fl[0] = 2000.f; m22->data.fl[1] = 1000.f;
    m22->data.fl[2] = 1000.f; m22->data.fl[3] = 500.0001f;
    CHECK(fiv_matrix_lu(m22, piv2) == FIV_RET_ERR_SINGULAR, "near-singular detected");

    /* zero column below a real pivot: [[1,1],[0,1]] stays nonsingular */
    m22->data.fl[0] = 1.f; m22->data.fl[1] = 1.f;
    m22->data.fl[2] = 0.f; m22->data.fl[3] = 1.f;
    CHECK(fiv_matrix_lu(m22, piv2) == FIV_RET_OK, "triangular input ok");

    /* handcrafted exact case [[0,1],[1,0]]: pivot swaps to the identity */
    m22->data.fl[0] = 0.f; m22->data.fl[1] = 1.f;
    m22->data.fl[2] = 1.f; m22->data.fl[3] = 0.f;
    CHECK(fiv_matrix_lu(m22, piv2) == FIV_RET_OK, "swap case ok");
    CHECK(piv2[0] == 1 && piv2[1] == 1, "swap case pivots");
    CHECK(fabsf_local(m22->data.fl[0] - 1.f) < 1e-6f &&
          fabsf_local(m22->data.fl[1] - 0.f) < 1e-6f &&
          fabsf_local(m22->data.fl[2] - 0.f) < 1e-6f &&
          fabsf_local(m22->data.fl[3] - 1.f) < 1e-6f,
          "swap case factors exact");

    /* handcrafted exact case [[2,1],[4,3]]:
       pivot row 1 up -> [[4,3],[2,1]], mult 0.5, U22 = 1-0.5*3 = -0.5 */
    m22->data.fl[0] = 2.f; m22->data.fl[1] = 1.f;
    m22->data.fl[2] = 4.f; m22->data.fl[3] = 3.f;
    CHECK(fiv_matrix_lu(m22, piv2) == FIV_RET_OK, "handcrafted ok");
    CHECK(piv2[0] == 1 && piv2[1] == 1, "handcrafted pivots");
    CHECK(fabsf_local(m22->data.fl[0] - 4.f) < 1e-6f &&
          fabsf_local(m22->data.fl[1] - 3.f) < 1e-6f &&
          fabsf_local(m22->data.fl[2] - 0.5f) < 1e-6f &&
          fabsf_local(m22->data.fl[3] + 0.5f) < 1e-6f,
          "handcrafted factors exact");

    fiv_release_tensor2d(&m22);
}

int main(void)
{
    printf("=== test_mat_lu ===\n");

    static const int kSquares[] = { 1, 2, 3, 7, 63, 64, 65, 66, 127, 128, 129, 200 };
    for (unsigned s = 0; s < sizeof(kSquares) / sizeof(kSquares[0]); s++)
        run_case(kSquares[s], kSquares[s], 0);

    static const int kRects[][2] = {
        { 3, 7 }, { 7, 3 }, { 33, 65 }, { 65, 33 }, { 128, 200 }, { 200, 128 }
    };
    for (unsigned s = 0; s < sizeof(kRects) / sizeof(kRects[0]); s++)
        run_case(kRects[s][0], kRects[s][1], 0);

    test_error_paths();

    run_case(1000, 1000, 1);

    printf("  pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
