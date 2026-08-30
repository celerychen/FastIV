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

/* Cross-language oracle harness for the 13 problems in tests/test_pdlp.py.
 *
 * The problem data lives in lp13_data.h (auto-generated from
 * lp13_problems.py, transcribed verbatim from test_pdlp.py). For each
 * problem we:
 *   1. solve through BOTH the DENSE fiv_mat backend and the SPARSE CSR/CSC
 *      backend -> dense/sparse dual-path consistency (project discipline).
 *   2. self-check the status against the expected status set, the solution
 *      against the analytic expected x (when unique), and (for OPTIMAL) the
 *      solver's own relative_gap / primal_residual / dual_residual.
 *   3. print an "ORACLE13 <name> <status> <primal_obj> <dual_obj> <x...>"
 *      line consumed by oracle_lp_solve13.py (real torch pdlp.py) and
 *      compare_oracle13.py, giving a genuine cross-language agreement check.
 *
 * This is the definitive validation target the project mandates: the 13
 * problems from tests/test_pdlp.py, matched status/x/obj against torch. */

#include "fiv_lp_solve.h"
#include "fiv_lp_mat.h"
#include "fiv_lp_vec.h"
#include "fiv_ctensor.h"   /* fiv_create_tensor1d / fiv_release_tensor1d */
#include "fiv_matrix.h"    /* fiv_create_tensor2d / fiv_release_tensor2d */
#include "fiv_sp_matrix.h" /* fiv_sparse_runtime_init */
#include "lp13_data.h"
#include "fiv_common.h"      /* fiv_malloc / fiv_free */

#include <stdio.h>
#include <stdlib.h>          /* NULL */
#include <string.h>
#include <math.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (cond) { g_pass++; }                                                 \
        else { g_fail++; printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); } \
    } while (0)

static int vec_close_f64(const fiv_vec *result_vec, const double *reference,
                         size_t length, double tol)
{
    if (result_vec->length != length) return 0;
    const double *data = result_vec->data.db;
    for (size_t index = 0; index < length; index++)
        if (fabs(data[index] - reference[index]) > tol * (1.0 + fabs(reference[index]))) return 0;
    return 1;
}

static int scalar_close(double value, double reference, double tol)
{
    return fabs(value - reference) <= tol * (1.0 + fabs(reference));
}

static fiv_vec *make_vec(size_t length, const double *init)
{
    fiv_vec *vec = fiv_create_tensor1d(length, FIV_64F1);
    if (vec == NULL) return NULL;
    if (init != NULL && length > 0) memcpy(vec->data.db, init, length * sizeof(double));
    else if (length > 0) memset(vec->data.db, 0, length * sizeof(double));
    return vec;
}

static void set_dense(fiv_mat *dense_matrix, int row, int col, double value)
{
    const size_t element_bytes = dense_matrix->element_bytes;
    const size_t flat = (size_t)row * dense_matrix->strides[0] / element_bytes
                      + (size_t)col * dense_matrix->strides[1] / element_bytes;
    dense_matrix->data.db[flat] = value;
}

/* Build the stacked K = [G; A] (m x n) as a DENSE fiv_mat. */
static fiv_lp_mat *build_dense_K(const lp13_problem_t *p)
{
    int m = p->m, n = p->n;
    fiv_mat *dense = fiv_create_tensor2d((size_t[2]){(size_t)m, (size_t)n}, FIV_64F1);
    if (dense == NULL) return NULL;
    for (int r = 0; r < p->m1; r++)
        for (int c = 0; c < n; c++)
            set_dense(dense, r, c, p->G[(size_t)r * n + c]);
    for (int r = 0; r < p->m2; r++)
        for (int c = 0; c < n; c++)
            set_dense(dense, p->m1 + r, c, p->A[(size_t)r * n + c]);
    return fiv_lp_mat_wrap_dense(dense);
}

/* Build the stacked K as a SPARSE CSR/CSC matrix (nonzero entries only). */
static fiv_lp_mat *build_sparse_K(const lp13_problem_t *p)
{
    int m = p->m, n = p->n;
    size_t nnz = 0;
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++) {
            double val = (r < p->m1) ? p->G[(size_t)r * n + c] : p->A[(size_t)(r - p->m1) * n + c];
            if (val != 0.0) nnz++;
        }
    int *coo_row = fiv_malloc(nnz * sizeof(int));
    int *coo_col = fiv_malloc(nnz * sizeof(int));
    double *coo_val = fiv_malloc(nnz * sizeof(double));
    if (coo_row == NULL || coo_col == NULL || coo_val == NULL) {
        fiv_free(coo_row); fiv_free(coo_col); fiv_free(coo_val);
        return NULL;
    }
    size_t k = 0;
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++) {
            double val = (r < p->m1) ? p->G[(size_t)r * n + c] : p->A[(size_t)(r - p->m1) * n + c];
            if (val != 0.0) {
                coo_row[k] = r; coo_col[k] = c; coo_val[k] = val; k++;
            }
        }
    fiv_lp_mat *mat = fiv_create_lp_mat_from_coo(coo_row, coo_col, coo_val, FIV_64F1,
                                                nnz, (size_t)m, (size_t)n);
    fiv_free(coo_row); fiv_free(coo_col); fiv_free(coo_val);
    return mat;
}

static void run_one(const lp13_problem_t *p)
{
    int trivial = (p->m == 0 || p->n == 0);
    int m = p->m, n = p->n;

    fiv_lp_mat *Kd, *Ks;
    if (trivial) {
        fiv_mat *dummy = fiv_create_tensor2d((size_t[2]){1, 1}, FIV_64F1);
        Kd = fiv_lp_mat_wrap_dense(dummy);
        Ks = Kd;
    } else {
        Kd = build_dense_K(p);
        Ks = build_sparse_K(p);
    }

    fiv_vec *c = make_vec(n, p->c);
    fiv_vec *l = make_vec(n, p->l);
    fiv_vec *u = make_vec(n, p->u);
    /* q = [h (m1 elems); b (m2 elems)] — build without over-reading p->h */
    fiv_vec *q = make_vec(m, NULL);
    if (m > 0) {
        for (int i = 0; i < p->m1; i++) q->data.db[i] = p->h[i];
        for (int i = 0; i < p->m2; i++) q->data.db[p->m1 + i] = p->b[i];
    }

    fiv_lp_solve_params params;
    memset(&params, 0, sizeof(params));
    params.iteration_limit        = 10000;   /* matches pdlp.py default */
    params.eps_tol                = 1e-4;    /* matches pdlp.py default */
    params.ruiz_iterations        = 10;
    params.pock_chambolle_alpha   = 1.0;
    params.primal_weight_smoothing = 0.5;
    params.eps_primal_infeasible  = 1e-8;
    params.eps_dual_infeasible    = 1e-8;
    params.time_sec_limit         = INFINITY;

    fiv_vec *xd = make_vec(n, NULL), *yd = make_vec(m, NULL);
    fiv_vec *xs = make_vec(n, NULL), *ys = make_vec(m, NULL);
    fiv_lp_solve_info id, is;
    memset(&id, 0, sizeof(id));
    memset(&is, 0, sizeof(is));

    fiv_lp_status sd = fiv_lp_solve(Kd, c, l, u, q, p->m1, &params, xd, yd, &id);
    fiv_lp_status ss =     fiv_lp_solve(Ks, c, l, u, q, p->m1, &params, xs, ys, &is);

    /* (1) DENSE vs SPARSE consistency.
     * Both backends solve the same problem; for a UNIQUE optimum they must
     * agree on the exact vertex (tight 1e-6 below). For a DEGENERATE optimum
     * (e.g. P13 transportation LP, a large optimal face) first-order methods
     * land on different vertices of the same face; there the meaningful
     * consistency test is identical status + identical optimal value (rel 1e-3)
     * + both feasible. P1/P2/P4/P6/P11 carry an analytic exp_x, so the unique
     * case is covered; here we require exact x agreement only when a unique
     * optimum is expected. */
    CHECK(sd == ss, "dense/sparse status match");
    CHECK(scalar_close(id.primal_obj, is.primal_obj, 1e-3), "dense/sparse primal_obj match");
    CHECK(scalar_close(id.dual_obj, is.dual_obj, 1e-3), "dense/sparse dual_obj match");
    /* Self-consistency residual tolerance matches PDLP's termination guarantee:
     * feas <= eps_tol * (1 + ||q||). For degenerate LPs (e.g. P13 transportation)
     * the solver correctly terminates at ~eps_tol*(1+||q||); a tighter absolute
     * 1e-3 would be stricter than pdlp.py itself. */
    ivf64 q_norm = 0.0;
    for (int i = 0; i < m; i++) q_norm += q->data.db[i] * q->data.db[i];
    q_norm = sqrt(q_norm);
    const ivf64 feas_tol = params.eps_tol * (1.0 + q_norm);
    if (sd == FIV_LP_STATUS_OPTIMAL) {
        CHECK(id.primal_residual <= feas_tol && is.primal_residual <= feas_tol,
              "dense/sparse both primal-feasible");
        CHECK(id.dual_residual <= feas_tol && is.dual_residual <= feas_tol,
              "dense/sparse both dual-feasible");
    }
    /* Exact vertex agreement only when the optimum is provably unique. */
    if (p->exp_x != NULL) {
        if (n > 0) CHECK(vec_close_f64(xd, xs->data.db, n, 1e-6), "dense/sparse x match (unique opt)");
        if (m > 0) CHECK(vec_close_f64(yd, ys->data.db, m, 1e-6), "dense/sparse y match (unique opt)");
    }

    /* (2) status against expected set */
    int status_ok = 0;
    if (sd == FIV_LP_STATUS_OPTIMAL && p->exp_optimal) status_ok = 1;
    if (sd == FIV_LP_STATUS_PRIMAL_INFEASIBLE && p->exp_primal_inf) status_ok = 1;
    if (sd == FIV_LP_STATUS_DUAL_INFEASIBLE && p->exp_dual_inf) status_ok = 1;
    if (sd == FIV_LP_STATUS_ITERATION_LIMIT && p->exp_iter_limit) status_ok = 1;
    if (sd == FIV_LP_STATUS_TIME_LIMIT && p->exp_iter_limit) status_ok = 1;
    CHECK(status_ok, "status matches expected set");

    /* (2) analytic solution */
    if (p->exp_x != NULL && n > 0)
        CHECK(vec_close_f64(xd, p->exp_x, n, p->x_tol), "x matches expected");

    /* (3) self-consistency for OPTIMAL */
    if (sd == FIV_LP_STATUS_OPTIMAL) {
        CHECK(id.relative_gap <= 1e-3, "relative_gap small (OPTIMAL claim)");
        CHECK(id.primal_residual <= feas_tol, "primal_residual small (OPTIMAL claim)");
        CHECK(id.dual_residual <= feas_tol, "dual_residual small (OPTIMAL claim)");
    }
    if (sd == FIV_LP_STATUS_PRIMAL_INFEASIBLE || sd == FIV_LP_STATUS_DUAL_INFEASIBLE)
        CHECK(isfinite(id.certificate_quality), "certificate_quality finite");

    /* ORACLE13 summary line (consumed by compare_oracle13.py vs real torch pdlp) */
    printf("ORACLE13 %s %d %.12g %.12g", p->name, (int)sd, id.primal_obj, id.dual_obj);
    for (int i = 0; i < n; i++) printf(" %.12g", xd->data.db[i]);
    printf("\n");

    fiv_release_tensor1d(&c); fiv_release_tensor1d(&l);
    fiv_release_tensor1d(&u); fiv_release_tensor1d(&q);
    fiv_release_tensor1d(&xd); fiv_release_tensor1d(&yd);
    fiv_release_tensor1d(&xs); fiv_release_tensor1d(&ys);
    if (!trivial) fiv_release_lp_mat(&Ks);
    fiv_release_lp_mat(&Kd);
}

int main(void)
{
    fiv_sparse_runtime_init();

    for (int i = 0; i < LP13_N; i++) {
        printf("--- %s ---\n", LP13_TABLE[i].name);
        run_one(&LP13_TABLE[i]);
    }

    printf("\nPASS=%d FAIL=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
