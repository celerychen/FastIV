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

/* Correctness tests for fiv_lp_solve (api/fiv_lp_solve.h) -- the M4 PDLP
 * driver that wires together M1 (fiv_lp_mat / fiv_lp_vec) + M2
 * (fiv_lp_rescale) + M3 (fiv_lp_pdhg) into the full solve() loop.
 *
 * Three independent cross-checks, matching the project's dense/sparse +
 * reference + oracle discipline:
 *   1. DENSE vs SPARSE path: the same problem is solved through both the DENSE
 *      fiv_mat backend and the SPARSE CSR/CSC backend; status / x / y / obj must
 *      agree (dual-path consistency).
 *   2. ANALYTIC reference: every bounded problem has a hand-derived closed-form
 *      optimum (status + objective +, for unique-vertex problems, the exact x).
 *      A wrong solver fails rather than a test that only re-checks itself.
 *   3. SELF-CONSISTENCY: an OPTIMAL result must report a small relative_gap and
 *      small primal/dual residuals (proof the returned point is genuinely
 *      feasible + stationary, not a false OPTIMAL).
 *
 * Additionally, each problem prints an "ORACLE <name> <status> <primal_obj>
 * <dual_obj>" line consumed by oracle_lp_solve.py (the real torch pdlp.py) and
 * compare_oracle.py, giving a genuine cross-language oracle agreement check. */

#include "fiv_lp_solve.h"
#include "fiv_lp_mat.h"
#include "fiv_lp_vec.h"
#include "fiv_ctensor.h"   /* fiv_create_tensor1d / fiv_release_tensor1d */
#include "fiv_matrix.h"    /* fiv_create_tensor2d / fiv_release_tensor2d */
#include "fiv_sp_matrix.h" /* fiv_sparse_runtime_init */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (cond) { g_pass++; }                                                 \
        else { g_fail++; printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); } \
    } while (0)

static int vec_close_f64(const fiv_vec *result_vec, const ivf64 *reference,
                         size_t length, ivf64 tol)
{
    if (result_vec->length != length) return 0;
    const ivf64 *data = result_vec->data.db;
    for (size_t index = 0; index < length; index++)
        if (fabs(data[index] - reference[index]) > tol * (1.0 + fabs(reference[index]))) return 0;
    return 1;
}

static int scalar_close(ivf64 value, ivf64 reference, ivf64 tol)
{
    return fabs(value - reference) <= tol * (1.0 + fabs(reference));
}

static fiv_vec *make_vec(size_t length, const ivf64 *init)
{
    fiv_vec *vec = fiv_create_tensor1d(length, FIV_64F1);
    if (vec == NULL) return NULL;
    if (init != NULL && length > 0) memcpy(vec->data.db, init, length * sizeof(ivf64));
    else if (length > 0) memset(vec->data.db, 0, length * sizeof(ivf64));
    return vec;
}

static void set_dense(fiv_mat *dense_matrix, int row, int col, ivf64 value)
{
    const size_t element_bytes = dense_matrix->element_bytes;
    const size_t flat = (size_t)row * dense_matrix->strides[0] / element_bytes
                      + (size_t)col * dense_matrix->strides[1] / element_bytes;
    dense_matrix->data.db[flat] = value;
}

static fiv_lp_mat *build_dense_K(int rows, int cols, const ivf64 *K_row_major)
{
    fiv_mat *dense = fiv_create_tensor2d((size_t[2]){(size_t)rows, (size_t)cols}, FIV_64F1);
    if (dense == NULL) return NULL;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            set_dense(dense, r, c, K_row_major[(size_t)r * cols + c]);
    return fiv_lp_mat_wrap_dense(dense);
}

static fiv_lp_mat *build_sparse_K(int rows, int cols, const int *coo_row,
                                  const int *coo_col, const ivf64 *coo_val, size_t nnz)
{
    return fiv_create_lp_mat_from_coo(coo_row, coo_col, coo_val, FIV_64F1,
                                     nnz, (size_t)rows, (size_t)cols);
}

/* =========================================================================
 * Problem definitions.  Convention: min c^T x s.t. G x >= h (first m1 rows),
 * A x = b (remaining rows), l <= x <= u.  K is the STACKED [G;A] (m x n).
 * ========================================================================= */

/* P-A: min -x1-2x2 s.t. x1+x2<=1.3, x in [0,1]^2.
 *   Unique vertex: x=(0.3,1.0), obj=-2.3 (inequality active at 1.3). */
static const ivf64 PA_K[2] = { -1.0, -1.0 };
static const int   PA_R[2] = { 0, 0 };
static const int   PA_C[2] = { 0, 1 };
static const ivf64 PA_V[2] = { -1.0, -1.0 };
static const ivf64 PA_c[2] = { -1.0, -2.0 };
static const ivf64 PA_l[2] = { 0.0, 0.0 };
static const ivf64 PA_u[2] = { 1.0, 1.0 };
static const ivf64 PA_q[1] = { -1.3 };
static const ivf64 PA_x[2] = { 0.3, 1.0 };

/* P-B: min x1+2x2 s.t. x1+x2>=1, x in [0,1]^2.
 *   Unique vertex: x=(1,0), obj=1 (inequality + box bounds active). */
static const ivf64 PB_K[2] = { 1.0, 1.0 };
static const int   PB_R[2] = { 0, 0 };
static const int   PB_C[2] = { 0, 1 };
static const ivf64 PB_V[2] = { 1.0, 1.0 };
static const ivf64 PB_c[2] = { 1.0, 2.0 };
static const ivf64 PB_l[2] = { 0.0, 0.0 };
static const ivf64 PB_u[2] = { 1.0, 1.0 };
static const ivf64 PB_q[1] = { 1.0 };
static const ivf64 PB_x[2] = { 1.0, 0.0 };

/* P-G: min -x1 s.t. x1>=0 (ineq), x1+x2=1 (eq), x in [0,1]^2.
 *   Unique vertex: x=(1,0), obj=-1 (inequality + equality + box active). */
static const ivf64 PG_K[4] = { 1.0, 0.0, 1.0, 1.0 };   /* row0=G[1,0], row1=A[1,1] */
static const int   PG_R[3] = { 0, 1, 1 };
static const int   PG_C[3] = { 0, 0, 1 };
static const ivf64 PG_V[3] = { 1.0, 1.0, 1.0 };
static const ivf64 PG_c[2] = { -1.0, 0.0 };
static const ivf64 PG_l[2] = { 0.0, 0.0 };
static const ivf64 PG_u[2] = { 1.0, 1.0 };
static const ivf64 PG_q[2] = { 0.0, 1.0 };
static const ivf64 PG_x[2] = { 1.0, 0.0 };

/* P-F: min x0+x1+x2 s.t. x0>=0, x1>=0 (ineq), x2=1 (eq), all vars FREE.
 *   Unique optimum: x=(0,0,1), obj=1 (exercises free bounds + ineq + eq). */
static const ivf64 PF_K[9] = { 1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0 };
static const int   PF_R[3] = { 0, 1, 2 };
static const int   PF_C[3] = { 0, 1, 2 };
static const ivf64 PF_V[3] = { 1.0, 1.0, 1.0 };
static const ivf64 PF_c[3] = { 1.0, 1.0, 1.0 };
static const ivf64 PF_l[3] = { -INFINITY, -INFINITY, -INFINITY };
static const ivf64 PF_u[3] = { INFINITY, INFINITY, INFINITY };
static const ivf64 PF_q[3] = { 0.0, 0.0, 1.0 };
static const ivf64 PF_x[3] = { 0.0, 0.0, 1.0 };

/* P-C: trivial n==0. q=[h=3, b=0], m1=1. 0>=h fails -> PRIMAL_INFEASIBLE,
 *   Farkas y_ray = [1, 0]. */
static const ivf64 PC_q[2] = { 3.0, 0.0 };
static const ivf64 PC_y[2] = { 1.0, 0.0 };   /* expected certificate */

/* P-D: trivial m==0, bounded. min x1+x2 over [0,1]^2 -> x=(0,0), obj=0. */
static const ivf64 PD_c[2] = { 1.0, 1.0 };
static const ivf64 PD_l[2] = { 0.0, 0.0 };
static const ivf64 PD_u[2] = { 1.0, 1.0 };
static const ivf64 PD_x[2] = { 0.0, 0.0 };

/* P-E: trivial m==0, unbounded. min -x1-x2 over x>=0, no upper -> DUAL_INFEASIBLE. */
static const ivf64 PE_c[2] = { -1.0, -1.0 };
static const ivf64 PE_l[2] = { 0.0, 0.0 };
static const ivf64 PE_u[2] = { INFINITY, INFINITY };

/* P-I: NON-TRIVIAL primal infeasible. x0>=1 AND x0<=0 (i.e. -x0>=0), free box.
 *   Contradictory inequalities -> PRIMAL_INFEASIBLE (Farkas certificate y=[1,1]). */
static const ivf64 PI_K[4] = { 1.0, 0.0, -1.0, 0.0 };   /* row0=[1,0], row1=[-1,0] */
static const int   PI_R[2] = { 0, 1 };
static const int   PI_C[2] = { 0, 0 };
static const ivf64 PI_V[2] = { 1.0, -1.0 };
static const ivf64 PI_c[2] = { 0.0, 0.0 };
static const ivf64 PI_l[2] = { -INFINITY, -INFINITY };
static const ivf64 PI_u[2] = { INFINITY, INFINITY };
static const ivf64 PI_q[2] = { 1.0, 0.0 };

/* P-J: NON-TRIVIAL dual infeasible (primal unbounded). min -x0 s.t. x0>=0,
 *   x0 free above -> x0 -> +inf drives obj to -inf -> DUAL_INFEASIBLE. */
static const ivf64 PJ_K[2] = { 1.0, 0.0 };               /* G=[1,0] (x0>=0) */
static const int   PJ_R[1] = { 0 };
static const int   PJ_C[1] = { 0 };
static const ivf64 PJ_V[1] = { 1.0 };
static const ivf64 PJ_c[2] = { -1.0, 0.0 };
static const ivf64 PJ_l[2] = { 0.0, -INFINITY };
static const ivf64 PJ_u[2] = { INFINITY, INFINITY };
static const ivf64 PJ_q[1] = { 0.0 };

typedef struct {
    const char *name;
    int n, m, m1;
    const ivf64 *K_dense;          /* m*n row-major (NULL ok if trivial) */
    size_t nnz;
    const int *coo_row, *coo_col;
    const ivf64 *coo_val;
    const ivf64 *c, *l, *u, *q;
    int   exp_status;
    ivf64 exp_obj;                 /* NAN to skip */
    const ivf64 *exp_x;            /* NULL to skip */
    ivf64 point_tol, obj_tol;
} problem_t;

static void run_one(const problem_t *p)
{
    int trivial = (p->m == 0 || p->n == 0);

    fiv_lp_mat *Kd, *Ks;
    if (trivial) {
        /* solve() returns inside the trivial branch before dereferencing K,
         * so a benign 1x1 dense matrix stands in for both paths. */
        fiv_mat *dummy = fiv_create_tensor2d((size_t[2]){1, 1}, FIV_64F1);
        Kd = fiv_lp_mat_wrap_dense(dummy);
        Ks = Kd;
    } else {
        Kd = build_dense_K(p->m, p->n, p->K_dense);
        Ks = build_sparse_K(p->m, p->n, p->coo_row, p->coo_col, p->coo_val, p->nnz);
    }

    fiv_vec *c = make_vec(p->n, p->c);
    fiv_vec *l = make_vec(p->n, p->l);
    fiv_vec *u = make_vec(p->n, p->u);
    fiv_vec *q = make_vec(p->m, p->q);

    fiv_lp_solve_params params;
    memset(&params, 0, sizeof(params));
    params.iteration_limit        = 20000;
    params.eps_tol                = 1e-4;
    params.ruiz_iterations        = 10;
    params.pock_chambolle_alpha   = 1.0;
    params.primal_weight_smoothing = 0.5;
    params.eps_primal_infeasible  = 1e-8;
    params.eps_dual_infeasible    = 1e-8;
    params.time_sec_limit         = INFINITY;

    fiv_vec *xd = make_vec(p->n, NULL), *yd = make_vec(p->m, NULL);
    fiv_vec *xs = make_vec(p->n, NULL), *ys = make_vec(p->m, NULL);
    fiv_lp_solve_info id, is;
    memset(&id, 0, sizeof(id));
    memset(&is, 0, sizeof(is));

    fiv_lp_status sd = fiv_lp_solve(Kd, c, l, u, q, p->m1, &params, xd, yd, &id);
    fiv_lp_status ss = fiv_lp_solve(Ks, c, l, u, q, p->m1, &params, xs, ys, &is);

    /* (1) DENSE vs SPARSE consistency */
    CHECK(sd == ss, "dense/sparse status match");
    CHECK(vec_close_f64(xd, xs->data.db, p->n, 1e-6), "dense/sparse x match");
    CHECK(vec_close_f64(yd, ys->data.db, p->m, 1e-6), "dense/sparse y match");
    CHECK(scalar_close(id.primal_obj, is.primal_obj, 1e-9), "dense/sparse primal_obj match");
    CHECK(scalar_close(id.dual_obj, is.dual_obj, 1e-9), "dense/sparse dual_obj match");

    /* (2) analytic status */
    CHECK(sd == p->exp_status, "status matches expected");

    /* (2) analytic objective */
    if (!isnan(p->exp_obj))
        CHECK(scalar_close(id.primal_obj, p->exp_obj, p->obj_tol), "primal_obj matches expected");

    /* (2) analytic solution */
    if (p->exp_x != NULL)
        CHECK(vec_close_f64(xd, p->exp_x, p->n, p->point_tol), "x matches expected");

    /* (3) self-consistency for OPTIMAL */
    if (sd == FIV_LP_STATUS_OPTIMAL) {
        CHECK(id.relative_gap <= 1e-3, "relative_gap small (OPTIMAL claim)");
        CHECK(id.primal_residual <= 1e-3, "primal_residual small (OPTIMAL claim)");
        CHECK(id.dual_residual <= 1e-3, "dual_residual small (OPTIMAL claim)");
    }
    /* certificate sanity for infeasible / unbounded */
    if (sd == FIV_LP_STATUS_PRIMAL_INFEASIBLE || sd == FIV_LP_STATUS_DUAL_INFEASIBLE)
        CHECK(isfinite(id.certificate_quality), "certificate_quality finite");

    /* explicit certificate check for the trivial infeasible case */
    if (p->exp_x == NULL && sd == FIV_LP_STATUS_PRIMAL_INFEASIBLE && p->n == 0)
        CHECK(vec_close_f64(yd, PC_y, p->m, 1e-9), "Farkas certificate matches [1,0]");

    /* ORACLE summary line (consumed by compare_oracle.py vs real torch pdlp) */
    printf("ORACLE %s %d %.12g %.12g\n", p->name, (int)sd, id.primal_obj, id.dual_obj);

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

    problem_t problems[] = {
        { "P-A", 2, 1, 1, PA_K, 2, PA_R, PA_C, PA_V, PA_c, PA_l, PA_u, PA_q,
          FIV_LP_STATUS_OPTIMAL, -2.3, PA_x, 1e-3, 1e-4 },
        { "P-B", 2, 1, 1, PB_K, 2, PB_R, PB_C, PB_V, PB_c, PB_l, PB_u, PB_q,
          FIV_LP_STATUS_OPTIMAL, 1.0, PB_x, 1e-3, 1e-4 },
        { "P-G", 2, 2, 1, PG_K, 3, PG_R, PG_C, PG_V, PG_c, PG_l, PG_u, PG_q,
          FIV_LP_STATUS_OPTIMAL, -1.0, PG_x, 1e-3, 1e-4 },
        { "P-F", 3, 3, 2, PF_K, 3, PF_R, PF_C, PF_V, PF_c, PF_l, PF_u, PF_q,
          FIV_LP_STATUS_OPTIMAL, 1.0, PF_x, 1e-3, 1e-4 },
        { "P-C", 0, 2, 1, NULL, 0, NULL, NULL, NULL, NULL, NULL, NULL, PC_q,
          FIV_LP_STATUS_PRIMAL_INFEASIBLE, NAN, NULL, 1e-9, 1e-4 },
        { "P-D", 2, 0, 0, NULL, 0, NULL, NULL, NULL, PD_c, PD_l, PD_u, NULL,
          FIV_LP_STATUS_OPTIMAL, 0.0, PD_x, 1e-3, 1e-4 },
        { "P-E", 2, 0, 0, NULL, 0, NULL, NULL, NULL, PE_c, PE_l, PE_u, NULL,
          FIV_LP_STATUS_DUAL_INFEASIBLE, NAN, NULL, 1e-3, 1e-4 },
        { "P-I", 2, 2, 2, PI_K, 2, PI_R, PI_C, PI_V, PI_c, PI_l, PI_u, PI_q,
          FIV_LP_STATUS_PRIMAL_INFEASIBLE, NAN, NULL, 1e-3, 1e-4 },
        { "P-J", 2, 1, 1, PJ_K, 1, PJ_R, PJ_C, PJ_V, PJ_c, PJ_l, PJ_u, PJ_q,
          FIV_LP_STATUS_DUAL_INFEASIBLE, NAN, NULL, 1e-3, 1e-4 },
    };
    const size_t num_problems = sizeof(problems) / sizeof(problems[0]);

    for (size_t i = 0; i < num_problems; i++) {
        printf("--- %s ---\n", problems[i].name);
        run_one(&problems[i]);
    }

    printf("\nPASS=%d FAIL=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
