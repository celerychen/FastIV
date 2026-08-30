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

/* Correctness tests for the PDLP PDHG core (api/fiv_lp_pdhg.h):
 *   - fiv_lp_adaptive_step_pdhg
 *   - fiv_lp_primal_weight_update
 *   - fiv_lp_compute_lambda_box
 *   - fiv_lp_kkt_error_sq
 *
 * Every check compares the library result against an INDEPENDENT dense
 * reference (plain ivf64 math, naive loops) re-derived straight from pdlp.py,
 * so a wrong implementation fails rather than a test that only re-checks its
 * own logic. The adaptive-step and KKT functions are additionally checked on
 * BOTH the DENSE and SPARSE K paths and the two must agree (dual-path
 * consistency), matching the project's dense/sparse cross-check discipline. */

#include "fiv_lp_pdhg.h"
#include "fiv_lp_mat.h"
#include "fiv_lp_vec.h"
#include "fiv_sp_matrix.h"
#include "fiv_ctensor.h"   /* fiv_create_tensor1d / fiv_release_tensor1d */
#include "fiv_matrix.h"    /* fiv_create_tensor2d / fiv_release_tensor2d */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 3            /* #variables */
#define M 3            /* #constraints (m1 + m2) */
#define M1 2           /* #inequality constraints */

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
    if (init) memcpy(vec->data.db, init, length * sizeof(ivf64));
    else memset(vec->data.db, 0, length * sizeof(ivf64));
    return vec;
}

static void set_dense(fiv_mat *dense_matrix, int row, int col, ivf64 value)
{
    const size_t element_bytes = dense_matrix->element_bytes;
    const size_t flat = (size_t)row * dense_matrix->strides[0] / element_bytes
                      + (size_t)col * dense_matrix->strides[1] / element_bytes;
    dense_matrix->data.db[flat] = value;
}

/* ============================ dense references ============================ */
static void dense_matvec(const ivf64 *K, int rows, int cols, const ivf64 *x, ivf64 *out)
{
    for (int row = 0; row < rows; row++) {
        ivf64 sum = 0.0;
        for (int col = 0; col < cols; col++) sum += K[(size_t)row * cols + col] * x[col];
        out[row] = sum;
    }
}
static void dense_matvec_t(const ivf64 *K, int rows, int cols, const ivf64 *x, ivf64 *out)
{
    for (int col = 0; col < cols; col++) {
        ivf64 sum = 0.0;
        for (int row = 0; row < rows; row++) sum += K[(size_t)row * cols + col] * x[row];
        out[col] = sum;
    }
}

/* Independent re-implementation of adaptive_step_pdhg (pdlp.py 386-421). */
static void ref_adaptive_step(const ivf64 *K, const ivf64 c[N], const ivf64 l[N], const ivf64 u[N],
                              const ivf64 q[M], const ivf64 x[N], const ivf64 y[M],
                              int m1, ivf64 w, ivf64 eta_hat, int k, ivf64 eps_zero,
                              ivf64 ref_xp[N], ivf64 ref_yp[M], ivf64 *ref_eta_used, ivf64 *ref_eta_hat)
{
    ivf64 weff = (w > eps_zero) ? w : eps_zero;
    ivf64 eta = (eta_hat > eps_zero) ? eta_hat : eps_zero;
    ivf64 kp1 = (ivf64)(k + 1);
    ivf64 fac1 = (k == 0) ? 1.0 : (1.0 - pow(kp1, -0.3));
    ivf64 fac2 = 1.0 + pow(kp1, -0.6);
    ivf64 KTy[N], grad_x[N], t[N], Kt[M], grad_y[M], dx[N], dy[M], Kdx[M];
    while (1) {
        dense_matvec_t(K, M, N, y, KTy);
        for (int i = 0; i < N; i++) grad_x[i] = c[i] - KTy[i];
        ivf64 step_x = eta / weff;
        for (int i = 0; i < N; i++) {
            ivf64 v = x[i] - step_x * grad_x[i];
            v = fmax(l[i], fmin(u[i], v));
            ref_xp[i] = v;
        }
        for (int i = 0; i < N; i++) t[i] = 2.0 * ref_xp[i] - x[i];
        dense_matvec(K, M, N, t, Kt);
        for (int i = 0; i < M; i++) grad_y[i] = q[i] - Kt[i];
        for (int i = 0; i < M; i++) {
            ivf64 v = y[i] + (eta * weff) * grad_y[i];
            if (i < m1 && v < 0.0) v = 0.0;
            ref_yp[i] = v;
        }
        for (int i = 0; i < N; i++) dx[i] = ref_xp[i] - x[i];
        for (int i = 0; i < M; i++) dy[i] = ref_yp[i] - y[i];
        ivf64 dx_sq = 0.0, dy_sq = 0.0;
        for (int i = 0; i < N; i++) dx_sq += dx[i] * dx[i];
        for (int i = 0; i < M; i++) dy_sq += dy[i] * dy[i];
        ivf64 movement = weff * dx_sq + dy_sq / weff;
        dense_matvec(K, M, N, dx, Kdx);
        ivf64 dyKdx = 0.0;
        for (int i = 0; i < M; i++) dyKdx += dy[i] * Kdx[i];
        ivf64 denom = 2.0 * fabs(dyKdx);
        ivf64 bar_eta = (denom <= eps_zero) ? INFINITY : (movement / denom);
        ivf64 eta_p = fac1 * bar_eta;
        ivf64 capped = fac2 * eta;
        if (eta_p > capped) eta_p = capped;
        if (eta_p < eps_zero) eta_p = eps_zero;
        if (eta <= bar_eta) { *ref_eta_used = eta; *ref_eta_hat = eta_p; return; }
        eta = eta_p;
    }
}

/* Independent re-implementation of kkt_error_sq (pdlp.py 267-303). */
static ivf64 ref_kkt(const ivf64 *K, const ivf64 c[N], const ivf64 l[N], const ivf64 u[N],
                     const ivf64 q[M], int m1, const ivf64 x[N], const ivf64 y[M],
                     ivf64 w, ivf64 eps_tol, ivf64 eps_zero)
{
    ivf64 weff = (w > eps_zero) ? w : eps_zero;
    ivf64 Kx[M], KTy[N];
    dense_matvec(K, M, N, x, Kx);
    dense_matvec_t(K, M, N, y, KTy);
    ivf64 r_eq_sq = 0.0, r_ineq_sq = 0.0;
    for (int i = 0; i < M; i++) {
        if (i < m1) { ivf64 r = q[i] - Kx[i]; if (r > 0.0) r_ineq_sq += r * r; }
        else { ivf64 r = Kx[i] - q[i]; r_eq_sq += r * r; }
    }
    ivf64 term1 = weff * weff * (r_eq_sq + r_ineq_sq);
    ivf64 g[N], lam[N];
    for (int i = 0; i < N; i++) g[i] = c[i] - KTy[i];
    for (int i = 0; i < N; i++) {
        ivf64 xi = x[i], gi = g[i], li = l[i], ui = u[i];
        int at_lower = (isfinite(li) && (xi <= li + eps_tol)) ? 1 : 0;
        int at_upper = (isfinite(ui) && (xi >= ui - eps_tol)) ? 1 : 0;
        if (at_lower && at_upper) {
            ivf64 dl = fabs(xi - li), du = fabs(ui - xi);
            at_lower = (dl <= du) ? 1 : 0;
            at_upper = (du < dl) ? 1 : 0;
        }
        ivf64 v = 0.0;
        if (at_lower) v = (gi > 0.0) ? gi : 0.0;
        else if (at_upper) v = (gi < 0.0) ? gi : 0.0;
        lam[i] = v;
    }
    ivf64 rs_sq = 0.0;
    for (int i = 0; i < N; i++) { ivf64 rs = g[i] - lam[i]; rs_sq += rs * rs; }
    ivf64 term2 = (1.0 / (weff * weff)) * rs_sq;
    ivf64 l_term = 0.0, u_term = 0.0;
    for (int i = 0; i < N; i++) {
        ivf64 li = lam[i];
        ivf64 lp = (li > 0.0) ? li : 0.0;
        ivf64 ln = (li < 0.0) ? -li : 0.0;
        if (isfinite(l[i])) l_term += l[i] * lp;
        if (isfinite(u[i])) u_term += u[i] * ln;
    }
    ivf64 qy = 0.0, cx = 0.0;
    for (int i = 0; i < M; i++) qy += q[i] * y[i];
    for (int i = 0; i < N; i++) cx += c[i] * x[i];
    ivf64 scalar = qy + l_term - u_term - cx;
    ivf64 term3 = scalar * scalar;
    return term1 + term2 + term3;
}

/* Independent re-implementation of compute_lambda_for_box (pdlp.py 234-246). */
static void ref_lambda(const ivf64 x[N], const ivf64 g[N], const ivf64 l[N], const ivf64 u[N],
                       ivf64 eps_tol, ivf64 out[N])
{
    for (int i = 0; i < N; i++) {
        ivf64 xi = x[i], gi = g[i], li = l[i], ui = u[i];
        int at_lower = (isfinite(li) && (xi <= li + eps_tol)) ? 1 : 0;
        int at_upper = (isfinite(ui) && (xi >= ui - eps_tol)) ? 1 : 0;
        if (at_lower && at_upper) {
            ivf64 dl = fabs(xi - li), du = fabs(ui - xi);
            at_lower = (dl <= du) ? 1 : 0;
            at_upper = (du < dl) ? 1 : 0;
        }
        ivf64 v = 0.0;
        if (at_lower) v = (gi > 0.0) ? gi : 0.0;
        else if (at_upper) v = (gi < 0.0) ? gi : 0.0;
        out[i] = v;
    }
}

/* Independent re-implementation of primal_weight_update (pdlp.py 219-231). */
static ivf64 ref_primal_weight(const ivf64 x_new[N], const ivf64 y_new[M],
                               const ivf64 x_old[N], const ivf64 y_old[M],
                               ivf64 w_old, ivf64 smoothing, ivf64 eps_zero)
{
    ivf64 dxn = 0.0, dyn = 0.0;
    for (int i = 0; i < N; i++) { ivf64 d = x_new[i] - x_old[i]; dxn += d * d; }
    for (int i = 0; i < M; i++) { ivf64 d = y_new[i] - y_old[i]; dyn += d * d; }
    dxn = sqrt(dxn); dyn = sqrt(dyn);
    if (dxn > eps_zero && dyn > eps_zero)
        return pow(dyn / dxn, smoothing) * pow(w_old, 1.0 - smoothing);
    return w_old;
}


/* ============================ shared problem ============================ */
/* K (M x N) stacked [G; A]; q = [h; b]; box l/u. */
static const ivf64 K_DENSE[M][N] = {
    { 1.0,  0.0,  1.0},   /* G row 0 */
    { 0.0,  1.0, -1.0},   /* G row 1 */
    { 1.0,  1.0,  1.0},   /* A row 0 */
};
static const ivf64 C[N] = { 1.0, -2.0, 0.5 };
static const ivf64 L[N] = { -1.0, -INFINITY, 0.0 };
static const ivf64 U[N] = {  2.0,  INFINITY, 1.0 };
static const ivf64 Q[M] = { 3.0, 1.0, 3.0 };   /* h0=3, h1=1, b=3 */

/* COO for K (7 nonzeros) */
static const int COO_ROW[] = {0, 0, 1, 1, 2, 2, 2};
static const int COO_COL[] = {0, 2, 1, 2, 0, 1, 2};
static const ivf64 COO_VAL[] = {1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0};
static const size_t COO_NNZ = sizeof(COO_ROW) / sizeof(COO_ROW[0]);

int main(void)
{
    fiv_sparse_runtime_init();
    const ivf64 eps_zero = 1e-12;
    const ivf64 eps_tol = 1e-4;

    /* ---- build DENSE K ---- */
    fiv_mat *dense_matrix = fiv_create_tensor2d((size_t[2]){M, N}, FIV_64F1);
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++) set_dense(dense_matrix, r, c, K_DENSE[r][c]);
    fiv_lp_mat *K_dense = fiv_lp_mat_wrap_dense(dense_matrix);

    /* ---- build SPARSE K (CSR) + CSC transpose view ---- */
    fiv_lp_mat *K_sparse = fiv_create_lp_mat_from_coo(COO_ROW, COO_COL, COO_VAL,
                                                     FIV_64F1, COO_NNZ, M, N);
    CHECK(K_sparse != NULL, "fiv_create_lp_mat_from_coo OK");
    CHECK(fiv_lp_mat_build_transpose(K_sparse) == FIV_RET_OK,
          "fiv_lp_mat_build_transpose (CSR->CSC) OK");
    CHECK(K_sparse->transpose_view != NULL, "transpose_view materialized");

    /* vectors */
    fiv_vec *c = make_vec(N, C);
    fiv_vec *l = make_vec(N, L);
    fiv_vec *u = make_vec(N, U);
    fiv_vec *q = make_vec(M, Q);

    /* =======================================================================
     * 1. fiv_lp_compute_lambda_box
     * ===================================================================== */
    {
        /* main problem, x at lower/upper/free boundaries */
        const ivf64 xv[N] = { -1.0, 0.0, 1.0 };   /* var0@lower, var1 free, var2@upper */
        const ivf64 g[N]  = {  2.0, -3.0, 4.0 };
        ivf64 ref[N];
        ref_lambda(xv, g, L, U, eps_tol, ref);

        fiv_vec *x_vec = make_vec(N, xv);
        fiv_vec *g_vec = make_vec(N, g);
        fiv_vec *lam = make_vec(N, NULL);
        CHECK(fiv_lp_compute_lambda_box(lam, x_vec, g_vec, l, u, eps_tol) == FIV_RET_OK,
              "compute_lambda_box OK");
        CHECK(vec_close_f64(lam, ref, N, 1e-12), "compute_lambda_box matches reference");
        fiv_release_tensor1d(&x_vec); fiv_release_tensor1d(&g_vec); fiv_release_tensor1d(&lam);
    }
    {
        /* tight box: a FIXED variable (l=u) hits both bounds -> nearer side wins */
        const int n2 = 2;
        const ivf64 xv[2] = { 0.0, 0.0 };
        const ivf64 gv[2] = { 1.0, 1.0 };
        const ivf64 lv[2] = { -2.0, 0.0 };
        const ivf64 uv[2] = {  3.0, 0.0 };   /* var1 fixed at 0 */
        ivf64 ref[2];
        ref_lambda(xv, gv, lv, uv, eps_tol, ref);   /* var0=0, var1 -> clamp(g,min0)=1 */

        fiv_vec *x_vec = make_vec(n2, xv);
        fiv_vec *g_vec = make_vec(n2, gv);
        fiv_vec *l_vec = make_vec(n2, lv);
        fiv_vec *u_vec = make_vec(n2, uv);
        fiv_vec *lam = make_vec(n2, NULL);
        CHECK(fiv_lp_compute_lambda_box(lam, x_vec, g_vec, l_vec, u_vec, eps_tol) == FIV_RET_OK,
              "compute_lambda_box (fixed var) OK");
        CHECK(vec_close_f64(lam, ref, n2, 1e-12), "compute_lambda_box (fixed var) matches reference");
        fiv_release_tensor1d(&x_vec); fiv_release_tensor1d(&g_vec);
        fiv_release_tensor1d(&l_vec); fiv_release_tensor1d(&u_vec); fiv_release_tensor1d(&lam);
    }

    /* =======================================================================
     * 2. fiv_lp_primal_weight_update
     * ===================================================================== */
    {
        const ivf64 x_new[N] = { 0.6, 0.1, 0.4 };
        const ivf64 y_new[M] = { 0.2, -0.1, 0.5 };
        const ivf64 x_old[N] = { 0.5, 0.0, 0.5 };
        const ivf64 y_old[M] = { 0.1, -0.2, 0.3 };
        const ivf64 w_old = 1.0, smoothing = 0.5;
        ivf64 ref = ref_primal_weight(x_new, y_new, x_old, y_old, w_old, smoothing, eps_zero);

        fiv_vec *xn = make_vec(N, x_new), *yn = make_vec(M, y_new);
        fiv_vec *xo = make_vec(N, x_old), *yo = make_vec(M, y_old);
        ivf64 got = fiv_lp_primal_weight_update(xn, yn, xo, yo, w_old, smoothing, eps_zero);
        CHECK(scalar_close(got, ref, 1e-12), "primal_weight_update matches reference");

        /* degenerate: dx == 0 -> returns w_old */
        fiv_vec *zero = make_vec(N, NULL);
        ivf64 got_degen = fiv_lp_primal_weight_update(xn, yn, xn, yo, w_old, smoothing, eps_zero);
        CHECK(scalar_close(got_degen, w_old, 1e-12), "primal_weight_update (dx=0) returns w_old");
        fiv_release_tensor1d(&xn); fiv_release_tensor1d(&yn);
        fiv_release_tensor1d(&xo); fiv_release_tensor1d(&yo); fiv_release_tensor1d(&zero);
    }

    /* =======================================================================
     * 3. fiv_lp_kkt_error_sq (dense vs sparse, vs reference)
     * ===================================================================== */
    {
        const ivf64 x[N] = { 0.5, 0.0, 0.5 };
        const ivf64 y[M] = { 0.1, -0.2, 0.3 };
        const ivf64 w = 1.0;
        ivf64 ref = ref_kkt((const ivf64 *)K_DENSE, C, L, U, Q, M1, x, y, w, eps_tol, eps_zero);

        fiv_vec *x_vec = make_vec(N, x);
        fiv_vec *y_vec = make_vec(M, y);
        ivf64 got_dense = 0.0, got_sparse = 0.0;
        CHECK(fiv_lp_kkt_error_sq(&got_dense, K_dense, c, l, u, q, M1, x_vec, y_vec,
                                  w, eps_tol, eps_zero, NULL, NULL) == FIV_RET_OK,
              "kkt_error_sq (dense) OK");
        CHECK(fiv_lp_kkt_error_sq(&got_sparse, K_sparse, c, l, u, q, M1, x_vec, y_vec,
                                  w, eps_tol, eps_zero, NULL, NULL) == FIV_RET_OK,
              "kkt_error_sq (sparse) OK");
        CHECK(scalar_close(got_dense, ref, 1e-10), "kkt_error_sq (dense) matches reference");
        CHECK(scalar_close(got_sparse, ref, 1e-10), "kkt_error_sq (sparse) matches reference");
        CHECK(scalar_close(got_dense, got_sparse, 1e-9), "kkt_error_sq dense==sparse");

        /* with cached Kx / KTy (must give same answer) */
        fiv_vec *Kx_cache = make_vec(M, NULL);
        fiv_vec *KTy_cache = make_vec(N, NULL);
        fiv_lp_mat_matvec(Kx_cache, K_dense, x_vec, 0);
        fiv_lp_mat_matvec(KTy_cache, K_dense, y_vec, 1);
        ivf64 got_cached = 0.0;
        CHECK(fiv_lp_kkt_error_sq(&got_cached, K_dense, c, l, u, q, M1, x_vec, y_vec,
                                  w, eps_tol, eps_zero, Kx_cache, KTy_cache) == FIV_RET_OK,
              "kkt_error_sq (cached) OK");
        CHECK(scalar_close(got_cached, ref, 1e-10), "kkt_error_sq (cached) matches reference");
        fiv_release_tensor1d(&x_vec); fiv_release_tensor1d(&y_vec);
        fiv_release_tensor1d(&Kx_cache); fiv_release_tensor1d(&KTy_cache);
    }

    /* =======================================================================
     * 4. fiv_lp_adaptive_step_pdhg (dense vs sparse, vs reference)
     * ===================================================================== */
    {
        const ivf64 x[N] = { 0.5, 0.0, 0.5 };
        const ivf64 y[M] = { 0.1, -0.2, 0.3 };
        const ivf64 w = 1.0, eta_hat = 0.5;
        const int k = 0;
        ivf64 ref_xp[N], ref_yp[M], ref_eta_used = 0.0, ref_eta_hat = 0.0;
        ref_adaptive_step((const ivf64 *)K_DENSE, C, L, U, Q, x, y, M1, w, eta_hat, k,
                          eps_zero, ref_xp, ref_yp, &ref_eta_used, &ref_eta_hat);

        fiv_vec *x_vec = make_vec(N, x);
        fiv_vec *y_vec = make_vec(M, y);
        fiv_vec *xp_d = make_vec(N, NULL), *yp_d = make_vec(M, NULL);
        fiv_vec *xp_s = make_vec(N, NULL), *yp_s = make_vec(M, NULL);
        fiv_lp_step_result res_d, res_s;

        CHECK(fiv_lp_adaptive_step_pdhg(x_vec, y_vec, xp_d, yp_d, K_dense, c, l, u, q,
                                        M1, w, eta_hat, k, eps_zero, &res_d) == FIV_RET_OK,
              "adaptive_step (dense) OK");
        CHECK(fiv_lp_adaptive_step_pdhg(x_vec, y_vec, xp_s, yp_s, K_sparse, c, l, u, q,
                                        M1, w, eta_hat, k, eps_zero, &res_s) == FIV_RET_OK,
              "adaptive_step (sparse) OK");

        CHECK(vec_close_f64(xp_d, ref_xp, N, 1e-10), "adaptive_step x_p (dense) matches reference");
        CHECK(vec_close_f64(yp_d, ref_yp, M, 1e-10), "adaptive_step y_p (dense) matches reference");
        CHECK(vec_close_f64(xp_s, ref_xp, N, 1e-10), "adaptive_step x_p (sparse) matches reference");
        CHECK(vec_close_f64(yp_s, ref_yp, M, 1e-10), "adaptive_step y_p (sparse) matches reference");
        CHECK(scalar_close(res_d.eta_used, ref_eta_used, 1e-10), "adaptive_step eta_used matches reference");
        CHECK(scalar_close(res_d.eta_hat_next, ref_eta_hat, 1e-10), "adaptive_step eta_hat_next matches reference");
        CHECK(scalar_close(res_d.eta_used, res_s.eta_used, 1e-9), "adaptive_step eta_used dense==sparse");
        CHECK(scalar_close(res_d.eta_hat_next, res_s.eta_hat_next, 1e-9), "adaptive_step eta_hat_next dense==sparse");
        CHECK(vec_close_f64(xp_d, xp_s->data.db, N, 1e-9), "adaptive_step x_p dense==sparse");
        CHECK(vec_close_f64(yp_d, yp_s->data.db, M, 1e-9), "adaptive_step y_p dense==sparse");

        /* k != 0: verify the fac1 / fac2 decay factors engage (no-error path). */
        const int k1 = 5;
        ivf64 ref_xp1[N], ref_yp1[M], ref_eu1 = 0.0, ref_eh1 = 0.0;
        ref_adaptive_step((const ivf64 *)K_DENSE, C, L, U, Q, x, y, M1, w, eta_hat, k1,
                          eps_zero, ref_xp1, ref_yp1, &ref_eu1, &ref_eh1);
        fiv_vec *xp_d1 = make_vec(N, NULL), *yp_d1 = make_vec(M, NULL);
        fiv_lp_step_result res_d1;
        CHECK(fiv_lp_adaptive_step_pdhg(x_vec, y_vec, xp_d1, yp_d1, K_dense, c, l, u, q,
                                        M1, w, eta_hat, k1, eps_zero, &res_d1) == FIV_RET_OK,
              "adaptive_step (k=5) OK");
        CHECK(vec_close_f64(xp_d1, ref_xp1, N, 1e-10), "adaptive_step x_p (k=5) matches reference");
        CHECK(scalar_close(res_d1.eta_used, ref_eu1, 1e-10), "adaptive_step eta_used (k=5) matches reference");

        fiv_release_tensor1d(&x_vec); fiv_release_tensor1d(&y_vec);
        fiv_release_tensor1d(&xp_d); fiv_release_tensor1d(&yp_d);
        fiv_release_tensor1d(&xp_s); fiv_release_tensor1d(&yp_s);
        fiv_release_tensor1d(&xp_d1); fiv_release_tensor1d(&yp_d1);
    }

    /* cleanup */
    fiv_release_lp_mat(&K_dense);
    fiv_release_lp_mat(&K_sparse);
    fiv_release_tensor2d(&dense_matrix);
    fiv_release_tensor1d(&c); fiv_release_tensor1d(&l);
    fiv_release_tensor1d(&u); fiv_release_tensor1d(&q);

    printf("\nPASS=%d FAIL=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
