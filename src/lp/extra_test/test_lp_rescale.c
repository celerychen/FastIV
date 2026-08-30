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

/* Correctness of the LP rescaling (Ruiz + Pock-Chambolle).
 *
 * The oracle is a PLAIN double[][] reference that re-implements the same
 * equilibration from scratch (it does NOT call fiv_lp_mat_reduce_abs_*), so a
 * wrong sign / wrong power / wrong combine-with-c would be caught: the rescale
 * factors and the scaled matrix must both match. */

#include "fiv_lp_rescale.h"
#include "fiv_lp_mat.h"
#include "fiv_ctensor.h"   /* fiv_create_tensor1d / fiv_release_tensor1d */

#include <stdio.h>
#include <stdlib.h>          /* NULL */
#include <string.h>
#include "fiv_common.h"      /* fiv_malloc / fiv_free */
#include <math.h>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg)                                                     \
    do { if (cond) g_pass++;                                                 \
         else { g_fail++; printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); } } \
    while (0)

static int vec_close(const fiv_vec *result, const ivf64 *reference, size_t len, ivf64 tol)
{
    if (result->length != len) return 0;
    const ivf64 *data = result->data.db;
    for (size_t i = 0; i < len; i++)
        if (fabs(data[i] - reference[i]) > tol * (1.0 + fabs(reference[i]))) return 0;
    return 1;
}

/* ---------------- independent dense reference ---------------- */
/* matrix stored row-major dense[m][n]. */
static void ref_reduce_max_row(const ivf64 *dense, size_t m, size_t n, ivf64 *row_max)
{
    for (size_t i = 0; i < m; i++) { ivf64 mx = 0.0;
        for (size_t j = 0; j < n; j++) mx = fmax(mx, fabs(dense[i*n+j])); row_max[i] = mx; }
}
static void ref_reduce_max_col(const ivf64 *dense, size_t m, size_t n, ivf64 *col_max)
{
    for (size_t j = 0; j < n; j++) { ivf64 mx = 0.0;
        for (size_t i = 0; i < m; i++) mx = fmax(mx, fabs(dense[i*n+j])); col_max[j] = mx; }
}
static void ref_reduce_pow_row(const ivf64 *dense, size_t m, size_t n, ivf64 p, ivf64 *row_sum)
{
    for (size_t i = 0; i < m; i++) { ivf64 s = 0.0;
        for (size_t j = 0; j < n; j++) s += pow(fabs(dense[i*n+j]), p); row_sum[i] = s; }
}
static void ref_reduce_pow_col(const ivf64 *dense, size_t m, size_t n, ivf64 p, ivf64 *col_sum)
{
    for (size_t j = 0; j < n; j++) { ivf64 s = 0.0;
        for (size_t i = 0; i < m; i++) s += pow(fabs(dense[i*n+j]), p); col_sum[j] = s; }
}

/* Run Ruiz(pass_cnt) + Pock-Chambolle(alpha) reference; fills var_rescale[n],
 * con_rescale[m] and scaled K (overwrites dense in place, same combine rule).
 * (Oracle inlined per test block below; helper reducers declared above.) */

static fiv_mat *make_dense(size_t m, size_t n, const ivf64 *init)
{
    size_t shape[2] = { m, n };
    fiv_mat *mat = fiv_create_tensor2d(shape, FIV_64F1);
    if (mat == NULL) return NULL;
    const size_t elem_bytes = mat->element_bytes;
    const size_t s0 = mat->strides[0] / elem_bytes;
    const size_t s1 = mat->strides[1] / elem_bytes;
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            mat->data.db[i*s0 + j*s1] = init[i*n + j];
    return mat;
}

static fiv_vec *make_vec(size_t len, const ivf64 *init)
{
    fiv_vec *v = fiv_create_tensor1d(len, FIV_64F1);
    if (v == NULL) return NULL;
    if (init) memcpy(v->data.db, init, len * sizeof(ivf64));
    else memset(v->data.db, 0, len * sizeof(ivf64));
    return v;
}

/* Verify every entry of the (dense) scaled K matches K_orig / (con ⊗ var). */
static int dense_K_matches(const fiv_mat *scaled, const ivf64 *orig, size_t m, size_t n,
                           const ivf64 *var, const ivf64 *con, ivf64 tol)
{
    const size_t elem_bytes = scaled->element_bytes;
    const size_t s0 = scaled->strides[0] / elem_bytes;
    const size_t s1 = scaled->strides[1] / elem_bytes;
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++) {
            const ivf64 expected = orig[i*n+j] / (con[i] * var[j]);
            const ivf64 got = scaled->data.db[i*s0 + j*s1];
            if (fabs(got - expected) > tol * (1.0 + fabs(expected))) return 0;
        }
    return 1;
}

int main(void)
{
    const size_t m = 5, n = 4;
    const ivf64 eps = 1e-12;

    /* original problem data */
    ivf64 K0[5*4] = {
        3.0, 0.0, -1.0, 2.0,
        0.0, 7.0,  4.0, 0.0,
        5.0, 0.0,  0.0, 6.0,
        0.0, 0.0, -2.0, 0.0,
        1.0, 8.0,  0.0, 3.0,
    };
    ivf64 c0[4] = { 2.0, 0.5, 9.0, 1.0 };
    ivf64 l0[4] = { 0.0, 0.0, 0.0, 0.0 };
    ivf64 u0[4] = { 5.0, 4.0, 10.0, 3.0 };
    ivf64 q0[5] = { 1.0, 2.0, 0.0, 4.0, 3.0 };

    /* ---- DENSE path ---- */
    {
        ivf64 Kref[5*4];
        memcpy(Kref, K0, sizeof K0);
        ivf64 var_ref[4], con_ref[5];
        for (size_t i = 0; i < n; i++) var_ref[i] = 1.0; /* reset; recompute below */
        for (size_t i = 0; i < m; i++) con_ref[i] = 1.0;
        /* recompute reference factors cleanly (ref_rescale threw them away) */
        {
            ivf64 *var2 = fiv_malloc(n*sizeof(ivf64)); ivf64 *con2 = fiv_malloc(m*sizeof(ivf64));
            for (size_t i=0;i<n;i++) var2[i]=1.0; for (size_t i=0;i<m;i++) con2[i]=1.0;
            ivf64 *rm=fiv_malloc(m*sizeof(ivf64)), *cm=fiv_malloc(n*sizeof(ivf64));
            ivf64 Kwork[5*4]; memcpy(Kwork,K0,sizeof K0);
            ivf64 cw[4],qw[5]; memcpy(cw,c0,sizeof c0); memcpy(qw,q0,sizeof q0);
            for (int p=0;p<10;p++){
                ref_reduce_max_row(Kwork,m,n,rm); ref_reduce_max_col(Kwork,m,n,cm);
                for(size_t i=0;i<m;i++) rm[i]=(rm[i]>eps)?sqrt(rm[i]):1.0;
                for(size_t j=0;j<n;j++){ivf64 v=fmax(cm[j],fabs(cw[j])); cm[j]=(v>eps)?sqrt(v):1.0;}
                for(size_t j=0;j<n;j++) cw[j]/=cm[j];
                for(size_t i=0;i<m;i++) qw[i]/=rm[i];
                for(size_t i=0;i<m;i++) for(size_t j=0;j<n;j++) Kwork[i*n+j]*=(1.0/rm[i])*(1.0/cm[j]);
                for(size_t i=0;i<n;i++) var2[i]*=cm[i]; for(size_t i=0;i<m;i++) con2[i]*=rm[i];
            }
            if (1.0>0.0){}
            ivf64 *rs=fiv_malloc(m*sizeof(ivf64)),*cs=fiv_malloc(n*sizeof(ivf64));
            ref_reduce_pow_row(Kwork,m,n,1.0,rs); ref_reduce_pow_col(Kwork,m,n,1.0,cs);
            for(size_t i=0;i<m;i++) rs[i]=(rs[i]>eps)?sqrt(rs[i]):1.0;
            for(size_t j=0;j<n;j++) cs[j]=(cs[j]>eps)?sqrt(cs[j]):1.0;
            for(size_t j=0;j<n;j++) cw[j]/=cs[j];
            for(size_t i=0;i<m;i++) qw[i]/=rs[i];
            for(size_t i=0;i<m;i++) for(size_t j=0;j<n;j++) Kwork[i*n+j]*=(1.0/rs[i])*(1.0/cs[j]);
            for(size_t i=0;i<n;i++) var2[i]*=cs[i]; for(size_t i=0;i<m;i++) con2[i]*=rs[i];
            memcpy(var_ref,var2,n*sizeof(ivf64)); memcpy(con_ref,con2,m*sizeof(ivf64));
            fiv_free(rm);fiv_free(cm);fiv_free(rs);fiv_free(cs);fiv_free(var2);fiv_free(con2);
        }

        fiv_mat *K = make_dense(m, n, K0);
        fiv_vec *c = make_vec(n, c0), *l = make_vec(n, l0), *u = make_vec(n, u0), *q = make_vec(m, q0);
        fiv_lp_mat *M = fiv_lp_mat_wrap_dense(K);
        fiv_lp_rescaling *R = fiv_create_lp_rescaling(n, m);
        fiv_ret rc = fiv_lp_rescale_solve(R, M, c, l, u, q, 10, 1.0, eps);
        CHECK(rc == FIV_RET_OK, "dense rescale_solve OK");
        CHECK(vec_close(R->variable_rescaling, var_ref, n, 1e-12), "dense var_rescale matches oracle");
        CHECK(vec_close(R->constraint_rescaling, con_ref, m, 1e-12), "dense con_rescale matches oracle");
        CHECK(dense_K_matches(K, K0, m, n, var_ref, con_ref, 1e-12), "dense K == K0/(con⊗var)");
        /* equilibration reduces scale: row/col max abs of scaled K <= ~1 */
        {
            ivf64 rmx[5], cmx[4]; ref_reduce_max_row(Kref,m,n,rmx); ref_reduce_max_col(Kref,m,n,cmx);
            ivf64 rmx2[5], cmx2[4];
            ivf64 *flat = fiv_malloc(m*n*sizeof(ivf64));
            const size_t eb=K->element_bytes, s0=K->strides[0]/eb, s1=K->strides[1]/eb;
            for(size_t i=0;i<m;i++) for(size_t j=0;j<n;j++) flat[i*n+j]=K->data.db[i*s0+j*s1];
            ref_reduce_max_row(flat,m,n,rmx2); ref_reduce_max_col(flat,m,n,cmx2);
            fiv_free(flat);
            ivf64 bound = 1.0 + 1e-9; ivf64 ok = 1.0;
            for(size_t i=0;i<m;i++) if(rmx2[i]>bound) ok=0.0;
            for(size_t j=0;j<n;j++) if(cmx2[j]>bound) ok=0.0;
            CHECK(ok>0.5, "dense row/col max of scaled K <= 1+eps");
        }
        /* unscale round-trip */
        {
            ivf64 xsc[4] = {1.3, -0.7, 2.1, 0.4};
            fiv_vec *x = make_vec(n, xsc);
            fiv_vec *xu = make_vec(n, NULL);
            rc = fiv_lp_unscale_primal(xu, R, x);
            CHECK(rc == FIV_RET_OK, "unscale_primal OK");
            ivf64 xexp[4]; for(size_t j=0;j<n;j++) xexp[j]=xsc[j]/var_ref[j];
            CHECK(vec_close(xu, xexp, n, 1e-12), "unscale_primal == x/var_rescale");
            fiv_release_tensor1d(&x); fiv_release_tensor1d(&xu);
        }
        fiv_release_lp_mat(&M); fiv_release_lp_rescaling(&R);
        fiv_release_tensor2d(&K);
        fiv_release_tensor1d(&c); fiv_release_tensor1d(&l); fiv_release_tensor1d(&u); fiv_release_tensor1d(&q);
    }

    /* ---- SPARSE path (exercises CSC transpose for col reductions) ---- */
    {
        /* Build COO directly from K0 so the sparse matrix is identical to the
           dense oracle's matrix (zero entries are harmless in CSR). */
        size_t total_nz = m * n;
        int *coo_r = fiv_malloc(total_nz * sizeof(int));
        int *coo_c = fiv_malloc(total_nz * sizeof(int));
        ivf64 *coo_v = fiv_malloc(total_nz * sizeof(ivf64));
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < n; j++) {
                size_t k = i * n + j;
                coo_r[k] = (int)i; coo_c[k] = (int)j; coo_v[k] = K0[k];
            }

        /* dense copy of original K for oracle compare */
        ivf64 Kdense[5*4];
        for (size_t i = 0; i < m * n; i++) Kdense[i] = K0[i];

        fiv_lp_mat *M = fiv_create_lp_mat_from_coo(coo_r, coo_c, coo_v, FIV_64F1, total_nz, m, n);
        CHECK(M != NULL, "sparse create_lp_mat_from_coo OK");
        fiv_vec *c = make_vec(n, c0), *l = make_vec(n, l0), *u = make_vec(n, u0), *q = make_vec(m, q0);
        fiv_lp_rescaling *R = fiv_create_lp_rescaling(n, m);
        fiv_ret rc = fiv_lp_rescale_solve(R, M, c, l, u, q, 10, 1.0, eps);
        CHECK(rc == FIV_RET_OK, "sparse rescale_solve OK");

        /* oracle on the dense Kdense */
        ivf64 var_ref[4], con_ref[5];
        {
            ivf64 *var2=fiv_malloc(n*sizeof(ivf64)); ivf64 *con2=fiv_malloc(m*sizeof(ivf64));
            for(size_t i=0;i<n;i++) var2[i]=1.0; for(size_t i=0;i<m;i++) con2[i]=1.0;
            ivf64 *rm=fiv_malloc(m*sizeof(ivf64)),*cm=fiv_malloc(n*sizeof(ivf64));
            ivf64 Kwork[5*4]; memcpy(Kwork,Kdense,sizeof Kdense);
            ivf64 cw[4],qw[5]; memcpy(cw,c0,sizeof c0); memcpy(qw,q0,sizeof q0);
            for(int p=0;p<10;p++){
                ref_reduce_max_row(Kwork,m,n,rm); ref_reduce_max_col(Kwork,m,n,cm);
                for(size_t i=0;i<m;i++) rm[i]=(rm[i]>eps)?sqrt(rm[i]):1.0;
                for(size_t j=0;j<n;j++){ivf64 v=fmax(cm[j],fabs(cw[j])); cm[j]=(v>eps)?sqrt(v):1.0;}
                for(size_t j=0;j<n;j++) cw[j]/=cm[j];
                for(size_t i=0;i<m;i++) qw[i]/=rm[i];
                for(size_t i=0;i<m;i++) for(size_t j=0;j<n;j++) Kwork[i*n+j]*=(1.0/rm[i])*(1.0/cm[j]);
                for(size_t i=0;i<n;i++) var2[i]*=cm[i]; for(size_t i=0;i<m;i++) con2[i]*=rm[i];
            }
            ivf64 *rs=fiv_malloc(m*sizeof(ivf64)),*cs=fiv_malloc(n*sizeof(ivf64));
            ref_reduce_pow_row(Kwork,m,n,1.0,rs); ref_reduce_pow_col(Kwork,m,n,1.0,cs);
            for(size_t i=0;i<m;i++) rs[i]=(rs[i]>eps)?sqrt(rs[i]):1.0;
            for(size_t j=0;j<n;j++) cs[j]=(cs[j]>eps)?sqrt(cs[j]):1.0;
            for(size_t j=0;j<n;j++) cw[j]/=cs[j];
            for(size_t i=0;i<m;i++) qw[i]/=rs[i];
            for(size_t i=0;i<m;i++) for(size_t j=0;j<n;j++) Kwork[i*n+j]*=(1.0/rs[i])*(1.0/cs[j]);
            for(size_t i=0;i<n;i++) var2[i]*=cs[i]; for(size_t i=0;i<m;i++) con2[i]*=rs[i];
            memcpy(var_ref,var2,n*sizeof(ivf64)); memcpy(con_ref,con2,m*sizeof(ivf64));
            fiv_free(rm);fiv_free(cm);fiv_free(rs);fiv_free(cs);fiv_free(var2);fiv_free(con2);
        }
        CHECK(vec_close(R->variable_rescaling, var_ref, n, 1e-12), "sparse var_rescale matches oracle");
        CHECK(vec_close(R->constraint_rescaling, con_ref, m, 1e-12), "sparse con_rescale matches oracle");

        /* verify scaled sparse K entries == Kdense / (con⊗var) */
        {
            fiv_sparse_mat *sp = M->as.sparse;
            ivf64 *vals = (ivf64*)sp->hdr.data.ptr;
            int ok = 1;
            for (size_t i = 0; i < sp->rows; i++)
                for (int e = sp->indptr[i]; e < sp->indptr[i+1]; e++) {
                    size_t j = sp->indices[e];
                    ivf64 expected = Kdense[i*n + j] / (con_ref[i] * var_ref[j]);
                    if (fabs(vals[e] - expected) > 1e-12*(1.0+fabs(expected))) ok = 0;
                }
            CHECK(ok, "sparse K entries == K0/(con⊗var)");
        }

        fiv_release_lp_mat(&M); fiv_release_lp_rescaling(&R);
        fiv_release_tensor1d(&c); fiv_release_tensor1d(&l); fiv_release_tensor1d(&u); fiv_release_tensor1d(&q);
        fiv_free(coo_r); fiv_free(coo_c); fiv_free(coo_v);
    }

    /* ---- alpha == 0 disables Pock-Chambolle (Ruiz only) ---- */
    {
        ivf64 Kref[5*4]; memcpy(Kref,K0,sizeof K0);
        ivf64 var_ref[4], con_ref[5];
        {
            ivf64 *var2=fiv_malloc(n*sizeof(ivf64)); ivf64 *con2=fiv_malloc(m*sizeof(ivf64));
            for(size_t i=0;i<n;i++) var2[i]=1.0; for(size_t i=0;i<m;i++) con2[i]=1.0;
            ivf64 *rm=fiv_malloc(m*sizeof(ivf64)),*cm=fiv_malloc(n*sizeof(ivf64));
            ivf64 Kwork[5*4]; memcpy(Kwork,K0,sizeof K0);
            ivf64 cw[4],qw[5]; memcpy(cw,c0,sizeof c0); memcpy(qw,q0,sizeof q0);
            for(int p=0;p<10;p++){
                ref_reduce_max_row(Kwork,m,n,rm); ref_reduce_max_col(Kwork,m,n,cm);
                for(size_t i=0;i<m;i++) rm[i]=(rm[i]>eps)?sqrt(rm[i]):1.0;
                for(size_t j=0;j<n;j++){ivf64 v=fmax(cm[j],fabs(cw[j])); cm[j]=(v>eps)?sqrt(v):1.0;}
                for(size_t j=0;j<n;j++) cw[j]/=cm[j];
                for(size_t i=0;i<m;i++) qw[i]/=rm[i];
                for(size_t i=0;i<m;i++) for(size_t j=0;j<n;j++) Kwork[i*n+j]*=(1.0/rm[i])*(1.0/cm[j]);
                for(size_t i=0;i<n;i++) var2[i]*=cm[i]; for(size_t i=0;i<m;i++) con2[i]*=rm[i];
            }
            memcpy(var_ref,var2,n*sizeof(ivf64)); memcpy(con_ref,con2,m*sizeof(ivf64));
            fiv_free(rm);fiv_free(cm);fiv_free(var2);fiv_free(con2);
        }
        fiv_mat *K = make_dense(m, n, K0);
        fiv_vec *c = make_vec(n, c0), *l = make_vec(n, l0), *u = make_vec(n, u0), *q = make_vec(m, q0);
        fiv_lp_mat *M = fiv_lp_mat_wrap_dense(K);
        fiv_lp_rescaling *R = fiv_create_lp_rescaling(n, m);
        fiv_ret rc = fiv_lp_rescale_solve(R, M, c, l, u, q, 10, 0.0, eps); /* alpha=0 -> PC skipped */
        CHECK(rc == FIV_RET_OK, "rescale_solve (alpha=0) OK");
        CHECK(vec_close(R->variable_rescaling, var_ref, n, 1e-12), "alpha=0: only Ruiz, var matches");
        CHECK(vec_close(R->constraint_rescaling, con_ref, m, 1e-12), "alpha=0: only Ruiz, con matches");
        fiv_release_lp_mat(&M); fiv_release_lp_rescaling(&R);
        fiv_release_tensor2d(&K);
        fiv_release_tensor1d(&c); fiv_release_tensor1d(&l); fiv_release_tensor1d(&u); fiv_release_tensor1d(&q);
    }

    printf("\nPASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
