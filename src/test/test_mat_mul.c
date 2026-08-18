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

/* Correctness tests for fiv_matrix_mul (api/fiv_matrix.h):
 * dst = alpha * op(A) * op(B) + beta * dst. Exercises both dispatch paths
 * (small non-blocked and blocked, the latter forced with
 * -DFIV_MAT_MUL_L3_LIMIT_BYTES=1), all 4 transpose combinations, alpha/beta
 * values, odd dims, k > 512, and the error paths. */

#include "fiv_matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static int g_pass = 0;
#define CHECK(c, msg)                                                           \
    do {                                                                        \
        if (!(c)) { printf("  [FAIL] %s @%d\n", (msg), __LINE__); g_fail++; }   \
        else       { g_pass++; }                                                \
    } while (0)
static float fabsf_local(float x) { return x < 0 ? -x : x; }

/* reference: C = alpha * op(A) * op(B) + beta * C, all row-major */
static void ref_gemm(int a_t, int b_t, int M, int N, int K,
                     float alpha, const float* A, int ra, int ca,
                     const float* B, int rb, int cb,
                     float beta, float* C)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) {
                float av = a_t ? A[k * ca + i] : A[i * ca + k];
                float bv = b_t ? B[j * cb + k] : B[k * cb + j];
                acc += av * bv;
            }
            C[i * N + j] = beta * C[i * N + j] + alpha * acc;
        }
    }
}

static unsigned g_seed = 12345u;
static float rnd(void)
{
    g_seed = g_seed * 1103515245u + 12345u;
    return (float)((g_seed >> 8) & 0xffff) / 4096.f - 4.f;
}

static void run_case(int a_t, int b_t, int ra, int ca, int rb, int cb,
                     float alpha, float beta, int fill_c, const char* name)
{
    int M = a_t ? ca : ra;
    int N = b_t ? rb : cb;
    size_t shA[2] = { (size_t)ra, (size_t)ca };
    size_t shB[2] = { (size_t)rb, (size_t)cb };
    size_t shC[2] = { (size_t)M, (size_t)N };

    fiv_mat* A = fiv_create_tensor2d(shA, FIV_32F1);
    fiv_mat* B = fiv_create_tensor2d(shB, FIV_32F1);
    fiv_mat* C = fiv_create_tensor2d(shC, FIV_32F1);
    CHECK(A != NULL && B != NULL && C != NULL, "alloc ok");

    for (int i = 0; i < ra * ca; i++) A->data.fl[i] = rnd();
    for (int i = 0; i < rb * cb; i++) B->data.fl[i] = rnd();
    for (int i = 0; i < M * N; i++)   C->data.fl[i] = fill_c ? rnd() : 0.f;

    float* exp = (float*)malloc((size_t)M * N * sizeof(float));
    memcpy(exp, C->data.fl, (size_t)M * N * sizeof(float));
    ref_gemm(a_t, b_t, M, N, a_t ? ra : ca, alpha,
             A->data.fl, ra, ca, B->data.fl, rb, cb, beta, exp);

    fiv_ret r = fiv_matrix_mul(C, A, B, a_t, b_t, FIV_SCALAR_FP32(alpha), FIV_SCALAR_FP32(beta));
    CHECK(r == FIV_RET_OK, name);

    if (r == FIV_RET_OK) {
        int bad = 0;
        for (int t = 0; t < M * N; t++) {
            float d = fabsf_local(C->data.fl[t] - exp[t]);
            float scale = exp[t] < 0 ? -exp[t] : exp[t];
            if (scale < 1.f) scale = 1.f;
            if (d > 1e-3f * scale) {
                if (bad < 3) printf("  [%s] elem %d: got %.6f exp %.6f\n",
                                    name, t, C->data.fl[t], exp[t]);
                bad++;
            }
        }
        CHECK(bad == 0, "result matches reference");
        CHECK(C->shapes[0] == (size_t)M && C->shapes[1] == (size_t)N,
              "metadata rewritten to M x N");
    }

    free(exp);
    fiv_release_tensor2d(&A);
    fiv_release_tensor2d(&B);
    fiv_release_tensor2d(&C);
}

static void test_error_paths(void)
{
    size_t sh2[2] = { 3, 3 };
    fiv_mat* a = fiv_create_tensor2d(sh2, FIV_32F1);
    fiv_mat* b = fiv_create_tensor2d(sh2, FIV_32F1);
    fiv_mat* c = fiv_create_tensor2d(sh2, FIV_32F1);
    CHECK(a != NULL && b != NULL && c != NULL, "alloc ok");

    CHECK(fiv_matrix_mul(c, a, b, 0, 0, FIV_SCALAR_FP32(1.f), FIV_SCALAR_FP32(0.f)) == FIV_RET_OK, "baseline 3x3 ok");
    CHECK(fiv_matrix_mul(NULL, a, b, 0, 0, FIV_SCALAR_FP32(1.f), FIV_SCALAR_FP32(0.f)) == FIV_RET_ERR_PARA, "null dst");
    CHECK(fiv_matrix_mul(c, NULL, b, 0, 0, FIV_SCALAR_FP32(1.f), FIV_SCALAR_FP32(0.f)) == FIV_RET_ERR_PARA, "null A");
    CHECK(fiv_matrix_mul(c, a, NULL, 0, 0, FIV_SCALAR_FP32(1.f), FIV_SCALAR_FP32(0.f)) == FIV_RET_ERR_PARA, "null B");
    CHECK(fiv_matrix_mul(a, a, b, 0, 0, FIV_SCALAR_FP32(1.f), FIV_SCALAR_FP32(0.f)) == FIV_RET_ERR_PARA, "in-place dst==A");
    CHECK(fiv_matrix_mul(b, a, b, 0, 0, FIV_SCALAR_FP32(1.f), FIV_SCALAR_FP32(0.f)) == FIV_RET_ERR_PARA, "in-place dst==B");

    /* unsupported dtype: 8U dst */
    fiv_mat* c8 = fiv_create_tensor2d(sh2, FIV_8U1);
    CHECK(fiv_matrix_mul(c8, a, b, 0, 0, FIV_SCALAR_FP32(1.f), FIV_SCALAR_FP32(0.f)) == FIV_RET_ERR_NOT_SUPPORT, "8U dst not supported");
    fiv_release_tensor2d(&c8);

    /* K mismatch: A(3x3) * B(5x5) */
    size_t sh5[2] = { 5, 5 };
    fiv_mat* b5 = fiv_create_tensor2d(sh5, FIV_32F1);
    CHECK(fiv_matrix_mul(c, a, b5, 0, 0, FIV_SCALAR_FP32(1.f), FIV_SCALAR_FP32(0.f)) == FIV_RET_ERR_PARA, "K mismatch");
    fiv_release_tensor2d(&b5);

    /* dst wrong shape: 3x4 vs 3x3 result */
    size_t shw[2] = { 3, 4 };
    fiv_mat* cw = fiv_create_tensor2d(shw, FIV_32F1);
    CHECK(fiv_matrix_mul(cw, a, b, 0, 0, FIV_SCALAR_FP32(1.f), FIV_SCALAR_FP32(0.f)) == FIV_RET_ERR_PARA, "dst wrong shape");
    fiv_release_tensor2d(&cw);

    /* alpha must be an fp32 scalar; a non-fp32 scalar is rejected */
    FIV_DECLAR_SCALAR_INT32(bad_a);
    bad_a.data.value_int32 = 1;
    CHECK(fiv_matrix_mul(c, a, b, 0, 0, bad_a, FIV_SCALAR_FP32(0.f)) == FIV_RET_ERR_NOT_SUPPORT, "non-fp32 alpha rejected");

    fiv_release_tensor2d(&a);
    fiv_release_tensor2d(&b);
    fiv_release_tensor2d(&c);
}

int main(void)
{
    printf("=== fiv_matrix_mul ===\n");

    /* small-matrix shapes (non-blocked path) */
    run_case(0, 0, 2, 3, 3, 4, 1.f, 0.f, 0, "A(2x3)*B(3x4) b=0");
    run_case(0, 1, 4, 5, 3, 5, 1.f, 0.f, 0, "A(4x5)*B^T(3x5) b=0");
    run_case(1, 0, 5, 4, 5, 3, 1.f, 0.f, 0, "A^T(5x4)*B(5x3) b=0");
    run_case(1, 1, 5, 4, 3, 5, 1.f, 0.f, 0, "A^T(5x4)*B^T(3x5) b=0");
    run_case(0, 0, 3, 3, 3, 3, 0.5f, 1.f, 1, "A(3x3)*B(3x3) a=.5 b=1 acc");
    run_case(0, 1, 3, 4, 5, 4, 1.f, 2.f, 1, "A(3x4)*B^T(5x4) a=1 b=2 oddN");
    run_case(1, 1, 4, 3, 6, 4, 1.f, 1.f, 1, "A^T(4x3)*B^T(6x4) b=1 acc");
    run_case(0, 0, 17, 13, 13, 19, 0.7f, 0.3f, 1, "A(17x13)*B(13x19) odd");
    run_case(0, 1, 16, 9, 21, 9, 1.f, 1.f, 1, "A(16x9)*B^T(21x9) b=1 acc");
    run_case(1, 0, 8, 5, 8, 11, 1.f, 0.f, 0, "A^T(8x5)*B(8x11)");
    run_case(0, 0, 1, 1, 1, 1, 2.f, 0.f, 0, "1x1");
    /* exactly one full 8x8 kernel tile (blocked path, no remainder logic) */
    run_case(0, 0, 8, 8, 8, 8, 1.f, 0.f, 0, "probe 8x8x8 full tile");
    run_case(0, 0, 16, 16, 16, 16, 1.f, 0.f, 0, "probe 16x16x16 2x2 tiles");
    run_case(0, 0, 8, 1024, 1024, 8, 1.f, 0.f, 0, "probe 8x1024x8 k=1024");
    run_case(1, 0, 8, 8, 8, 8, 1.f, 0.f, 0, "probe A^T 8x8x8");
    run_case(0, 1, 8, 8, 8, 8, 1.f, 0.f, 0, "probe B^T 8x8x8");

    /* shapes large enough for the blocked path (L3 limit 8MB default) */
    run_case(0, 0, 1024, 1024, 1024, 1024, 1.f, 0.f, 0, "blocked 1024^2 b=0");
    run_case(1, 0, 1024, 1024, 1024, 1024, 1.f, 0.f, 0, "blocked 1024^2 A^T");
    run_case(0, 0, 900, 900, 900, 900, 1.f, 0.f, 0, "blocked 900^2 (m%8=4,k>512)");
    run_case(1, 0, 1200, 900, 1200, 1000, 1.f, 0.f, 0, "blocked A^T 1200x900x1000");
    run_case(0, 0, 700, 700, 700, 700, 1.f, 1.f, 1, "blocked 700^2 b=1 acc");

    test_error_paths();
    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
