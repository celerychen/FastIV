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

/*
 * test_matvet_cuda.cu - CUDA 版综合测试 (正确性 + 性能对比, 含 v1/v2/v3 优化版)
 *
 * 正确性: 朴素 CPU 参考对比 CUDA 输出, 覆盖多尺寸 (含非对齐/padding),
 *         对 naive 版、v1、v2、v3 都验。
 * 性能:   多种矩阵尺寸, 分别计时 H2D / kernel / D2H / GPU总 / CPU,
 *         并同框对比 naive / v1 / v2 / v3 四种 GPU 实现 (mat·vec 与 mat^T·vec)。
 *
 * 编译 (CUDA 12.9 / RTX 3060, sm_86):
 *   nvcc -O2 -arch=sm_86 -ccbin "C:/.../14.44.35207/bin/HostX64/x64" -I. ^
 *     test_matvet_cuda.cu matvet_cuda.cu matvet_cuda_v1.cu matvet_cuda_v2.cu matvet_v0.c -o test_cuda.exe
 */

#include "matvet_cuda.h"
#include "matvet_cuda_v1.h"
#include "matvet_cuda_v2.h"
#include "matvet_cuda_v3.h"
#include "matvet_cuda_v4.h"
#include "matvet_cuda_v5.h"
#include "matvet_cuda_cublas.h"
#include "matvet.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#ifndef IVF32_DEFINED
typedef float ivf32;
#endif

#define CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error @ %s (line %d): %s\n", #e, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

/* ---------- 朴素 CPU 参考 (oracle) ---------- */
static void ref_mat_mul(ivf32* dst, const ivf32* mat, int rows, int cols, int stride, const ivf32* vec)
{
    for (int i = 0; i < rows; i++)
    {
        const ivf32* r = mat + (size_t)i * stride;
        float s = 0.0f;
        for (int j = 0; j < cols; j++) s += (float)r[j] * (float)vec[j];
        dst[i] = (ivf32)s;
    }
}
static void ref_mat_t_mul(ivf32* dst, const ivf32* mat, int rows, int cols, int stride, const ivf32* vec)
{
    memset(dst, 0, (size_t)cols * sizeof(ivf32));
    for (int i = 0; i < rows; i++)
    {
        const ivf32* r = mat + (size_t)i * stride;
        float vi = (float)vec[i];
        for (int j = 0; j < cols; j++) dst[j] += (ivf32)((float)r[j] * vi);
    }
}

static int fequal(float a, float b)
{
    float d = fabsf(a - b);
    return d <= 1e-2f || d <= 1e-3f * fabsf(a);
}
static int check_arr(const char* tag, const ivf32* got, const ivf32* ref, int n, int* fail)
{
    for (int i = 0; i < n; i++)
        if (!fequal((float)got[i], (float)ref[i]))
        {
            printf("    FAIL %s[%d] = %g, expect %g\n", tag, i, (float)got[i], (float)ref[i]);
            if (fail) (*fail) += 1;
            return 0;
        }
    return 1;
}
static float rnd_val(void) { return (float)((rand() % 1000) / 250.0 - 2.0); }

/* ===================== 正确性用例 ===================== */
static int run_case(const char* impl, int rows, int cols, int stride,
                    void (*cuda_mm)(ivf32*, const ivf32*, int, int, int, const ivf32*),
                    void (*cuda_mt)(ivf32*, const ivf32*, int, int, int, const ivf32*))
{
    int fail = 0;
    int mat_cap = (int)((size_t)rows * stride);
    ivf32* mat  = (ivf32*)malloc((size_t)mat_cap * sizeof(ivf32));
    ivf32* vec  = (ivf32*)malloc((size_t)cols * sizeof(ivf32));
    ivf32* vecR = (ivf32*)malloc((size_t)rows * sizeof(ivf32));
    for (int k = 0; k < mat_cap; k++) mat[k] = (ivf32)rnd_val();
    for (int k = 0; k < cols;    k++) vec[k] = (ivf32)rnd_val();
    for (int k = 0; k < rows;    k++) vecR[k] = (ivf32)rnd_val();

    ivf32* dC   = (ivf32*)malloc((size_t)rows * sizeof(ivf32));
    ivf32* dRef = (ivf32*)malloc((size_t)rows * sizeof(ivf32));
    ivf32* dtC  = (ivf32*)malloc((size_t)cols * sizeof(ivf32));
    ivf32* dtRef= (ivf32*)malloc((size_t)cols * sizeof(ivf32));

    ref_mat_mul(dRef, mat, rows, cols, stride, vec);
    cuda_mm(dC, mat, rows, cols, stride, vec);
    check_arr("mat_mul(CUDA)", dC, dRef, rows, &fail);

    ref_mat_t_mul(dtRef, mat, rows, cols, stride, vecR);
    cuda_mt(dtC, mat, rows, cols, stride, vecR);
    check_arr("mat_t_mul(CUDA)", dtC, dtRef, cols, &fail);

    free(mat); free(vec); free(vecR);
    free(dC); free(dRef); free(dtC); free(dtRef);
    return fail;
}

/* ===================== 计时器 (CPU) ===================== */
#if defined(_WIN32)
#include <windows.h>
static double now_sec(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

/* ===================== 性能对比 ===================== */
typedef void (*launch_fn)(const float*, const float*, float*, int, int, int);
typedef void (*wrap_fn)(ivf32*, const ivf32*, int, int, int, const ivf32*);

struct GpuVar {
    const char* name;
    launch_fn   launch;
    wrap_fn     wrap;
};

static void perf_one(const char* op, int rows, int cols, int stride, int transpose,
                     GpuVar vars[], int nv, int cpu_fail_check)
{
    (void)cpu_fail_check;
    int mat_cap = (int)((size_t)rows * stride);
    int vecN = transpose ? rows : cols;
    int dstN = transpose ? cols : rows;

    ivf32* mat = (ivf32*)malloc((size_t)mat_cap * sizeof(ivf32));
    ivf32* vec = (ivf32*)malloc((size_t)vecN    * sizeof(ivf32));
    ivf32* dC  = (ivf32*)malloc((size_t)dstN    * sizeof(ivf32));
    ivf32* ref = (ivf32*)malloc((size_t)dstN    * sizeof(ivf32));
    for (int k = 0; k < mat_cap; k++) mat[k] = (ivf32)rnd_val();
    for (int k = 0; k < vecN;    k++) vec[k] = (ivf32)rnd_val();

    float *d_mat, *d_vec, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_mat, (size_t)mat_cap * sizeof(ivf32)));
    CUDA_CHECK(cudaMalloc(&d_vec, (size_t)vecN    * sizeof(ivf32)));
    CUDA_CHECK(cudaMalloc(&d_dst, (size_t)dstN    * sizeof(ivf32)));

    long flops = (long)rows * cols;
    int iters = (flops >= 16L * 1024 * 1024) ? 20
              : (flops >=  4L * 1024 * 1024) ? 50
              : (flops >=  1L * 1024 * 1024) ? 100
              : 300;

    /* warmup */
    CUDA_CHECK(cudaMemcpy(d_mat, mat, (size_t)mat_cap * sizeof(ivf32), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, vec, (size_t)vecN    * sizeof(ivf32), cudaMemcpyHostToDevice));
    vars[0].launch(d_mat, d_vec, d_dst, rows, cols, stride);
    CUDA_CHECK(cudaMemcpy(dC, d_dst, (size_t)dstN * sizeof(ivf32), cudaMemcpyDeviceToHost));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    /* H2D: 上传 mat+vec (两种实现一致, 测一次) */
    CUDA_CHECK(cudaEventRecord(e0));
    for (int k = 0; k < iters; k++)
    {
        CUDA_CHECK(cudaMemcpy(d_mat, mat, (size_t)mat_cap * sizeof(ivf32), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vec, vec, (size_t)vecN    * sizeof(ivf32), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms_h2d = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, e0, e1));
    ms_h2d /= iters;

    /* D2H: 回传 dst (一致, 测一次) */
    CUDA_CHECK(cudaEventRecord(e0));
    for (int k = 0; k < iters; k++)
        CUDA_CHECK(cudaMemcpy(dC, d_dst, (size_t)dstN * sizeof(ivf32), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms_d2h = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, e0, e1));
    ms_d2h /= iters;

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    /* CPU 参考 (纯 C 优化版) */
    if (!transpose) mat_mul_vet_real32(ref, mat, rows, cols, stride, vec);
    else            mat_t_mul_vet_real32(ref, mat, rows, cols, stride, vec);
    double t0 = now_sec();
    for (int k = 0; k < iters; k++)
    {
        if (!transpose) mat_mul_vet_real32(ref, mat, rows, cols, stride, vec);
        else            mat_t_mul_vet_real32(ref, mat, rows, cols, stride, vec);
    }
    double ms_cpu = (now_sec() - t0) / iters * 1e3;

    printf("  %-9s %5dx%-5d | H2D %7.3f | D2H %7.3f | CPU %7.3f | ",
           op, rows, cols, ms_h2d, ms_d2h, ms_cpu);

    for (int v = 0; v < nv; v++)
    {
        /* kernel 计时 */
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        for (int k = 0; k < iters; k++)
            vars[v].launch(d_mat, d_vec, d_dst, rows, cols, stride);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms_kernel = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, e0, e1));
        ms_kernel /= iters;
        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));

        /* 端到端 (封装单次) */
        double t1 = now_sec();
        vars[v].wrap(dC, mat, rows, cols, stride, vec);
        double ms_e2e = (now_sec() - t1) * 1e3;

        float ms_gtot = ms_h2d + ms_kernel + ms_d2h;
        double gf = 2.0 * rows * cols / (ms_kernel * 1e-3) / 1e9;
        int pf = 0;
        check_arr("perf(CUDA)", dC, ref, dstN, &pf);
        if (pf) printf("[%s MISMATCH!] ", vars[v].name);

        printf("[%s] k %7.3f(%5.0fGF) tot %7.3f e2e %7.3f k/CPU %5.2fx ",
               vars[v].name, ms_kernel, gf, ms_gtot, ms_e2e, ms_cpu / ms_kernel);
    }
    printf("\n");

    CUDA_CHECK(cudaFree(d_mat));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_dst));
    free(mat); free(vec); free(dC); free(ref);
}

int main(void)
{
    /* ---------- 1) 正确性 ---------- */
    int sizes[] = { 1, 3, 7, 8, 16, 33, 100, 255, 256, 257, 1000 };
    int n = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int fail = 0, cases = 0;
    srand(12345);

    for (int a = 0; a < n; a++)
        for (int b = 0; b < n; b++)
        {
            int rows = sizes[a], cols = sizes[b];
            int strides[2] = { cols, cols + 3 };
            for (int si = 0; si < 2; si++)
            {
                fail += run_case("naive", rows, cols, strides[si],
                                 mat_mul_vet_real32_cuda,  mat_t_mul_vet_real32_cuda);
                fail += run_case("v1",    rows, cols, strides[si],
                                 mat_mul_vet_real32_cuda_v1, mat_t_mul_vet_real32_cuda_v1);
                fail += run_case("v2",    rows, cols, strides[si],
                                 mat_mul_vet_real32_cuda_v2, mat_t_mul_vet_real32_cuda_v2);
                fail += run_case("v3",    rows, cols, strides[si],
                                 mat_mul_vet_real32_cuda_v3, mat_t_mul_vet_real32_cuda_v3);
                fail += run_case("v4",    rows, cols, strides[si],
                                 mat_mul_vet_real32_cuda_v4, mat_t_mul_vet_real32_cuda_v4);
                fail += run_case("v5",    rows, cols, strides[si],
                                 mat_mul_vet_real32_cuda_v5, mat_t_mul_vet_real32_cuda_v5);
                fail += run_case("cublas",rows, cols, strides[si],
                                 mat_mul_vet_real32_cuda_cublas, mat_t_mul_vet_real32_cuda_cublas);
                cases += 7;
            }
        }
    printf("CUDA correctness: cases=%d, failures=%d\n", cases, fail);
    printf(fail == 0 ? "ALL CUDA TESTS PASS\n" : "SOME CUDA TESTS FAILED\n");

    /* ---------- 2) 性能 (naive vs v1) ---------- */
    printf("\n================ PERFORMANCE (ms/call) ================\n");
    printf("  op        rows x cols | H2D      | D2H      | CPU      | [impl] k / GF / tot / e2e / k-CPU  \n");

    struct Cfg { int rows, cols; } cfgs[] = {
        {256, 256}, {512, 512}, {1024, 1024}, {2048, 2048},
        {4096, 4096}, {8192, 8192},
        {1024, 8192}, {8192, 1024},
    };
    int nc = (int)(sizeof(cfgs) / sizeof(cfgs[0]));

    GpuVar v_mm[7] = {
        {"naive", (launch_fn)matvec_naive_launch,      (wrap_fn)mat_mul_vet_real32_cuda},
        {"v1",    (launch_fn)matvec_v1_launch,         (wrap_fn)mat_mul_vet_real32_cuda_v1},
        {"v2",    (launch_fn)matvec_v2_launch,         (wrap_fn)mat_mul_vet_real32_cuda_v2},
        {"v3",    (launch_fn)matvec_v3_launch,         (wrap_fn)mat_mul_vet_real32_cuda_v3},
        {"v4",    (launch_fn)matvec_v4_launch,         (wrap_fn)mat_mul_vet_real32_cuda_v4},
        {"v5",    (launch_fn)matvec_v5_launch,         (wrap_fn)mat_mul_vet_real32_cuda_v5},
        {"cublas",(launch_fn)cublas_matvec_launch,     (wrap_fn)mat_mul_vet_real32_cuda_cublas},
    };
    GpuVar v_mt[7] = {
        {"naive", (launch_fn)mat_t_vec_naive_launch,   (wrap_fn)mat_t_mul_vet_real32_cuda},
        {"v1",    (launch_fn)mat_t_vec_v1_launch,      (wrap_fn)mat_t_mul_vet_real32_cuda_v1},
        {"v2",    (launch_fn)mat_t_vec_v2_launch,      (wrap_fn)mat_t_mul_vet_real32_cuda_v2},
        {"v3",    (launch_fn)mat_t_vec_v3_launch,      (wrap_fn)mat_t_mul_vet_real32_cuda_v3},
        {"v4",    (launch_fn)mat_t_vec_v4_launch,      (wrap_fn)mat_t_mul_vet_real32_cuda_v4},
        {"v5",    (launch_fn)mat_t_vec_v5_launch,      (wrap_fn)mat_t_mul_vet_real32_cuda_v5},
        {"cublas",(launch_fn)cublas_mat_t_vec_launch,  (wrap_fn)mat_t_mul_vet_real32_cuda_cublas},
    };

    printf("  -- dst = mat*vec --\n");
    for (int i = 0; i < nc; i++)
        perf_one("mat*vec", cfgs[i].rows, cfgs[i].cols, cfgs[i].cols, 0, v_mm, 7, 0);

    printf("  -- dst = matT*vec --\n");
    for (int i = 0; i < nc; i++)
        perf_one("matT*vec", cfgs[i].rows, cfgs[i].cols, cfgs[i].cols, 1, v_mt, 7, 0);

    return fail == 0 ? 0 : 1;
}
