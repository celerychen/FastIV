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
 * matvet_cuda_cublas.cu - cuBLAS (cublasSgemv) 对标实现
 *
 * 行主序 -> 列主序 reinterpret 说明见 matvet_cuda_cublas.h。
 * cuBLAS handle 懒初始化 (首次调用时创建), 全程复用默认流, 与 perf 计时事件一致。
 */

#include "matvet_cuda_cublas.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasHandle_t g_h = NULL;

static void cublas_ensure(void)
{
    if (g_h == NULL)
    {
        cublasStatus_t st = cublasCreate(&g_h);
        if (st != CUBLAS_STATUS_SUCCESS)
        {
            fprintf(stderr, "cublasCreate failed: %d\n", (int)st);
            exit(1);
        }
    }
}

static void cuda_check(cudaError_t err, const char* where)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error @ %s: %s\n", where, cudaGetErrorString(err));
        exit(1);
    }
}

/* dst = mat · vec
 * 行主序 mat (rows x cols, lda=stride) 当作列主序矩阵 reinterpret:
 *   m = cols, n = rows, lda = stride, op = T  (op(A)=A^T 才是我们要的 rows x cols 乘法)
 * 这样 lda(=stride=cols) 始终 >= m(=cols), 满足 cuBLAS 的 lda 约束。 */
void cublas_matvec_launch(const float* d_mat, const float* d_vec, float* d_dst,
                          int rows, int cols, int stride)
{
    cublas_ensure();
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasSgemv(g_h, CUBLAS_OP_T, cols, rows,
                                    &alpha, d_mat, stride, d_vec, 1,
                                    &beta, d_dst, 1);
    if (st != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cublasSgemv(T) failed: %d\n", (int)st);
        exit(1);
    }
}

/* dst = mat^T · vec
 * m = cols, n = rows, lda = stride, op = N  (op(A)=A 即 cols x rows = mat^T 的维度) */
void cublas_mat_t_vec_launch(const float* d_mat, const float* d_vec, float* d_dst,
                             int rows, int cols, int stride)
{
    cublas_ensure();
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasSgemv(g_h, CUBLAS_OP_N, cols, rows,
                                    &alpha, d_mat, stride, d_vec, 1,
                                    &beta, d_dst, 1);
    if (st != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cublasSgemv(N) failed: %d\n", (int)st);
        exit(1);
    }
}

/* ===================== 主机封装: dst = mat · vec ===================== */
void mat_mul_vet_real32_cuda_cublas(ivf32* dst, const ivf32* mat, int rows, int cols, int stride, const ivf32* vec)
{
    const size_t mat_bytes = (size_t)rows * stride * sizeof(ivf32);
    const size_t vec_bytes = (size_t)cols   * sizeof(ivf32);
    const size_t dst_bytes = (size_t)rows   * sizeof(ivf32);

    float *d_mat, *d_vec, *d_dst;
    cuda_check(cudaMalloc(&d_mat, mat_bytes), "cublas cudaMalloc mat");
    cuda_check(cudaMalloc(&d_vec, vec_bytes), "cublas cudaMalloc vec");
    cuda_check(cudaMalloc(&d_dst, dst_bytes), "cublas cudaMalloc dst");

    cuda_check(cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice), "cublas H2D mat");
    cuda_check(cudaMemcpy(d_vec, vec, vec_bytes, cudaMemcpyHostToDevice), "cublas H2D vec");

    cublas_matvec_launch(d_mat, d_vec, d_dst, rows, cols, stride);
    cuda_check(cudaGetLastError(), "cublas launch matvec");

    cuda_check(cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost), "cublas D2H dst");

    cuda_check(cudaFree(d_mat), "cublas free mat");
    cuda_check(cudaFree(d_vec), "cublas free vec");
    cuda_check(cudaFree(d_dst), "cublas free dst");
}

/* ===================== 主机封装: dst = mat^T · vec ===================== */
void mat_t_mul_vet_real32_cuda_cublas(ivf32* dst, const ivf32* mat, int rows, int cols, int stride, const ivf32* vec)
{
    const size_t mat_bytes = (size_t)rows * stride * sizeof(ivf32);
    const size_t vec_bytes = (size_t)rows   * sizeof(ivf32);
    const size_t dst_bytes = (size_t)cols   * sizeof(ivf32);

    float *d_mat, *d_vec, *d_dst;
    cuda_check(cudaMalloc(&d_mat, mat_bytes), "cublas cudaMalloc mat");
    cuda_check(cudaMalloc(&d_vec, vec_bytes), "cublas cudaMalloc vec");
    cuda_check(cudaMalloc(&d_dst, dst_bytes), "cublas cudaMalloc dst");

    cuda_check(cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice), "cublas H2D mat");
    cuda_check(cudaMemcpy(d_vec, vec, vec_bytes, cudaMemcpyHostToDevice), "cublas H2D vec");

    cublas_mat_t_vec_launch(d_mat, d_vec, d_dst, rows, cols, stride);
    cuda_check(cudaGetLastError(), "cublas launch mat_t_vec");

    cuda_check(cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost), "cublas D2H dst");

    cuda_check(cudaFree(d_mat), "cublas free mat");
    cuda_check(cudaFree(d_vec), "cublas free vec");
    cuda_check(cudaFree(d_dst), "cublas free dst");
}
