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
 * matvet_cuda.cu - 朴素 CUDA 版矩阵-向量乘法 (实现)
 *
 * 设计: 朴素到极点 —— 每个输出元素由一个线程计算, 不做 shared-memory 分块、
 * 不做向量化加载、不做归约优化。目的只是给出"能跑且正确"的 GPU 基线,
 * 方便和 CPU 版、以及后续的优化版 (cooperative / shared-mem / tensor-core) 对比。
 *
 * 核函数:
 *   matvec_naive_kernel     每个线程算一行点积 -> dst[i] = Σ_j mat[i*stride+j]*vec[j]
 *   mat_t_vec_naive_kernel  每个线程算一个输出列 -> dst[j] = Σ_i mat[i*stride+j]*vec[i]
 *
 * 主机封装: 分配设备内存 -> H2D 拷贝 -> 启动核 -> D2H 拷回 -> 释放。
 * 线程块固定 256, grid 按输出元素数向上取整。
 */

#include "matvet_cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* ---------- 朴素核函数 ---------- */

/* dst = mat · vec : 第 i 个线程负责第 i 行 */
__global__ void matvec_naive_kernel(const float* mat, const float* vec, float* dst,
                                     int rows, int cols, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    const float* row = mat + (size_t)i * stride;
    float s = 0.0f;
    for (int j = 0; j < cols; j++) s += row[j] * vec[j];
    dst[i] = s;
}

/* dst = mat^T · vec : 第 j 个线程负责第 j 列, 沿行累加 */
__global__ void mat_t_vec_naive_kernel(const float* mat, const float* vec, float* dst,
                                        int rows, int cols, int stride)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float s = 0.0f;
    for (int i = 0; i < rows; i++) s += mat[(size_t)i * stride + j] * vec[i];
    dst[j] = s;
}

/* 仅启动核函数 (供 benchmark 计时; 设备指针由调用方管理, 不分配/不拷) */
void matvec_naive_launch(const float* d_mat, const float* d_vec, float* d_dst,
                         int rows, int cols, int stride)
{
    const int block = 256;
    const int grid  = (rows + block - 1) / block;
    matvec_naive_kernel<<<grid, block>>>(d_mat, d_vec, d_dst, rows, cols, stride);
}
void mat_t_vec_naive_launch(const float* d_mat, const float* d_vec, float* d_dst,
                            int rows, int cols, int stride)
{
    const int block = 256;
    const int grid  = (cols + block - 1) / block;
    mat_t_vec_naive_kernel<<<grid, block>>>(d_mat, d_vec, d_dst, rows, cols, stride);
}

/* CUDA 错误检查, 出错即退出 */
static void cuda_check(cudaError_t err, const char* where)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error @ %s: %s\n", where, cudaGetErrorString(err));
        exit(1);
    }
}

/* ---------- 主机封装: dst = mat · vec ---------- */
void mat_mul_vet_real32_cuda(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec)
{
    const size_t mat_bytes = (size_t)rows * mat_stride * sizeof(ivf32);
    const size_t vec_bytes = (size_t)cols * sizeof(ivf32);
    const size_t dst_bytes = (size_t)rows * sizeof(ivf32);

    float *d_mat = NULL, *d_vec = NULL, *d_dst = NULL;
    cuda_check(cudaMalloc(&d_mat, mat_bytes), "cudaMalloc mat");
    cuda_check(cudaMalloc(&d_vec, vec_bytes), "cudaMalloc vec");
    cuda_check(cudaMalloc(&d_dst, dst_bytes), "cudaMalloc dst");

    cuda_check(cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice), "H2D mat");
    cuda_check(cudaMemcpy(d_vec, vec, vec_bytes, cudaMemcpyHostToDevice), "H2D vec");

    const int block = 256;
    const int grid  = (rows + block - 1) / block;
    matvec_naive_kernel<<<grid, block>>>(d_mat, d_vec, d_dst, rows, cols, mat_stride);
    cuda_check(cudaGetLastError(), "launch matvec_naive");

    cuda_check(cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost), "D2H dst");

    cuda_check(cudaFree(d_mat), "free mat");
    cuda_check(cudaFree(d_vec), "free vec");
    cuda_check(cudaFree(d_dst), "free dst");
}

/* ---------- 主机封装: dst = mat^T · vec ---------- */
void mat_t_mul_vet_real32_cuda(ivf32* dst, const ivf32* mat, int rows, int cols, int mat_stride, const ivf32* vec)
{
    const size_t mat_bytes = (size_t)rows * mat_stride * sizeof(ivf32);
    const size_t vec_bytes = (size_t)rows * sizeof(ivf32);   /* vec 长度 = rows */
    const size_t dst_bytes = (size_t)cols * sizeof(ivf32);   /* dst 长度 = cols */

    float *d_mat = NULL, *d_vec = NULL, *d_dst = NULL;
    cuda_check(cudaMalloc(&d_mat, mat_bytes), "cudaMalloc mat");
    cuda_check(cudaMalloc(&d_vec, vec_bytes), "cudaMalloc vec");
    cuda_check(cudaMalloc(&d_dst, dst_bytes), "cudaMalloc dst");

    cuda_check(cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice), "H2D mat");
    cuda_check(cudaMemcpy(d_vec, vec, vec_bytes, cudaMemcpyHostToDevice), "H2D vec");

    const int block = 256;
    const int grid  = (cols + block - 1) / block;
    mat_t_vec_naive_kernel<<<grid, block>>>(d_mat, d_vec, d_dst, rows, cols, mat_stride);
    cuda_check(cudaGetLastError(), "launch mat_t_vec_naive");

    cuda_check(cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost), "D2H dst");

    cuda_check(cudaFree(d_mat), "free mat");
    cuda_check(cudaFree(d_vec), "free vec");
    cuda_check(cudaFree(d_dst), "free dst");
}
