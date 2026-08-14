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
 * matvet_cuda_v3.cu - CUDA 矩阵-向量乘法 (v3: 严格照 SGEMV 指南 sgemv_opt)
 *
 * 关键点: 共享内存是固定 __shared__ float xs[256] / vs[256], 不是跟向量一样大的缓存。
 * 向量 x 按 256 大小的 tile 一块块协作搬进共享内存, 内层只扫当前 tile, 扫完再搬下一块。
 *
 * mat·vec   : 一个 block 负责一行 (blockIdx.x = row, blockDim = 256),
 *             把 x(长 cols) 按 tile 搬进 xs[256], 内层 A[row*stride+tile+j]*xs[j] 行内连续读(合并),
 *             块内归约 (sdata[256] 折半相加), tid==0 写 dst[row]。
 * mat^T·vec : 一个 block 负责一列 (blockIdx.x = col, blockDim = 256),
 *             把 x(长 rows) 按 tile 搬进 vs[256], 内层 A[(tile+j)*stride+col]*vs[j],
 *             块内归约, tid==0 写 dst[col]。
 */

#include "matvet_cuda_v3.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static const int BS = 256;   /* block 大小 = 共享 tile 大小, 与指南一致 */

/* ===================== mat·vec ===================== */
__global__ void matvec_v3_kernel(const float* __restrict__ mat,
                                  const float* __restrict__ vec,
                                  float* dst, int rows, int cols, int stride)
{
    __shared__ float xs[BS];          /* 固定 256, 不是 cols! */
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < cols; tile += BS)
    {
        /* 协作把 x 的这一 tile 搬进共享内存 (合并加载) */
        if (tile + tid < cols) xs[tid] = vec[tile + tid];
        __syncthreads();

        int end = (cols - tile < BS) ? (cols - tile) : BS;
        /* 每个线程只取自己的 stride 子集, 否则 256 线程各算一遍整行 -> 块归约放大 256x */
        for (int j = tid; j < end; j += BS)
            sum += mat[(size_t)row * stride + tile + j] * xs[j];
        __syncthreads();
    }

    /* 块内归约 */
    __shared__ float sdata[BS];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = BS / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) dst[row] = sdata[0];
}

/* ===================== mat^T·vec ===================== */
/* 指南只给了 mat·vec 的代码; matT·vec 若照"一个 block 算一个输出"直译,
 * A 会按 stride 跨步读(不合并)而极慢。正确快速写法: 一个 block 管 BS 列,
 * 每线程负责一列, x(长 rows) 按 tile 搬进共享 vs[BS]; 固定行号时各线程读连续列 -> 合并。 */
__global__ void mat_t_vec_v3_kernel(const float* __restrict__ mat,
                                     const float* __restrict__ vec,
                                     float* dst, int rows, int cols, int stride)
{
    __shared__ float vs[BS];              /* 固定 256, 不是 rows! */
    int col = blockIdx.x * BS + threadIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int tile = 0; tile < rows; tile += BS)
    {
        if (tile + tid < rows) vs[tid] = vec[tile + tid];   /* 协作搬 x 的 tile */
        __syncthreads();

        int end = (rows - tile < BS) ? (rows - tile) : BS;
        /* 所有线程都参与同步; 只有 col<cols 的线程累加/写回, 避免 early-return 破坏 __syncthreads */
        if (col < cols)
        {
            for (int k = 0; k < end; k++)
            {
                int i = tile + k;
                sum += mat[(size_t)i * stride + col] * vs[k];  /* 列连续 -> 合并读 A */
            }
        }
        __syncthreads();
    }
    if (col < cols) dst[col] = sum;
}

/* ===================== 仅启动核 (供 benchmark) ===================== */
void matvec_v3_launch(const float* d_mat, const float* d_vec, float* d_dst,
                      int rows, int cols, int stride)
{
    matvec_v3_kernel<<<rows, BS>>>(d_mat, d_vec, d_dst, rows, cols, stride);
}

void mat_t_vec_v3_launch(const float* d_mat, const float* d_vec, float* d_dst,
                          int rows, int cols, int stride)
{
    int grid = (cols + BS - 1) / BS;   /* 每 block 管 BS 列 */
    mat_t_vec_v3_kernel<<<grid, BS>>>(d_mat, d_vec, d_dst, rows, cols, stride);
}

static void cuda_check(cudaError_t err, const char* where)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error @ %s: %s\n", where, cudaGetErrorString(err));
        exit(1);
    }
}

/* ===================== 主机封装: dst = mat · vec ===================== */
void mat_mul_vet_real32_cuda_v3(ivf32* dst, const ivf32* mat, int rows, int cols, int stride, const ivf32* vec)
{
    const size_t mat_bytes = (size_t)rows * stride * sizeof(ivf32);
    const size_t vec_bytes = (size_t)cols   * sizeof(ivf32);
    const size_t dst_bytes = (size_t)rows   * sizeof(ivf32);

    float *d_mat, *d_vec, *d_dst;
    cuda_check(cudaMalloc(&d_mat, mat_bytes), "v3 cudaMalloc mat");
    cuda_check(cudaMalloc(&d_vec, vec_bytes), "v3 cudaMalloc vec");
    cuda_check(cudaMalloc(&d_dst, dst_bytes), "v3 cudaMalloc dst");

    cuda_check(cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice), "v3 H2D mat");
    cuda_check(cudaMemcpy(d_vec, vec, vec_bytes, cudaMemcpyHostToDevice), "v3 H2D vec");

    matvec_v3_launch(d_mat, d_vec, d_dst, rows, cols, stride);
    cuda_check(cudaGetLastError(), "v3 launch matvec");

    cuda_check(cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost), "v3 D2H dst");

    cuda_check(cudaFree(d_mat), "v3 free mat");
    cuda_check(cudaFree(d_vec), "v3 free vec");
    cuda_check(cudaFree(d_dst), "v3 free dst");
}

/* ===================== 主机封装: dst = mat^T · vec ===================== */
void mat_t_mul_vet_real32_cuda_v3(ivf32* dst, const ivf32* mat, int rows, int cols, int stride, const ivf32* vec)
{
    const size_t mat_bytes = (size_t)rows * stride * sizeof(ivf32);
    const size_t vec_bytes = (size_t)rows   * sizeof(ivf32);   /* vec 长 = rows */
    const size_t dst_bytes = (size_t)cols   * sizeof(ivf32);   /* dst 长 = cols */

    float *d_mat, *d_vec, *d_dst;
    cuda_check(cudaMalloc(&d_mat, mat_bytes), "v3 cudaMalloc mat");
    cuda_check(cudaMalloc(&d_vec, vec_bytes), "v3 cudaMalloc vec");
    cuda_check(cudaMalloc(&d_dst, dst_bytes), "v3 cudaMalloc dst");

    cuda_check(cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice), "v3 H2D mat");
    cuda_check(cudaMemcpy(d_vec, vec, vec_bytes, cudaMemcpyHostToDevice), "v3 H2D vec");

    mat_t_vec_v3_launch(d_mat, d_vec, d_dst, rows, cols, stride);
    cuda_check(cudaGetLastError(), "v3 launch mat_t_vec");

    cuda_check(cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost), "v3 D2H dst");

    cuda_check(cudaFree(d_mat), "v3 free mat");
    cuda_check(cudaFree(d_vec), "v3 free vec");
    cuda_check(cudaFree(d_dst), "v3 free dst");
}
