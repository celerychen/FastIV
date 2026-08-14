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
 * matvet_cuda_v2.cu - 优化版 v2 矩阵-向量乘法 (shared-memory + block reduction)
 *
 * 严格参照 SGEMV 指南的 "After (Optimized Implementation)" / sgemv_opt 思路:
 *   - 一个 block 负责一行 (blockIdx.x = row), blockDim.x = 256。
 *   - 把向量 x(vec) 按 tile 协作搬进 __shared__ (合并加载), 内层 A[row][tile+j]*xs[j]
 *     中 A 行内连续 -> 合并, xs 来自共享内存 -> 高带宽。
 *   - 块内归约: 写 sdata[tid] 后循环折半相加 (shared memory block reduction)。
 *   - inner 循环按 threadIdx.x 跨步 (j = tid; j += BS), 每线程只累加自身 stride 子集,
 *     避免归约时重复放大 (这是严格遵循指南的关键, 不做 float4 以免 stride 错位)。
 *
 * mat^T·vec: 转置对应版 —— 每个线程负责一个输出列 j, 把 vec(长度 rows) 按 tile
 * 搬进共享内存 vs[], 内层 mat[(tile+k)*stride + j] 在固定 k 时跨线程 j 连续 -> 合并。
 *
 * 注意 blockDim.x 必须是 2 的幂且 <= 256 (与 xs/sdata 大小一致)。
 */

#include "matvet_cuda_v2.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static const int BS = 256;   /* block 尺寸 = 256 线程 (= xs / sdata 数组长度) */

/* ---------- dst = mat · vec : 一个 block 一行 + 共享内存缓存 vec + 块归约 ---------- */
__global__ void matvec_v2_kernel(const float* __restrict__ mat,
                                  const float* __restrict__ vec,
                                  float* dst, int rows, int cols, int stride)
{
    __shared__ float xs[BS];     /* 当前 tile 的 vec */
    __shared__ float sdata[BS];  /* 块归约临时 */

    int i = blockIdx.x;          /* 一行一个 block */
    int tid = threadIdx.x;
    const float* row = mat + (size_t)i * stride;

    float sum = 0.0f;

    /* 指南 sgemv_opt 思路: 一个 block 一行, 把 x(vec) 协作搬进共享内存,
     * 内层每个线程按 threadIdx.x 跨步 (j = tid; j < end; j += BS) 只算自己那份列,
     * 最后块内归约。注意: 每个线程只累加自身 stride 子集, 否则归约会重复放大。 */
    for (int tile = 0; tile < cols; tile += BS)
    {
        /* 协作把 vec 这一 tile 搬进共享内存 (合并加载) */
        if (tile + tid < cols) xs[tid] = vec[tile + tid];
        __syncthreads();

        int end = (cols - tile < BS) ? (cols - tile) : BS;
        /* 行内连续读 (合并) * xs[j] (共享内存, 广播)。每线程 stride 子集, 不与邻居重叠 */
        for (int j = tid; j < end; j += BS)
            sum += row[tile + j] * xs[j];

        __syncthreads();
    }

    /* 块内归约 (shared memory block reduction) */
    sdata[tid] = sum;
    __syncthreads();
    for (int s = BS / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) dst[i] = sdata[0];
}

/* ---------- dst = mat^T · vec : 每线程一列 + 共享内存缓存 vec ---------- */
__global__ void mat_t_vec_v2_kernel(const float* __restrict__ mat,
                                     const float* __restrict__ vec,
                                     float* dst, int rows, int cols, int stride)
{
    __shared__ float vs[BS];   /* 当前 tile 的 vec (长度 rows) */

    int j = blockIdx.x * blockDim.x + threadIdx.x;  /* 一个线程一个输出列 */
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < rows; tile += BS)
    {
        /* 协作把 vec 这一 tile 搬进共享内存 */
        if (tile + tid < rows) vs[tid] = vec[tile + tid];
        __syncthreads();

        int end = (rows - tile < BS) ? (rows - tile) : BS;
        /* 固定 k 时跨线程 j 连续 -> mat 读取合并; vs[k] 来自共享内存 */
        for (int k = 0; k < end; k++)
            sum += mat[(size_t)(tile + k) * stride + j] * vs[k];
        __syncthreads();
    }

    if (j < cols) dst[j] = sum;
}

/* ---------- 仅启动核 (供 benchmark) ---------- */
void matvec_v2_launch(const float* d_mat, const float* d_vec, float* d_dst,
                      int rows, int cols, int stride)
{
    matvec_v2_kernel<<<rows, BS>>>(d_mat, d_vec, d_dst, rows, cols, stride);
}
void mat_t_vec_v2_launch(const float* d_mat, const float* d_vec, float* d_dst,
                         int rows, int cols, int stride)
{
    int grid = (cols + BS - 1) / BS;
    mat_t_vec_v2_kernel<<<grid, BS>>>(d_mat, d_vec, d_dst, rows, cols, stride);
}

static void cuda_check(cudaError_t err, const char* where)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error @ %s: %s\n", where, cudaGetErrorString(err));
        exit(1);
    }
}

/* ---------- 主机封装: dst = mat · vec ---------- */
void mat_mul_vet_real32_cuda_v2(ivf32* dst, const ivf32* mat, int rows, int cols, int stride, const ivf32* vec)
{
    const size_t mat_bytes = (size_t)rows * stride * sizeof(ivf32);
    const size_t vec_bytes = (size_t)cols   * sizeof(ivf32);
    const size_t dst_bytes = (size_t)rows   * sizeof(ivf32);

    float *d_mat, *d_vec, *d_dst;
    cuda_check(cudaMalloc(&d_mat, mat_bytes), "v2 cudaMalloc mat");
    cuda_check(cudaMalloc(&d_vec, vec_bytes), "v2 cudaMalloc vec");
    cuda_check(cudaMalloc(&d_dst, dst_bytes), "v2 cudaMalloc dst");

    cuda_check(cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice), "v2 H2D mat");
    cuda_check(cudaMemcpy(d_vec, vec, vec_bytes, cudaMemcpyHostToDevice), "v2 H2D vec");

    matvec_v2_kernel<<<rows, BS>>>(d_mat, d_vec, d_dst, rows, cols, stride);
    cuda_check(cudaGetLastError(), "v2 launch matvec");

    cuda_check(cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost), "v2 D2H dst");

    cuda_check(cudaFree(d_mat), "v2 free mat");
    cuda_check(cudaFree(d_vec), "v2 free vec");
    cuda_check(cudaFree(d_dst), "v2 free dst");
}

/* ---------- 主机封装: dst = mat^T · vec ---------- */
void mat_t_mul_vet_real32_cuda_v2(ivf32* dst, const ivf32* mat, int rows, int cols, int stride, const ivf32* vec)
{
    const size_t mat_bytes = (size_t)rows * stride * sizeof(ivf32);
    const size_t vec_bytes = (size_t)rows   * sizeof(ivf32);   /* vec 长 = rows */
    const size_t dst_bytes = (size_t)cols   * sizeof(ivf32);   /* dst 长 = cols */

    float *d_mat, *d_vec, *d_dst;
    cuda_check(cudaMalloc(&d_mat, mat_bytes), "v2 cudaMalloc mat");
    cuda_check(cudaMalloc(&d_vec, vec_bytes), "v2 cudaMalloc vec");
    cuda_check(cudaMalloc(&d_dst, dst_bytes), "v2 cudaMalloc dst");

    cuda_check(cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice), "v2 H2D mat");
    cuda_check(cudaMemcpy(d_vec, vec, vec_bytes, cudaMemcpyHostToDevice), "v2 H2D vec");

    int grid = (cols + BS - 1) / BS;
    mat_t_vec_v2_kernel<<<grid, BS>>>(d_mat, d_vec, d_dst, rows, cols, stride);
    cuda_check(cudaGetLastError(), "v2 launch mat_t_vec");

    cuda_check(cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost), "v2 D2H dst");

    cuda_check(cudaFree(d_mat), "v2 free mat");
    cuda_check(cudaFree(d_vec), "v2 free vec");
    cuda_check(cudaFree(d_dst), "v2 free dst");
}
