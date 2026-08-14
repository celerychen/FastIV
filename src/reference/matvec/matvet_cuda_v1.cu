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
 * matvet_cuda_v1.cu - 优化版 CUDA 矩阵-向量乘法 (v1 实现)
 *
 * 思路 (参照 SGEMV 指南的 warp-reduction 法, 不用 shared memory):
 *   mat·vec : 一个 warp(32 线程) 算一行。每个 lane 负责列
 *             j = lane; j += 32 (跨 lane 连续 -> 合并访问)。内层做 float4
 *             向量化加载 (仅当 stride 为 4 的倍数, 保证 row 与 vec 都 16B 对齐),
 *             不足 4 列或 stride 非对齐时退回标量。warp 内 shuffle 归约, lane0
 *             直接写 dst[i] (无需 atomic / memset)。
 *   mat^T·vec : 一个线程算一个输出列 j (该布局对 mat 已合并读); 对内循环 over i
 *             做 float4 向量化加载 vec (i 始终 4 对齐, vec 起点即对齐)。每线程独立,
 *             无需归约。
 */

#include "matvet_cuda_v1.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static const int WARP = 32;   /* mat·vec 每行一个 warp */
static const int BLOCK = 256; /* mat^T·vec 每线程一列, 用 256 线程/block */

/* ---------- dst = mat · vec : 每行一个 warp + warp shuffle 归约 ---------- */
__global__ void matvec_v1_kernel(const float* __restrict__ mat,
                                  const float* __restrict__ vec,
                                  float* dst, int rows, int cols, int stride)
{
    int i = blockIdx.x;            /* 一行一个 block (blockDim.x == WARP) */
    int lane = threadIdx.x;        /* 0..31 */
    if (i >= rows) return;
    const float* row = mat + (size_t)i * stride;

    float sum = 0.0f;

    /* stride 为 4 的倍数时, row 与 vec 的 float4 加载都能 16B 对齐 */
    if ((stride & 3) == 0)
    {
        int c = lane * 4;
        int step = WARP * 4;       /* 每 warp 一轮推进 128 列 */
        int n4 = cols & ~3;        /* 4 对齐的列边界 */
        for (; c + 3 < n4; c += step)
        {
            float4 rv = __ldg((const float4*)&row[c]);
            float4 vv = __ldg((const float4*)&vec[c]);
            sum += rv.x * vv.x + rv.y * vv.y + rv.z * vv.z + rv.w * vv.w;
        }
        /* 尾部: 最后 (cols - n4) 列由前几个 lane 各管一列, 不重不漏 */
        int rem = cols - n4;
        if (lane < rem) sum += row[n4 + lane] * vec[n4 + lane];
    }
    else
    {
        /* stride 非 4 对齐: 退回标量 (仍合并, 无对齐风险) */
        for (int j = lane; j < cols; j += WARP)
            sum += row[j] * vec[j];
    }

    /* 块内归约: 每 warp 内 butterfly shuffle, lane0 持有该 warp 部分和 */
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) dst[i] = sum;   /* 直接写, 无需 memset / atomic */
}

/* ---------- dst = mat^T · vec : 每线程一列 ----------
 * 说明: 该布局下 warp 内 j 连续, 固定 i 时各线程读 mat[i*stride+j] (j 连续)
 * -> 天然合并, 朴素版已接近最优。不用 shared memory 时再向量化收益很小,
 * 故 v1 此处保持与朴素等价的标量循环 (正确性/性能与 naive 持平)。 */
__global__ void mat_t_vec_v1_kernel(const float* __restrict__ mat,
                                     const float* __restrict__ vec,
                                     float* dst, int rows, int cols, int stride)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;

    float sum = 0.0f;
    for (int i = 0; i < rows; i++)
        sum += mat[(size_t)i * stride + j] * vec[i];
    dst[j] = sum;
}

/* ---------- 仅启动核 (供 benchmark) ---------- */
void matvec_v1_launch(const float* d_mat, const float* d_vec, float* d_dst,
                      int rows, int cols, int stride)
{
    matvec_v1_kernel<<<rows, WARP>>>(d_mat, d_vec, d_dst, rows, cols, stride);
}
void mat_t_vec_v1_launch(const float* d_mat, const float* d_vec, float* d_dst,
                         int rows, int cols, int stride)
{
    int grid = (cols + BLOCK - 1) / BLOCK;
    mat_t_vec_v1_kernel<<<grid, BLOCK>>>(d_mat, d_vec, d_dst, rows, cols, stride);
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
void mat_mul_vet_real32_cuda_v1(ivf32* dst, const ivf32* mat, int rows, int cols, int stride, const ivf32* vec)
{
    const size_t mat_bytes = (size_t)rows * stride * sizeof(ivf32);
    const size_t vec_bytes = (size_t)cols   * sizeof(ivf32);
    const size_t dst_bytes = (size_t)rows   * sizeof(ivf32);

    float *d_mat, *d_vec, *d_dst;
    cuda_check(cudaMalloc(&d_mat, mat_bytes), "v1 cudaMalloc mat");
    cuda_check(cudaMalloc(&d_vec, vec_bytes), "v1 cudaMalloc vec");
    cuda_check(cudaMalloc(&d_dst, dst_bytes), "v1 cudaMalloc dst");

    cuda_check(cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice), "v1 H2D mat");
    cuda_check(cudaMemcpy(d_vec, vec, vec_bytes, cudaMemcpyHostToDevice), "v1 H2D vec");

    matvec_v1_kernel<<<rows, WARP>>>(d_mat, d_vec, d_dst, rows, cols, stride);
    cuda_check(cudaGetLastError(), "v1 launch matvec");

    cuda_check(cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost), "v1 D2H dst");

    cuda_check(cudaFree(d_mat), "v1 free mat");
    cuda_check(cudaFree(d_vec), "v1 free vec");
    cuda_check(cudaFree(d_dst), "v1 free dst");
}

/* ---------- 主机封装: dst = mat^T · vec ---------- */
void mat_t_mul_vet_real32_cuda_v1(ivf32* dst, const ivf32* mat, int rows, int cols, int stride, const ivf32* vec)
{
    const size_t mat_bytes = (size_t)rows * stride * sizeof(ivf32);
    const size_t vec_bytes = (size_t)rows   * sizeof(ivf32);   /* vec 长 = rows */
    const size_t dst_bytes = (size_t)cols   * sizeof(ivf32);   /* dst 长 = cols */

    float *d_mat, *d_vec, *d_dst;
    cuda_check(cudaMalloc(&d_mat, mat_bytes), "v1 cudaMalloc mat");
    cuda_check(cudaMalloc(&d_vec, vec_bytes), "v1 cudaMalloc vec");
    cuda_check(cudaMalloc(&d_dst, dst_bytes), "v1 cudaMalloc dst");

    cuda_check(cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice), "v1 H2D mat");
    cuda_check(cudaMemcpy(d_vec, vec, vec_bytes, cudaMemcpyHostToDevice), "v1 H2D vec");

    int grid = (cols + BLOCK - 1) / BLOCK;
    mat_t_vec_v1_kernel<<<grid, BLOCK>>>(d_mat, d_vec, d_dst, rows, cols, stride);
    cuda_check(cudaGetLastError(), "v1 launch mat_t_vec");

    cuda_check(cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost), "v1 D2H dst");

    cuda_check(cudaFree(d_mat), "v1 free mat");
    cuda_check(cudaFree(d_vec), "v1 free vec");
    cuda_check(cudaFree(d_dst), "v1 free dst");
}
