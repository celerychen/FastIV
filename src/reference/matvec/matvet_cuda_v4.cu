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
 * matvet_cuda_v4.cu - 共享内存 tile + float4 向量化叠加 (v4)
 *
 * 设计要点 (为什么这样叠加):
 *   v3 把 x 按 256 一块 tile 协作搬进 __shared__, 内层每线程管一列、块内归约 -> 合并好、无大 buffer。
 *   v1 用 float4 一次吃 4 个连续元素, 减少指令与访存事务 -> 算力最高。
 *   v4 = 二者叠加: tile 协作搬 x 时用 float4, 内层点积也用 float4; 仅在对齐满足时走向量化,
 *   否则退回 v3 标量等价 (保证正确性, 不踩 v1 曾踩过的 misaligned address 坑)。
 */

#include "matvet_cuda_v4.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static const int BS = 256;   /* block 大小 = 共享 tile 大小 */

/* 对齐判定: 指针 16B 对齐 且 stride 为 4 的倍数, float4 才安全 */
static __device__ __forceinline__ bool ok_float4(const float* p, const float* q, int stride)
{
    return (((size_t)p & 15u) == 0u) && (((size_t)q & 15u) == 0u) && ((stride & 3) == 0);
}

/* ===================== mat·vec ===================== */
__global__ void matvec_v4_kernel(const float* __restrict__ mat,
                                  const float* __restrict__ vec,
                                  float* dst, int rows, int cols, int stride)
{
    __shared__ float xs[BS];          /* 固定 256, 不是 cols! */
    int row = blockIdx.x;
    int tid = threadIdx.x;
    bool aligned = ok_float4(mat, vec, stride) && (((size_t)&mat[(size_t)row * stride] & 15u) == 0u);

    float sum = 0.0f;

    for (int tile = 0; tile < cols; tile += BS)
    {
        int end = (cols - tile < BS) ? (cols - tile) : BS;

        /* 协作把 x 这一 tile 搬进共享内存: 64 线程各 float4, 不足 4 列或末块标量补齐 */
        if (tid < 64)
        {
            int base = tid * 4;
            if (aligned && base + 4 <= end)
            {
                float4 v = __ldg((const float4*)&vec[tile + base]);
                xs[base] = v.x; xs[base + 1] = v.y; xs[base + 2] = v.z; xs[base + 3] = v.w;
            }
            else
            {
                for (int q = base; q < end && q < base + 4; q++) xs[q] = vec[tile + q];
            }
        }
        __syncthreads();

        /* 每线程管 4 连续列 (0..63 -> 覆盖整块 256 列); 线程 64..255 空闲(归约补 0) */
        float tsum = 0.0f;
        if (tid < 64)
        {
            int base = tid * 4;
            if (aligned && base + 4 <= end)
            {
                float4 a = __ldg((const float4*)&mat[(size_t)row * stride + tile + base]);
                float4 x = *(const float4*)&xs[base];
                tsum = a.x * x.x + a.y * x.y + a.z * x.z + a.w * x.w;
            }
            else
            {
                for (int q = base; q < end && q < base + 4; q++)
                    tsum += mat[(size_t)row * stride + tile + q] * xs[q];
            }
        }
        sum += tsum;
        __syncthreads();
    }

    /* 块内归约 (256 线程, 空闲者补 0) */
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
/* 一个 block 管 BS 列, 每线程一列; x(长 rows) 按 tile 用 float4 搬进共享 vs;
 * 固定列号时各线程读连续列 -> 合并; 内层过行用 unroll-4 (A 行间跨 stride 无法 float4)。 */
__global__ void mat_t_vec_v4_kernel(const float* __restrict__ mat,
                                     const float* __restrict__ vec,
                                     float* dst, int rows, int cols, int stride)
{
    __shared__ float vs[BS];              /* 固定 256, 不是 rows! */
    int col = blockIdx.x * BS + threadIdx.x;
    int tid = threadIdx.x;
    bool vec_aligned = (((size_t)vec & 15u) == 0u);

    float sum = 0.0f;
    for (int tile = 0; tile < rows; tile += BS)
    {
        int end = (rows - tile < BS) ? (rows - tile) : BS;

        /* 协作搬 x 的 tile 进共享: 64 线程 float4, 末块/非对齐标量 */
        if (tid < 64)
        {
            int base = tid * 4;
            if (vec_aligned && base + 4 <= end)
            {
                float4 v = __ldg((const float4*)&vec[tile + base]);
                vs[base] = v.x; vs[base + 1] = v.y; vs[base + 2] = v.z; vs[base + 3] = v.w;
            }
            else
            {
                for (int q = base; q < end && q < base + 4; q++) vs[q] = vec[tile + q];
            }
        }
        __syncthreads();

        /* 所有线程参与同步; 仅 col<cols 累加/写回, 避免 early-return 破坏 __syncthreads */
        if (col < cols)
        {
            for (int k = 0; k < end; k += 4)
            {
                if (k + 3 < end)
                {
                    sum += mat[(size_t)(tile + k)     * stride + col] * vs[k]
                         + mat[(size_t)(tile + k + 1) * stride + col] * vs[k + 1]
                         + mat[(size_t)(tile + k + 2) * stride + col] * vs[k + 2]
                         + mat[(size_t)(tile + k + 3) * stride + col] * vs[k + 3];
                }
                else
                {
                    for (int q = k; q < end; q++)
                        sum += mat[(size_t)(tile + q) * stride + col] * vs[q];
                }
            }
        }
        __syncthreads();
    }
    if (col < cols) dst[col] = sum;
}

/* ===================== 仅启动核 (供 benchmark) ===================== */
void matvec_v4_launch(const float* d_mat, const float* d_vec, float* d_dst,
                      int rows, int cols, int stride)
{
    matvec_v4_kernel<<<rows, BS>>>(d_mat, d_vec, d_dst, rows, cols, stride);
}

void mat_t_vec_v4_launch(const float* d_mat, const float* d_vec, float* d_dst,
                          int rows, int cols, int stride)
{
    int grid = (cols + BS - 1) / BS;   /* 每 block 管 BS 列 */
    mat_t_vec_v4_kernel<<<grid, BS>>>(d_mat, d_vec, d_dst, rows, cols, stride);
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
void mat_mul_vet_real32_cuda_v4(ivf32* dst, const ivf32* mat, int rows, int cols, int stride, const ivf32* vec)
{
    const size_t mat_bytes = (size_t)rows * stride * sizeof(ivf32);
    const size_t vec_bytes = (size_t)cols   * sizeof(ivf32);
    const size_t dst_bytes = (size_t)rows   * sizeof(ivf32);

    float *d_mat, *d_vec, *d_dst;
    cuda_check(cudaMalloc(&d_mat, mat_bytes), "v4 cudaMalloc mat");
    cuda_check(cudaMalloc(&d_vec, vec_bytes), "v4 cudaMalloc vec");
    cuda_check(cudaMalloc(&d_dst, dst_bytes), "v4 cudaMalloc dst");

    cuda_check(cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice), "v4 H2D mat");
    cuda_check(cudaMemcpy(d_vec, vec, vec_bytes, cudaMemcpyHostToDevice), "v4 H2D vec");

    matvec_v4_launch(d_mat, d_vec, d_dst, rows, cols, stride);
    cuda_check(cudaGetLastError(), "v4 launch matvec");

    cuda_check(cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost), "v4 D2H dst");

    cuda_check(cudaFree(d_mat), "v4 free mat");
    cuda_check(cudaFree(d_vec), "v4 free vec");
    cuda_check(cudaFree(d_dst), "v4 free dst");
}

/* ===================== 主机封装: dst = mat^T · vec ===================== */
void mat_t_mul_vet_real32_cuda_v4(ivf32* dst, const ivf32* mat, int rows, int cols, int stride, const ivf32* vec)
{
    const size_t mat_bytes = (size_t)rows * stride * sizeof(ivf32);
    const size_t vec_bytes = (size_t)rows   * sizeof(ivf32);   /* vec 长 = rows */
    const size_t dst_bytes = (size_t)cols   * sizeof(ivf32);   /* dst 长 = cols */

    float *d_mat, *d_vec, *d_dst;
    cuda_check(cudaMalloc(&d_mat, mat_bytes), "v4 cudaMalloc mat");
    cuda_check(cudaMalloc(&d_vec, vec_bytes), "v4 cudaMalloc vec");
    cuda_check(cudaMalloc(&d_dst, dst_bytes), "v4 cudaMalloc dst");

    cuda_check(cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice), "v4 H2D mat");
    cuda_check(cudaMemcpy(d_vec, vec, vec_bytes, cudaMemcpyHostToDevice), "v4 H2D vec");

    mat_t_vec_v4_launch(d_mat, d_vec, d_dst, rows, cols, stride);
    cuda_check(cudaGetLastError(), "v4 launch mat_t_vec");

    cuda_check(cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost), "v4 D2H dst");

    cuda_check(cudaFree(d_mat), "v4 free mat");
    cuda_check(cudaFree(d_vec), "v4 free vec");
    cuda_check(cudaFree(d_dst), "v4 free dst");
}
