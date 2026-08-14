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
 * matvet_cuda_v5.cu - v5 实现
 *   mat·vec   : 复用 v4 的 float4 + 256-tile 共享内存写法 (已到 cuBLAS 水平)
 *   mat^T·vec : 自适应 block 粒度的单 kernel 写法, 专治高瘦矩阵占用率不足
 *
 *   诊断结论 (来自基准): 旧版 matT 按"输出列"并行, 一个 block 管 256 列 ->
 *     高瘦形状 cols 小 -> block 极少 -> SM 大量空转 -> 仅发挥 ~1/3 算力
 *     (8192x1024: v3 60GF vs cuBLAS 153GF; 而方阵 v3 已 161GF 打平 cuBLAS)
 *   改法: 保留 v3 已验证的"每线程一列、按 tile 搬 vec 进共享、内层列连续合并读 A"结构,
 *     仅把 block 覆盖的列数自适应: cols>4096 用 256 列/block (稳住房方阵),
 *     否则用 32 列/block -> 高瘦形状也能有几十个 block 喂满 28 个 SM, 零额外访存。
 */

#include "matvet_cuda_v5.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* ============ mat·vec (复用 v4 思路) ============ */
static const int BS = 256;
static __device__ __forceinline__ bool ok_float4(const float* p, const float* q, int stride)
{
    return (((size_t)p & 15u) == 0u) && (((size_t)q & 15u) == 0u) && ((stride & 3) == 0);
}

__global__ void matvec_v5_kernel(const float* __restrict__ mat,
                                  const float* __restrict__ vec,
                                  float* dst, int rows, int cols, int stride)
{
    __shared__ float xs[BS];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    bool aligned = ok_float4(mat, vec, stride) && (((size_t)&mat[(size_t)row * stride] & 15u) == 0u);

    float sum = 0.0f;
    for (int tile = 0; tile < cols; tile += BS)
    {
        int end = (cols - tile < BS) ? (cols - tile) : BS;
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

/* ============ mat^T·vec (v5: 2D GEMM 式分块 + 部分积缓冲) ============
 * 纯输出并行的硬上限 = cols 个线程(本机 1024 列 -> 仅 32 warp), 无论怎么切 block 都喂不饱 28 个 SM。
 * 必须把收缩维(rows)也切开成 2D 网格, 总 warp 数才能爆炸(8192x1024 从 32 -> 8192 warp)。
 * 每个 block 算 [rowTile × colTile] 子块的部分积写入 part[rowTileIdx*cols+col],
 * 再由极轻归约核求和。part 缓冲 = rowTiles×cols = A 的 1/rowTile (≤1MB), 带宽代价可忽略。
 * 关键: blockDim=256(8 warp) 而非 32; vec 段由全体线程并行搬入共享(非单线程串行)。 */
#define COL_TILE_V5 256
#define ROW_TILE_V5 32

__global__ void mat_t_vec_v5_kernel(const float* __restrict__ mat,
                                     const float* __restrict__ vec,
                                     float* __restrict__ part,
                                     int rows, int cols, int stride)
{
    __shared__ float Ash[ROW_TILE_V5][COL_TILE_V5];   /* [32][256] = 32KB */
    __shared__ float vsh[ROW_TILE_V5];                /* [32] = 128B */

    int c0 = blockIdx.x * COL_TILE_V5;   /* 列块起点 */
    int r0 = blockIdx.y * ROW_TILE_V5;   /* 行(收缩)块起点 */
    int t  = threadIdx.x;                /* 0..255, 每线程管 1 列 */

    int rows_in = (rows - r0 < ROW_TILE_V5) ? (rows - r0) : ROW_TILE_V5;
    int cols_in = (cols - c0 < COL_TILE_V5) ? (cols - c0) : COL_TILE_V5;

    /* 并行把 vec 的 rowTile 段搬进共享 (256 线程 strided 覆盖 [0,rows_in)) */
    for (int j = t; j < rows_in; j += blockDim.x) vsh[j] = vec[r0 + j];
    __syncthreads();

    float sum = 0.0f;
    for (int jr = 0; jr < rows_in; jr++)
    {
        if (t < cols_in) Ash[jr][t] = mat[(size_t)(r0 + jr) * stride + (c0 + t)]; /* 列连续->合并 */
        __syncthreads();
        if (t < cols_in) sum += Ash[jr][t] * vsh[jr];
        __syncthreads();
    }
    if (t < cols_in) part[(size_t)blockIdx.y * cols + (c0 + t)] = sum;
}

/* 沿 rowTile 维归约部分积 -> dst */
__global__ void mat_t_vec_v5_reduce(const float* __restrict__ part,
                                     float* __restrict__ dst,
                                     int numRowTiles, int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float s = 0.0f;
    for (int r = 0; r < numRowTiles; r++)
        s += part[(size_t)r * cols + c];
    dst[c] = s;
}

/* ============ 小矩阵单 kernel 路径 (matT·vec) ============
 * 与 part 路径同结构 (2D 分块 + 列连续合并读 A + vec 搬共享),
 * 但 block 内算完局部和直接用 atomicAdd 累加进 dst, 不再写 part 缓冲、不再跑第二 kernel。
 * 关键: 列块故意取小 (COL_TILE_A=32), 让小矩阵也能产生大量 block 拉高占用率。
 *   256x256 -> grid (8 x 8)=64 block (占用率 ~28%), 而非固定 256 列/block 时的 1x8=8 block (3.6%)。
 *   每列仍只被 rowTiles 个 row 块 atomicAdd 竞争 (256x256 仅 8 次), 开销可忽略。
 * 大矩阵 (每列被很多 rowTile 块竞争 atomic) 仍走 part 路径, 见下方 dispatch。 */
#define COL_TILE_A 32
__global__ void mat_t_vec_v5_kernel_atomic(const float* __restrict__ mat,
                                           const float* __restrict__ vec,
                                           float* __restrict__ dst,
                                           int rows, int cols, int stride)
{
    __shared__ float Ash[ROW_TILE_V5][COL_TILE_A];   /* [32][32] = 4KB */
    __shared__ float vsh[ROW_TILE_V5];

    int c0 = blockIdx.x * COL_TILE_A;          /* 列块起点 */
    int r0 = blockIdx.y * ROW_TILE_V5;         /* 行(收缩)块起点 */
    int t  = threadIdx.x;                      /* 0..255, 仅前 COL_TILE_A 个算 */

    int rows_in = (rows - r0 < ROW_TILE_V5) ? (rows - r0) : ROW_TILE_V5;
    int cols_in = (cols - c0 < COL_TILE_A)  ? (cols - c0) : COL_TILE_A;

    /* 并行把 vec 的 rowTile 段搬进共享 (256 线程 strided 覆盖 [0,rows_in)) */
    for (int j = t; j < rows_in; j += blockDim.x) vsh[j] = vec[r0 + j];
    __syncthreads();

    float sum = 0.0f;
    if (t < cols_in)
    {
        int mycol = c0 + t;
        for (int jr = 0; jr < rows_in; jr++)
        {
            Ash[jr][t] = mat[(size_t)(r0 + jr) * stride + mycol]; /* 列连续 -> 合并 */
            __syncthreads();
            sum += Ash[jr][t] * vsh[jr];
            __syncthreads();
        }
        atomicAdd(&dst[mycol], sum);
    }
}

/* ============ 仅启动核 (供 benchmark, part 缓冲缓存复用) ============ */
static float* g_part = nullptr;
static size_t g_part_cap = 0;
static void ensure_part(size_t need_bytes)
{
    if (need_bytes > g_part_cap)
    {
        if (g_part) cudaFree(g_part);
        cudaMalloc(&g_part, need_bytes);
        g_part_cap = need_bytes;
    }
}

void matvec_v5_launch(const float* d_mat, const float* d_vec, float* d_dst,
                      int rows, int cols, int stride)
{
    matvec_v5_kernel<<<rows, BS>>>(d_mat, d_vec, d_dst, rows, cols, stride);
}

void mat_t_vec_v5_launch(const float* d_mat, const float* d_vec, float* d_dst,
                         int rows, int cols, int stride)
{
    /* 极小矩阵 (<=256x256): 走单 kernel + atomicAdd (高占用率, 无 part 全局往返)。
     * 其余: 走原 2D 分块 + part 缓冲 + 轻归约 (避免每列被大量 rowTile 块竞争 atomic + 多一次 memset launch)。
     * 说明: 256x256 实测仍差 cuBLAS 约一倍, 根因是 GPU 小矩阵 GEMV 的 kernel launch 固定开销
     *       (~3-5us/次) 占 0.13MFLOP 计算的主导, 任何 2-launch 手写路径都卡在 ~8us(≈17GF),
     *       需 CUDA Graph 消除 launch 开销才能真正追平 cuBLAS 的 4us(≈35GF)。 */
    if ((long)rows * cols <= 256L * 256)
    {
        dim3 grid((cols + COL_TILE_A - 1) / COL_TILE_A, (rows + ROW_TILE_V5 - 1) / ROW_TILE_V5);
        cudaMemsetAsync((void*)d_dst, 0, (size_t)cols * sizeof(float), cudaStreamDefault);
        mat_t_vec_v5_kernel_atomic<<<grid, 256>>>(d_mat, d_vec, d_dst, rows, cols, stride);
    }
    else
    {
        int rowTiles = (rows + ROW_TILE_V5 - 1) / ROW_TILE_V5;
        size_t need = (size_t)rowTiles * cols * sizeof(float);
        ensure_part(need);
        dim3 grid((cols + COL_TILE_V5 - 1) / COL_TILE_V5, rowTiles);
        mat_t_vec_v5_kernel<<<grid, COL_TILE_V5>>>(d_mat, d_vec, g_part, rows, cols, stride);
        int reduce_blocks = (cols + 255) / 256;
        mat_t_vec_v5_reduce<<<reduce_blocks, 256>>>(g_part, d_dst, rowTiles, cols);
    }
}

/* ============ 主机封装 ============ */
static void cuda_check(cudaError_t err, const char* where)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error @ %s: %s\n", where, cudaGetErrorString(err));
        exit(1);
    }
}

void mat_mul_vet_real32_cuda_v5(ivf32* dst, const ivf32* mat, int rows, int cols, int stride, const ivf32* vec)
{
    const size_t mat_bytes = (size_t)rows * stride * sizeof(ivf32);
    const size_t vec_bytes = (size_t)cols   * sizeof(ivf32);
    const size_t dst_bytes = (size_t)rows   * sizeof(ivf32);

    float *d_mat, *d_vec, *d_dst;
    cuda_check(cudaMalloc(&d_mat, mat_bytes), "v5 cudaMalloc mat");
    cuda_check(cudaMalloc(&d_vec, vec_bytes), "v5 cudaMalloc vec");
    cuda_check(cudaMalloc(&d_dst, dst_bytes), "v5 cudaMalloc dst");

    cuda_check(cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice), "v5 H2D mat");
    cuda_check(cudaMemcpy(d_vec, vec, vec_bytes, cudaMemcpyHostToDevice), "v5 H2D vec");

    matvec_v5_launch(d_mat, d_vec, d_dst, rows, cols, stride);
    cuda_check(cudaGetLastError(), "v5 launch matvec");

    cuda_check(cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost), "v5 D2H dst");

    cuda_check(cudaFree(d_mat), "v5 free mat");
    cuda_check(cudaFree(d_vec), "v5 free vec");
    cuda_check(cudaFree(d_dst), "v5 free dst");
}

void mat_t_mul_vet_real32_cuda_v5(ivf32* dst, const ivf32* mat, int rows, int cols, int stride, const ivf32* vec)
{
    const size_t mat_bytes = (size_t)rows * stride * sizeof(ivf32);
    const size_t vec_bytes = (size_t)rows   * sizeof(ivf32);   /* vec 长 = rows */
    const size_t dst_bytes = (size_t)cols   * sizeof(ivf32);   /* dst 长 = cols */
    int rowTiles = (rows + ROW_TILE_V5 - 1) / ROW_TILE_V5;
    const size_t part_bytes = (size_t)rowTiles * cols * sizeof(float);

    float *d_mat, *d_vec, *d_dst, *d_part;
    cuda_check(cudaMalloc(&d_mat, mat_bytes), "v5 cudaMalloc mat");
    cuda_check(cudaMalloc(&d_vec, vec_bytes), "v5 cudaMalloc vec");
    cuda_check(cudaMalloc(&d_dst, dst_bytes), "v5 cudaMalloc dst");
    cuda_check(cudaMalloc(&d_part, part_bytes), "v5 cudaMalloc part");

    cuda_check(cudaMemcpy(d_mat, mat, mat_bytes, cudaMemcpyHostToDevice), "v5 H2D mat");
    cuda_check(cudaMemcpy(d_vec, vec, vec_bytes, cudaMemcpyHostToDevice), "v5 H2D vec");

    mat_t_vec_v5_launch(d_mat, d_vec, d_dst, rows, cols, stride);
    cuda_check(cudaGetLastError(), "v5 launch mat_t_vec");

    cuda_check(cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost), "v5 D2H dst");

    cuda_check(cudaFree(d_mat), "v5 free mat");
    cuda_check(cudaFree(d_vec), "v5 free vec");
    cuda_check(cudaFree(d_dst), "v5 free dst");
    cuda_check(cudaFree(d_part), "v5 free part");
}
