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

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef float ivf32;

#ifdef __aarch64__
#include <arm_neon.h>
typedef struct { float32x4_t lo, hi; } v8f;
static inline v8f v8_load(const ivf32* p){ v8f r; r.lo=vld1q_f32(p); r.hi=vld1q_f32(p+4); return r; }
static inline v8f v8_set1(ivf32 c){ v8f r; r.lo=vdupq_n_f32(c); r.hi=vdupq_n_f32(c); return r; }
static inline v8f v8_mul(v8f a, v8f b){ v8f r; r.lo=vmulq_f32(a.lo,b.lo); r.hi=vmulq_f32(a.hi,b.hi); return r; }
static inline v8f v8_fma(v8f a, v8f b, v8f c){ v8f r; r.lo=vfmaq_f32(a.lo,b.lo,c.lo); r.hi=vfmaq_f32(a.hi,b.hi,c.hi); return r; }
static inline void v8_store(ivf32* p, v8f a){ vst1q_f32(p,a.lo); vst1q_f32(p+4,a.hi); }
#elif defined(__AVX2__)
#include <immintrin.h>
typedef __m256 v8f;
static inline v8f v8_load(const ivf32* p){ return _mm256_loadu_ps(p); }
static inline v8f v8_set1(ivf32 c){ return _mm256_set1_ps(c); }
static inline v8f v8_mul(v8f a, v8f b){ return _mm256_mul_ps(a,b); }
static inline v8f v8_fma(v8f a, v8f b, v8f c){ return _mm256_fmadd_ps(a,b,c); }
static inline void v8_store(ivf32* p, v8f a){ _mm256_storeu_ps(p,a); }
#else
typedef struct { ivf32 s0,s1,s2,s3,s4,s5,s6,s7; } v8f;
static inline v8f v8_load(const ivf32* p){ v8f r; r.s0=p[0];r.s1=p[1];r.s2=p[2];r.s3=p[3];r.s4=p[4];r.s5=p[5];r.s6=p[6];r.s7=p[7]; return r; }
static inline v8f v8_set1(ivf32 c){ v8f r; r.s0=r.s1=r.s2=r.s3=r.s4=r.s5=r.s6=r.s7=c; return r; }
static inline v8f v8_mul(v8f a, v8f b){ v8f r; r.s0=a.s0*b.s0;r.s1=a.s1*b.s1;r.s2=a.s2*b.s2;r.s3=a.s3*b.s3;r.s4=a.s4*b.s4;r.s5=a.s5*b.s5;r.s6=a.s6*b.s6;r.s7=a.s7*b.s7; return r; }
static inline v8f v8_fma(v8f a, v8f b, v8f c){ v8f r; r.s0=a.s0*b.s0+c.s0;r.s1=a.s1*b.s1+c.s1;r.s2=a.s2*b.s2+c.s2;r.s3=a.s3*b.s3+c.s3;r.s4=a.s4*b.s4+c.s4;r.s5=a.s5*b.s5+c.s5;r.s6=a.s6*b.s6+c.s6;r.s7=a.s7*b.s7+c.s7; return r; }
static inline void v8_store(ivf32* p, v8f a){ p[0]=a.s0;p[1]=a.s1;p[2]=a.s2;p[3]=a.s3;p[4]=a.s4;p[5]=a.s5;p[6]=a.s6;p[7]=a.s7; }
#endif

/* ============================================================================
 * conv2d_3x1_1x3_v2 —— 在 v1 架构之上，手写 SIMD intrinsics 显式向量化
 *
 * 输入为 6 个系数：coef[0..2] = 行滤波(横向 1x3)，coef[3..5] = 列滤波(纵向 3x1)。
 *
 * 架构完全沿用 v1（保证 in-place 与正确性契约不变）：
 *   6 行环形缓冲 buf、外层 y += 4、avail 分派、发射前保证 src[oo+1] 已入 buf。
 * 仅把两个热点用 SIMD 重写（这是“我自己的向量化思路”，不是交给编译器自动向量化）：
 *
 *   1) 行滤波 row_filter：横向 3-tap stencil。
 *        - NEON：错位加载三 4-向量 A/B/C，用 vext 对齐出 {L,C,R}，8/次，0.375 次 load/像素。
 *        - AVX2：三路重叠 loadu L=srow[x-1..x+6]、C=srow[x..x+7]、R=srow[x+1..x+8]，
 *                out = H0*L + H1*C + H2*R，8/次，0.375 次 load/像素（同 AVX2 无跨道 extract）。
 *        - 标量：v1 的同构体。
 *        左/右边界各一次标量 clamp，内部零分支。
 *
 *   2) 列滤波发射（avail 分派的每个分支内，稳态 avail>=4）：
 *        emit_rows 用 v8f 抽象（NEON=双 4-向量 / AVX2=单 8 向量 / 标量=8 标量）一次处理 8 列，
 *        加载相邻 6 个缓冲行向量 t0..t5，算出 4 路输出
 *            d0 = V0*t0 + V1*t1 + V2*t2
 *            d1 = V0*t1 + V1*t2 + V2*t3
 *            d2 = V0*t2 + V1*t3 + V2*t4
 *            d3 = V0*t3 + V1*t4 + V2*t5
 *        相邻输出共享 t 向量，比逐路独立加载更省带宽；每 8 列一次 store 4 行。
 *
 * 每输出像素 6 次乘累加，等价于可分离 3x3 二维卷积。
 * ==========================================================================*/

/* 6 行环形缓冲里，源行 j（已行滤波）所在槽位；j 越界按 replicate 钳制。
 * w = 最近一次写入的源行 last_y 的槽位，故源行 j 的槽位 = (w - (last_y - j) + 6) % 6。 */
static inline ivf32* tmp_row(ivf32* buf, int w, int last_y, int ws, int hs, int j) {
    if (j < 0)   j = 0;
    if (j >= hs) j = hs - 1;
    int slot = (w - (last_y - j) + 6) % 6;
    return buf + (size_t)slot * ws;
}

/* 行滤波：对一条源行做横向 1x3，写入 brow。
 * 左边界 x=0 与右边界 x=W-1 各一次标量 clamp（replicate），内部零分支。
 * 按架构分支：NEON（错位 vext 三 4-向量）/ AVX2（三路重叠 loadu 8-向量）/ 标量。 */
#ifdef __aarch64__
static inline void row_filter(ivf32* brow, const ivf32* srow, int W,
                              ivf32 h0, ivf32 h1, ivf32 h2) {
    float32x4_t H0 = vdupq_n_f32(h0), H1 = vdupq_n_f32(h1), H2 = vdupq_n_f32(h2);
    if (W == 1) {                                   /* 退化：邻居全钳到自身 */
        brow[0] = (h0 + h1 + h2) * srow[0];
        return;
    }
    brow[0] = (h0 + h1) * srow[0] + h2 * srow[1];    /* 左边界：xl 钳到 0 */
    int x = 1;
    for (; x + 8 <= W - 4; x += 8) {
        float32x4_t A = vld1q_f32(srow + x - 1);    /* s[x-1 .. x+2] */
        float32x4_t B = vld1q_f32(srow + x + 3);    /* s[x+3 .. x+6] */
        float32x4_t C = vld1q_f32(srow + x + 7);    /* s[x+7 .. x+10] */
        float32x4_t C0 = vextq_f32(A, B, 1);        /* s[x   .. x+3] */
        float32x4_t R0 = vextq_f32(A, B, 2);        /* s[x+1 .. x+4] */
        vst1q_f32(brow + x,     vfmaq_f32(vfmaq_f32(vmulq_f32(A, H0), C0, H1), R0, H2));
        float32x4_t C1 = vextq_f32(B, C, 1);        /* s[x+4 .. x+7] */
        float32x4_t R1 = vextq_f32(B, C, 2);        /* s[x+5 .. x+8] */
        vst1q_f32(brow + x + 4, vfmaq_f32(vfmaq_f32(vmulq_f32(B, H0), C1, H1), R1, H2));
    }
    for (; x <= W - 2; x++)                         /* 尾部剩余标量补 */
        brow[x] = h0 * srow[x - 1] + h1 * srow[x] + h2 * srow[x + 1];
    brow[W - 1] = h0 * srow[W - 2] + (h1 + h2) * srow[W - 1];  /* 右边界：xr 钳到 W-1 */
}
#elif defined(__AVX2__)
static inline void row_filter(ivf32* brow, const ivf32* srow, int W,
                              ivf32 h0, ivf32 h1, ivf32 h2) {
    __m256 H0 = _mm256_set1_ps(h0), H1 = _mm256_set1_ps(h1), H2 = _mm256_set1_ps(h2);
    if (W == 1) {                                   /* 退化：邻居全钳到自身 */
        brow[0] = (h0 + h1 + h2) * srow[0];
        return;
    }
    brow[0] = (h0 + h1) * srow[0] + h2 * srow[1];    /* 左边界：xl 钳到 0 */
    int x = 1;
    /* 内部：8 个/次。三路重叠 loadu L=srow[x-1..x+6]、C=srow[x..x+7]、R=srow[x+1..x+8]，
     * out = H0*L + H1*C + H2*R。每 8 输出仅 3 次 load（0.375/像素）。
     * 循环界 x+8 <= W-1 保证 L/C/R 三次 load 均不越界（最大读 s[x+8] <= s[W-1]）。 */
    for (; x + 8 <= W - 1; x += 8) {
        __m256 L = _mm256_loadu_ps(srow + x - 1);
        __m256 C = _mm256_loadu_ps(srow + x);
        __m256 R = _mm256_loadu_ps(srow + x + 1);
        _mm256_storeu_ps(brow + x,
            _mm256_fmadd_ps(H2, R, _mm256_fmadd_ps(H1, C, _mm256_mul_ps(H0, L))));
    }
    for (; x <= W - 2; x++)                         /* 尾部剩余标量补 */
        brow[x] = h0 * srow[x - 1] + h1 * srow[x] + h2 * srow[x + 1];
    brow[W - 1] = h0 * srow[W - 2] + (h1 + h2) * srow[W - 1];  /* 右边界：xr 钳到 W-1 */
}
#else
static inline void row_filter(ivf32* brow, const ivf32* srow, int W,
                              ivf32 h0, ivf32 h1, ivf32 h2) {
    if (W == 1) {                                   /* 退化：邻居全钳到自身 */
        brow[0] = (h0 + h1 + h2) * srow[0];
        return;
    }
    brow[0]   = (h0 + h1) * srow[0] + h2 * srow[1];             /* 左边界：xl 钳到 0   */
    for (int x = 1; x < W - 1; x++)                            /* 内部：无钳制        */
        brow[x] = h0 * srow[x - 1] + h1 * srow[x] + h2 * srow[x + 1];
    brow[W-1] = h0 * srow[W - 2] + (h1 + h2) * srow[W - 1];     /* 右边界：xr 钳到 W-1 */
}
#endif

/* 行滤波一个源行 yr 入 buf[ring]，并推进 ring / last_y（取代原 RF 宏）。
 * 仅传标量系数；SIMD 向量在 row_filter 内按架构构造，避免 SIMD 类型泄漏到签名。 */
static inline void rf(ivf32* buf, int* ring, int* last_y, int yr,
                      const ivf32* src, int ws, int ss,
                      ivf32 h0, ivf32 h1, ivf32 h2) {
    int w = *ring;
    row_filter(buf + (size_t)w * ws, src + (size_t)yr * ss, ws, h0, h1, h2);
    *ring = (w + 1) % 6;
    *last_y = yr;
}

/* 列滤波发射：用 v8f 抽象一次处理 8 列，相邻输出共享 t 向量。
 * 按 avail（本 4 行块可发射的输出行数 1..4）分派；未用到的 t/d 指针可为 NULL（永不解引用）。
 * v8f 在 NEON 下为双 4-向量、AVX2 下为单 8 向量、标量下为 8 标量结构体，故一份代码三架构通用。 */
static inline void emit_rows(ivf32* t0, ivf32* t1, ivf32* t2, ivf32* t3, ivf32* t4, ivf32* t5,
                             ivf32* d0, ivf32* d1, ivf32* d2, ivf32* d3,
                             int avail, int W, ivf32 v0, ivf32 v1, ivf32 v2) {
    v8f V0 = v8_set1(v0), V1 = v8_set1(v1), V2 = v8_set1(v2);
    int x = 0;
    if (avail >= 4) {
        for (; x + 8 <= W; x += 8) {
            v8f a0 = v8_load(t0 + x), a1 = v8_load(t1 + x), a2 = v8_load(t2 + x),
                a3 = v8_load(t3 + x), a4 = v8_load(t4 + x), a5 = v8_load(t5 + x);
            v8_store(d0 + x, v8_fma(V2, a2, v8_fma(V1, a1, v8_mul(V0, a0))));
            v8_store(d1 + x, v8_fma(V2, a3, v8_fma(V1, a2, v8_mul(V0, a1))));
            v8_store(d2 + x, v8_fma(V2, a4, v8_fma(V1, a3, v8_mul(V0, a2))));
            v8_store(d3 + x, v8_fma(V2, a5, v8_fma(V1, a4, v8_mul(V0, a3))));
        }
        for (; x < W; x++) {
            d0[x] = v0 * t0[x] + v1 * t1[x] + v2 * t2[x];
            d1[x] = v0 * t1[x] + v1 * t2[x] + v2 * t3[x];
            d2[x] = v0 * t2[x] + v1 * t3[x] + v2 * t4[x];
            d3[x] = v0 * t3[x] + v1 * t4[x] + v2 * t5[x];
        }
    } else if (avail == 3) {
        for (; x + 8 <= W; x += 8) {
            v8f a0 = v8_load(t0 + x), a1 = v8_load(t1 + x), a2 = v8_load(t2 + x),
                a3 = v8_load(t3 + x), a4 = v8_load(t4 + x);
            v8_store(d0 + x, v8_fma(V2, a2, v8_fma(V1, a1, v8_mul(V0, a0))));
            v8_store(d1 + x, v8_fma(V2, a3, v8_fma(V1, a2, v8_mul(V0, a1))));
            v8_store(d2 + x, v8_fma(V2, a4, v8_fma(V1, a3, v8_mul(V0, a2))));
        }
        for (; x < W; x++) {
            d0[x] = v0 * t0[x] + v1 * t1[x] + v2 * t2[x];
            d1[x] = v0 * t1[x] + v1 * t2[x] + v2 * t3[x];
            d2[x] = v0 * t2[x] + v1 * t3[x] + v2 * t4[x];
        }
    } else if (avail == 2) {
        for (; x + 8 <= W; x += 8) {
            v8f a0 = v8_load(t0 + x), a1 = v8_load(t1 + x), a2 = v8_load(t2 + x),
                a3 = v8_load(t3 + x);
            v8_store(d0 + x, v8_fma(V2, a2, v8_fma(V1, a1, v8_mul(V0, a0))));
            v8_store(d1 + x, v8_fma(V2, a3, v8_fma(V1, a2, v8_mul(V0, a1))));
        }
        for (; x < W; x++) {
            d0[x] = v0 * t0[x] + v1 * t1[x] + v2 * t2[x];
            d1[x] = v0 * t1[x] + v1 * t2[x] + v2 * t3[x];
        }
    } else { /* avail == 1 */
        for (; x + 8 <= W; x += 8) {
            v8f a0 = v8_load(t0 + x), a1 = v8_load(t1 + x), a2 = v8_load(t2 + x);
            v8_store(d0 + x, v8_fma(V2, a2, v8_fma(V1, a1, v8_mul(V0, a0))));
        }
        for (; x < W; x++) {
            d0[x] = v0 * t0[x] + v1 * t1[x] + v2 * t2[x];
        }
    }
}

void conv2d_3x1_1x3_v2(ivf32* dst, int width_dst, int height_dst, int stride_dst,
                       ivf32* src, int width_src, int height_src, int stride_src,
                       ivf32 coef[6])
{
    /* 直接取 6 个系数：前 3 个行滤波，后 3 个列滤波 */
    ivf32 h0 = coef[0], h1 = coef[1], h2 = coef[2];   /* 行滤波(横向) 标量 */
    ivf32 v0 = coef[3], v1 = coef[4], v2 = coef[5];   /* 列滤波(纵向) 标量 */

    /* 6 行小缓冲（行滤波结果滑动窗口），取代整图 tmp */
    ivf32* buf = (ivf32*)malloc((size_t)6 * width_src * sizeof(ivf32));

    int ring = 0;        /* 下一个写入槽 */
    int w = 0;           /* 最近一次写入的源行 last_y 的槽位 */
    int last_y = -1;     /* 最近一次写入的源行号（收尾用） */
    int next_out = 0;    /* 下一个待输出的输出行 */

    /* 主循环：y += 4，每次处理 4 个源行，发射最多 4 行输出 */
    for (int y = 0; y < height_src; y += 4) {
        /* ---- 行滤波 4 个源行（越界跳过） ---- */
        rf(buf, &ring, &last_y, y, src, width_src, stride_src, h0, h1, h2);
        if (y + 1 < height_src) rf(buf, &ring, &last_y, y + 1, src, width_src, stride_src, h0, h1, h2);
        if (y + 2 < height_src) rf(buf, &ring, &last_y, y + 2, src, width_src, stride_src, h0, h1, h2);
        if (y + 3 < height_src) rf(buf, &ring, &last_y, y + 3, src, width_src, stride_src, h0, h1, h2);
        w = (ring - 1 + 6) % 6;            /* w = 最近写入源行 last_y 的槽位 */

        /* ---- 列滤波：buf 就绪后一次发射最多 4 行 [next_out .. next_out+3] ---- */
        int ready = last_y - 1;                   /* 当前可发射的最高输出行（含） */
        if (ready > height_dst - 1) ready = height_dst - 1;
        if (ready < 0) ready = -1;
        int avail = height_dst - next_out;        /* 剩余待输出行数 */
        if (avail > 4) avail = 4;
        if (avail > ready - next_out + 1) avail = ready - next_out + 1;  /* 不超过就绪范围 */
        if (avail > 0) {
            int j0 = next_out;
            ivf32* t0 = tmp_row(buf, w, last_y, width_src, height_src, j0 - 1);
            ivf32* t1 = tmp_row(buf, w, last_y, width_src, height_src, j0);
            ivf32* t2 = tmp_row(buf, w, last_y, width_src, height_src, j0 + 1);
            ivf32* t3 = tmp_row(buf, w, last_y, width_src, height_src, j0 + 2);
            ivf32* t4 = tmp_row(buf, w, last_y, width_src, height_src, j0 + 3);
            ivf32* t5 = tmp_row(buf, w, last_y, width_src, height_src, j0 + 4);
            ivf32* d0 = dst + (size_t)(j0 + 0) * stride_dst;
            ivf32* d1 = dst + (size_t)(j0 + 1) * stride_dst;
            ivf32* d2 = dst + (size_t)(j0 + 2) * stride_dst;
            ivf32* d3 = dst + (size_t)(j0 + 3) * stride_dst;
            /* 循环外提(loop unswitching)：avail 每 4 行块只算一次，在此按值分派，
             * emit_rows 内最内层 SIMD 循环（v8f）纯直线（load/fma/store），零分支。 */
            int W = width_dst;
            emit_rows(t0, t1, t2, t3, t4, t5, d0, d1, d2, d3, avail, W, v0, v1, v2);
            next_out += avail;
        }
    }

    /* ---- 收尾：输出剩余行（用 buf 现存行 + 钳制），v8f 单路 + 标量尾部 ---- */
    while (next_out < height_dst) {
        int j = next_out;
        ivf32* tT = tmp_row(buf, w, last_y, width_src, height_src, j - 1);
        ivf32* tM = tmp_row(buf, w, last_y, width_src, height_src, j);
        ivf32* tB = tmp_row(buf, w, last_y, width_src, height_src, j + 1);
        ivf32* drow = dst + (size_t)j * stride_dst;
        emit_rows(tT, tM, tB, NULL, NULL, NULL, drow, NULL, NULL, NULL, 1, width_dst, v0, v1, v2);
        next_out++;
    }

    free(buf);
}
