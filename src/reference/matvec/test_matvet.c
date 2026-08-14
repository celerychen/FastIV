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
 * test_matvet.c - 独立测试 (不参与库实现)
 *
 * 用朴素参考实现 (oracle) 校验各指令集版本:
 *   - mat_mul_vet_real32      (纯 C)
 *   - mat_mul_vet_real32_avx  (AVX2+FMA, 编译目标支持时)
 *   - mat_mul_vet_real32_neon (NEON, ARM 目标时)
 *   以及对应的 mat^T · vec 版本。
 *
 * 覆盖: 多种行列尺寸 (含非 4/8 倍数以触发尾部处理), 以及 mat_stride > cols (带 padding)
 *       以验证 stride 修复。
 */

#include "matvet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef IVF32_DEFINED
typedef float ivf32;
#endif

/* ---- 朴素参考实现 (oracle), 严格按 mat_stride 访问 ---- */
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

/* 容差比较: 允许绝对误差或相对误差 */
static int fequal(float a, float b)
{
    float d = fabsf(a - b);
    return d <= 1e-2f || d <= 1e-3f * fabsf(a);
}

static int check_arr(const char* tag, const ivf32* got, const ivf32* ref, int n, int* fail)
{
    for (int i = 0; i < n; i++)
    {
        if (!fequal((float)got[i], (float)ref[i]))
        {
            printf("    FAIL %s[%d] = %g, expect %g\n", tag, i, (float)got[i], (float)ref[i]);
            *fail += 1;
            return 0;
        }
    }
    return 1;
}

static float rnd_val(void)
{
    /* [-2, 2] */
    return (float)((rand() % 1000) / 250.0 - 2.0);
}

static void run_case(int rows, int cols, int stride, int* fail)
{
    int mat_cap = (int)((size_t)rows * stride);
    ivf32* mat  = (ivf32*)malloc((size_t)mat_cap * sizeof(ivf32));
    ivf32* vec  = (ivf32*)malloc((size_t)cols * sizeof(ivf32));   /* mat·vec 用, 长度 = cols */
    ivf32* vecR = (ivf32*)malloc((size_t)rows * sizeof(ivf32));   /* mat^T·vec 用, 长度 = rows */
    for (int k = 0; k < mat_cap; k++) mat[k] = (ivf32)rnd_val();
    for (int k = 0; k < cols; k++)    vec[k] = (ivf32)rnd_val();
    for (int k = 0; k < rows; k++)    vecR[k] = (ivf32)rnd_val();

    ivf32* dC   = (ivf32*)malloc((size_t)rows * sizeof(ivf32));
    ivf32* dRef = (ivf32*)malloc((size_t)rows * sizeof(ivf32));
    ivf32* dtC  = (ivf32*)malloc((size_t)cols * sizeof(ivf32));
    ivf32* dtRef= (ivf32*)malloc((size_t)cols * sizeof(ivf32));

    /* ---- mat · vec ---- */
    ref_mat_mul(dRef, mat, rows, cols, stride, vec);

    mat_mul_vet_real32(dC, mat, rows, cols, stride, vec);
    check_arr("mat_mul(C)", dC, dRef, rows, fail);

#ifdef __AVX2__
    {
        ivf32* dA = (ivf32*)malloc((size_t)rows * sizeof(ivf32));
        mat_mul_vet_real32_avx(dA, mat, rows, cols, stride, vec);
        check_arr("mat_mul(AVX)", dA, dRef, rows, fail);
        free(dA);
    }
#endif
#ifdef __ARM_NEON
    {
        ivf32* dN = (ivf32*)malloc((size_t)rows * sizeof(ivf32));
        mat_mul_vet_real32_neon(dN, mat, rows, cols, stride, vec);
        check_arr("mat_mul(NEON)", dN, dRef, rows, fail);
        free(dN);
    }
#endif

    /* ---- mat^T · vec ---- */
    ref_mat_t_mul(dtRef, mat, rows, cols, stride, vecR);

    mat_t_mul_vet_real32(dtC, mat, rows, cols, stride, vecR);
    check_arr("mat_t_mul(C)", dtC, dtRef, cols, fail);

#ifdef __AVX2__
    {
        ivf32* dtA = (ivf32*)malloc((size_t)cols * sizeof(ivf32));
        mat_t_mul_vet_real32_avx(dtA, mat, rows, cols, stride, vecR);
        check_arr("mat_t_mul(AVX)", dtA, dtRef, cols, fail);
        free(dtA);
    }
#endif
#ifdef __ARM_NEON
    {
        ivf32* dtN = (ivf32*)malloc((size_t)cols * sizeof(ivf32));
        mat_t_mul_vet_real32_neon(dtN, mat, rows, cols, stride, vecR);
        check_arr("mat_t_mul(NEON)", dtN, dtRef, cols, fail);
        free(dtN);
    }
#endif

    free(mat); free(vec); free(vecR);
    free(dC); free(dRef); free(dtC); free(dtRef);
}

int main(void)
{
    int sizes[] = { 1, 3, 7, 8, 16, 33, 100 };
    int n = sizeof(sizes) / sizeof(sizes[0]);
    int fail = 0;
    int cases = 0;

    srand(12345);
    for (int a = 0; a < n; a++)
    {
        for (int b = 0; b < n; b++)
        {
            int rows = sizes[a];
            int cols = sizes[b];
            /* stride = cols (无 padding) 与 cols+3 (带 padding, 校验 stride 修复) */
            int strides[2] = { cols, cols + 3 };
            for (int si = 0; si < 2; si++)
            {
                run_case(rows, cols, strides[si], &fail);
                cases++;
            }
        }
    }

    printf("cases=%d, failures=%d\n", cases, fail);
#if defined(__AVX2__)
    printf("AVX2+FMA build: active\n");
#else
    printf("AVX2+FMA build: inactive (this compile target)\n");
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
    printf("NEON build: active\n");
#else
    printf("NEON build: inactive (this compile target, expected on x86)\n");
#endif
    if (fail == 0) printf("ALL TESTS PASS\n");
    else           printf("SOME TESTS FAILED\n");
    return fail == 0 ? 0 : 1;
}
