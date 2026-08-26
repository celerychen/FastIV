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

#include "fiv_mat_mul.h"
#include "fiv_common.h"
#include "fiv_mat_mul_db.h"

#include <string.h>   /* memset */
#include <stdio.h>    /* printf (kept verbatim from the verified kernels) */

#if defined(FIV_USE_AVX) || defined(FIV_USE_AVX2)
#include <immintrin.h>
#endif

#ifndef FIV_INLINE
#define FIV_INLINE static inline
#endif

#ifndef FIV_PRINT_LOG
#define FIV_PRINT_LOG(msg) ((void)0)
#endif

/* ============================================================================
   Small-matrix kernels. Copied verbatim from the verified
   fiv_small_matrix_mul.c (real32 only; the real64/db variants are out of scope
   for now). The only edits are: SIMD feature macros mapped to the project's
   canonical FIV_USE_AVX2 / FIV_USE_AVX / FIV_USE_ARM_NEON, and the functions
   made static (they live inside this translation unit now).
   ========================================================================== */

#if defined(FIV_USE_AVX)
FIV_INLINE ivf32 fiv_mm_hsum_ps_sse3(__m128 v)
{
	__m128 shuf = _mm_movehdup_ps(v);
	__m128 sums = _mm_add_ps(v, shuf);
	shuf = _mm_movehl_ps(shuf, sums);
	sums = _mm_add_ps(sums, shuf);
	return _mm_cvtss_f32(sums);
}
#endif

#if defined(FIV_USE_AVX2)
FIV_INLINE ivf32 fiv_mm256_hsum_ps(__m256 v)
{
	__m128 vlow  = _mm256_castps256_ps128(v);
	__m128 vhigh = _mm256_extractf128_ps(v, 1);
	vlow = _mm_add_ps(vlow, vhigh);
	return fiv_mm_hsum_ps_sse3(vlow);
}
#endif

#if defined(FIV_USE_ARM_NEON)
/* Reduce 4 accumulators (4 outputs x 4 lane-partials) to one vector holding
   the 4 complete output sums, via pairwise adds. Keeps the results in the
   vector register file (no scalar extraction); the caller stores the vector
   directly, which is cheaper than 4 horizontal sums + 4 scalar stores. */
FIV_INLINE float32x4_t fiv_neon_hsum_4x4(float32x4_t a0, float32x4_t a1,
                                         float32x4_t a2, float32x4_t a3)
{
	float32x4_t p0 = vpaddq_f32(a0, a1);
	float32x4_t p1 = vpaddq_f32(a2, a3);
	return vpaddq_f32(p0, p1);
}
#endif

/* C = A * B, A: rows_a x cols_a, B: cols_a x cols_b */
static void fiv_small_matrix_mul_matrix_real32(
	ivf32* data_a, int rows_a, int cols_a, int stride_a,
	ivf32* data_b, int cols_b, int stride_b,
	ivf32* data_c, int stride_c, ivf32 alpha, ivf32 beta)
{
	int i = 0;
	for (; i <= rows_a - 4; i += 4){
		ivf32* ptr_a[4], *ptr_c[4];
		ptr_a[0] = &data_a[i * stride_a];
		ptr_a[1] = ptr_a[0] + stride_a;
		ptr_a[2] = ptr_a[1] + stride_a;
		ptr_a[3] = ptr_a[2] + stride_a;

		ptr_c[0] = &data_c[i * stride_c];
		ptr_c[1] = ptr_c[0] + stride_c;
		ptr_c[2] = ptr_c[1] + stride_c;
		ptr_c[3] = ptr_c[2] + stride_c;

		if (beta == 0.f) {
			memset(ptr_c[0], 0, sizeof(ivf32) * cols_b);
			memset(ptr_c[1], 0, sizeof(ivf32) * cols_b);
			memset(ptr_c[2], 0, sizeof(ivf32) * cols_b);
			memset(ptr_c[3], 0, sizeof(ivf32) * cols_b);
		}  else if (beta != 1.f) {
			for (int l = 0; l < cols_b; l++){
				ptr_c[0][l] *= beta;
				ptr_c[1][l] *= beta;
				ptr_c[2][l] *= beta;
				ptr_c[3][l] *= beta;
			}
		}
		int k = 0;
		for (; k <= cols_a - 2; k += 2)	{
			ivf32 t_a[8];
			if (alpha != 1.f) {
				t_a[0] = ptr_a[0][k + 0] * alpha;
				t_a[1] = ptr_a[1][k + 0] * alpha;
				t_a[2] = ptr_a[2][k + 0] * alpha;
				t_a[3] = ptr_a[3][k + 0] * alpha;
				t_a[4] = ptr_a[0][k + 1] * alpha;
				t_a[5] = ptr_a[1][k + 1] * alpha;
				t_a[6] = ptr_a[2][k + 1] * alpha;
				t_a[7] = ptr_a[3][k + 1] * alpha;
			}	else {
				t_a[0] = ptr_a[0][k + 0] ;
				t_a[1] = ptr_a[1][k + 0] ;
				t_a[2] = ptr_a[2][k + 0] ;
				t_a[3] = ptr_a[3][k + 0] ;
				t_a[4] = ptr_a[0][k + 1] ;
				t_a[5] = ptr_a[1][k + 1] ;
				t_a[6] = ptr_a[2][k + 1] ;
				t_a[7] = ptr_a[3][k + 1] ;

			}
			ivf32* ptr_b[2] = {&data_b[k * stride_b], &data_b[(k + 1)* stride_b]};
			int j = 0;
#if defined(FIV_USE_AVX2)
			__m256 m_t_a1 = _mm256_broadcast_ss(&t_a[0]);
			__m256 m_t_a2 = _mm256_broadcast_ss(&t_a[1]);
			__m256 m_t_a3 = _mm256_broadcast_ss(&t_a[2]);
			__m256 m_t_a4 = _mm256_broadcast_ss(&t_a[3]);
			__m256 m_t_a5 = _mm256_broadcast_ss(&t_a[4]);
			__m256 m_t_a6 = _mm256_broadcast_ss(&t_a[5]);
			__m256 m_t_a7 = _mm256_broadcast_ss(&t_a[6]);
			__m256 m_t_a8 = _mm256_broadcast_ss(&t_a[7]);

			for (; j <= cols_b - 8; j += 8)	{
				__m256 m_t_c0 = _mm256_loadu_ps(&ptr_c[0][j]);
				__m256 m_t_c1 = _mm256_loadu_ps(&ptr_c[1][j]);
				__m256 m_t_c2 = _mm256_loadu_ps(&ptr_c[2][j]);
				__m256 m_t_c3 = _mm256_loadu_ps(&ptr_c[3][j]);
				__m256 m_t_b0 = _mm256_loadu_ps(&ptr_b[0][j]);
				__m256 m_t_b1 = _mm256_loadu_ps(&ptr_b[1][j]);

				m_t_c0 = _mm256_fmadd_ps(m_t_a1, m_t_b0, m_t_c0);
				m_t_c1 = _mm256_fmadd_ps(m_t_a2, m_t_b0, m_t_c1);
				m_t_c2 = _mm256_fmadd_ps(m_t_a3, m_t_b0, m_t_c2);
				m_t_c3 = _mm256_fmadd_ps(m_t_a4, m_t_b0, m_t_c3);

				m_t_c0 = _mm256_fmadd_ps(m_t_a5, m_t_b1, m_t_c0);
				m_t_c1 = _mm256_fmadd_ps(m_t_a6, m_t_b1, m_t_c1);
				m_t_c2 = _mm256_fmadd_ps(m_t_a7, m_t_b1, m_t_c2);
				m_t_c3 = _mm256_fmadd_ps(m_t_a8, m_t_b1, m_t_c3);

				_mm256_storeu_ps(&ptr_c[0][j], m_t_c0);
				_mm256_storeu_ps(&ptr_c[1][j], m_t_c1);
				_mm256_storeu_ps(&ptr_c[2][j], m_t_c2);
				_mm256_storeu_ps(&ptr_c[3][j], m_t_c3);

			}
#elif defined(FIV_USE_ARM_NEON)
			for (; j <= cols_b - 8; j += 8) {
				float32x4x2_t b0 = { { vld1q_f32(&ptr_b[0][j]),     vld1q_f32(&ptr_b[0][j + 4]) } };
				float32x4x2_t b1 = { { vld1q_f32(&ptr_b[1][j]),     vld1q_f32(&ptr_b[1][j + 4]) } };
				float32x4x2_t c0 = { { vld1q_f32(&ptr_c[0][j]),     vld1q_f32(&ptr_c[0][j + 4]) } };
				float32x4x2_t c1 = { { vld1q_f32(&ptr_c[1][j]),     vld1q_f32(&ptr_c[1][j + 4]) } };
				float32x4x2_t c2 = { { vld1q_f32(&ptr_c[2][j]),     vld1q_f32(&ptr_c[2][j + 4]) } };
				float32x4x2_t c3 = { { vld1q_f32(&ptr_c[3][j]),     vld1q_f32(&ptr_c[3][j + 4]) } };

				c0.val[0] = vmlaq_n_f32(c0.val[0], b0.val[0], t_a[0]);
				c0.val[1] = vmlaq_n_f32(c0.val[1], b0.val[1], t_a[0]);
				c1.val[0] = vmlaq_n_f32(c1.val[0], b0.val[0], t_a[1]);
				c1.val[1] = vmlaq_n_f32(c1.val[1], b0.val[1], t_a[1]);
				c2.val[0] = vmlaq_n_f32(c2.val[0], b0.val[0], t_a[2]);
				c2.val[1] = vmlaq_n_f32(c2.val[1], b0.val[1], t_a[2]);
				c3.val[0] = vmlaq_n_f32(c3.val[0], b0.val[0], t_a[3]);
				c3.val[1] = vmlaq_n_f32(c3.val[1], b0.val[1], t_a[3]);

				c0.val[0] = vmlaq_n_f32(c0.val[0], b1.val[0], t_a[4]);
				c0.val[1] = vmlaq_n_f32(c0.val[1], b1.val[1], t_a[4]);
				c1.val[0] = vmlaq_n_f32(c1.val[0], b1.val[0], t_a[5]);
				c1.val[1] = vmlaq_n_f32(c1.val[1], b1.val[1], t_a[5]);
				c2.val[0] = vmlaq_n_f32(c2.val[0], b1.val[0], t_a[6]);
				c2.val[1] = vmlaq_n_f32(c2.val[1], b1.val[1], t_a[6]);
				c3.val[0] = vmlaq_n_f32(c3.val[0], b1.val[0], t_a[7]);
				c3.val[1] = vmlaq_n_f32(c3.val[1], b1.val[1], t_a[7]);

				vst1q_f32(&ptr_c[0][j],     c0.val[0]);
				vst1q_f32(&ptr_c[0][j + 4], c0.val[1]);
				vst1q_f32(&ptr_c[1][j],     c1.val[0]);
				vst1q_f32(&ptr_c[1][j + 4], c1.val[1]);
				vst1q_f32(&ptr_c[2][j],     c2.val[0]);
				vst1q_f32(&ptr_c[2][j + 4], c2.val[1]);
				vst1q_f32(&ptr_c[3][j],     c3.val[0]);
				vst1q_f32(&ptr_c[3][j + 4], c3.val[1]);
			}
#endif
			for (; j < cols_b; j++){
				ptr_c[0][j] += t_a[0] * ptr_b[0][j];
				ptr_c[1][j] += t_a[1] * ptr_b[0][j];
				ptr_c[2][j] += t_a[2] * ptr_b[0][j];
				ptr_c[3][j] += t_a[3] * ptr_b[0][j];
				
				ptr_c[0][j] += t_a[4] * ptr_b[1][j];
				ptr_c[1][j] += t_a[5] * ptr_b[1][j];
				ptr_c[2][j] += t_a[6] * ptr_b[1][j];
				ptr_c[3][j] += t_a[7] * ptr_b[1][j];
			}
		}

		for (; k < cols_a; k++)	{
			ivf32 t_a[4] = {
				ptr_a[0][k] * alpha,ptr_a[1][k] * alpha,
				ptr_a[2][k] * alpha,ptr_a[3][k] * alpha
			};
			ivf32* ptr_b = &data_b[k * stride_b];

			int j = 0;
#if defined(FIV_USE_AVX2)
			__m256 m_t_a1 = _mm256_broadcast_ss(&t_a[0]);
			__m256 m_t_a2 = _mm256_broadcast_ss(&t_a[1]);
			__m256 m_t_a3 = _mm256_broadcast_ss(&t_a[2]);
			__m256 m_t_a4 = _mm256_broadcast_ss(&t_a[3]);

			for (; j <= cols_b - 8; j += 8){
				__m256 m_t_c0 = _mm256_loadu_ps(&ptr_c[0][j]);
				__m256 m_t_c1 = _mm256_loadu_ps(&ptr_c[1][j]);
				__m256 m_t_c2 = _mm256_loadu_ps(&ptr_c[2][j]);
				__m256 m_t_c3 = _mm256_loadu_ps(&ptr_c[3][j]);
				__m256 m_t_b1 = _mm256_loadu_ps(&ptr_b[j]);

				m_t_c0 = _mm256_fmadd_ps(m_t_a1, m_t_b1, m_t_c0);
				m_t_c1 = _mm256_fmadd_ps(m_t_a2, m_t_b1, m_t_c1);
				m_t_c2 = _mm256_fmadd_ps(m_t_a3, m_t_b1, m_t_c2);
				m_t_c3 = _mm256_fmadd_ps(m_t_a4, m_t_b1, m_t_c3);

				_mm256_storeu_ps(&ptr_c[0][j], m_t_c0);
				_mm256_storeu_ps(&ptr_c[1][j], m_t_c1);
				_mm256_storeu_ps(&ptr_c[2][j], m_t_c2);
				_mm256_storeu_ps(&ptr_c[3][j], m_t_c3);
			}
#elif defined(FIV_USE_ARM_NEON)
			for (; j <= cols_b - 8; j += 8) {
				float32x4x2_t b  = { { vld1q_f32(&ptr_b[j]),         vld1q_f32(&ptr_b[j + 4]) } };
				float32x4x2_t c0 = { { vld1q_f32(&ptr_c[0][j]),     vld1q_f32(&ptr_c[0][j + 4]) } };
				float32x4x2_t c1 = { { vld1q_f32(&ptr_c[1][j]),     vld1q_f32(&ptr_c[1][j + 4]) } };
				float32x4x2_t c2 = { { vld1q_f32(&ptr_c[2][j]),     vld1q_f32(&ptr_c[2][j + 4]) } };
				float32x4x2_t c3 = { { vld1q_f32(&ptr_c[3][j]),     vld1q_f32(&ptr_c[3][j + 4]) } };

				c0.val[0] = vmlaq_n_f32(c0.val[0], b.val[0], t_a[0]);
				c0.val[1] = vmlaq_n_f32(c0.val[1], b.val[1], t_a[0]);
				c1.val[0] = vmlaq_n_f32(c1.val[0], b.val[0], t_a[1]);
				c1.val[1] = vmlaq_n_f32(c1.val[1], b.val[1], t_a[1]);
				c2.val[0] = vmlaq_n_f32(c2.val[0], b.val[0], t_a[2]);
				c2.val[1] = vmlaq_n_f32(c2.val[1], b.val[1], t_a[2]);
				c3.val[0] = vmlaq_n_f32(c3.val[0], b.val[0], t_a[3]);
				c3.val[1] = vmlaq_n_f32(c3.val[1], b.val[1], t_a[3]);

				vst1q_f32(&ptr_c[0][j],     c0.val[0]);
				vst1q_f32(&ptr_c[0][j + 4], c0.val[1]);
				vst1q_f32(&ptr_c[1][j],     c1.val[0]);
				vst1q_f32(&ptr_c[1][j + 4], c1.val[1]);
				vst1q_f32(&ptr_c[2][j],     c2.val[0]);
				vst1q_f32(&ptr_c[2][j + 4], c2.val[1]);
				vst1q_f32(&ptr_c[3][j],     c3.val[0]);
				vst1q_f32(&ptr_c[3][j + 4], c3.val[1]);
			}
#endif
          for (; j < cols_b; j++){
			  ptr_c[0][j] += t_a[0] * ptr_b[j];
			  ptr_c[1][j] += t_a[1] * ptr_b[j];
			  ptr_c[2][j] += t_a[2] * ptr_b[j];
			  ptr_c[3][j] += t_a[3] * ptr_b[j];
          }
		}
	}

	for (; i < rows_a; i++){
		ivf32* ptr_a = &data_a[i * stride_a];
		ivf32* ptr_c = &data_c[i * stride_c];
		if (beta == 0.f){
			memset(ptr_c, 0, sizeof(ivf32) * cols_b);
		}	else if (beta != 1.f) {
			for (int l = 0; l < cols_b; l++){
				ptr_c[l] *= beta;
			}
		}
		int k = 0;
		for (; k < cols_a; k++) {
			ivf32 t_a_d = ptr_a[k] * alpha;
			ivf32* ptr_b = &data_b[k * stride_b];
			for (int j = 0; j < cols_b; j++){
				ptr_c[j] += t_a_d * ptr_b[j];
			}
		}
	}
}

/* C = A * B^T, A: rows_a x cols_a, B: rows_b x cols_a, result rows_a x rows_b */
static void fiv_small_matrix_mul_matrix_t_real32(
	ivf32* data_a, int rows_a, int cols_a, int stride_a,
	ivf32* data_b, int rows_b, int stride_b,
	ivf32* data_c, int stride_c, ivf32 alpha, ivf32 beta)
{
	int i = 0;
	for (; i <= rows_a - 4; i += 4) {
		ivf32* ptr_lines_a[4], *ptr_lines_c[4];
		ptr_lines_a[0] = &data_a[i * stride_a];
		ptr_lines_a[1] = ptr_lines_a[0] + stride_a;
		ptr_lines_a[2] = ptr_lines_a[1] + stride_a;
		ptr_lines_a[3] = ptr_lines_a[2] + stride_a;

		ptr_lines_c[0] = &data_c[i * stride_c];
		ptr_lines_c[1] = ptr_lines_c[0] + stride_c;
		ptr_lines_c[2] = ptr_lines_c[1] + stride_c;
		ptr_lines_c[3] = ptr_lines_c[2] + stride_c;

		int j = 0;

		for (; j <= rows_b - 2; j += 2) {
			ivf32* ptr_lines_b[2];
			ptr_lines_b[0] = &data_b[j * stride_b];
			ptr_lines_b[1] = ptr_lines_b[0] + stride_b;

			ivf32 FIV_DALIGNED sum[4 * 2] = { 0 };
			int k = 0;
#if defined(FIV_USE_AVX2)
			__m256 m_s[4 * 2] = { 0 };
			for (; k <= cols_a - 8; k += 8) {
				__m256 t_b1 = _mm256_loadu_ps(&ptr_lines_b[0][k]);
				__m256 t_b2 = _mm256_loadu_ps(&ptr_lines_b[1][k]);

				m_s[0] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_lines_a[0][k]), t_b1, m_s[0]);
				m_s[1] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_lines_a[1][k]), t_b1, m_s[1]);
				m_s[2] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_lines_a[2][k]), t_b1, m_s[2]);
				m_s[3] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_lines_a[3][k]), t_b1, m_s[3]);

				m_s[4] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_lines_a[0][k]), t_b2, m_s[4]);
				m_s[5] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_lines_a[1][k]), t_b2, m_s[5]);
				m_s[6] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_lines_a[2][k]), t_b2, m_s[6]);
				m_s[7] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_lines_a[3][k]), t_b2, m_s[7]);
			}
			sum[0] = fiv_mm256_hsum_ps(m_s[0]);
			sum[1] = fiv_mm256_hsum_ps(m_s[1]);
			sum[2] = fiv_mm256_hsum_ps(m_s[2]);
			sum[3] = fiv_mm256_hsum_ps(m_s[3]);
			sum[4] = fiv_mm256_hsum_ps(m_s[4]);
			sum[5] = fiv_mm256_hsum_ps(m_s[5]);
			sum[6] = fiv_mm256_hsum_ps(m_s[6]);
			sum[7] = fiv_mm256_hsum_ps(m_s[7]);
#elif defined(FIV_USE_ARM_NEON)
			/* 8-way unroll: 8 outputs x 2 k-halves grouped as float32x4x2_t */
			float32x4x2_t m_s[8];
			for (int r = 0; r < 8; r++) {
				m_s[r].val[0] = vdupq_n_f32(0);
				m_s[r].val[1] = vdupq_n_f32(0);
			}
			for (; k <= cols_a - 8; k += 8) {
				float32x4x2_t t_b1 = { { vld1q_f32(&ptr_lines_b[0][k]),
				                         vld1q_f32(&ptr_lines_b[0][k + 4]) } };
				float32x4x2_t t_b2 = { { vld1q_f32(&ptr_lines_b[1][k]),
				                         vld1q_f32(&ptr_lines_b[1][k + 4]) } };
				float32x4x2_t a0 = { { vld1q_f32(&ptr_lines_a[0][k]),
				                       vld1q_f32(&ptr_lines_a[0][k + 4]) } };
				float32x4x2_t a1 = { { vld1q_f32(&ptr_lines_a[1][k]),
				                       vld1q_f32(&ptr_lines_a[1][k + 4]) } };
				float32x4x2_t a2 = { { vld1q_f32(&ptr_lines_a[2][k]),
				                       vld1q_f32(&ptr_lines_a[2][k + 4]) } };
				float32x4x2_t a3 = { { vld1q_f32(&ptr_lines_a[3][k]),
				                       vld1q_f32(&ptr_lines_a[3][k + 4]) } };

				m_s[0].val[0] = vmlaq_f32(m_s[0].val[0], a0.val[0], t_b1.val[0]);
				m_s[0].val[1] = vmlaq_f32(m_s[0].val[1], a0.val[1], t_b1.val[1]);
				m_s[1].val[0] = vmlaq_f32(m_s[1].val[0], a1.val[0], t_b1.val[0]);
				m_s[1].val[1] = vmlaq_f32(m_s[1].val[1], a1.val[1], t_b1.val[1]);
				m_s[2].val[0] = vmlaq_f32(m_s[2].val[0], a2.val[0], t_b1.val[0]);
				m_s[2].val[1] = vmlaq_f32(m_s[2].val[1], a2.val[1], t_b1.val[1]);
				m_s[3].val[0] = vmlaq_f32(m_s[3].val[0], a3.val[0], t_b1.val[0]);
				m_s[3].val[1] = vmlaq_f32(m_s[3].val[1], a3.val[1], t_b1.val[1]);

				m_s[4].val[0] = vmlaq_f32(m_s[4].val[0], a0.val[0], t_b2.val[0]);
				m_s[4].val[1] = vmlaq_f32(m_s[4].val[1], a0.val[1], t_b2.val[1]);
				m_s[5].val[0] = vmlaq_f32(m_s[5].val[0], a1.val[0], t_b2.val[0]);
				m_s[5].val[1] = vmlaq_f32(m_s[5].val[1], a1.val[1], t_b2.val[1]);
				m_s[6].val[0] = vmlaq_f32(m_s[6].val[0], a2.val[0], t_b2.val[0]);
				m_s[6].val[1] = vmlaq_f32(m_s[6].val[1], a2.val[1], t_b2.val[1]);
				m_s[7].val[0] = vmlaq_f32(m_s[7].val[0], a3.val[0], t_b2.val[0]);
				m_s[7].val[1] = vmlaq_f32(m_s[7].val[1], a3.val[1], t_b2.val[1]);
			}
			/* reduce each 4-lane accumulator pair to a full output sum, staying
			   in vector registers: two vector stores fill sum[0..7] with no
			   scalar extraction */
			vst1q_f32(sum,
				fiv_neon_hsum_4x4(vaddq_f32(m_s[0].val[0], m_s[0].val[1]),
				                  vaddq_f32(m_s[1].val[0], m_s[1].val[1]),
				                  vaddq_f32(m_s[2].val[0], m_s[2].val[1]),
				                  vaddq_f32(m_s[3].val[0], m_s[3].val[1])));
			vst1q_f32(sum + 4,
				fiv_neon_hsum_4x4(vaddq_f32(m_s[4].val[0], m_s[4].val[1]),
				                  vaddq_f32(m_s[5].val[0], m_s[5].val[1]),
				                  vaddq_f32(m_s[6].val[0], m_s[6].val[1]),
				                  vaddq_f32(m_s[7].val[0], m_s[7].val[1])));
#endif
			for (; k < cols_a; k++){
				sum[0] += ptr_lines_a[0][k] * ptr_lines_b[0][k];
				sum[1] += ptr_lines_a[1][k] * ptr_lines_b[0][k];
				sum[2] += ptr_lines_a[2][k] * ptr_lines_b[0][k];
				sum[3] += ptr_lines_a[3][k] * ptr_lines_b[0][k];
				sum[4] += ptr_lines_a[0][k] * ptr_lines_b[1][k];
				sum[5] += ptr_lines_a[1][k] * ptr_lines_b[1][k];
				sum[6] += ptr_lines_a[2][k] * ptr_lines_b[1][k];
				sum[7] += ptr_lines_a[3][k] * ptr_lines_b[1][k];
			}

			if (alpha != 1.f){
				sum[0] *= alpha;
				sum[1] *= alpha;
				sum[2] *= alpha;
				sum[3] *= alpha;
				sum[4] *= alpha;
				sum[5] *= alpha;
				sum[6] *= alpha;
				sum[7] *= alpha;
			}
			if (beta == 0.f){
				ptr_lines_c[0][j] = sum[0];
				ptr_lines_c[1][j] = sum[1];
				ptr_lines_c[2][j] = sum[2];
				ptr_lines_c[3][j] = sum[3];
				ptr_lines_c[0][j + 1] = sum[4];
				ptr_lines_c[1][j + 1] = sum[5];
				ptr_lines_c[2][j + 1] = sum[6];
				ptr_lines_c[3][j + 1] = sum[7];
			}  else {
				ptr_lines_c[0][j] = beta * ptr_lines_c[0][j] + sum[0];
				ptr_lines_c[1][j] = beta * ptr_lines_c[1][j] + sum[1];
				ptr_lines_c[2][j] = beta * ptr_lines_c[2][j] + sum[2];
				ptr_lines_c[3][j] = beta * ptr_lines_c[3][j] + sum[3];
				ptr_lines_c[0][j + 1] = beta * ptr_lines_c[0][j + 1] + sum[4];
				ptr_lines_c[1][j + 1] = beta * ptr_lines_c[1][j + 1] + sum[5];
				ptr_lines_c[2][j + 1] = beta * ptr_lines_c[2][j + 1] + sum[6];
				ptr_lines_c[3][j + 1] = beta * ptr_lines_c[3][j + 1] + sum[7];


			}


		}

		for (; j < rows_b; j++){

			ivf32* ptr_line_b = &data_b[j * stride_b];
			ivf32 sum[4] = { 0 };
			int k = 0;
#if defined(FIV_USE_AVX2)
			__m256 m_s[4] = { 0 };
			for (; k <= cols_a - 8; k += 8){
				__m256 t_b = _mm256_loadu_ps(&ptr_line_b[k]);
				m_s[0] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_lines_a[0][k]), t_b, m_s[0]);
				m_s[1] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_lines_a[1][k]), t_b, m_s[1]);
				m_s[2] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_lines_a[2][k]), t_b, m_s[2]);
				m_s[3] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_lines_a[3][k]), t_b, m_s[3]);
			}
			sum[0] = fiv_mm256_hsum_ps(m_s[0]);
			sum[1] = fiv_mm256_hsum_ps(m_s[1]);
			sum[2] = fiv_mm256_hsum_ps(m_s[2]);
			sum[3] = fiv_mm256_hsum_ps(m_s[3]);
#elif defined(FIV_USE_ARM_NEON)
			/* 8-way unroll: 4 rows x 2 k-halves grouped as float32x4x2_t */
			float32x4x2_t m_s[4];
			for (int r = 0; r < 4; r++) {
				m_s[r].val[0] = vdupq_n_f32(0);
				m_s[r].val[1] = vdupq_n_f32(0);
			}
			for (; k <= cols_a - 8; k += 8){
				float32x4x2_t t_b = { { vld1q_f32(&ptr_line_b[k]),
				                        vld1q_f32(&ptr_line_b[k + 4]) } };
				float32x4x2_t a0 = { { vld1q_f32(&ptr_lines_a[0][k]),
				                       vld1q_f32(&ptr_lines_a[0][k + 4]) } };
				float32x4x2_t a1 = { { vld1q_f32(&ptr_lines_a[1][k]),
				                       vld1q_f32(&ptr_lines_a[1][k + 4]) } };
				float32x4x2_t a2 = { { vld1q_f32(&ptr_lines_a[2][k]),
				                       vld1q_f32(&ptr_lines_a[2][k + 4]) } };
				float32x4x2_t a3 = { { vld1q_f32(&ptr_lines_a[3][k]),
				                       vld1q_f32(&ptr_lines_a[3][k + 4]) } };

				m_s[0].val[0] = vmlaq_f32(m_s[0].val[0], a0.val[0], t_b.val[0]);
				m_s[0].val[1] = vmlaq_f32(m_s[0].val[1], a0.val[1], t_b.val[1]);
				m_s[1].val[0] = vmlaq_f32(m_s[1].val[0], a1.val[0], t_b.val[0]);
				m_s[1].val[1] = vmlaq_f32(m_s[1].val[1], a1.val[1], t_b.val[1]);
				m_s[2].val[0] = vmlaq_f32(m_s[2].val[0], a2.val[0], t_b.val[0]);
				m_s[2].val[1] = vmlaq_f32(m_s[2].val[1], a2.val[1], t_b.val[1]);
				m_s[3].val[0] = vmlaq_f32(m_s[3].val[0], a3.val[0], t_b.val[0]);
				m_s[3].val[1] = vmlaq_f32(m_s[3].val[1], a3.val[1], t_b.val[1]);
			}
			/* one vector store fills sum[0..3], no scalar extraction */
			vst1q_f32(sum,
				fiv_neon_hsum_4x4(vaddq_f32(m_s[0].val[0], m_s[0].val[1]),
				                  vaddq_f32(m_s[1].val[0], m_s[1].val[1]),
				                  vaddq_f32(m_s[2].val[0], m_s[2].val[1]),
				                  vaddq_f32(m_s[3].val[0], m_s[3].val[1])));
#endif
			for (; k < cols_a; k++){
				sum[0] += ptr_lines_a[0][k] * ptr_line_b[k];
				sum[1] += ptr_lines_a[1][k] * ptr_line_b[k];
				sum[2] += ptr_lines_a[2][k] * ptr_line_b[k];
				sum[3] += ptr_lines_a[3][k] * ptr_line_b[k];
			}

			if (beta == 0.f){
				ptr_lines_c[0][j] = alpha * sum[0];
				ptr_lines_c[1][j] = alpha * sum[1];
				ptr_lines_c[2][j] = alpha * sum[2];
				ptr_lines_c[3][j] = alpha * sum[3];
			}  else {
				ptr_lines_c[0][j] = beta * ptr_lines_c[0][j] + alpha * sum[0];
				ptr_lines_c[1][j] = beta * ptr_lines_c[1][j] + alpha * sum[1];
				ptr_lines_c[2][j] = beta * ptr_lines_c[2][j] + alpha * sum[2];
				ptr_lines_c[3][j] = beta * ptr_lines_c[3][j] + alpha * sum[3];
			}
		}
	}
	for (; i < rows_a; i++) {
		ivf32* ptr_line_a = &data_a[i * stride_a];
		ivf32* ptr_line_c = &data_c[i * stride_c];
		for (int j = 0; j < rows_b; j++){
			ivf32* ptr_line_b = &data_b[j * stride_b];
			ivf32 sum = 0;
			for (int k = 0; k < cols_a; k++) {
				sum += ptr_line_a[k] * ptr_line_b[k];
			}
			ptr_line_c[j] = beta * ptr_line_c[j] + alpha * sum;
		}
	}
}

/* C = A^T * B, A: rows_a x cols_a, B: rows_a x cols_b, result cols_a x cols_b */
static void fiv_small_matrix_t_mul_matrix_real32(
	ivf32* data_a, int rows_a, int cols_a, int stride_a,
	ivf32* data_b, int cols_b, int stride_b,
	ivf32* data_c, int stride_c, ivf32 alpha, ivf32 beta)
{
	int i = 0;
	for (; i <= cols_a - 4; i += 4){
		ivf32* ptr_c_lines[4];
		ptr_c_lines[0] = &data_c[i * stride_c];
		ptr_c_lines[1] = ptr_c_lines[0] + stride_c;
		ptr_c_lines[2] = ptr_c_lines[1] + stride_c;
		ptr_c_lines[3] = ptr_c_lines[2] + stride_c;

		if (beta == 0.f){
			memset(ptr_c_lines[0], 0, sizeof(ivf32) * cols_b);
			memset(ptr_c_lines[1], 0, sizeof(ivf32) * cols_b);
			memset(ptr_c_lines[2], 0, sizeof(ivf32) * cols_b);
			memset(ptr_c_lines[3], 0, sizeof(ivf32) * cols_b);
		}	else if (beta != 1.f) {
			for (int l = 0; l < cols_b; l++){
				ptr_c_lines[0][l] *= beta;
				ptr_c_lines[1][l] *= beta;
				ptr_c_lines[2][l] *= beta;
				ptr_c_lines[3][l] *= beta;
			}
		}
		int k = 0;
		for (; k <= rows_a - 2; k += 2){
			ivf32* ptr_data_a_tmp = &data_a[k * stride_a + i];
			ivf32 t_a[8] = {ptr_data_a_tmp[0],ptr_data_a_tmp[1], ptr_data_a_tmp[2], ptr_data_a_tmp[3],
			ptr_data_a_tmp[0 + stride_a], ptr_data_a_tmp[1 + stride_a],
			ptr_data_a_tmp[2 + stride_a], ptr_data_a_tmp[3 + stride_a]};
			if (alpha != 1.f) {
				t_a[0] *= alpha; t_a[1] *= alpha;
				t_a[2] *= alpha; t_a[3] *= alpha;
				t_a[4] *= alpha; t_a[5] *= alpha;
				t_a[6] *= alpha; t_a[7] *= alpha;
			}
			ivf32* ptr_b_line1 = &data_b[k *stride_b];
			ivf32* ptr_b_line2 = &data_b[k *stride_b + stride_b];
			int j = 0;
#if defined(FIV_USE_AVX2)
			__m256 m_t_a1 = _mm256_broadcast_ss(&t_a[0]);
			__m256 m_t_a2 = _mm256_broadcast_ss(&t_a[1]);
			__m256 m_t_a3 = _mm256_broadcast_ss(&t_a[2]);
			__m256 m_t_a4 = _mm256_broadcast_ss(&t_a[3]);
			__m256 m_t_a5 = _mm256_broadcast_ss(&t_a[4]);
			__m256 m_t_a6 = _mm256_broadcast_ss(&t_a[5]);
			__m256 m_t_a7 = _mm256_broadcast_ss(&t_a[6]);
			__m256 m_t_a8 = _mm256_broadcast_ss(&t_a[7]);

			for (; j <= cols_b - 8; j += 8){
				__m256 m_t_c0 = _mm256_loadu_ps(&ptr_c_lines[0][j]);
				__m256 m_t_c1 = _mm256_loadu_ps(&ptr_c_lines[1][j]);
				__m256 m_t_c2 = _mm256_loadu_ps(&ptr_c_lines[2][j]);
				__m256 m_t_c3 = _mm256_loadu_ps(&ptr_c_lines[3][j]);

				__m256 m_t_b1 = _mm256_loadu_ps(&ptr_b_line1[j]);
				__m256 m_t_b2 = _mm256_loadu_ps(&ptr_b_line2[j]);

				m_t_c0 = _mm256_fmadd_ps(m_t_a1, m_t_b1, m_t_c0);
				m_t_c1 = _mm256_fmadd_ps(m_t_a2, m_t_b1, m_t_c1);
				m_t_c2 = _mm256_fmadd_ps(m_t_a3, m_t_b1, m_t_c2);
				m_t_c3 = _mm256_fmadd_ps(m_t_a4, m_t_b1, m_t_c3);

				m_t_c0 = _mm256_fmadd_ps(m_t_a5, m_t_b2, m_t_c0);
				m_t_c1 = _mm256_fmadd_ps(m_t_a6, m_t_b2, m_t_c1);
				m_t_c2 = _mm256_fmadd_ps(m_t_a7, m_t_b2, m_t_c2);
				m_t_c3 = _mm256_fmadd_ps(m_t_a8, m_t_b2, m_t_c3);

				_mm256_storeu_ps(&ptr_c_lines[0][j], m_t_c0);
				_mm256_storeu_ps(&ptr_c_lines[1][j], m_t_c1);
				_mm256_storeu_ps(&ptr_c_lines[2][j], m_t_c2);
				_mm256_storeu_ps(&ptr_c_lines[3][j], m_t_c3);

			}
#elif defined(FIV_USE_ARM_NEON)
			for (; j <= cols_b - 8; j += 8){
				float32x4x2_t b1 = { { vld1q_f32(&ptr_b_line1[j]),     vld1q_f32(&ptr_b_line1[j + 4]) } };
				float32x4x2_t b2 = { { vld1q_f32(&ptr_b_line2[j]),     vld1q_f32(&ptr_b_line2[j + 4]) } };
				float32x4x2_t c0 = { { vld1q_f32(&ptr_c_lines[0][j]), vld1q_f32(&ptr_c_lines[0][j + 4]) } };
				float32x4x2_t c1 = { { vld1q_f32(&ptr_c_lines[1][j]), vld1q_f32(&ptr_c_lines[1][j + 4]) } };
				float32x4x2_t c2 = { { vld1q_f32(&ptr_c_lines[2][j]), vld1q_f32(&ptr_c_lines[2][j + 4]) } };
				float32x4x2_t c3 = { { vld1q_f32(&ptr_c_lines[3][j]), vld1q_f32(&ptr_c_lines[3][j + 4]) } };

				c0.val[0] = vmlaq_n_f32(c0.val[0], b1.val[0], t_a[0]);
				c0.val[1] = vmlaq_n_f32(c0.val[1], b1.val[1], t_a[0]);
				c1.val[0] = vmlaq_n_f32(c1.val[0], b1.val[0], t_a[1]);
				c1.val[1] = vmlaq_n_f32(c1.val[1], b1.val[1], t_a[1]);
				c2.val[0] = vmlaq_n_f32(c2.val[0], b1.val[0], t_a[2]);
				c2.val[1] = vmlaq_n_f32(c2.val[1], b1.val[1], t_a[2]);
				c3.val[0] = vmlaq_n_f32(c3.val[0], b1.val[0], t_a[3]);
				c3.val[1] = vmlaq_n_f32(c3.val[1], b1.val[1], t_a[3]);

				c0.val[0] = vmlaq_n_f32(c0.val[0], b2.val[0], t_a[4]);
				c0.val[1] = vmlaq_n_f32(c0.val[1], b2.val[1], t_a[4]);
				c1.val[0] = vmlaq_n_f32(c1.val[0], b2.val[0], t_a[5]);
				c1.val[1] = vmlaq_n_f32(c1.val[1], b2.val[1], t_a[5]);
				c2.val[0] = vmlaq_n_f32(c2.val[0], b2.val[0], t_a[6]);
				c2.val[1] = vmlaq_n_f32(c2.val[1], b2.val[1], t_a[6]);
				c3.val[0] = vmlaq_n_f32(c3.val[0], b2.val[0], t_a[7]);
				c3.val[1] = vmlaq_n_f32(c3.val[1], b2.val[1], t_a[7]);

				vst1q_f32(&ptr_c_lines[0][j],     c0.val[0]);
				vst1q_f32(&ptr_c_lines[0][j + 4], c0.val[1]);
				vst1q_f32(&ptr_c_lines[1][j],     c1.val[0]);
				vst1q_f32(&ptr_c_lines[1][j + 4], c1.val[1]);
				vst1q_f32(&ptr_c_lines[2][j],     c2.val[0]);
				vst1q_f32(&ptr_c_lines[2][j + 4], c2.val[1]);
				vst1q_f32(&ptr_c_lines[3][j],     c3.val[0]);
				vst1q_f32(&ptr_c_lines[3][j + 4], c3.val[1]);

			}
#endif
			for (; j < cols_b; j++){
				ptr_c_lines[0][j] += t_a[0] * ptr_b_line1[j];
				ptr_c_lines[0][j] += t_a[4] * ptr_b_line2[j];

				ptr_c_lines[1][j] += t_a[1] * ptr_b_line1[j];
				ptr_c_lines[1][j] += t_a[5] * ptr_b_line2[j];

				ptr_c_lines[2][j] += t_a[2] * ptr_b_line1[j];
				ptr_c_lines[2][j] += t_a[6] * ptr_b_line2[j];

				ptr_c_lines[3][j] += t_a[3] * ptr_b_line1[j];
				ptr_c_lines[3][j] += t_a[7] * ptr_b_line2[j];
			}
		}

		for (; k < rows_a; k++){
			ivf32 t_a1 = data_a[k * stride_a + i + 0];
			ivf32 t_a2 = data_a[k * stride_a + i + 1];
			ivf32 t_a3 = data_a[k * stride_a + i + 2];
			ivf32 t_a4 = data_a[k * stride_a + i + 3];
			if (alpha != 1.f){
				t_a1 *= alpha; t_a2 *= alpha;
				t_a3 *= alpha; t_a4 *= alpha;
			}
			ivf32* ptr_b_line = &data_b[k * stride_b];
			int j = 0;
#if defined(FIV_USE_AVX2)
			__m256 m_t_a1 = _mm256_broadcast_ss(&t_a1);
			__m256 m_t_a2 = _mm256_broadcast_ss(&t_a2);
			__m256 m_t_a3 = _mm256_broadcast_ss(&t_a3);
			__m256 m_t_a4 = _mm256_broadcast_ss(&t_a4);

			for (; j <= cols_b - 8; j += 8){
				__m256 m_t_c0 = _mm256_loadu_ps(&ptr_c_lines[0][j]);
				__m256 m_t_c1 = _mm256_loadu_ps(&ptr_c_lines[1][j]);
				__m256 m_t_c2 = _mm256_loadu_ps(&ptr_c_lines[2][j]);
				__m256 m_t_c3 = _mm256_loadu_ps(&ptr_c_lines[3][j]);
				__m256 m_t_b = _mm256_loadu_ps(&ptr_b_line[j]);

				m_t_c0 = _mm256_fmadd_ps(m_t_a1, m_t_b, m_t_c0);
				m_t_c1 = _mm256_fmadd_ps(m_t_a2, m_t_b, m_t_c1);
				m_t_c2 = _mm256_fmadd_ps(m_t_a3, m_t_b, m_t_c2);
				m_t_c3 = _mm256_fmadd_ps(m_t_a4, m_t_b, m_t_c3);

				_mm256_storeu_ps(&ptr_c_lines[0][j], m_t_c0);
				_mm256_storeu_ps(&ptr_c_lines[1][j], m_t_c1);
				_mm256_storeu_ps(&ptr_c_lines[2][j], m_t_c2);
				_mm256_storeu_ps(&ptr_c_lines[3][j], m_t_c3);

			}
#elif defined(FIV_USE_ARM_NEON)
			for (; j <= cols_b - 8; j += 8){
				float32x4x2_t b  = { { vld1q_f32(&ptr_b_line[j]),     vld1q_f32(&ptr_b_line[j + 4]) } };
				float32x4x2_t c0 = { { vld1q_f32(&ptr_c_lines[0][j]), vld1q_f32(&ptr_c_lines[0][j + 4]) } };
				float32x4x2_t c1 = { { vld1q_f32(&ptr_c_lines[1][j]), vld1q_f32(&ptr_c_lines[1][j + 4]) } };
				float32x4x2_t c2 = { { vld1q_f32(&ptr_c_lines[2][j]), vld1q_f32(&ptr_c_lines[2][j + 4]) } };
				float32x4x2_t c3 = { { vld1q_f32(&ptr_c_lines[3][j]), vld1q_f32(&ptr_c_lines[3][j + 4]) } };

				c0.val[0] = vmlaq_n_f32(c0.val[0], b.val[0], t_a1);
				c0.val[1] = vmlaq_n_f32(c0.val[1], b.val[1], t_a1);
				c1.val[0] = vmlaq_n_f32(c1.val[0], b.val[0], t_a2);
				c1.val[1] = vmlaq_n_f32(c1.val[1], b.val[1], t_a2);
				c2.val[0] = vmlaq_n_f32(c2.val[0], b.val[0], t_a3);
				c2.val[1] = vmlaq_n_f32(c2.val[1], b.val[1], t_a3);
				c3.val[0] = vmlaq_n_f32(c3.val[0], b.val[0], t_a4);
				c3.val[1] = vmlaq_n_f32(c3.val[1], b.val[1], t_a4);

				vst1q_f32(&ptr_c_lines[0][j],     c0.val[0]);
				vst1q_f32(&ptr_c_lines[0][j + 4], c0.val[1]);
				vst1q_f32(&ptr_c_lines[1][j],     c1.val[0]);
				vst1q_f32(&ptr_c_lines[1][j + 4], c1.val[1]);
				vst1q_f32(&ptr_c_lines[2][j],     c2.val[0]);
				vst1q_f32(&ptr_c_lines[2][j + 4], c2.val[1]);
				vst1q_f32(&ptr_c_lines[3][j],     c3.val[0]);
				vst1q_f32(&ptr_c_lines[3][j + 4], c3.val[1]);

			}
#endif
			for (; j < cols_b; j++){
				ptr_c_lines[0][j] += t_a1 * ptr_b_line[j];
				ptr_c_lines[1][j] += t_a2 * ptr_b_line[j];
				ptr_c_lines[2][j] += t_a3 * ptr_b_line[j];
				ptr_c_lines[3][j] += t_a4 * ptr_b_line[j];
			}
		}
	}

	for (; i < cols_a; i++){
		ivf32* ptr_c_line = &data_c[i * stride_c];
		if (beta == 0.f){
			memset(ptr_c_line, 0, sizeof(ivf32) * cols_b);
		}	else if(beta != 1.f){
			for (int k = 0; k < cols_b; k++){
				ptr_c_line[k] *= beta;
			}
		}
		for (int k = 0; k < rows_a; k++){
			ivf32 t_a = data_a[k * stride_a + i] * alpha;
			ivf32* ptr_b_line = &data_b[k * stride_b];
			for (int j = 0; j < cols_b; j++) {
				ptr_c_line[j] += t_a * ptr_b_line[j];
			}
		}
	}
}

/* C = A^T * B^T, A: rows_a x cols_a, B: rows_b x cols_a, result cols_a x rows_b */
static void fiv_small_matrix_t_mul_matrix_t_real32(
	ivf32* data_a, int rows_a, int cols_a, int stride_a,
	ivf32* data_b, int rows_b, int stride_b,
	ivf32* data_c, int stride_c, ivf32 alpha, ivf32 beta)
{
	int i = 0;
	ivf32* ptr_mem_a = (ivf32*)fiv_malloc(sizeof(ivf32) * 8 * rows_a);
	if (ptr_mem_a == NULL) {
		printf("MEM ERROR\n");
		return;
	}
	for (; i <= cols_a - 8; i += 8){
		ivf32* ptr_c_lines[8];
		ptr_c_lines[0] = &data_c[i * stride_c];
		ptr_c_lines[1] = ptr_c_lines[0] + stride_c;
		ptr_c_lines[2] = ptr_c_lines[1] + stride_c;
		ptr_c_lines[3] = ptr_c_lines[2] + stride_c;
		ptr_c_lines[4] = ptr_c_lines[3] + stride_c;
		ptr_c_lines[5] = ptr_c_lines[4] + stride_c;
		ptr_c_lines[6] = ptr_c_lines[5] + stride_c;
		ptr_c_lines[7] = ptr_c_lines[6] + stride_c;
		ivf32* ptr_a_i_col = &data_a[i];
		for (int k = 0; k < rows_a; k++) {
			ptr_mem_a[8 * k + 0] = ptr_a_i_col[k * stride_a + 0];
			ptr_mem_a[8 * k + 1] = ptr_a_i_col[k * stride_a + 1];
			ptr_mem_a[8 * k + 2] = ptr_a_i_col[k * stride_a + 2];
			ptr_mem_a[8 * k + 3] = ptr_a_i_col[k * stride_a + 3];
			ptr_mem_a[8 * k + 4] = ptr_a_i_col[k * stride_a + 4];
			ptr_mem_a[8 * k + 5] = ptr_a_i_col[k * stride_a + 5];
			ptr_mem_a[8 * k + 6] = ptr_a_i_col[k * stride_a + 6];
			ptr_mem_a[8 * k + 7] = ptr_a_i_col[k * stride_a + 7];

		}

		int j = 0;
		for (; j <= rows_b - 2; j += 2) {
			ivf32* ptr_a = ptr_mem_a;
			ivf32* ptr_b[2] = {&data_b[j * stride_b], &data_b[(j + 1) * stride_b]};
			ivf32 FIV_DALIGNED sum[16] = { 0 };
			int k = 0;
#if defined(FIV_USE_AVX2)
			__m256 m_sum[8] = { 0 };
			for (; k <= rows_a - 4; k += 4) {
				m_sum[0] = _mm256_fmadd_ps(_mm256_loadu_ps(ptr_a), _mm256_broadcast_ss(&ptr_b[0][k]), m_sum[0]);
				m_sum[4] = _mm256_fmadd_ps(_mm256_loadu_ps(ptr_a), _mm256_broadcast_ss(&ptr_b[1][k]), m_sum[4]);

				m_sum[1] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_a[8]), _mm256_broadcast_ss(&ptr_b[0][k + 1]), m_sum[1]);
				m_sum[5] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_a[8]), _mm256_broadcast_ss(&ptr_b[1][k + 1]), m_sum[5]);

				m_sum[2] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_a[16]), _mm256_broadcast_ss(&ptr_b[0][k + 2]), m_sum[2]);
				m_sum[6] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_a[16]), _mm256_broadcast_ss(&ptr_b[1][k + 2]), m_sum[6]);

				m_sum[3] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_a[24]), _mm256_broadcast_ss(&ptr_b[0][k + 3]), m_sum[3]);
				m_sum[7] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_a[24]), _mm256_broadcast_ss(&ptr_b[1][k + 3]), m_sum[7]);
				ptr_a += 32;
			}
			m_sum[0] = _mm256_add_ps(m_sum[0], m_sum[1]);
			m_sum[2] = _mm256_add_ps(m_sum[2], m_sum[3]);
			m_sum[0] = _mm256_add_ps(m_sum[0], m_sum[2]);

			m_sum[4] = _mm256_add_ps(m_sum[4], m_sum[5]);
			m_sum[6] = _mm256_add_ps(m_sum[6], m_sum[7]);
			m_sum[4] = _mm256_add_ps(m_sum[4], m_sum[6]);

			for (; k < rows_a; k++){
				m_sum[0] = _mm256_fmadd_ps(_mm256_loadu_ps(ptr_a), _mm256_broadcast_ss(&ptr_b[0][k]), m_sum[0]);
				m_sum[4] = _mm256_fmadd_ps(_mm256_loadu_ps(ptr_a), _mm256_broadcast_ss(&ptr_b[1][k]), m_sum[4]);
				ptr_a += 8;
			}
			_mm256_storeu_ps(sum    , m_sum[0]);
			_mm256_storeu_ps(sum + 8, m_sum[4]);
#elif defined(FIV_USE_ARM_NEON)
			float32x4_t s0a = vdupq_n_f32(0), s0b = vdupq_n_f32(0);
			float32x4_t s1a = vdupq_n_f32(0), s1b = vdupq_n_f32(0);
			for (; k <= rows_a - 4; k += 4) {
				float32x4x2_t a0 = { { vld1q_f32(&ptr_mem_a[8 * k]),       vld1q_f32(&ptr_mem_a[8 * k + 4]) } };
				float32x4x2_t a1 = { { vld1q_f32(&ptr_mem_a[8 * (k + 1)]), vld1q_f32(&ptr_mem_a[8 * (k + 1) + 4]) } };
				float32x4x2_t a2 = { { vld1q_f32(&ptr_mem_a[8 * (k + 2)]), vld1q_f32(&ptr_mem_a[8 * (k + 2) + 4]) } };
				float32x4x2_t a3 = { { vld1q_f32(&ptr_mem_a[8 * (k + 3)]), vld1q_f32(&ptr_mem_a[8 * (k + 3) + 4]) } };

				s0a = vmlaq_n_f32(s0a, a0.val[0], ptr_b[0][k]);
				s0b = vmlaq_n_f32(s0b, a0.val[1], ptr_b[0][k]);
				s1a = vmlaq_n_f32(s1a, a0.val[0], ptr_b[1][k]);
				s1b = vmlaq_n_f32(s1b, a0.val[1], ptr_b[1][k]);

				s0a = vmlaq_n_f32(s0a, a1.val[0], ptr_b[0][k + 1]);
				s0b = vmlaq_n_f32(s0b, a1.val[1], ptr_b[0][k + 1]);
				s1a = vmlaq_n_f32(s1a, a1.val[0], ptr_b[1][k + 1]);
				s1b = vmlaq_n_f32(s1b, a1.val[1], ptr_b[1][k + 1]);

				s0a = vmlaq_n_f32(s0a, a2.val[0], ptr_b[0][k + 2]);
				s0b = vmlaq_n_f32(s0b, a2.val[1], ptr_b[0][k + 2]);
				s1a = vmlaq_n_f32(s1a, a2.val[0], ptr_b[1][k + 2]);
				s1b = vmlaq_n_f32(s1b, a2.val[1], ptr_b[1][k + 2]);

				s0a = vmlaq_n_f32(s0a, a3.val[0], ptr_b[0][k + 3]);
				s0b = vmlaq_n_f32(s0b, a3.val[1], ptr_b[0][k + 3]);
				s1a = vmlaq_n_f32(s1a, a3.val[0], ptr_b[1][k + 3]);
				s1b = vmlaq_n_f32(s1b, a3.val[1], ptr_b[1][k + 3]);
			}
			for (; k < rows_a; k++){
				float32x4x2_t a0 = { { vld1q_f32(&ptr_mem_a[8 * k]),       vld1q_f32(&ptr_mem_a[8 * k + 4]) } };
				s0a = vmlaq_n_f32(s0a, a0.val[0], ptr_b[0][k]);
				s0b = vmlaq_n_f32(s0b, a0.val[1], ptr_b[0][k]);
				s1a = vmlaq_n_f32(s1a, a0.val[0], ptr_b[1][k]);
				s1b = vmlaq_n_f32(s1b, a0.val[1], ptr_b[1][k]);
			}
			vst1q_f32(sum, s0a);
			vst1q_f32(sum + 4, s0b);
			vst1q_f32(sum + 8, s1a);
			vst1q_f32(sum + 12, s1b);
#endif
			for (; k < rows_a; k++){
				sum[0] += ptr_a[0] * ptr_b[0][k];
				sum[1] += ptr_a[1] * ptr_b[0][k];
				sum[2] += ptr_a[2] * ptr_b[0][k];
				sum[3] += ptr_a[3] * ptr_b[0][k];
				sum[4] += ptr_a[4] * ptr_b[0][k];
				sum[5] += ptr_a[5] * ptr_b[0][k];
				sum[6] += ptr_a[6] * ptr_b[0][k];
				sum[7] += ptr_a[7] * ptr_b[0][k];

				sum[8]  += ptr_a[0] * ptr_b[1][k];
				sum[9]  += ptr_a[1] * ptr_b[1][k];
				sum[10] += ptr_a[2] * ptr_b[1][k];
				sum[11] += ptr_a[3] * ptr_b[1][k];
				sum[12] += ptr_a[4] * ptr_b[1][k];
				sum[13] += ptr_a[5] * ptr_b[1][k];
				sum[14] += ptr_a[6] * ptr_b[1][k];
				sum[15] += ptr_a[7] * ptr_b[1][k];
				ptr_a += 8;
			}

			sum[0] *= alpha; sum[1] *= alpha;
			sum[2] *= alpha; sum[3] *= alpha;
			sum[4] *= alpha; sum[5] *= alpha;
			sum[6] *= alpha; sum[7] *= alpha;

			sum[8]  *= alpha; sum[9]  *= alpha;
			sum[10] *= alpha; sum[11] *= alpha;
			sum[12] *= alpha; sum[13] *= alpha;
			sum[14] *= alpha; sum[15] *= alpha;

			if (beta == 0.f){
				ptr_c_lines[0][j + 0] = sum[0];
				ptr_c_lines[0][j + 1] = sum[8];
				ptr_c_lines[1][j + 0] = sum[1];
				ptr_c_lines[1][j + 1] = sum[9];
				ptr_c_lines[2][j + 0] = sum[2];
				ptr_c_lines[2][j + 1] = sum[10];
				ptr_c_lines[3][j + 0] = sum[3];
				ptr_c_lines[3][j + 1] = sum[11];
				ptr_c_lines[4][j + 0] = sum[4];
				ptr_c_lines[4][j + 1] = sum[12];
				ptr_c_lines[5][j + 0] = sum[5];
				ptr_c_lines[5][j + 1] = sum[13];
				ptr_c_lines[6][j + 0] = sum[6];
				ptr_c_lines[6][j + 1] = sum[14];
				ptr_c_lines[7][j + 0] = sum[7];
				ptr_c_lines[7][j + 1] = sum[15];
			}	else {
				ptr_c_lines[0][j + 0] = beta * ptr_c_lines[0][j + 0] + sum[0];
				ptr_c_lines[0][j + 1] = beta * ptr_c_lines[0][j + 1] + sum[8];
				ptr_c_lines[1][j + 0] = beta * ptr_c_lines[1][j + 0] + sum[1];
				ptr_c_lines[1][j + 1] = beta * ptr_c_lines[1][j + 1] + sum[9];
				ptr_c_lines[2][j + 0] = beta * ptr_c_lines[2][j + 0] + sum[2];
				ptr_c_lines[2][j + 1] = beta * ptr_c_lines[2][j + 1] + sum[10];
				ptr_c_lines[3][j + 0] = beta * ptr_c_lines[3][j + 0] + sum[3];
				ptr_c_lines[3][j + 1] = beta * ptr_c_lines[3][j + 1] + sum[11];
				ptr_c_lines[4][j + 0] = beta * ptr_c_lines[4][j + 0] + sum[4];
				ptr_c_lines[4][j + 1] = beta * ptr_c_lines[4][j + 1] + sum[12];
				ptr_c_lines[5][j + 0] = beta * ptr_c_lines[5][j + 0] + sum[5];
				ptr_c_lines[5][j + 1] = beta * ptr_c_lines[5][j + 1] + sum[13];
				ptr_c_lines[6][j + 0] = beta * ptr_c_lines[6][j + 0] + sum[6];
				ptr_c_lines[6][j + 1] = beta * ptr_c_lines[6][j + 1] + sum[14];
				ptr_c_lines[7][j + 0] = beta * ptr_c_lines[7][j + 0] + sum[7];
				ptr_c_lines[7][j + 1] = beta * ptr_c_lines[7][j + 1] + sum[15];
			}
		}

		for (; j < rows_b; j++){
			ivf32* ptr_a = ptr_mem_a;
			ivf32* ptr_b = &data_b[j * stride_b];
			ivf32 FIV_DALIGNED sum[8] = { 0 };
			int k = 0;
#if defined(FIV_USE_AVX2)
			__m256 m_sum[4] = { 0 };
			for (; k <= rows_a - 4; k += 4) {
				m_sum[0] = _mm256_fmadd_ps(_mm256_loadu_ps(ptr_a), _mm256_broadcast_ss(&ptr_b[k]), m_sum[0]);
				m_sum[1] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_a[8]), _mm256_broadcast_ss(&ptr_b[k + 1]), m_sum[1]);
				m_sum[2] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_a[16]), _mm256_broadcast_ss(&ptr_b[k + 2]), m_sum[2]);
				m_sum[3] = _mm256_fmadd_ps(_mm256_loadu_ps(&ptr_a[24]), _mm256_broadcast_ss(&ptr_b[k + 3]), m_sum[3]);
				ptr_a += 32;
			}
			m_sum[0] = _mm256_add_ps(m_sum[0], m_sum[1]);
			m_sum[2] = _mm256_add_ps(m_sum[2], m_sum[3]);
			m_sum[0] = _mm256_add_ps(m_sum[0], m_sum[2]);

			for (; k < rows_a; k++){
				m_sum[0] = _mm256_fmadd_ps(_mm256_loadu_ps(ptr_a), _mm256_broadcast_ss(&ptr_b[k]), m_sum[0]);
				ptr_a += 8;
			}
			_mm256_store_ps(sum, m_sum[0]);
#elif defined(FIV_USE_ARM_NEON)
			float32x4_t s0a = vdupq_n_f32(0), s0b = vdupq_n_f32(0);
			for (; k <= rows_a - 4; k += 4) {
				float32x4x2_t a0 = { { vld1q_f32(&ptr_mem_a[8 * k]),       vld1q_f32(&ptr_mem_a[8 * k + 4]) } };
				float32x4x2_t a1 = { { vld1q_f32(&ptr_mem_a[8 * (k + 1)]), vld1q_f32(&ptr_mem_a[8 * (k + 1) + 4]) } };
				float32x4x2_t a2 = { { vld1q_f32(&ptr_mem_a[8 * (k + 2)]), vld1q_f32(&ptr_mem_a[8 * (k + 2) + 4]) } };
				float32x4x2_t a3 = { { vld1q_f32(&ptr_mem_a[8 * (k + 3)]), vld1q_f32(&ptr_mem_a[8 * (k + 3) + 4]) } };

				s0a = vmlaq_n_f32(s0a, a0.val[0], ptr_b[k]);
				s0b = vmlaq_n_f32(s0b, a0.val[1], ptr_b[k]);
				s0a = vmlaq_n_f32(s0a, a1.val[0], ptr_b[k + 1]);
				s0b = vmlaq_n_f32(s0b, a1.val[1], ptr_b[k + 1]);
				s0a = vmlaq_n_f32(s0a, a2.val[0], ptr_b[k + 2]);
				s0b = vmlaq_n_f32(s0b, a2.val[1], ptr_b[k + 2]);
				s0a = vmlaq_n_f32(s0a, a3.val[0], ptr_b[k + 3]);
				s0b = vmlaq_n_f32(s0b, a3.val[1], ptr_b[k + 3]);
			}
			for (; k < rows_a; k++){
				float32x4x2_t a0 = { { vld1q_f32(&ptr_mem_a[8 * k]),       vld1q_f32(&ptr_mem_a[8 * k + 4]) } };
				s0a = vmlaq_n_f32(s0a, a0.val[0], ptr_b[k]);
				s0b = vmlaq_n_f32(s0b, a0.val[1], ptr_b[k]);
			}
			vst1q_f32(sum, s0a);
			vst1q_f32(sum + 4, s0b);
#endif
			for (; k < rows_a; k++){
				sum[0] += ptr_a[0] * ptr_b[k];
				sum[1] += ptr_a[1] * ptr_b[k];
				sum[2] += ptr_a[2] * ptr_b[k];
				sum[3] += ptr_a[3] * ptr_b[k];
				sum[4] += ptr_a[4] * ptr_b[k];
				sum[5] += ptr_a[5] * ptr_b[k];
				sum[6] += ptr_a[6] * ptr_b[k];
				sum[7] += ptr_a[7] * ptr_b[k];
				ptr_a += 8;
			}

			sum[0] *= alpha; sum[1] *= alpha;
			sum[2] *= alpha; sum[3] *= alpha;
			sum[4] *= alpha; sum[5] *= alpha;
			sum[6] *= alpha; sum[7] *= alpha;

			if (beta == 0.f){
				ptr_c_lines[0][j] = sum[0];
				ptr_c_lines[1][j] = sum[1];
				ptr_c_lines[2][j] = sum[2];
				ptr_c_lines[3][j] = sum[3];
				ptr_c_lines[4][j] = sum[4];
				ptr_c_lines[5][j] = sum[5];
				ptr_c_lines[6][j] = sum[6];
				ptr_c_lines[7][j] = sum[7];
			}	else {
				ptr_c_lines[0][j] = beta * ptr_c_lines[0][j] + sum[0];
				ptr_c_lines[1][j] = beta * ptr_c_lines[1][j] + sum[1];
				ptr_c_lines[2][j] = beta * ptr_c_lines[2][j] + sum[2];
				ptr_c_lines[3][j] = beta * ptr_c_lines[3][j] + sum[3];
				ptr_c_lines[4][j] = beta * ptr_c_lines[4][j] + sum[4];
				ptr_c_lines[5][j] = beta * ptr_c_lines[5][j] + sum[5];
				ptr_c_lines[6][j] = beta * ptr_c_lines[6][j] + sum[6];
				ptr_c_lines[7][j] = beta * ptr_c_lines[7][j] + sum[7];

			}
		}
	}
	for (; i < cols_a; i++){
		ivf32* ptr_c_line = &data_c[i * stride_c];
		ivf32* ptr_a_i_col = &data_a[i];
		for (int k = 0; k < rows_a; k++) {
			ptr_mem_a[k] = ptr_a_i_col[k * stride_a];
		}
		for (int j = 0; j < rows_b; j++) {
			ivf32* ptr_a = ptr_mem_a;
			ivf32* ptr_b = &data_b[j * stride_b];
			ivf32 sum = 0;
			for (int k = 0; k < rows_a; k++) {
				sum += ptr_a[k] * ptr_b[k];
			}
			ptr_c_line[j] = beta * ptr_c_line[j] + alpha * sum;
		}
	}
	fiv_free(ptr_mem_a);
}

/* ============================================================================
   Blocked matrix multiplication. Copied verbatim from the verified
   fiv_matrix_mul_blocked_fl.c (real32 only). SIMD feature macros mapped to
   FIV_USE_AVX2 / FIV_USE_ARM_NEON; the 4x24 kernel below is kept for
   completeness but is never selected (the driver always picks 8x8), so it is
   guarded to compile only where its AVX2 body is valid.
   ========================================================================== */

typedef void(*ptr_func_mat_mul_mxkxn_kernel)
(int kc, ivf32 alpha, ivf32* a, ivf32* b, ivf32 beta, ivf32* c, int inc_row_c, int inc_col_c);

/* Blocking is derived from the L3 cache size: the two packed panels
   (A: FIV_M_BLOCK*K, B: FIV_N_BLOCK*K, 4 bytes each) together should fit
   about half of L3. Default 8MB; override with -DFIV_L3_CACHE_BYTES=. */
#ifndef FIV_L3_CACHE_BYTES
#define FIV_L3_CACHE_BYTES (8 * 1024 * 1024)
#endif
#if _DEBUG
#define FIV_M_BLOCK 32
#define FIV_K_BLOCK 32
#define FIV_N_BLOCK 32
#else
#define FIV_M_BLOCK 512
#define FIV_N_BLOCK 480
#define FIV_BLOCK_K_CALC() \
    ((((FIV_L3_CACHE_BYTES / 2) / (4 * (FIV_M_BLOCK + FIV_N_BLOCK))) / 128) * 128)
#define FIV_K_BLOCK (FIV_BLOCK_K_CALC() < 64 ? 64 : FIV_BLOCK_K_CALC())
#endif

static void copy_mrxk_blocked(
    int k, int kernel_m, ivf32* a, int inc_row_a, int inc_col_a, ivf32* buffer)
{
    int i, j;
    for (j = 0; j < k; j++){
        for (i = 0; i < kernel_m; i++){
            buffer[i] = a[i * inc_row_a];
        }
        buffer += kernel_m;
        a += inc_col_a;
    }
}

static void copy_a_blocked(int mc, int kc, int kernel_m, ivf32* a, int inc_row_a, int inc_col_a, ivf32* buffer)
{
    int mp = mc / kernel_m;
    int mr = mc % kernel_m;
    int i, j;
    for (i = 0; i < mp; i++) {
        copy_mrxk_blocked(kc, kernel_m, a, inc_row_a, inc_col_a, buffer);
        buffer += kc * kernel_m;
        a += kernel_m * inc_row_a;
    } 
    if (mr > 0) {
        for (j = 0; j < kc; j++){
            for (i = 0; i < mr; i++) {
                buffer[i] = a[i * inc_row_a];
            }
            for (i = mr; i < kernel_m; i++){
                buffer[i] = 0.f;
            }
            buffer += kernel_m;
            a += inc_col_a;
        }
    }
}

static void copy_nrxk_blocked(int k, int kernel_n, ivf32* b, int inc_row_b, int inc_col_b, ivf32* buffer)
{
    int i, j;
    for (i = 0; i < k; i++){
        for (j = 0; j < kernel_n; j++){
            buffer[j] = b[j * inc_col_b];
        }
        buffer += kernel_n;
        b += inc_row_b;
    }
}

static void copy_b_blocked(int kc, int nc, int kernel_n, ivf32* b, int inc_row_b, int inc_col_b, ivf32* buffer)
{
    int np = nc / kernel_n;
    int nr = nc % kernel_n;
    int i, j;
    for (j = 0; j < np; j++){
        copy_nrxk_blocked(kc, kernel_n, b, inc_row_b, inc_col_b, buffer);
        buffer += kc       * kernel_n;
        b      += kernel_n * inc_col_b;
    }
    if (nr > 0) {
        for (i = 0; i < kc; i++){
            for (j = 0; j < nr; j++) {
                buffer[j] = b[j * inc_col_b];
            }
            for (j = nr; j < kernel_n; j++) {
                buffer[j] = 0.f;
            }
            buffer += kernel_n;
            b      += inc_row_b;

        }
    }

}

#define FIV_KERNEL_M_8  (8)
#define FIV_KERNEL_N_8  (8)

static void mat_mul_8xkx8_kernel(
            int kc, ivf32 alpha, ivf32* a, ivf32* b, ivf32 beta,
            ivf32* c, int inc_row_c, int inc_col_c)
{
    ivf32 FIV_DALIGNED ab[FIV_KERNEL_M_8 * FIV_KERNEL_N_8];
    int i, j, l;
#if defined(FIV_USE_AVX2)
    __m256 ab0, ab1, ab2, ab3;
    __m256 ab4, ab5, ab6, ab7;
    ab0 = _mm256_setzero_ps(); ab1 = _mm256_setzero_ps();
    ab2 = _mm256_setzero_ps(); ab3 = _mm256_setzero_ps();
    ab4 = _mm256_setzero_ps(); ab5 = _mm256_setzero_ps();
    ab6 = _mm256_setzero_ps(); ab7 = _mm256_setzero_ps();
    
    __m256 b0 = _mm256_load_ps(b);
    b += 8;
    for (l = 0; l < kc; l++){
        __m256 a0 = _mm256_broadcast_ss(a);
        __m256 a1 = _mm256_broadcast_ss(a + 1);
        __m256 a2 = _mm256_broadcast_ss(a + 2);
        __m256 a3 = _mm256_broadcast_ss(a + 3);

        ab0 = _mm256_fmadd_ps(b0, a0, ab0);
        ab1 = _mm256_fmadd_ps(b0, a1, ab1);
        ab2 = _mm256_fmadd_ps(b0, a2, ab2);
        ab3 = _mm256_fmadd_ps(b0, a3, ab3);

        __m256 a4 = _mm256_broadcast_ss(a + 4);
        __m256 a5 = _mm256_broadcast_ss(a + 5);

         a0 = _mm256_broadcast_ss(a + 6);
         a1 = _mm256_broadcast_ss(a + 7);
        a += 8;

        ab4 = _mm256_fmadd_ps(b0, a4, ab4);
        ab5 = _mm256_fmadd_ps(b0, a5, ab5);
        ab6 = _mm256_fmadd_ps(b0, a0, ab6);
        ab7 = _mm256_fmadd_ps(b0, a1, ab7);

        b0 = _mm256_load_ps(b);
        b += 8;

    }

    _mm256_store_ps(ab + 0,  ab0);
    _mm256_store_ps(ab + 8,  ab1);
    _mm256_store_ps(ab + 16, ab2);
    _mm256_store_ps(ab + 24, ab3);
    _mm256_store_ps(ab + 32, ab4);
    _mm256_store_ps(ab + 40, ab5);
    _mm256_store_ps(ab + 48, ab6);
    _mm256_store_ps(ab + 56, ab7);

#elif defined(FIV_USE_ARM_NEON)
    float32x4_t ab0_0123 = vdupq_n_f32(0);
    float32x4_t ab0_4567 = vdupq_n_f32(0);

    float32x4_t ab1_0123 = vdupq_n_f32(0);
    float32x4_t ab1_4567 = vdupq_n_f32(0);

    float32x4_t ab2_0123 = vdupq_n_f32(0);
    float32x4_t ab2_4567 = vdupq_n_f32(0);

    float32x4_t ab3_0123 = vdupq_n_f32(0);
    float32x4_t ab3_4567 = vdupq_n_f32(0);

    float32x4_t ab4_0123 = vdupq_n_f32(0);
    float32x4_t ab4_4567 = vdupq_n_f32(0);

    float32x4_t ab5_0123 = vdupq_n_f32(0);
    float32x4_t ab5_4567 = vdupq_n_f32(0);

    float32x4_t ab6_0123 = vdupq_n_f32(0);
    float32x4_t ab6_4567 = vdupq_n_f32(0);

    float32x4_t ab7_0123 = vdupq_n_f32(0);
    float32x4_t ab7_4567 = vdupq_n_f32(0);

    float32x4x2_t b0 = { { vld1q_f32(b), vld1q_f32(b + 4) } };

    for (l = 0; l < kc; l++){
        b += 8;
        ab0_0123 = vmlaq_n_f32(ab0_0123, b0.val[0], a[0]);
        ab0_4567 = vmlaq_n_f32(ab0_4567, b0.val[1], a[0]);

        ab1_0123 = vmlaq_n_f32(ab1_0123, b0.val[0], a[1]);
        ab1_4567 = vmlaq_n_f32(ab1_4567, b0.val[1], a[1]);

        ab2_0123 = vmlaq_n_f32(ab2_0123, b0.val[0], a[2]);
        ab2_4567 = vmlaq_n_f32(ab2_4567, b0.val[1], a[2]);

        ab3_0123 = vmlaq_n_f32(ab3_0123, b0.val[0], a[3]);
        ab3_4567 = vmlaq_n_f32(ab3_4567, b0.val[1], a[3]);

        ab4_0123 = vmlaq_n_f32(ab4_0123, b0.val[0], a[4]);
        ab4_4567 = vmlaq_n_f32(ab4_4567, b0.val[1], a[4]);

        ab5_0123 = vmlaq_n_f32(ab5_0123, b0.val[0], a[5]);
        ab5_4567 = vmlaq_n_f32(ab5_4567, b0.val[1], a[5]);

        ab6_0123 = vmlaq_n_f32(ab6_0123, b0.val[0], a[6]);
        ab6_4567 = vmlaq_n_f32(ab6_4567, b0.val[1], a[6]);

        ab7_0123 = vmlaq_n_f32(ab7_0123, b0.val[0], a[7]);
        ab7_4567 = vmlaq_n_f32(ab7_4567, b0.val[1], a[7]);

        a += 8;

        b0.val[0] = vld1q_f32(b);
        b0.val[1] = vld1q_f32(b + 4);


    }
    vst1q_f32(ab + 0,  ab0_0123);
    vst1q_f32(ab + 4,  ab0_4567);
    vst1q_f32(ab + 8,  ab1_0123);
    vst1q_f32(ab + 12, ab1_4567);
    vst1q_f32(ab + 16, ab2_0123);
    vst1q_f32(ab + 20, ab2_4567);
    vst1q_f32(ab + 24, ab3_0123);
    vst1q_f32(ab + 28, ab3_4567);
    vst1q_f32(ab + 32, ab4_0123);
    vst1q_f32(ab + 36, ab4_4567);
    vst1q_f32(ab + 40, ab5_0123);
    vst1q_f32(ab + 44, ab5_4567);
    vst1q_f32(ab + 48, ab6_0123);
    vst1q_f32(ab + 52, ab6_4567);
    vst1q_f32(ab + 56, ab7_0123);
    vst1q_f32(ab + 60, ab7_4567);
#else
    memset(ab, 0, sizeof(ivf32) * FIV_KERNEL_M_8 * FIV_KERNEL_N_8);
    for (l = 0; l < kc; l++){
        for (j = 0; j < FIV_KERNEL_N_8; j++) {
            for (i = 0; i < FIV_KERNEL_M_8; i++) {
                ab[i * FIV_KERNEL_M_8 + j] += a[i] * b[j];
            }
        }
        a += FIV_KERNEL_M_8;
        b += FIV_KERNEL_N_8;
    }

#endif
    if (beta == 0.f){
        if (inc_row_c > inc_col_c) {
            for (i = 0; i < FIV_KERNEL_M_8; i++) {
                for (j = 0; j < FIV_KERNEL_N_8; j++) {
                    c[i * inc_row_c + j * inc_col_c] = 0;
                }
            }
        }    else {
            for (j = 0; j < FIV_KERNEL_N_8; j++) {
                for (i = 0; i < FIV_KERNEL_M_8; i++) {
                    c[i * inc_row_c + j * inc_col_c] = 0;
                }
            }
        }
    }    else if(beta != 1.f){
        for (j = 0; j < FIV_KERNEL_N_8; j++) {
            for (i = 0; i < FIV_KERNEL_M_8; i++) {
                c[i * inc_row_c + j * inc_col_c] *= beta;
            }
        }

    }

    if (alpha == 1.f){
        for (j = 0; j < FIV_KERNEL_N_8; j++) {
            for (i = 0; i < FIV_KERNEL_M_8; i++) {
                c[j * inc_row_c + i * inc_col_c] += ab[i + j * FIV_KERNEL_M_8];
            }
        }
    }  else {
        for (j = 0; j < FIV_KERNEL_N_8; j++) {
            for (i = 0; i < FIV_KERNEL_M_8; i++) {
                c[j * inc_row_c + i * inc_col_c] += alpha * ab[i + j * FIV_KERNEL_M_8];
            }
        }
    }

}


#undef FIV_KERNEL_M_8
#undef FIV_KERNEL_N_8

/* 4x24 kernel: kept verbatim from the verified source, but never selected
   (the driver always uses the 8x8 kernel); compile only where its AVX2 body
   is valid to avoid a bogus uninitialized warning on non-AVX2 targets. */
#if defined(FIV_USE_AVX2)

#define FIV_KERNEL_M_4   (4)
#define FIV_KERNEL_N_24  (24)

static void mat_mul_4xkx24_kernel(
            int kc, ivf32 alpha,
            ivf32* a, ivf32* b,
            ivf32 beta, ivf32*c,
            int inc_row_c, int inc_col_c)
{
    ivf32 FIV_DALIGNED ab[FIV_KERNEL_M_4 * FIV_KERNEL_N_24];
    int i, j, l;
    __m256 ab0, ab1, ab2, ab3;
    __m256 ab4, ab5, ab6, ab7;
    __m256 ab8, ab9, ab10, ab11;

    ab0 = _mm256_setzero_ps();
    ab1 = _mm256_setzero_ps();
    ab2 = _mm256_setzero_ps();
    ab3 = _mm256_setzero_ps();
    ab4 = _mm256_setzero_ps();
    ab5 = _mm256_setzero_ps();
    ab6 = _mm256_setzero_ps();
    ab7 = _mm256_setzero_ps();
    ab8 = _mm256_setzero_ps();
    ab9 = _mm256_setzero_ps();
    ab10 = _mm256_setzero_ps();
    ab11 = _mm256_setzero_ps();

    __m256 b0 = _mm256_load_ps(b);
    __m256 b1 = _mm256_load_ps(b + 8);
    __m256 b2 = _mm256_load_ps(b + 16);
    b += 24;
    for (l = 0; l < kc; l++) {
        __m256 a0 = _mm256_broadcast_ss(a);

        ab0 = _mm256_fmadd_ps(a0, b0, ab0);
        ab1 = _mm256_fmadd_ps(a0, b1, ab1);
        ab2 = _mm256_fmadd_ps(a0, b2, ab2);

        a0 = _mm256_broadcast_ss(a + 1);
        ab3 = _mm256_fmadd_ps(a0, b0, ab3);
        ab4 = _mm256_fmadd_ps(a0, b1, ab4);
        ab5 = _mm256_fmadd_ps(a0, b2, ab5);

        a0 = _mm256_broadcast_ss(a + 2);
        ab6 = _mm256_fmadd_ps(a0, b0, ab6);
        ab7 = _mm256_fmadd_ps(a0, b1, ab7);
        ab8 = _mm256_fmadd_ps(a0, b2, ab8);

        a0 = _mm256_broadcast_ss(a + 3);
        a += 4;
        ab9  = _mm256_fmadd_ps(a0, b0, ab9);
        ab10 = _mm256_fmadd_ps(a0, b1, ab10);
        ab11 = _mm256_fmadd_ps(a0, b2, ab11);

        b0 = _mm256_load_ps(b);
        b1 = _mm256_load_ps(b + 8);
        b2 = _mm256_load_ps(b + 16);

        b += 24;

    }
    _mm256_store_ps(ab + 0,  ab0);
    _mm256_store_ps(ab + 8,  ab1);
    _mm256_store_ps(ab + 16, ab2);

    _mm256_store_ps(ab + 24, ab3);
    _mm256_store_ps(ab + 32, ab4);
    _mm256_store_ps(ab + 40, ab5);

    _mm256_store_ps(ab + 48, ab6);
    _mm256_store_ps(ab + 56, ab7);
    _mm256_store_ps(ab + 64, ab8);

    _mm256_store_ps(ab + 72, ab9);
    _mm256_store_ps(ab + 80, ab10);
    _mm256_store_ps(ab + 88, ab11);

    if (beta == 0.f){
        if (inc_row_c > inc_col_c){
            for (i = 0; i < FIV_KERNEL_M_4; i++) {
                for (j = 0; j < FIV_KERNEL_N_24; j++) {
                    c[i * inc_row_c + j * inc_col_c] = 0.f;
                }
            }
        }    else {
            for (j = 0; j < FIV_KERNEL_N_24; j++) {
                for (i = 0; i < FIV_KERNEL_M_4; i++) {
                    c[i * inc_row_c + j * inc_col_c] = 0.f;
                }
            }
        }
    } else if (beta != 1.f){
        for (j = 0; j < FIV_KERNEL_N_24; j++) {
            for (i = 0; i < FIV_KERNEL_M_4; i++) {
                c[i * inc_row_c + j * inc_col_c] *= beta;
            }
        }
    }

    if (alpha == 1.f){
        for (i = 0; i < FIV_KERNEL_M_4; i++) {
            for (j = 0; j < FIV_KERNEL_N_24; j++) {
                c[i * inc_row_c + j * inc_col_c] += ab[j + i * FIV_KERNEL_N_24];
            }
        }
    }   else {
        for (i = 0; i < FIV_KERNEL_M_4; i++) {
            for (j = 0; j < FIV_KERNEL_N_24; j++) {
                c[i * inc_row_c + j * inc_col_c] += alpha * ab[j + i * FIV_KERNEL_N_24];
            }
        }
    }
}

#undef FIV_KERNEL_M_4
#undef FIV_KERNEL_N_24

#endif  /* FIV_USE_AVX2 */

#define FIV_MAX_KERNEL_SIZE 96   /* fits the 8x8 kernel (needs >= 64) */

static void dgeaxpy_row_major(
            int m, int n, ivf32 alpha,
            ivf32* x, int inc_row_x, int inc_col_x,
            ivf32* y, int inc_row_y, int inc_col_y)
{
    int i, j;
    if (alpha != 1.f){
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                y[i * inc_row_y + j * inc_col_y] += alpha * x[i * inc_row_x + j * inc_col_x];
            }
        }
    }    else {
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                y[i * inc_row_y + j * inc_col_y] += x[i * inc_row_x + j * inc_col_x];
            }
        }
    }
}

static void dgescal_row_major(
            int m, int n, ivf32 alpha,
            ivf32* x, int inc_row_x, int inc_col_x)
{
    int i, j;
    if (alpha != 1.f){
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                x[i * inc_row_x + j * inc_col_x] *= alpha;
            }
        }
    }   else {
        /* alpha == 1: keep x unchanged */
    }
}


static void mat_mul_kernel_row_major(
    int mc, int nc, int kc,
    ivf32 alpha, ivf32 beta,
    ivf32* c, int inc_row_c, int inc_col_c,
    ivf32* blocked_a, ivf32* blocked_b,
    int kernel_m_size, int kernel_n_size,
    ptr_func_mat_mul_mxkxn_kernel kernel)
{
    int mp = (mc + kernel_m_size - 1) / kernel_m_size;
    int np = (nc + kernel_n_size - 1) / kernel_n_size;
    int _mr = mc % kernel_m_size;
    int _nr = nc % kernel_n_size;
    int i;
#pragma omp parallel for schedule(static), shared(mp, np, _mr, _nr)
    for (i = 0; i < mp; i++) {
        int j, mr, nr;
        ivf32* ptr_a = &blocked_a[i * kc * kernel_m_size];
        ivf32* ptr_c = &c[i * kernel_m_size * inc_row_c];
        mr = (i != mp - 1 || _mr == 0) ? kernel_m_size : _mr;
        for (j = 0; j < np; j++) {
            ivf32* ptr_b_j = &blocked_b[j * kc * kernel_n_size];
            ivf32* ptr_c_j = &ptr_c[j * kernel_n_size * inc_col_c];
            nr = (j != np - 1 || _nr == 0) ? kernel_n_size : _nr;
            if (mr == kernel_m_size && nr == kernel_n_size){
                kernel(kc, alpha, ptr_a, ptr_b_j, beta, ptr_c_j, inc_row_c, inc_col_c);
            }    else {
                ivf32 FIV_DALIGNED blocked_c[FIV_MAX_KERNEL_SIZE];
                kernel(kc, alpha, ptr_a, ptr_b_j, 0.f, blocked_c, 1, kernel_m_size);
                dgescal_row_major(mr, nr, beta, ptr_c_j, inc_row_c, inc_col_c);
                dgeaxpy_row_major(mr, nr, 1.f, blocked_c, 1, kernel_m_size, ptr_c_j, inc_row_c, inc_col_c);

            }
        }

    }

}


static void blocked_mat_mul_row_major_real32(
            int m, int n, int k,
            ivf32 alpha,
            ivf32* a, int inc_row_a, int inc_col_a,
            ivf32* b, int inc_row_b, int inc_col_b,
            ivf32 beta,
            ivf32* c, int inc_row_c, int inc_col_c)
{
    int mb = (m + FIV_M_BLOCK - 1) / FIV_M_BLOCK;
    int nb = (n + FIV_N_BLOCK - 1) / FIV_N_BLOCK;
    int kb = (k + FIV_K_BLOCK - 1) / FIV_K_BLOCK;
    int _mc = m % FIV_M_BLOCK;
    int _nc = n % FIV_N_BLOCK;
    int _kc = k % FIV_K_BLOCK;

    if (alpha == 0.f || k == 0) {
        dgescal_row_major(m, n, beta, c, inc_row_c, inc_col_c);
        return;
    }

    int mc, nc, kc, i, j, l;
#if defined(FIV_USE_AVX2)
    int kernel_m_size = 8, kernel_n_size = 8;
#else
    int kernel_m_size = 8, kernel_n_size = 8;
#endif
    ptr_func_mat_mul_mxkxn_kernel kernel_func;
#if defined(FIV_USE_AVX2)
    if (kernel_m_size == 4 && kernel_n_size == 24){
        kernel_func = mat_mul_4xkx24_kernel;
    }    else
#endif
    if(kernel_m_size == 8 && kernel_n_size == 8)    {
        kernel_func = mat_mul_8xkx8_kernel;
    }    else {
        FIV_PRINT_LOG("no kernel function");
        return;
    }



    int a_zero_rest = FIV_M_BLOCK % kernel_m_size == 0 ? 0 : kernel_m_size - FIV_M_BLOCK % kernel_m_size;
    int b_zero_rest = FIV_N_BLOCK % kernel_n_size == 0 ? 0 : kernel_n_size - FIV_N_BLOCK % kernel_n_size;

    ivf32* blocked_a = (ivf32*)fiv_malloc(sizeof(ivf32) * (FIV_M_BLOCK + a_zero_rest) * FIV_K_BLOCK);
    ivf32* blocked_b = (ivf32*)fiv_malloc(sizeof(ivf32) * (FIV_N_BLOCK + b_zero_rest) * FIV_K_BLOCK);
    if (blocked_a && blocked_b) {
        for (i = 0; i < mb; i++) {
            ivf32* ptr_a = &a[i * FIV_M_BLOCK * inc_row_a];
            ivf32* ptr_c = &c[i * FIV_M_BLOCK * inc_row_c];
            mc = (i != mb - 1 || _mc == 0) ? FIV_M_BLOCK : _mc;
            for (l = 0; l < kb; l++) {
                ivf32* ptr_a_l = &ptr_a[l * FIV_K_BLOCK * inc_col_a];
                ivf32* ptr_b   = &b[l * FIV_K_BLOCK * inc_row_b];
                kc = (l != kb - 1 || _kc == 0) ? FIV_K_BLOCK : _kc;
                ivf32 _beta = (l == 0) ? beta : 1.f;
                copy_a_blocked(mc, kc, kernel_m_size, ptr_a_l, inc_row_a, inc_col_a, blocked_a);
                for (j = 0; j < nb; j++) {
                    ivf32* ptr_c_j = &ptr_c[j * FIV_N_BLOCK * inc_col_c];
                    ivf32* ptr_b_j = &ptr_b[j * FIV_N_BLOCK * inc_col_b];
                    nc = (j != nb - 1 || _nc == 0) ? FIV_N_BLOCK : _nc;
                    copy_b_blocked(kc, nc, kernel_n_size, ptr_b_j, inc_row_b, inc_col_b, blocked_b);
                    mat_mul_kernel_row_major(
                    mc, nc, kc, alpha, _beta,
                    ptr_c_j, inc_row_c, inc_col_c,
                    blocked_a, blocked_b,
                    kernel_m_size, kernel_n_size,
                    kernel_func);
                }
            }
      }
    }    else {
        FIV_PRINT_LOG("memory alloc error!");
    }

    fiv_free(blocked_b);
    fiv_free(blocked_a);
}


/* Blocked GEMM entry (real32). a_t/b_t: 1 means the operand is used
   transposed. m/n/k are the effective dims of op(A)/op(B)/op(C). */
static void fiv_matrix_mul_real32(
    int a_t, int b_t, int m, int n, int k,
    ivf32 alpha,
    ivf32* a, int lda,
    ivf32* b, int ldb,
    ivf32 beta,
    ivf32* c, int ldc)
{
    int i, j;
    if (m <= 0 || n <= 0 || ((alpha == 0.f || k <= 0) && (beta == 1.f))) {
        return;
    }
    if (alpha == 0.f){
        if (beta == 0.f) {
            for (j = 0; j < n; j++) {
                for (i = 0; i < m; i++) {
                    c[j * ldc + i] = 0;
                }
            }
        }    else {
            for (j = 0; j < n; j++) {
                for (i = 0; i < m; i++) {
                    c[j * ldc + i] *= beta;
                }
            }
        }
        return;
    }
    if (a_t == 0 && b_t == 0){
        blocked_mat_mul_row_major_real32(
        m, n, k,
        alpha,
        a, lda, 1,
        b, ldb, 1,
        beta,
         c, ldc, 1);
    } else if (a_t && b_t == 0){
        blocked_mat_mul_row_major_real32(
        m, n, k,
        alpha,
        a, 1, lda,
        b, ldb, 1,
        beta,
        c, ldc, 1);
    } else if (a_t == 0 && b_t){
        blocked_mat_mul_row_major_real32(
        m, n, k,
        alpha,
        a, lda, 1,
        b, 1, ldb,
        beta,
        c, ldc, 1);
    } else if (a_t && b_t) {
        blocked_mat_mul_row_major_real32(
        m, n, k,
        alpha,
        a, 1, lda,
        b, 1, ldb,
        beta,
        c, ldc, 1);
    }
}

/* ============================================================================
   Public API: dst = alpha * op(A) * op(B) + beta * dst, dispatched to the
   small (non-blocked) path when A+B+C fit in the L3-cache budget, otherwise
   to the blocked path.
   ========================================================================== */

fiv_ret fiv_matrix_mul(fiv_mat* dst, const fiv_mat* A, const fiv_mat* B,
                       int a_transpose, int b_transpose, fiv_scalar alpha, fiv_scalar beta)
{
    if (dst == NULL || A == NULL || B == NULL) return FIV_RET_ERR_PARA;
    if (dst->data.ptr == NULL || A->data.ptr == NULL || B->data.ptr == NULL) return FIV_RET_ERR_PARA;
    if (dst->data_continue == 0 || A->data_continue == 0 || B->data_continue == 0) return FIV_RET_ERR_PARA;
    /* dtype dispatch: 64F defers to the double-precision backend; the float32
       path below then re-checks that A/B/dst are all FIV_32F1. */
    if (A->dtype == FIV_64F1)
        return fiv_matrix_mul_real64(dst, A, B, a_transpose, b_transpose, alpha, beta);
    if (A->dtype != FIV_32F1 || B->dtype != FIV_32F1 || dst->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    /* alpha/beta must be fp32 scalars (FIV_32F1); any other type is unsupported */
    if (alpha.id != FIV_ID_SCALAR || alpha.dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (beta.id != FIV_ID_SCALAR || beta.dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    ivf32 alpha_f = alpha.data.value_fp32;
    ivf32 beta_f  = beta.data.value_fp32;
    /* in-place (dst aliasing A or B) is not supported */
    if (dst->data.ptr == A->data.ptr || dst->data.ptr == B->data.ptr) return FIV_RET_ERR_PARA;

    const int ra = (int)A->shapes[0];
    const int ca = (int)A->shapes[1];
    const int rb = (int)B->shapes[0];
    const int cb = (int)B->shapes[1];
    if (ra <= 0 || ca <= 0 || rb <= 0 || cb <= 0) return FIV_RET_ERR_PARA;

    const int M = a_transpose ? ca : ra;
    const int N = b_transpose ? rb : cb;
    const int K = a_transpose ? ra : ca;   /* cols of op(A) */
    const int Kb = b_transpose ? cb : rb;  /* rows of op(B) */
    if (K != Kb) return FIV_RET_ERR_PARA;

    if ((int)dst->shapes[0] != M || (int)dst->shapes[1] != N) return FIV_RET_ERR_PARA;
    if (dst->total_bytes < (size_t)M * (size_t)N * (size_t)dst->element_bytes) return FIV_RET_ERR_PARA;

    ivf32* a = (ivf32*)A->data.ptr;
    ivf32* b = (ivf32*)B->data.ptr;
    ivf32* c = (ivf32*)dst->data.ptr;

    /* alpha/beta are passed straight through to the kernels, no scratch buffer:
       every variant handles all beta values itself (see the small A*B^T beta
       handling and the blocked remainder-tile accumulation). */
    const size_t ws_bytes = ((size_t)ra * (size_t)ca + (size_t)rb * (size_t)cb +
                             (size_t)M * (size_t)N) * (size_t)A->element_bytes;

    if (ws_bytes <= FIV_MAT_MUL_L3_LIMIT_BYTES) {
        /* small (non-blocked) path */
        if (!a_transpose && !b_transpose)
            fiv_small_matrix_mul_matrix_real32(a, ra, ca, ca, b, cb, cb, c, N, alpha_f, beta_f);
        else if (!a_transpose && b_transpose)
            fiv_small_matrix_mul_matrix_t_real32(a, ra, ca, ca, b, rb, cb, c, N, alpha_f, beta_f);
        else if (a_transpose && !b_transpose)
            fiv_small_matrix_t_mul_matrix_real32(a, ra, ca, ca, b, cb, cb, c, N, alpha_f, beta_f);
        else
            fiv_small_matrix_t_mul_matrix_t_real32(a, ra, ca, ca, b, rb, cb, c, N, alpha_f, beta_f);
    } else {
        /* large (blocked) path */
        fiv_matrix_mul_real32(a_transpose, b_transpose, M, N, K, alpha_f, a, ca, b, cb, beta_f, c, N);
    }

    dst->shapes[0]   = (size_t)M;
    dst->shapes[1]   = (size_t)N;
    dst->strides[0]  = (size_t)N * (size_t)dst->element_bytes;
    dst->strides[1]  = (size_t)dst->element_bytes;
    dst->total_bytes = (size_t)M * (size_t)N * (size_t)dst->element_bytes;
    dst->data_continue = 1;
    return FIV_RET_OK;
}
