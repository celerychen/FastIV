/*
	Copyright (c) 2022 - LTS  HAOQIN. All Rights Reserved.

	This file is a part of FIV which means Fast Image and Vision Project.

	This FIV is free software : you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.If not, see < https://www.gnu.org/licenses/>.

	This program is also available under a commercial proprietary license.
	For more information, contact us at celerychen2000@126.com

====================================================================================
   This file is created by celery July.15th, 2022
====================================================================================
*/

#include "fiv_small_matrix_mul.h"
#include "fiv_common.h"

void fiv_small_matrix_mul_matrix_real32(
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
#if defined(FIV_AVX_OPTED)
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
#if defined(FIV_AVX_OPTED)
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

#if defined(FIV_SSE_OPTED)
static FIV_INLINE ivf32 fiv_mm_hsum_ps_sse3(__m128 v) 
{
	__m128 shuf = _mm_movehdup_ps(v);
	__m128 sums = _mm_add_ps(v, shuf);
	shuf = _mm_movehl_ps(shuf, sums);
	sums = _mm_add_ps(sums, shuf);
	return _mm_cvtss_f32(sums);
}
#endif


#if defined(FIV_AVX_OPTED)
static FIV_INLINE ivf32 fiv_mm256_hsum_ps(__m256 v)
{
	__m128 vlow  = _mm256_castps256_ps128(v);
	__m128 vhigh = _mm256_extractf128_ps(v, 1);
	vlow = _mm_add_ps(vlow, vhigh);
	return fiv_mm_hsum_ps_sse3(vlow);
}
#endif

void fiv_small_matrix_mul_matrix_t_real32(
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
#if defined(FIV_AVX_OPTED)
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
			}  else if(beta != 1.f){
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
#if defined(FIV_AVX_OPTED)
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
			}	else {
				ptr_lines_c[0][j] = alpha * sum[0] + beta * ptr_lines_c[0][j];
				ptr_lines_c[1][j] = alpha * sum[1] + beta * ptr_lines_c[1][j];
				ptr_lines_c[2][j] = alpha * sum[2] + beta * ptr_lines_c[2][j];
				ptr_lines_c[3][j] = alpha * sum[3] + beta * ptr_lines_c[3][j];
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


void fiv_small_matrix_t_mul_matrix_real32(
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
#if defined(FIV_AVX_OPTED)
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
#if defined(FIV_AVX_OPTED)
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


void fiv_small_matrix_t_mul_matrix_t_real32(
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
#if defined(FIV_AVX_OPTED)
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
#if defined(FIV_AVX_OPTED)
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


/************************************************************************/
/* 
    double data type
*/
/************************************************************************/


void fiv_small_matrix_mul_matrix_real64(
	ivf64* data_a, int rows_a, int cols_a, int stride_a,
	ivf64* data_b, int cols_b, int stride_b,
	ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta)
{
	int i = 0;
	for (; i <= rows_a - 4; i += 4) {
		ivf64* ptr_a[4], *ptr_c[4];
		ptr_a[0] = &data_a[i * stride_a];
		ptr_a[1] = ptr_a[0] + stride_a;
		ptr_a[2] = ptr_a[1] + stride_a;
		ptr_a[3] = ptr_a[2] + stride_a;

		ptr_c[0] = &data_c[i * stride_c];
		ptr_c[1] = ptr_c[0] + stride_c;
		ptr_c[2] = ptr_c[1] + stride_c;
		ptr_c[3] = ptr_c[2] + stride_c;

		if (beta == 0.0) {
			memset(ptr_c[0], 0, sizeof(ivf64) * cols_b);
			memset(ptr_c[1], 0, sizeof(ivf64) * cols_b);
			memset(ptr_c[2], 0, sizeof(ivf64) * cols_b);
			memset(ptr_c[3], 0, sizeof(ivf64) * cols_b);
		}	else if (beta != 1.0) {
			for (int l = 0; l < cols_b; l++) {
				ptr_c[0][l] *= beta;
				ptr_c[1][l] *= beta;
				ptr_c[2][l] *= beta;
				ptr_c[3][l] *= beta;
			}
		}
		int k = 0;
		for (; k <= cols_a - 2; k += 2) {
			ivf64 t_a[8];
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
				t_a[0] = ptr_a[0][k + 0];
				t_a[1] = ptr_a[1][k + 0];
				t_a[2] = ptr_a[2][k + 0];
				t_a[3] = ptr_a[3][k + 0];
				t_a[4] = ptr_a[0][k + 1];
				t_a[5] = ptr_a[1][k + 1];
				t_a[6] = ptr_a[2][k + 1];
				t_a[7] = ptr_a[3][k + 1];
			}
			ivf64* ptr_b[2] = { &data_b[k * stride_b], &data_b[(k + 1)* stride_b] };
			int j = 0;
#if defined(FIV_AVX_OPTED)
			__m256d m_t_a1 = _mm256_broadcast_sd(&t_a[0]);
			__m256d m_t_a2 = _mm256_broadcast_sd(&t_a[1]);
			__m256d m_t_a3 = _mm256_broadcast_sd(&t_a[2]);
			__m256d m_t_a4 = _mm256_broadcast_sd(&t_a[3]);
			__m256d m_t_a5 = _mm256_broadcast_sd(&t_a[4]);
			__m256d m_t_a6 = _mm256_broadcast_sd(&t_a[5]);
			__m256d m_t_a7 = _mm256_broadcast_sd(&t_a[6]);
			__m256d m_t_a8 = _mm256_broadcast_sd(&t_a[7]);

			for (; j <= cols_b - 4; j += 4) {
				__m256d m_t_c0 = _mm256_loadu_pd(&ptr_c[0][j]);
				__m256d m_t_c1 = _mm256_loadu_pd(&ptr_c[1][j]);
				__m256d m_t_c2 = _mm256_loadu_pd(&ptr_c[2][j]);
				__m256d m_t_c3 = _mm256_loadu_pd(&ptr_c[3][j]);
				__m256d m_t_b0 = _mm256_loadu_pd(&ptr_b[0][j]);
				__m256d m_t_b1 = _mm256_loadu_pd(&ptr_b[1][j]);

				m_t_c0 = _mm256_fmadd_pd(m_t_a1, m_t_b0, m_t_c0);
				m_t_c1 = _mm256_fmadd_pd(m_t_a2, m_t_b0, m_t_c1);
				m_t_c2 = _mm256_fmadd_pd(m_t_a3, m_t_b0, m_t_c2);
				m_t_c3 = _mm256_fmadd_pd(m_t_a4, m_t_b0, m_t_c3);

				m_t_c0 = _mm256_fmadd_pd(m_t_a5, m_t_b1, m_t_c0);
				m_t_c1 = _mm256_fmadd_pd(m_t_a6, m_t_b1, m_t_c1);
				m_t_c2 = _mm256_fmadd_pd(m_t_a7, m_t_b1, m_t_c2);
				m_t_c3 = _mm256_fmadd_pd(m_t_a8, m_t_b1, m_t_c3);

				_mm256_storeu_pd(&ptr_c[0][j], m_t_c0);
				_mm256_storeu_pd(&ptr_c[1][j], m_t_c1);
				_mm256_storeu_pd(&ptr_c[2][j], m_t_c2);
				_mm256_storeu_pd(&ptr_c[3][j], m_t_c3);
			}
#endif
			for (; j < cols_b; j++) {
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

		for (; k < cols_a; k++) {
			ivf64 t_a[4] = {
				ptr_a[0][k] * alpha,ptr_a[1][k] * alpha,
				ptr_a[2][k] * alpha,ptr_a[3][k] * alpha
			};
			ivf64* ptr_b = &data_b[k * stride_b];

			int j = 0;
#if defined(FIV_AVX_OPTED)
			__m256d m_t_a1 = _mm256_broadcast_sd(&t_a[0]);
			__m256d m_t_a2 = _mm256_broadcast_sd(&t_a[1]);
			__m256d m_t_a3 = _mm256_broadcast_sd(&t_a[2]);
			__m256d m_t_a4 = _mm256_broadcast_sd(&t_a[3]);

			for (; j <= cols_b - 4; j += 4) {
				__m256d m_t_c0 = _mm256_loadu_pd(&ptr_c[0][j]);
				__m256d m_t_c1 = _mm256_loadu_pd(&ptr_c[1][j]);
				__m256d m_t_c2 = _mm256_loadu_pd(&ptr_c[2][j]);
				__m256d m_t_c3 = _mm256_loadu_pd(&ptr_c[3][j]);
				__m256d m_t_b1 = _mm256_loadu_pd(&ptr_b[j]);

				m_t_c0 = _mm256_fmadd_pd(m_t_a1, m_t_b1, m_t_c0);
				m_t_c1 = _mm256_fmadd_pd(m_t_a2, m_t_b1, m_t_c1);
				m_t_c2 = _mm256_fmadd_pd(m_t_a3, m_t_b1, m_t_c2);
				m_t_c3 = _mm256_fmadd_pd(m_t_a4, m_t_b1, m_t_c3);

				_mm256_storeu_pd(&ptr_c[0][j], m_t_c0);
				_mm256_storeu_pd(&ptr_c[1][j], m_t_c1);
				_mm256_storeu_pd(&ptr_c[2][j], m_t_c2);
				_mm256_storeu_pd(&ptr_c[3][j], m_t_c3);
			}
#endif
			for (; j < cols_b; j++) {
				ptr_c[0][j] += t_a[0] * ptr_b[j];
				ptr_c[1][j] += t_a[1] * ptr_b[j];
				ptr_c[2][j] += t_a[2] * ptr_b[j];
				ptr_c[3][j] += t_a[3] * ptr_b[j];
			}
		}
	}

	for (; i < rows_a; i++) {
		ivf64* ptr_a = &data_a[i * stride_a];
		ivf64* ptr_c = &data_c[i * stride_c];
		if (beta == 0.0) {
			memset(ptr_c, 0, sizeof(ivf64) * cols_b);
		}	else if (beta != 1.0) {
			for (int l = 0; l < cols_b; l++) {
				ptr_c[l] *= beta;
			}
		}
		int k = 0;
		for (; k < cols_a; k++) {
			ivf64 t_a_d = ptr_a[k] * alpha;
			ivf64* ptr_b = &data_b[k * stride_b];
			for (int j = 0; j < cols_b; j++) {
				ptr_c[j] += t_a_d * ptr_b[j];
			}
		}
	}
}

#if defined(FIV_AVX_OPTED)
static FIV_INLINE ivf64 fiv_mm256_hsum_pd(__m256d v)
{
	__m128d vlow = _mm256_castpd256_pd128(v);
	__m128d vhigh = _mm256_extractf128_pd(v, 1);
	vlow = _mm_add_pd(vlow, vhigh);
	__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
	return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
}
#endif


void fiv_small_matrix_mul_matrix_t_real64(
	ivf64* data_a, int rows_a, int cols_a, int stride_a,
	ivf64* data_b, int rows_b, int stride_b,
	ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta)
{
	int i = 0;
	for (; i <= rows_a - 4; i += 4) {
		ivf64* ptr_lines_a[4], *ptr_lines_c[4];
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
			ivf64* ptr_lines_b[2];
			ptr_lines_b[0] = &data_b[j * stride_b];
			ptr_lines_b[1] = ptr_lines_b[0] + stride_b;

			ivf64 FIV_DALIGNED sum[4 * 2] = { 0 };
			int k = 0;
#if defined(FIV_AVX_OPTED)
			__m256d m_s[4 * 2] = { 0 };
			for (; k <= cols_a - 4; k += 4) {
				__m256d t_b1 = _mm256_loadu_pd(&ptr_lines_b[0][k]);
				__m256d t_b2 = _mm256_loadu_pd(&ptr_lines_b[1][k]);

				m_s[0] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[0][k]), t_b1, m_s[0]);
				m_s[1] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[1][k]), t_b1, m_s[1]);
				m_s[2] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[2][k]), t_b1, m_s[2]);
				m_s[3] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[3][k]), t_b1, m_s[3]);

				m_s[4] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[0][k]), t_b2, m_s[4]);
				m_s[5] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[1][k]), t_b2, m_s[5]);
				m_s[6] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[2][k]), t_b2, m_s[6]);
				m_s[7] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[3][k]), t_b2, m_s[7]);
			}
			sum[0] = fiv_mm256_hsum_pd(m_s[0]);
			sum[1] = fiv_mm256_hsum_pd(m_s[1]);
			sum[2] = fiv_mm256_hsum_pd(m_s[2]);
			sum[3] = fiv_mm256_hsum_pd(m_s[3]);
			sum[4] = fiv_mm256_hsum_pd(m_s[4]);
			sum[5] = fiv_mm256_hsum_pd(m_s[5]);
			sum[6] = fiv_mm256_hsum_pd(m_s[6]);
			sum[7] = fiv_mm256_hsum_pd(m_s[7]);
#endif
			for (; k < cols_a; k++) {
				sum[0] += ptr_lines_a[0][k] * ptr_lines_b[0][k];
				sum[1] += ptr_lines_a[1][k] * ptr_lines_b[0][k];
				sum[2] += ptr_lines_a[2][k] * ptr_lines_b[0][k];
				sum[3] += ptr_lines_a[3][k] * ptr_lines_b[0][k];
				sum[4] += ptr_lines_a[0][k] * ptr_lines_b[1][k];
				sum[5] += ptr_lines_a[1][k] * ptr_lines_b[1][k];
				sum[6] += ptr_lines_a[2][k] * ptr_lines_b[1][k];
				sum[7] += ptr_lines_a[3][k] * ptr_lines_b[1][k];
			}

			if (alpha != 1.0) {
				sum[0] *= alpha;
				sum[1] *= alpha;
				sum[2] *= alpha;
				sum[3] *= alpha;
				sum[4] *= alpha;
				sum[5] *= alpha;
				sum[6] *= alpha;
				sum[7] *= alpha;
			}
			if (beta == 0.0) {
				ptr_lines_c[0][j] = sum[0];
				ptr_lines_c[1][j] = sum[1];
				ptr_lines_c[2][j] = sum[2];
				ptr_lines_c[3][j] = sum[3];
				ptr_lines_c[0][j + 1] = sum[4];
				ptr_lines_c[1][j + 1] = sum[5];
				ptr_lines_c[2][j + 1] = sum[6];
				ptr_lines_c[3][j + 1] = sum[7];
			}	else if (beta != 1.f) {
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

		for (; j < rows_b; j++) {

			ivf64* ptr_line_b = &data_b[j * stride_b];
			ivf64 sum[4] = { 0 };
			int k = 0;
#if defined(FIV_AVX_OPTED)
			__m256d m_s[4] = { 0 };
			for (; k <= cols_a - 4; k += 4) {
				__m256d t_b = _mm256_loadu_pd(&ptr_line_b[k]);
				m_s[0] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[0][k]), t_b, m_s[0]);
				m_s[1] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[1][k]), t_b, m_s[1]);
				m_s[2] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[2][k]), t_b, m_s[2]);
				m_s[3] = _mm256_fmadd_pd(_mm256_loadu_pd(&ptr_lines_a[3][k]), t_b, m_s[3]);
			}
			sum[0] = fiv_mm256_hsum_pd(m_s[0]);
			sum[1] = fiv_mm256_hsum_pd(m_s[1]);
			sum[2] = fiv_mm256_hsum_pd(m_s[2]);
			sum[3] = fiv_mm256_hsum_pd(m_s[3]);
#endif
			for (; k < cols_a; k++) {
				sum[0] += ptr_lines_a[0][k] * ptr_line_b[k];
				sum[1] += ptr_lines_a[1][k] * ptr_line_b[k];
				sum[2] += ptr_lines_a[2][k] * ptr_line_b[k];
				sum[3] += ptr_lines_a[3][k] * ptr_line_b[k];
			}

			if (beta == 0.0) {
				ptr_lines_c[0][j] = alpha * sum[0];
				ptr_lines_c[1][j] = alpha * sum[1];
				ptr_lines_c[2][j] = alpha * sum[2];
				ptr_lines_c[3][j] = alpha * sum[3];
			}	else {
				ptr_lines_c[0][j] = alpha * sum[0] + beta * ptr_lines_c[0][j];
				ptr_lines_c[1][j] = alpha * sum[1] + beta * ptr_lines_c[1][j];
				ptr_lines_c[2][j] = alpha * sum[2] + beta * ptr_lines_c[2][j];
				ptr_lines_c[3][j] = alpha * sum[3] + beta * ptr_lines_c[3][j];
			}
		}
	}
	for (; i < rows_a; i++) {
		ivf64* ptr_line_a = &data_a[i * stride_a];
		ivf64* ptr_line_c = &data_c[i * stride_c];
		for (int j = 0; j < rows_b; j++) {
			ivf64* ptr_line_b = &data_b[j * stride_b];
			ivf64 sum = 0;
			for (int k = 0; k < cols_a; k++) {
				sum += ptr_line_a[k] * ptr_line_b[k];
			}
			ptr_line_c[j] = beta * ptr_line_c[j] + alpha * sum;
		}
	}
}


void fiv_small_matrix_t_mul_matrix_real64(
	ivf64* data_a, int rows_a, int cols_a, int stride_a,
	ivf64* data_b, int cols_b, int stride_b,
	ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta)
{
	int i = 0;
	for (; i <= cols_a - 4; i += 4) {
		ivf64* ptr_c_lines[4];
		ptr_c_lines[0] = &data_c[i * stride_c];
		ptr_c_lines[1] = ptr_c_lines[0] + stride_c;
		ptr_c_lines[2] = ptr_c_lines[1] + stride_c;
		ptr_c_lines[3] = ptr_c_lines[2] + stride_c;

		if (beta == 0.) {
			memset(ptr_c_lines[0], 0, sizeof(ivf64) * cols_b);
			memset(ptr_c_lines[1], 0, sizeof(ivf64) * cols_b);
			memset(ptr_c_lines[2], 0, sizeof(ivf64) * cols_b);
			memset(ptr_c_lines[3], 0, sizeof(ivf64) * cols_b);
		}	else if (beta != 1.) {
			for (int l = 0; l < cols_b; l++) {
				ptr_c_lines[0][l] *= beta;
				ptr_c_lines[1][l] *= beta;
				ptr_c_lines[2][l] *= beta;
				ptr_c_lines[3][l] *= beta;
			}
		}
		int k = 0;
		for (; k <= rows_a - 2; k += 2) {
			ivf64* ptr_data_a_tmp = &data_a[k * stride_a + i];
			ivf64 t_a[8] = { ptr_data_a_tmp[0],ptr_data_a_tmp[1], ptr_data_a_tmp[2], ptr_data_a_tmp[3],
			ptr_data_a_tmp[0 + stride_a], ptr_data_a_tmp[1 + stride_a],
			ptr_data_a_tmp[2 + stride_a], ptr_data_a_tmp[3 + stride_a] };
			if (alpha != 1.) {
				t_a[0] *= alpha; t_a[1] *= alpha;
				t_a[2] *= alpha; t_a[3] *= alpha;
				t_a[4] *= alpha; t_a[5] *= alpha;
				t_a[6] *= alpha; t_a[7] *= alpha;
			}
			ivf64* ptr_b_line1 = &data_b[k *stride_b];
			ivf64* ptr_b_line2 = &data_b[k *stride_b + stride_b];
			int j = 0;
#if defined(FIV_AVX_OPTED)
			__m256d m_t_a1 = _mm256_broadcast_sd(&t_a[0]);
			__m256d m_t_a2 = _mm256_broadcast_sd(&t_a[1]);
			__m256d m_t_a3 = _mm256_broadcast_sd(&t_a[2]);
			__m256d m_t_a4 = _mm256_broadcast_sd(&t_a[3]);
			__m256d m_t_a5 = _mm256_broadcast_sd(&t_a[4]);
			__m256d m_t_a6 = _mm256_broadcast_sd(&t_a[5]);
			__m256d m_t_a7 = _mm256_broadcast_sd(&t_a[6]);
			__m256d m_t_a8 = _mm256_broadcast_sd(&t_a[7]);

			for (; j <= cols_b - 4; j += 4) {
				__m256d m_t_c0 = _mm256_loadu_pd(&ptr_c_lines[0][j]);
				__m256d m_t_c1 = _mm256_loadu_pd(&ptr_c_lines[1][j]);
				__m256d m_t_c2 = _mm256_loadu_pd(&ptr_c_lines[2][j]);
				__m256d m_t_c3 = _mm256_loadu_pd(&ptr_c_lines[3][j]);

				__m256d m_t_b1 = _mm256_loadu_pd(&ptr_b_line1[j]);
				__m256d m_t_b2 = _mm256_loadu_pd(&ptr_b_line2[j]);

				m_t_c0 = _mm256_fmadd_pd(m_t_a1, m_t_b1, m_t_c0);
				m_t_c1 = _mm256_fmadd_pd(m_t_a2, m_t_b1, m_t_c1);
				m_t_c2 = _mm256_fmadd_pd(m_t_a3, m_t_b1, m_t_c2);
				m_t_c3 = _mm256_fmadd_pd(m_t_a4, m_t_b1, m_t_c3);

				m_t_c0 = _mm256_fmadd_pd(m_t_a5, m_t_b2, m_t_c0);
				m_t_c1 = _mm256_fmadd_pd(m_t_a6, m_t_b2, m_t_c1);
				m_t_c2 = _mm256_fmadd_pd(m_t_a7, m_t_b2, m_t_c2);
				m_t_c3 = _mm256_fmadd_pd(m_t_a8, m_t_b2, m_t_c3);

				_mm256_storeu_pd(&ptr_c_lines[0][j], m_t_c0);
				_mm256_storeu_pd(&ptr_c_lines[1][j], m_t_c1);
				_mm256_storeu_pd(&ptr_c_lines[2][j], m_t_c2);
				_mm256_storeu_pd(&ptr_c_lines[3][j], m_t_c3);

			}
#endif
			for (; j < cols_b; j++) {
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

		for (; k < rows_a; k++) {
			ivf64 t_a1 = data_a[k * stride_a + i + 0];
			ivf64 t_a2 = data_a[k * stride_a + i + 1];
			ivf64 t_a3 = data_a[k * stride_a + i + 2];
			ivf64 t_a4 = data_a[k * stride_a + i + 3];
			if (alpha != 1.) {
				t_a1 *= alpha; t_a2 *= alpha;
				t_a3 *= alpha; t_a4 *= alpha;
			}
			ivf64* ptr_b_line = &data_b[k * stride_b];
			int j = 0;
#if defined(FIV_AVX_OPTED)
			__m256d m_t_a1 = _mm256_broadcast_sd(&t_a1);
			__m256d m_t_a2 = _mm256_broadcast_sd(&t_a2);
			__m256d m_t_a3 = _mm256_broadcast_sd(&t_a3);
			__m256d m_t_a4 = _mm256_broadcast_sd(&t_a4);

			for (; j <= cols_b - 4; j += 4) {
				__m256d m_t_c0 = _mm256_loadu_pd(&ptr_c_lines[0][j]);
				__m256d m_t_c1 = _mm256_loadu_pd(&ptr_c_lines[1][j]);
				__m256d m_t_c2 = _mm256_loadu_pd(&ptr_c_lines[2][j]);
				__m256d m_t_c3 = _mm256_loadu_pd(&ptr_c_lines[3][j]);
				__m256d m_t_b = _mm256_loadu_pd(&ptr_b_line[j]);

				m_t_c0 = _mm256_fmadd_pd(m_t_a1, m_t_b, m_t_c0);
				m_t_c1 = _mm256_fmadd_pd(m_t_a2, m_t_b, m_t_c1);
				m_t_c2 = _mm256_fmadd_pd(m_t_a3, m_t_b, m_t_c2);
				m_t_c3 = _mm256_fmadd_pd(m_t_a4, m_t_b, m_t_c3);

				_mm256_storeu_pd(&ptr_c_lines[0][j], m_t_c0);
				_mm256_storeu_pd(&ptr_c_lines[1][j], m_t_c1);
				_mm256_storeu_pd(&ptr_c_lines[2][j], m_t_c2);
				_mm256_storeu_pd(&ptr_c_lines[3][j], m_t_c3);

			}
#endif
			for (; j < cols_b; j++) {
				ptr_c_lines[0][j] += t_a1 * ptr_b_line[j];
				ptr_c_lines[1][j] += t_a2 * ptr_b_line[j];
				ptr_c_lines[2][j] += t_a3 * ptr_b_line[j];
				ptr_c_lines[3][j] += t_a4 * ptr_b_line[j];
			}
		}
	}

	for (; i < cols_a; i++) {
		ivf64* ptr_c_line = &data_c[i * stride_c];
		if (beta == 0.) {
			memset(ptr_c_line, 0, sizeof(ivf64) * cols_b);
		}	else if (beta != 1.) {
			for (int k = 0; k < cols_b; k++) {
				ptr_c_line[k] *= beta;
			}
		}
		for (int k = 0; k < rows_a; k++) {
			ivf64 t_a = data_a[k * stride_a + i] * alpha;
			ivf64* ptr_b_line = &data_b[k * stride_b];
			for (int j = 0; j < cols_b; j++) {
				ptr_c_line[j] += t_a * ptr_b_line[j];
			}
		}
	}
}




void fiv_small_matrix_t_mul_matrix_t_real64(
	ivf64* data_a, int rows_a, int cols_a, int stride_a,
	ivf64* data_b, int rows_b, int stride_b,
	ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta)
{
	int i = 0;
	ivf64* ptr_mem_a = (ivf64*)fiv_malloc(sizeof(ivf64) * 8 * rows_a);
	if (ptr_mem_a == NULL) {
		printf("MEM ERROR\n");
		return;
	}
	for (; i <= cols_a - 8; i += 8) {
		ivf64* ptr_c_lines[8];
		ptr_c_lines[0] = &data_c[i * stride_c];
		ptr_c_lines[1] = ptr_c_lines[0] + stride_c;
		ptr_c_lines[2] = ptr_c_lines[1] + stride_c;
		ptr_c_lines[3] = ptr_c_lines[2] + stride_c;
		ptr_c_lines[4] = ptr_c_lines[3] + stride_c;
		ptr_c_lines[5] = ptr_c_lines[4] + stride_c;
		ptr_c_lines[6] = ptr_c_lines[5] + stride_c;
		ptr_c_lines[7] = ptr_c_lines[6] + stride_c;
		ivf64* ptr_a_i_col = &data_a[i];
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
			ivf64* ptr_a = ptr_mem_a;
			ivf64* ptr_b[2] = { &data_b[j * stride_b], &data_b[(j + 1) * stride_b] };
			ivf64 FIV_DALIGNED sum[16] = { 0 };
			int k = 0;
#if defined(FIV_AVX_OPTED)
			__m256d m_sum[8] = { 0 };
			for (; k <= rows_a - 2; k += 2) {
				__m256d m_b0 = _mm256_broadcast_sd(&ptr_b[0][k]);
				__m256d m_b1 = _mm256_broadcast_sd(&ptr_b[1][k]);
				__m256d m_b2 = _mm256_broadcast_sd(&ptr_b[0][k + 1]);
				__m256d m_b3 = _mm256_broadcast_sd(&ptr_b[1][k + 1]);

				__m256d m_a0 = _mm256_loadu_pd(&ptr_a[0]);
				__m256d m_a1 = _mm256_loadu_pd(&ptr_a[4]);
				__m256d m_a2 = _mm256_loadu_pd(&ptr_a[8]);
				__m256d m_a3 = _mm256_loadu_pd(&ptr_a[12]);

				ptr_a += 16;

				m_sum[0] = _mm256_fmadd_pd(m_a0, m_b0, m_sum[0]);
				m_sum[1] = _mm256_fmadd_pd(m_a1, m_b0, m_sum[1]);
				m_sum[2] = _mm256_fmadd_pd(m_a0, m_b1, m_sum[2]);
				m_sum[3] = _mm256_fmadd_pd(m_a1, m_b1, m_sum[3]);

				m_sum[4] = _mm256_fmadd_pd(m_a2, m_b2, m_sum[4]);
				m_sum[5] = _mm256_fmadd_pd(m_a3, m_b2, m_sum[5]);
				m_sum[6] = _mm256_fmadd_pd(m_a2, m_b3, m_sum[6]);
				m_sum[7] = _mm256_fmadd_pd(m_a3, m_b3, m_sum[7]);



			}
			m_sum[0] = _mm256_add_pd(m_sum[0], m_sum[4]);
			m_sum[1] = _mm256_add_pd(m_sum[1], m_sum[5]);
			m_sum[2] = _mm256_add_pd(m_sum[2], m_sum[6]);
			m_sum[3] = _mm256_add_pd(m_sum[3], m_sum[7]);

			_mm256_storeu_pd(sum,      m_sum[0]);
			_mm256_storeu_pd(sum + 4,  m_sum[1]);
			_mm256_storeu_pd(sum + 8,  m_sum[2]);
			_mm256_storeu_pd(sum + 12, m_sum[3]);


#endif
			for (; k < rows_a; k++) {
				sum[0] += ptr_a[0] * ptr_b[0][k];
				sum[1] += ptr_a[1] * ptr_b[0][k];
				sum[2] += ptr_a[2] * ptr_b[0][k];
				sum[3] += ptr_a[3] * ptr_b[0][k];
				sum[4] += ptr_a[4] * ptr_b[0][k];
				sum[5] += ptr_a[5] * ptr_b[0][k];
				sum[6] += ptr_a[6] * ptr_b[0][k];
				sum[7] += ptr_a[7] * ptr_b[0][k];

				sum[8] += ptr_a[0] * ptr_b[1][k];
				sum[9] += ptr_a[1] * ptr_b[1][k];
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

			sum[8] *= alpha; sum[9] *= alpha;
			sum[10] *= alpha; sum[11] *= alpha;
			sum[12] *= alpha; sum[13] *= alpha;
			sum[14] *= alpha; sum[15] *= alpha;

			if (beta == 0.0) {
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

		for (; j < rows_b; j++) {
			ivf64* ptr_a = ptr_mem_a;
			ivf64* ptr_b = &data_b[j * stride_b];
			ivf64 FIV_DALIGNED sum[8] = { 0 };
			int k = 0;

			for (; k < rows_a; k++) {
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

			if (beta == 0.) {
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
	for (; i < cols_a; i++) {
		ivf64* ptr_c_line = &data_c[i * stride_c];
		ivf64* ptr_a_i_col = &data_a[i];
		for (int k = 0; k < rows_a; k++) {
			ptr_mem_a[k] = ptr_a_i_col[k * stride_a];
		}
		for (int j = 0; j < rows_b; j++) {
			ivf64* ptr_a = ptr_mem_a;
			ivf64* ptr_b = &data_b[j * stride_b];
			ivf64 sum = 0;
			for (int k = 0; k < rows_a; k++) {
				sum += ptr_a[k] * ptr_b[k];
			}
			ptr_c_line[j] = beta * ptr_c_line[j] + alpha * sum;
		}
	}
	fiv_free(ptr_mem_a);
}






















