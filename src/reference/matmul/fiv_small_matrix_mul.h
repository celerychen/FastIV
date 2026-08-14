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
   This file is created by celery July.14th, 2022
====================================================================================
*/

#ifndef _FIV_SMALL_MATRIX_MUL_H_
#define _FIV_SMALL_MATRIX_MUL_H_


#include "fiv_data_typedefs.h"

#ifdef __cplusplus
extern "C" {
#endif


	void fiv_small_matrix_mul_matrix_real32(
		ivf32* data_a, int rows_a, int cols_a, int stride_a,
		ivf32* data_b, int cols_b, int stride_b,
		ivf32* data_c, int stride_c, ivf32 alpha, ivf32 beta);

	void fiv_small_matrix_mul_matrix_t_real32(
		ivf32* data_a, int rows_a, int cols_a, int stride_a,
		ivf32* data_b, int rows_b, int stride_b,
		ivf32* data_c, int stride_c, ivf32 alpha, ivf32 beta);

	void fiv_small_matrix_t_mul_matrix_real32(
		ivf32* data_a, int rows_a, int cols_a, int stride_a,
		ivf32* data_b, int cols_b, int stride_b,
		ivf32* data_c, int stride_c, ivf32 alpha, ivf32 beta);

	void fiv_small_matrix_t_mul_matrix_t_real32(
		ivf32* data_a, int rows_a, int cols_a, int stride_a,
		ivf32* data_b, int rows_b, int stride_b,
		ivf32* data_c, int stride_c, ivf32 alpha, ivf32 beta);


	void fiv_small_matrix_mul_matrix_real64(
		ivf64* data_a, int rows_a, int cols_a, int stride_a,
		ivf64* data_b, int cols_b, int stride_b,
		ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta);

	void fiv_small_matrix_mul_matrix_t_real64(
		ivf64* data_a, int rows_a, int cols_a, int stride_a,
		ivf64* data_b, int rows_b, int stride_b,
		ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta);

	void fiv_small_matrix_t_mul_matrix_real64(
		ivf64* data_a, int rows_a, int cols_a, int stride_a,
		ivf64* data_b, int cols_b, int stride_b,
		ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta);

	void fiv_small_matrix_t_mul_matrix_t_real64(
		ivf64* data_a, int rows_a, int cols_a, int stride_a,
		ivf64* data_b, int rows_b, int stride_b,
		ivf64* data_c, int stride_c, ivf64 alpha, ivf64 beta);




#ifdef __cplusplus
}
#endif






#endif