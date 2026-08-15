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

#ifndef _FIV_MAT_MUL_H_
#define _FIV_MAT_MUL_H_

#include "fiv_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Working set (A + B + C in bytes) below which the non-blocked small-matrix
   path is used; above it the blocked path takes over. The default is chosen so
   the three matrices fit in a typical CPU L3 cache. Override with
   -DFIV_MAT_MUL_L3_LIMIT_BYTES=<n> to tune for a specific CPU. */
#ifndef FIV_MAT_MUL_L3_LIMIT_BYTES
#define FIV_MAT_MUL_L3_LIMIT_BYTES (8u * 1024u * 1024u)
#endif

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_MAT_MUL_H_ */
