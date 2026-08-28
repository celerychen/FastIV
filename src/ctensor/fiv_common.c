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



#include "fiv_common.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>


void* fiv_malloc(size_t size)
{
    iv8u* udata, *udata_offset;
    iv8u** adata;
    if (size == 0) return NULL;
    udata = (iv8u*)malloc(size + sizeof(void*) + FIV_STRIDE_ALIGN + sizeof(size_t));
    if (udata == NULL) return NULL;
    udata_offset = udata + sizeof(void*)+sizeof(size_t);
    adata = (iv8u**)FIV_PTR_ALIGN(udata_offset, FIV_STRIDE_ALIGN);
    ((size_t*)adata)[-1] = size;
    adata[-2] = udata;
    return (void*)adata;
}

void* fiv_calloc(size_t num, size_t size)
{
    void* ptr = fiv_malloc(num * size);
    if (ptr) {
        memset(ptr, 0, num * size);
    }
    return ptr;
}

void fiv_free(void *p)
{
    if (p) {
        iv8u* udata = ((iv8u**)p)[-2];
        if (udata) {
            free(udata);
        }

    }
}

/* Returns the current monotonic time in milliseconds (double, sub-ms precision).
   Subtract two readings to time a span, e.g. (t1 - t0) gives elapsed ms. */
ivf64 fiv_get_current_system_time(void)
{
#if defined(_WIN32)
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (ivf64)cnt.QuadPart / (ivf64)freq.QuadPart * 1e3;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ivf64)ts.tv_sec * 1e3 + (ivf64)ts.tv_nsec * 1e-6;
#endif
}


void* fiv_realloc(void* p, size_t size)
{
    size_t* ptr_t;
    size_t old_size;
    if (p == NULL) return fiv_malloc(size);

    ptr_t = (size_t*)p;
    old_size = ptr_t[-1];
    if (old_size < size) {
        void* ptr_t2 = fiv_malloc(size);
        if (ptr_t2 == NULL) return NULL;

        memcpy(ptr_t2, p, old_size);
        fiv_free(p);

        return ptr_t2;
    }    else {
        return p;
    }

}



/* Deterministic LCG in [-1, 1]; shared seed stream keeps training reproducible. */
float fiv_nn_rand(void)
{
    static unsigned s = 12345u;
    s = s * 1103515245u + 12345u;
    return ((float)((s >> 8) & 0xffffff) / 16777216.0f) * 2.0f - 1.0f;   /* [-1, 1] */
}
