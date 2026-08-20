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



#ifndef   _FIV_COMMON_H_
#define   _FIV_COMMON_H_

#include <stddef.h>
#include "fiv_data_typedefs.h"



    /************************************************************************/
    /*
    deterministic LCG in [-1, 1] shared by weight initializers; the shared
    seed stream keeps training reproducible across nodes
    */
    /************************************************************************/
    float fiv_nn_rand(void);


#ifdef __cplusplus
extern "C" {
#endif


    /************************************************************************/
    /*
    memory allocate and free functions
    */
    /************************************************************************/
    void* fiv_malloc(size_t size);
    void* fiv_calloc(size_t num, size_t size);
    void  fiv_free(void *p);
    void* fiv_realloc(void* p, size_t size);






#ifdef __cplusplus
}
#endif


#endif