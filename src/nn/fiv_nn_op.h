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

#ifndef _FIV_NN_OP_H_
#define _FIV_NN_OP_H_

#include "fiv_nn.h"
#include "fiv_matrix.h"

/* fiv_nn_op_base must be the first member of every op struct so the engine
   can upcast any op pointer to fiv_nn_op_base* for dispatch. */
typedef void*  (*fiv_nn_op_create_fn)(void* params);
typedef void   (*fiv_nn_op_release_fn)(void* op_state);
typedef fiv_ret(*fiv_nn_op_forward_fn)(void* op_state, void* output, void* input);
typedef fiv_ret(*fiv_nn_op_backward_fn)(void* op_state, void* grad_input, const void* grad_output, const void* input);
typedef fiv_ret(*fiv_nn_op_inference_fn)(void* op_state, void* output, void* input);
typedef void*  (*fiv_nn_op_alloc_out_fn)(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret);

typedef struct fiv_nn_op_base {
    fiv_nn_op_create_fn     create_fn;
    fiv_nn_op_release_fn    release_fn;
    fiv_nn_op_forward_fn    forward_fn;    /* training forward (may cache state) */
    fiv_nn_op_backward_fn   backward_fn;   /* training backward (accumulates into grad_input) */
    fiv_nn_op_inference_fn  inference_fn;  /* inference forward */
    fiv_nn_op_alloc_out_fn  alloc_out_fn;
} fiv_nn_op_base;

#endif  /* _FIV_NN_OP_H_ */
