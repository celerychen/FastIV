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

#ifndef _FIV_NN_TRAIN_H_
#define _FIV_NN_TRAIN_H_

#include "fiv_nn.h"
#include "fiv_nn_infer.h"

#ifdef __cplusplus
extern "C" {
#endif

fiv_ret fiv_neural_network_train(void* nn_context, fiv_nn_train_params* nn_train_params, fiv_load_dataset_fn load_dataset);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_NN_TRAIN_H_ */
