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

#ifndef _FIV_NN_TOPO_H_
#define _FIV_NN_TOPO_H_

#include "fiv_nn_infer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Kahn topological sort (shared by inference and training). A cycle or a
   node unreachable from node 0 (isolated subgraph) leaves fewer than
   node_count ids and returns an error. */
fiv_ret fiv_nn_topo_sort(fiv_nn_network_context* net);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_NN_TOPO_H_ */
