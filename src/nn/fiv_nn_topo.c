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

#include "fiv_nn_topo.h"
#include "fiv_common.h"

fiv_ret fiv_nn_topo_sort(fiv_nn_network_context* net)
{
    int n = net->node_count;
    int* indeg = (int*)fiv_calloc((size_t)n, sizeof(int));
    if (!indeg) return FIV_RET_ERR_MEM;

    for (int i = 1; i < n; i++) {
        int s = net->nodes[i].input_src;
        if (s >= 0 && s < n) indeg[i]++;
    }

    int* q = (int*)fiv_malloc(sizeof(int) * (size_t)n);
    if (!q) { fiv_free(indeg); return FIV_RET_ERR_MEM; }
    int head = 0, tail = 0;
    for (int i = 0; i < n; i++) if (indeg[i] == 0) q[tail++] = i;

    int cnt = 0;
    while (head < tail) {
        int u = q[head++];
        net->topo_order[cnt++] = u;
        for (int j = 1; j < n; j++) {
            if (net->nodes[j].input_src == u && --indeg[j] == 0) q[tail++] = j;
        }
    }
    fiv_free(indeg);
    fiv_free(q);

    net->topo_count = cnt;
    if (cnt != n) return FIV_RET_ERR_DATA_UNINITED;
    net->topo_valid = 1;
    return FIV_RET_OK;
}
