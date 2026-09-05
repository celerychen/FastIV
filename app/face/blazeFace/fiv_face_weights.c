/*
 * FastIV - Fast image and vision
 * Copyright (C) 2026 Celery Chen
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * See LICENSE file in project root for full license text.
 *
 * BlazeFace weights loader. Self-contained replacement of reference weights.c;
 * no dependency on src/reference/.
 */

#include "fiv_face_weights.h"

#include "fiv_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIV_WT_MAGIC 0x424C5A46u

static int fiv_read_u32(FILE* f, unsigned int* v) {
    unsigned char byte_buf[4];
    if (fread(byte_buf, 1, 4, f) != 4) return 0;
    *v = ((unsigned int)byte_buf[0])       | ((unsigned int)byte_buf[1] << 8) |
         ((unsigned int)byte_buf[2] << 16) | ((unsigned int)byte_buf[3] << 24);
    return 1;
}

int fiv_weights_load(const char* path, fiv_weights* w) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "fiv_weights_load: cannot open %s\n", path); return 0; }

    unsigned int magic = 0, num_tensors = 0;
    if (!fiv_read_u32(f, &magic) || magic != FIV_WT_MAGIC) {
        fprintf(stderr, "fiv_weights_load: bad magic (not blazeface_weights.bin)\n");
        fclose(f); return 0;
    }
    if (!fiv_read_u32(f, &num_tensors)) { fclose(f); return 0; }

    w->n     = (int)num_tensors;
    w->items = (fiv_weight_tensor*)fiv_calloc((size_t)num_tensors, sizeof(fiv_weight_tensor));
    if (!w->items) { fclose(f); return 0; }

    unsigned int tensor_idx = 0;
    for (tensor_idx = 0; tensor_idx < num_tensors; tensor_idx++) {
        fiv_weight_tensor* tensor = &w->items[tensor_idx];
        unsigned int name_len = 0;
        if (!fiv_read_u32(f, &name_len)) goto fail;
        if (name_len >= FIV_WT_NAME_MAX) name_len = FIV_WT_NAME_MAX - 1;
        if (fread(tensor->name, 1, name_len, f) != name_len) goto fail;
        tensor->name[name_len] = '\0';

        unsigned int ndim = 0;
        if (!fiv_read_u32(f, &ndim)) goto fail;
        tensor->ndim = (int)ndim;

        unsigned int dim_buf[4] = {0, 0, 0, 0};
        for (int k = 0; k < 4; k++) {
            if (!fiv_read_u32(f, &dim_buf[k])) goto fail;
            tensor->dims[k] = (int)dim_buf[k];
        }

        size_t elems = 1;
        for (int k = 0; k < tensor->ndim; k++) elems *= (size_t)tensor->dims[k];
        tensor->data = (ivf32*)fiv_malloc(elems * sizeof(ivf32));
        if (!tensor->data) goto fail;
        if (fread(tensor->data, sizeof(ivf32), elems, f) != elems) goto fail;
    }
    fclose(f);
    return 1;

fail:
    fprintf(stderr, "fiv_weights_load: corrupt at tensor %u (%s)\n", num_tensors,
            (tensor_idx < num_tensors) ? w->items[tensor_idx].name : "");
    fclose(f);
    fiv_weights_free(w);
    return 0;
}

const ivf32* fiv_weights_get(const fiv_weights* w, const char* name,
                             int* ndim, int* dims) {
    for (int i = 0; i < w->n; i++) {
        if (strcmp(w->items[i].name, name) == 0) {
            if (ndim) *ndim = w->items[i].ndim;
            if (dims) for (int k = 0; k < 4; k++) dims[k] = w->items[i].dims[k];
            return w->items[i].data;
        }
    }
    return NULL;
}

void fiv_weights_free(fiv_weights* w) {
    if (!w) return;
    if (w->items) {
        for (int i = 0; i < w->n; i++) fiv_free(w->items[i].data);
        fiv_free(w->items);
        w->items = NULL;
    }
    w->n = 0;
}
