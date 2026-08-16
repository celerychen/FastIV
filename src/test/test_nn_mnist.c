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

/* End-to-end correctness test on MNIST: train a 2-layer MLP (784->128 ReLU->10)
 * with cross entropy + SGD backprop (lr 0.05) and evaluate classification
 * accuracy on the test set (target > 97%). Everything goes through the single
 * fiv_test_dataset_callback interface: each call yields one batch, the training
 * framework loops until END_OF_FILE, and evaluation reuses the same callback.
 * If the data files are absent the test prints SKIP and passes. */

#include "fiv_nn.h"
#include "fiv_matrix.h"
#include "fiv_ctensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- MNIST IDX data loading, inlined here (test-only, MNIST-specific) ----
   fiv_test_dataset_callback is the single fiv_load_dataset_fn the training and
   evaluation loops call; each invocation fills one batch and returns OK, or
   FIV_RET_ERR_END_OF_FILE when the current epoch is exhausted. ---- */

#define FIV_TEST_DATASET_BATCH 64

typedef struct {
    const char* name;
    fiv_mat* inputs;        /* [total, feature] float32 in [0,1] */
    fiv_vec* labels;        /* class indices */
    int total;
    int feature;
    int pos;                /* batch cursor */
    int cur_epoch;          /* last shuffled epoch, -1 = not yet */
    fiv_tensor2d batch_in;  /* zero-copy view of the current batch */
    fiv_tensor1d batch_lab;
} fiv_test_dataset;

/* where each named set is loaded from (IDX file paths) */
typedef struct { const char* name; const char* img; const char* lab; int max; } fiv_ds_spec;
static const fiv_ds_spec g_specs[] = {
    { "mnist_train", "../data/mnist/train-images-idx3-ubyte",
                      "../data/mnist/train-labels-idx1-ubyte", 59968 },
    { "mnist_test",  "../data/mnist/t10k-images-idx3-ubyte",
                      "../data/mnist/t10k-labels-idx1-ubyte", 10000 },
};
#define G_N_SPECS ((int)(sizeof(g_specs) / sizeof(g_specs[0])))

static fiv_test_dataset g_datasets[8];
static int g_dataset_count = 0;

static unsigned rd_u32_be(const unsigned char* p)
{
    return ((unsigned)p[0] << 24) | ((unsigned)p[1] << 16) |
           ((unsigned)p[2] << 8) | (unsigned)p[3];
}

/* 32-bit LCG step. */
static unsigned fiv_ds_lcg_next(unsigned* s)
{
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

/* In-place Fisher-Yates over every sample, seeded by the epoch index so each
   epoch sees a different order. Input rows and the parallel labels swap together. */
static void fiv_test_dataset_shuffle(fiv_test_dataset* ds, int current_epoch_index)
{
    int n = ds->total;
    if (n <= 1) return;
    unsigned s = 0x9E3779B9u ^ ((unsigned)(current_epoch_index + 1) * 2654435761u);
    float* x = ds->inputs->data.fl;
    float* y = ds->labels->data.fl;
    int f = ds->feature;
    float* tmp = (float*)malloc(sizeof(float) * (size_t)f);
    if (!tmp) return;
    for (int i = n - 1; i > 0; i--) {
        unsigned j = fiv_ds_lcg_next(&s) % (unsigned)(i + 1);
        float* ri = x + (size_t)i * f;
        float* rj = x + (size_t)j * f;
        memcpy(tmp, ri, sizeof(float) * f);
        memcpy(ri, rj, sizeof(float) * f);
        memcpy(rj, tmp, sizeof(float) * f);
        float tl = y[i]; y[i] = y[j]; y[j] = tl;
    }
    free(tmp);
}

static float* load_idx_images(const char* path, int want, int* out_feature)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;
    unsigned char hdr[16];
    if (fread(hdr, 1, 16, fp) != 16) { fclose(fp); return NULL; }
    int n = (int)rd_u32_be(hdr + 4);
    int px = (int)rd_u32_be(hdr + 8) * (int)rd_u32_be(hdr + 12);
    if (n < want || px <= 0) { fclose(fp); return NULL; }

    float* x = (float*)malloc(sizeof(float) * (size_t)want * (size_t)px);
    unsigned char* buf = (unsigned char*)malloc((size_t)px);
    if (!x || !buf) { free(x); free(buf); fclose(fp); return NULL; }
    for (int i = 0; i < want; i++) {
        if (fread(buf, 1, (size_t)px, fp) != (size_t)px) {
            free(x); free(buf); fclose(fp); return NULL;
        }
        for (int k = 0; k < px; k++) x[(size_t)i * px + k] = buf[k] / 255.0f;
    }
    free(buf);
    fclose(fp);
    *out_feature = px;
    return x;
}

static float* load_idx_labels(const char* path, int want)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;
    unsigned char hdr[8];
    if (fread(hdr, 1, 8, fp) != 8) { fclose(fp); return NULL; }
    int n = (int)rd_u32_be(hdr + 4);
    if (n < want) { fclose(fp); return NULL; }

    float* y = (float*)malloc(sizeof(float) * (size_t)want);
    unsigned char* buf = (unsigned char*)malloc((size_t)want);
    if (!y || !buf) { free(y); free(buf); fclose(fp); return NULL; }
    if (fread(buf, 1, (size_t)want, fp) != (size_t)want) {
        free(y); free(buf); fclose(fp); return NULL;
    }
    for (int i = 0; i < want; i++) y[i] = (float)buf[i];
    free(buf);
    fclose(fp);
    return y;
}

/* Lazily load the named set from its spec into the internal registry.
   Returns the dataset index, or -1 if the files cannot be opened. */
static int fiv_test_dataset_load(const char* name)
{
    int idx = -1;
    for (int i = 0; i < G_N_SPECS; i++)
        if (strcmp(g_specs[i].name, name) == 0) { idx = i; break; }
    if (idx < 0 || g_dataset_count >= 8) return -1;

    int feature = 0;
    float* x = load_idx_images(g_specs[idx].img, g_specs[idx].max, &feature);
    float* y = load_idx_labels(g_specs[idx].lab, g_specs[idx].max);
    if (!x || !y || feature <= 0) { free(x); free(y); return -1; }

    size_t xsh[2] = { (size_t)g_specs[idx].max, (size_t)feature };
    fiv_mat* inputs = fiv_create_tensor2d(xsh, FIV_32F1);
    fiv_vec* labels = fiv_create_tensor1d((size_t)g_specs[idx].max, FIV_32F1);
    if (!inputs || !labels) {
        if (inputs) fiv_release_tensor((void**)&inputs);
        if (labels) fiv_release_tensor((void**)&labels);
        free(x); free(y);
        return -1;
    }
    memcpy(inputs->data.ptr, x, sizeof(float) * (size_t)g_specs[idx].max * (size_t)feature);
    memcpy(labels->data.ptr, y, sizeof(float) * (size_t)g_specs[idx].max);
    free(x);
    free(y);

    fiv_test_dataset* ds = &g_datasets[g_dataset_count];
    ds->name = g_specs[idx].name;
    ds->inputs = inputs;
    ds->labels = labels;
    ds->total = g_specs[idx].max;
    ds->feature = feature;
    ds->pos = 0;
    ds->cur_epoch = -1;
    return g_dataset_count++;
}

static fiv_ret fiv_test_dataset_callback(fiv_dadaset_info* data_set_info)
{
    if (!data_set_info) return FIV_RET_ERR_PARA;

    /* name == NULL means the default (first) set, used by training */
    const char* name = data_set_info->data_set_name ? data_set_info->data_set_name
                                                     : g_specs[0].name;

    fiv_test_dataset* ds = NULL;
    for (int i = 0; i < g_dataset_count; i++) {
        if (strcmp(g_datasets[i].name, name) == 0) { ds = &g_datasets[i]; break; }
    }
    if (!ds) {
        int di = fiv_test_dataset_load(name);
        if (di < 0) return FIV_RET_DATA_NOT_FOUND;
        ds = &g_datasets[di];
    }

    /* reshuffle once per epoch so consecutive epochs see different orders */
    if (ds->cur_epoch != data_set_info->current_epoch_index) {
        fiv_test_dataset_shuffle(ds, data_set_info->current_epoch_index);
        ds->cur_epoch = data_set_info->current_epoch_index;
        ds->pos = 0;
    }

    if (ds->pos >= ds->total) return FIV_RET_ERR_END_OF_FILE;

    int n = ds->total - ds->pos;
    if (n > FIV_TEST_DATASET_BATCH) n = FIV_TEST_DATASET_BATCH;

    size_t off2[2] = { (size_t)ds->pos, 0 };
    size_t sz2[2]  = { (size_t)n, (size_t)ds->feature };
    fiv_ret r = fiv_tensor_view(&ds->batch_in, ds->inputs, off2, sz2);
    if (r != FIV_RET_OK) return r;
    size_t off1[1] = { (size_t)ds->pos };
    size_t sz1[1]  = { (size_t)n };
    r = fiv_tensor_view(&ds->batch_lab, ds->labels, off1, sz1);
    if (r != FIV_RET_OK) return r;
    ds->pos += n;

    data_set_info->data_set_name   = (char*)ds->name;
    data_set_info->total_data_size = ds->total;
    data_set_info->feature_size    = ds->feature;
    data_set_info->label_size      = 1;
    data_set_info->current_batch_inputs   = &ds->batch_in;
    data_set_info->current_batch_outputs = &ds->batch_lab;
    return FIV_RET_OK;
}

static int g_fail = 0;
static int g_pass = 0;
#define CHECK(c, msg)                                                           \
    do {                                                                        \
        if (!(c)) { printf("  [FAIL] %s @%d\n", (msg), __LINE__); g_fail++; }   \
        else       { g_pass++; }                                                \
    } while (0)

#define BATCH  64
#define PX     784

int main(void)
{
    void* net = fiv_create_neural_network();
    fiv_linear_node_params p1 = { 784, 128 };
    fiv_linear_node_params p2 = { 128, 10 };
    CHECK(fiv_neural_network_add_node(net, FIV_NN_NODE_LINEAR, 0, 1, &p1) == FIV_RET_OK, "mnist: add LINEAR 784->128");
    CHECK(fiv_neural_network_add_node(net, FIV_NN_NODE_RELU, 1, 2, NULL) == FIV_RET_OK, "mnist: add RELU");
    CHECK(fiv_neural_network_add_node(net, FIV_NN_NODE_LINEAR, 2, 3, &p2) == FIV_RET_OK, "mnist: add LINEAR 128->10");

    /* train through the single callback; missing data files => SKIP */
    fiv_nn_train_params tp = { 14, BATCH, 0.05f, 0 };   /* 0 = cross entropy */
    fiv_ret tr = fiv_neural_network_train(net, &tp, fiv_test_dataset_callback);
    if (tr == FIV_RET_DATA_NOT_FOUND) {
        printf("  [SKIP] mnist data files not found under ../data/mnist/\n");
        void* netp = net;
        fiv_release_neural_network(&netp);
        printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
        return 0;
    }
    CHECK(tr == FIV_RET_OK, "mnist: train OK");

    /* verify per-epoch shuffle: first-sample label sequence differs between
       epoch 0 and epoch 1 (driven purely via the callback) */
    {
        int e0[8], e1[8];
        fiv_dadaset_info gi;
        for (int k = 0; k < 8; k++) {
            memset(&gi, 0, sizeof(gi));
            gi.data_set_name = "mnist_train";
            gi.current_epoch_index = 0;
            CHECK(fiv_test_dataset_callback(&gi) == FIV_RET_OK, "shuffle: epoch 0 batch");
            e0[k] = (int)((fiv_tensor1d*)gi.current_batch_outputs)->data.fl[0];
        }
        for (int k = 0; k < 8; k++) {
            memset(&gi, 0, sizeof(gi));
            gi.data_set_name = "mnist_train";
            gi.current_epoch_index = 1;
            CHECK(fiv_test_dataset_callback(&gi) == FIV_RET_OK, "shuffle: epoch 1 batch");
            e1[k] = (int)((fiv_tensor1d*)gi.current_batch_outputs)->data.fl[0];
        }
        int differs = 0;
        for (int k = 0; k < 8; k++) if (e0[k] != e1[k]) differs = 1;
        CHECK(differs, "shuffle: epoch 0 and 1 sample orders differ");
    }

    /* evaluate on the test set, again only through the callback */
    int correct = 0;
    int eval_ok = 1;
    size_t ish[2] = { 1, PX };
    size_t osh[2] = { 1, 10 };
    fiv_mat* in = fiv_create_tensor2d(ish, FIV_32F1);
    fiv_mat* out = fiv_create_tensor2d(osh, FIV_32F1);
    fiv_dadaset_info gi;
    memset(&gi, 0, sizeof(gi));
    gi.data_set_name = "mnist_test";
    gi.current_epoch_index = 0;
    for (;;) {
        fiv_ret rr = fiv_test_dataset_callback(&gi);
        if (rr == FIV_RET_ERR_END_OF_FILE) break;
        if (rr != FIV_RET_OK) { eval_ok = 0; break; }
        fiv_tensor2d* bin = (fiv_tensor2d*)gi.current_batch_inputs;
        fiv_tensor1d* blab = (fiv_tensor1d*)gi.current_batch_outputs;
        int rows = (int)bin->shapes[0];
        for (int i = 0; i < rows; i++) {
            memcpy(in->data.fl, bin->data.fl + (size_t)i * PX, sizeof(float) * PX);
            if (fiv_neural_network_inference(out, in, net) != FIV_RET_OK) { eval_ok = 0; break; }
            int pred = 0;
            for (int j = 1; j < 10; j++) if (out->data.fl[j] > out->data.fl[pred]) pred = j;
            if (pred == (int)blab->data.fl[i]) correct++;
        }
        if (!eval_ok) break;
    }
    CHECK(eval_ok, "mnist: test inference OK");
    printf("  mnist test accuracy: %d/10000\n", correct);
    CHECK(correct > 10000 * 97 / 100, "mnist: test accuracy > 97%");

    fiv_release_tensor((void**)&in);
    fiv_release_tensor((void**)&out);
    void* netp = net;
    fiv_release_neural_network(&netp);

    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
