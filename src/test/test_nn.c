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

/* Correctness tests for the inference-only neural network (api/fiv_nn.h).
 * Only public API headers are used: architecture assembly, argument
 * validation, zero-weight inference, model save/load (numeric checks via
 * hand-written model files), shape-mismatch rejection and release. */

#include "fiv_nn.h"
#include "fiv_matrix.h"
#include "fiv_ctensor.h"
#include <stdio.h>
#include <string.h>

static int g_fail = 0;
static int g_pass = 0;
#define CHECK(c, msg)                                                           \
    do {                                                                        \
        if (!(c)) { printf("  [FAIL] %s @%d\n", (msg), __LINE__); g_fail++; }   \
        else       { g_pass++; }                                                \
    } while (0)
static float fabsf_local(float x) { return x < 0 ? -x : x; }
#define NEAR(a, b) (fabsf_local((a) - (b)) < 1e-4f)

#define MODEL_MAGIC 0x4649564Eu   /* "FIVN" */

/* Model file v1: int32 magic, version, node_count, then per node:
   int32 node_type, index_start, in_features, out_features; LINEAR appends
   float weight[out*in] and float bias[out]. */
static void put_i32(FILE* fp, int v) { fwrite(&v, 4, 1, fp); }

static int write_model_linear(const char* path)
{
    FILE* fp = fopen(path, "wb");
    if (!fp) return 0;
    put_i32(fp, (int)MODEL_MAGIC);
    put_i32(fp, 1);
    put_i32(fp, 2);                    /* node 0 + node 1 */
    put_i32(fp, FIV_NN_NODE_LINEAR);   /* node 1: LINEAR, src 0, 3 -> 2 */
    put_i32(fp, 0);
    put_i32(fp, 3);
    put_i32(fp, 2);
    float W[6] = { 1, 2, 3, 4, 5, 6 };
    float B[2] = { 0.5f, -1.0f };
    fwrite(W, 4, 6, fp);
    fwrite(B, 4, 2, fp);
    fclose(fp);
    return 1;
}

static int write_model_linear_relu(const char* path)
{
    FILE* fp = fopen(path, "wb");
    if (!fp) return 0;
    put_i32(fp, (int)MODEL_MAGIC);
    put_i32(fp, 1);
    put_i32(fp, 3);                    /* node 0 + nodes 1,2 */
    put_i32(fp, FIV_NN_NODE_LINEAR);   /* node 1: LINEAR, src 0, 3 -> 2 */
    put_i32(fp, 0);
    put_i32(fp, 3);
    put_i32(fp, 2);
    float W[6] = { 1, 2, 3, 4, 5, 6 };
    float B[2] = { 0.5f, -1.0f };
    fwrite(W, 4, 6, fp);
    fwrite(B, 4, 2, fp);
    put_i32(fp, FIV_NN_NODE_RELU);     /* node 2: RELU, src 1 */
    put_i32(fp, 1);
    put_i32(fp, -1);
    put_i32(fp, -1);
    fclose(fp);
    return 1;
}

static int write_model_linear_relu6(const char* path)
{
    FILE* fp = fopen(path, "wb");
    if (!fp) return 0;
    put_i32(fp, (int)MODEL_MAGIC);
    put_i32(fp, 1);
    put_i32(fp, 3);                    /* node 0 + nodes 1,2 */
    put_i32(fp, FIV_NN_NODE_LINEAR);   /* node 1: LINEAR, src 0, 3 -> 2 */
    put_i32(fp, 0);
    put_i32(fp, 3);
    put_i32(fp, 2);
    float W[6] = { 1, 2, 3, 4, 5, 6 };
    float B[2] = { 0.5f, -1.0f };
    fwrite(W, 4, 6, fp);
    fwrite(B, 4, 2, fp);
    put_i32(fp, FIV_NN_NODE_RELU6);    /* node 2: RELU6, src 1 */
    put_i32(fp, 1);
    put_i32(fp, -1);
    put_i32(fp, -1);
    fclose(fp);
    return 1;
}

static int write_model_bad_magic(const char* path)
{
    FILE* fp = fopen(path, "wb");
    if (!fp) return 0;
    put_i32(fp, 0xDEADBEEFu);
    put_i32(fp, 1);
    put_i32(fp, 2);
    fclose(fp);
    return 1;
}

static int write_model_bad_type(const char* path)
{
    FILE* fp = fopen(path, "wb");
    if (!fp) return 0;
    put_i32(fp, (int)MODEL_MAGIC);
    put_i32(fp, 1);
    put_i32(fp, 2);
    put_i32(fp, 99);                   /* unknown node type */
    put_i32(fp, 0);
    put_i32(fp, -1);
    put_i32(fp, -1);
    fclose(fp);
    return 1;
}

static fiv_mat* mk_in(void)
{
    size_t sh[2] = { 2, 3 };
    fiv_mat* m = fiv_create_tensor2d(sh, FIV_32F1);
    float v[6] = { 1, 0, -1, 2, 1, 0 };
    memcpy(m->data.fl, v, sizeof(v));
    return m;
}

static fiv_mat* mk_out(void)
{
    size_t sh[2] = { 2, 2 };
    return fiv_create_tensor2d(sh, FIV_32F1);
}

/* ---- training: a toy data set; the callback hands out one batch per call
   (the whole 4-sample set in one batch) and signals EOF with cursor reset ---- */
static fiv_mat* g_ds_in;   /* [N, feature] */
static void*   g_ds_lab;   /* fiv_vec [N] (CE) or fiv_mat [N, out] (MSE) */
static int     g_ds_total;
static int     g_ds_feature;
static int     g_ds_pos;

static fiv_ret test_ds_callback(fiv_dadaset_info* info)
{
    if (g_ds_pos >= g_ds_total) {
        g_ds_pos = 0;   /* reset: next epoch replays */
        return FIV_RET_ERR_END_OF_FILE;
    }
    g_ds_pos = g_ds_total;
    info->data_set_name  = "toy";
    info->total_data_size = g_ds_total;
    info->feature_size    = g_ds_feature;
    info->label_size      = 1;
    info->current_batch_inputs   = g_ds_in;
    info->current_batch_outputs = g_ds_lab;
    return FIV_RET_OK;
}

int main(void)
{
    /* ---- assembly, validation, zero-weight inference (public API) ---- */
    void* net = fiv_create_neural_network();
    CHECK(net != NULL, "network create");
    fiv_linear_node_params p32 = { 3, 2 };
    fiv_linear_node_params p22 = { 2, 2 };
    CHECK(fiv_neural_network_add_node(net, FIV_NN_NODE_LINEAR, 0, 1, &p32) == FIV_RET_OK, "add LINEAR 0->1");
    CHECK(fiv_neural_network_add_node(net, FIV_NN_NODE_RELU, 1, 2, NULL) == FIV_RET_OK, "add RELU 1->2");
    CHECK(fiv_neural_network_add_node(net, FIV_NN_NODE_RELU, 2, 3, NULL) == FIV_RET_OK, "add RELU 2->3");
    CHECK(fiv_neural_network_add_node(net, FIV_NN_NODE_LINEAR, 0, 4, &p32) == FIV_RET_OK, "add branch LINEAR 0->4");
    CHECK(fiv_neural_network_add_node(net, FIV_NN_NODE_LINEAR, 3, 4, &p32) == FIV_RET_ERR_PARA, "non-sequential index_end rejected");
    CHECK(fiv_neural_network_add_node(net, 99, 4, 5, NULL) == FIV_RET_ERR_PARA, "bad node_type rejected");
    CHECK(fiv_neural_network_add_node(net, FIV_NN_NODE_LINEAR, 4, 5, &p22) == FIV_RET_OK, "add LINEAR 4->5");
    CHECK(fiv_neural_network_add_node(net, FIV_NN_NODE_LINEAR, 99, 6, &p22) == FIV_RET_ERR_PARA, "index_start >= index_end rejected");

    fiv_mat* in = mk_in();
    fiv_mat* out = mk_out();
    memset(out->data.fl, 0x7f, 2 * 2 * sizeof(float));
    CHECK(fiv_neural_network_inference(out, in, net) == FIV_RET_OK, "network inference run 1");
    float r1[4];
    memcpy(r1, out->data.fl, sizeof(r1));
    CHECK(r1[0] != 0.0f || r1[1] != 0.0f || r1[2] != 0.0f || r1[3] != 0.0f, "random init: output non-zero");
    CHECK(fiv_neural_network_inference(out, in, net) == FIV_RET_OK, "network inference run 2 (reuse)");
    CHECK(out->data.fl[0] == r1[0] && out->data.fl[1] == r1[1]
       && out->data.fl[2] == r1[2] && out->data.fl[3] == r1[3], "run 2 deterministic (same output)");

    /* shape-inconsistent chain (LINEAR 2-out feeds LINEAR 3-in) is rejected */
    void* net2 = fiv_create_neural_network();
    fiv_neural_network_add_node(net2, FIV_NN_NODE_LINEAR, 0, 1, &p32);
    CHECK(fiv_neural_network_add_node(net2, FIV_NN_NODE_LINEAR, 1, 2, &p32) == FIV_RET_OK, "add shape-mismatch LINEAR 1->2");
    fiv_mat* out2 = mk_out();
    CHECK(fiv_neural_network_inference(out2, in, net2) == FIV_RET_ERR_PARA, "inference rejects shape-mismatched chain");
    fiv_release_tensor((void**)&out2);
    void* net2p = net2;
    fiv_release_neural_network(&net2p);

    fiv_release_tensor((void**)&in);
    fiv_release_tensor((void**)&out);
    void* netp = net;
    CHECK(fiv_release_neural_network(&netp) == FIV_RET_OK && netp == NULL, "network release nulls ctx");

    /* ---- model load: hand-written LINEAR model -> y = x*W^T + b ---- */
    CHECK(write_model_linear("fiv_nn_m_linear.bin"), "write LINEAR model file");
    void* mnet = fiv_create_neural_network_from_model("fiv_nn_m_linear.bin");
    CHECK(mnet != NULL, "from_model loads LINEAR model");
    in = mk_in();
    out = mk_out();
    CHECK(fiv_neural_network_inference(out, in, mnet) == FIV_RET_OK, "loaded model inference");
    CHECK(NEAR(out->data.fl[0], -1.5f) && NEAR(out->data.fl[1], -3.0f)
       && NEAR(out->data.fl[2], 4.5f) && NEAR(out->data.fl[3], 12.0f), "loaded model linear numerics y=xW^T+b");

    /* ---- model save -> reload roundtrip gives the same result ---- */
    CHECK(fiv_neural_network_save_model("fiv_nn_m_saved.bin", mnet) == FIV_RET_OK, "save_model writes file");
    void* mnet2 = fiv_create_neural_network_from_model("fiv_nn_m_saved.bin");
    CHECK(mnet2 != NULL, "from_model reloads saved model");
    fiv_mat* out_r = mk_out();
    CHECK(fiv_neural_network_inference(out_r, in, mnet2) == FIV_RET_OK, "reloaded model inference");
    int same = NEAR(out_r->data.fl[0], out->data.fl[0]) && NEAR(out_r->data.fl[1], out->data.fl[1])
            && NEAR(out_r->data.fl[2], out->data.fl[2]) && NEAR(out_r->data.fl[3], out->data.fl[3]);
    CHECK(same, "save/load roundtrip output identical");
    fiv_release_tensor((void**)&out_r);
    void* m2p = mnet2;
    fiv_release_neural_network(&m2p);

    /* ---- model with RELU: relu applied after linear ---- */
    CHECK(write_model_linear_relu("fiv_nn_m_relu.bin"), "write LINEAR+RELU model file");
    void* rnet = fiv_create_neural_network_from_model("fiv_nn_m_relu.bin");
    CHECK(rnet != NULL, "from_model loads LINEAR+RELU model");
    CHECK(fiv_neural_network_inference(out, in, rnet) == FIV_RET_OK, "LINEAR+RELU model inference");
    CHECK(NEAR(out->data.fl[0], 0.0f) && NEAR(out->data.fl[1], 0.0f)
       && NEAR(out->data.fl[2], 4.5f) && NEAR(out->data.fl[3], 12.0f), "relu in network (clamps neg, no cap)");

    /* ---- model with RELU6: same weights, but caps at 6 (distinct from RELU) ---- */
    CHECK(write_model_linear_relu6("fiv_nn_m_relu6.bin"), "write LINEAR+RELU6 model file");
    void* r6net = fiv_create_neural_network_from_model("fiv_nn_m_relu6.bin");
    CHECK(r6net != NULL, "from_model loads LINEAR+RELU6 model");
    CHECK(fiv_neural_network_inference(out, in, r6net) == FIV_RET_OK, "LINEAR+RELU6 model inference");
    CHECK(NEAR(out->data.fl[0], 0.0f) && NEAR(out->data.fl[1], 0.0f)
       && NEAR(out->data.fl[2], 4.5f) && NEAR(out->data.fl[3], 6.0f), "relu6 in network (clamps neg, caps at 6)");
    void* r6p = r6net;
    fiv_release_neural_network(&r6p);

    /* ---- bad files -> NULL ---- */
    CHECK(fiv_create_neural_network_from_model("fiv_nn_no_such_file.bin") == NULL, "missing file -> NULL");
    CHECK(write_model_bad_magic("fiv_nn_m_badmagic.bin"), "write bad-magic file");
    CHECK(fiv_create_neural_network_from_model("fiv_nn_m_badmagic.bin") == NULL, "bad magic -> NULL");
    CHECK(write_model_bad_type("fiv_nn_m_badtype.bin"), "write bad-type file");
    CHECK(fiv_create_neural_network_from_model("fiv_nn_m_badtype.bin") == NULL, "bad node type -> NULL");

    /* ---- train: parameter validation ---- */
    void* tnet = fiv_create_neural_network();
    fiv_linear_node_params p31 = { 3, 1 };
    CHECK(fiv_neural_network_add_node(tnet, FIV_NN_NODE_LINEAR, 0, 1, &p31) == FIV_RET_OK, "train: add LINEAR 3->1");
    fiv_nn_train_params bad;
    bad.epoch_num = 0; bad.bach_size = 4; bad.learning_rate = 0.05f; bad.loss_fn_type = 1;
    CHECK(fiv_neural_network_train(tnet, &bad, test_ds_callback) == FIV_RET_ERR_PARA, "train: epoch<=0 rejected");
    bad.epoch_num = 300; bad.learning_rate = 0.0f;
    CHECK(fiv_neural_network_train(tnet, &bad, test_ds_callback) == FIV_RET_ERR_PARA, "train: lr<=0 rejected");
    bad.learning_rate = 0.05f; bad.loss_fn_type = 2;
    CHECK(fiv_neural_network_train(tnet, &bad, test_ds_callback) == FIV_RET_ERR_PARA, "train: bad loss_fn_type rejected");
    CHECK(fiv_neural_network_train(tnet, &bad, NULL) == FIV_RET_ERR_PARA, "train: null callback rejected");

    /* ---- train: MSE linear regression converges (numerical-gradient SGD) ---- */
    size_t rxsh[2] = { 4, 3 };
    fiv_mat* rx = fiv_create_tensor2d(rxsh, FIV_32F1);
    float RX[12] = { 1,2,3, 0,-1,2, 2,0,1, -1,1,0 };
    memcpy(rx->data.fl, RX, sizeof(RX));
    size_t rysh[2] = { 4, 1 };
    fiv_mat* ry = fiv_create_tensor2d(rysh, FIV_32F1);
    float Wstar[3] = { 0.5f, -1.0f, 2.0f };
    for (int i = 0; i < 4; i++) {
        float y = 0.25f;
        for (int j = 0; j < 3; j++) y += RX[i * 3 + j] * Wstar[j];
        ry->data.fl[i] = y;
    }
    g_ds_in = rx;
    g_ds_lab = ry;
    g_ds_total = 4;
    g_ds_feature = 3;
    g_ds_pos = 0;
    fiv_nn_train_params good = { 300, 4, 0.05f, 1 };
    fiv_ret tr = fiv_neural_network_train(tnet, &good, test_ds_callback);
    printf("  [dbg] MSE train ret=%d\n", (int)tr);
    CHECK(tr == FIV_RET_OK, "train: MSE regression OK");
    size_t p1sh[2] = { 4, 1 };
    fiv_mat* pred1 = fiv_create_tensor2d(p1sh, FIV_32F1);
    CHECK(fiv_neural_network_inference(pred1, rx, tnet) == FIV_RET_OK, "train: post-train inference");
    int fit = 1;
    for (int i = 0; i < 4; i++)
        if (fabsf_local(pred1->data.fl[i] - ry->data.fl[i]) > 0.05f) fit = 0;
    CHECK(fit, "train: regression converged (|pred-y|<0.05)");

    /* ---- train: CE classification (2 classes, linear separable) ---- */
    void* cnet = fiv_create_neural_network();
    fiv_linear_node_params p22c = { 2, 2 };
    fiv_neural_network_add_node(cnet, FIV_NN_NODE_LINEAR, 0, 1, &p22c);
    size_t cxsh[2] = { 4, 2 };
    fiv_mat* cx = fiv_create_tensor2d(cxsh, FIV_32F1);
    float CX[8] = { 0,0, 0,1, 1,0, 1,1 };
    memcpy(cx->data.fl, CX, sizeof(CX));
    fiv_vec* cy = fiv_create_tensor1d(4, FIV_32F1);
    float CY[4] = { 0, 0, 1, 1 };
    memcpy(cy->data.fl, CY, sizeof(CY));
    g_ds_in = cx;
    g_ds_lab = cy;
    g_ds_total = 4;
    g_ds_feature = 2;
    g_ds_pos = 0;
    fiv_nn_train_params cp = { 800, 4, 0.3f, 0 };
    CHECK(fiv_neural_network_train(cnet, &cp, test_ds_callback) == FIV_RET_OK, "train: CE classification OK");
    size_t cysh[2] = { 4, 2 };
    fiv_mat* cpred = fiv_create_tensor2d(cysh, FIV_32F1);
    CHECK(fiv_neural_network_inference(cpred, cx, cnet) == FIV_RET_OK, "train: CE post-train inference");
    int cls_ok = 1;
    for (int i = 0; i < 4; i++) {
        float* row = cpred->data.fl + i * 2;
        int a = row[0] > row[1] ? 0 : 1;
        if (a != (int)CY[i]) cls_ok = 0;
    }
    CHECK(cls_ok, "train: CE classifies all samples");

    /* ---- train: MLP (LINEAR->RELU->LINEAR) regression, BP through relu ---- */
    void* mlp = fiv_create_neural_network();
    fiv_linear_node_params p34 = { 3, 8 };
    fiv_linear_node_params p41 = { 8, 1 };
    CHECK(fiv_neural_network_add_node(mlp, FIV_NN_NODE_LINEAR, 0, 1, &p34) == FIV_RET_OK, "train MLP: add LINEAR 3->8");
    CHECK(fiv_neural_network_add_node(mlp, FIV_NN_NODE_RELU, 1, 2, NULL) == FIV_RET_OK, "train MLP: add RELU");
    CHECK(fiv_neural_network_add_node(mlp, FIV_NN_NODE_LINEAR, 2, 3, &p41) == FIV_RET_OK, "train MLP: add LINEAR 8->1");
    g_ds_in = rx;
    g_ds_lab = ry;
    g_ds_total = 4;
    g_ds_feature = 3;
    g_ds_pos = 0;
    fiv_nn_train_params mparams = { 3000, 4, 0.05f, 1 };
    CHECK(fiv_neural_network_train(mlp, &mparams, test_ds_callback) == FIV_RET_OK, "train MLP: regression OK");
    size_t mp1sh[2] = { 4, 1 };
    fiv_mat* mpred = fiv_create_tensor2d(mp1sh, FIV_32F1);
    CHECK(fiv_neural_network_inference(mpred, rx, mlp) == FIV_RET_OK, "train MLP: post-train inference");
    int mfit = 1;
    for (int i = 0; i < 4; i++)
        if (fabsf_local(mpred->data.fl[i] - ry->data.fl[i]) > 0.05f) mfit = 0;
    CHECK(mfit, "train MLP: regression converged (|pred-y|<0.05)");

    /* ---- trained-model roundtrip: save -> load -> infer must match in-memory infer ---- */
    CHECK(fiv_neural_network_save_model("fiv_nn_mlp_trained.bin", mlp) == FIV_RET_OK, "train MLP: save trained model");
    void* mlp_ld = fiv_create_neural_network_from_model("fiv_nn_mlp_trained.bin");
    CHECK(mlp_ld != NULL, "train MLP: load trained model");
    fiv_mat* mloaded = fiv_create_tensor2d(mp1sh, FIV_32F1);
    CHECK(fiv_neural_network_inference(mloaded, rx, mlp_ld) == FIV_RET_OK, "train MLP: loaded-model inference");
    int mrt = 1;
    for (int i = 0; i < 4; i++)
        if (fabsf_local(mloaded->data.fl[i] - mpred->data.fl[i]) > 1e-4f) mrt = 0;
    CHECK(mrt, "train MLP: loaded-model inference matches in-memory");
    void* mldp = mlp_ld;
    fiv_release_neural_network(&mldp);
    fiv_release_tensor((void**)&mloaded);

    void* mnp = mlp;
    fiv_release_neural_network(&mnp);
    fiv_release_tensor((void**)&mpred);

    void* tnp = tnet;
    fiv_release_neural_network(&tnp);
    void* cnp = cnet;
    fiv_release_neural_network(&cnp);
    fiv_release_tensor((void**)&rx);
    fiv_release_tensor((void**)&ry);
    fiv_release_tensor((void**)&pred1);
    fiv_release_tensor((void**)&cx);
    fiv_release_tensor((void**)&cy);
    fiv_release_tensor((void**)&cpred);

    void* mp = mnet;
    CHECK(fiv_release_neural_network(&mp) == FIV_RET_OK && mp == NULL, "model network release nulls ctx");
    fiv_release_tensor((void**)&in);
    fiv_release_tensor((void**)&out);
    void* rp = rnet;
    fiv_release_neural_network(&rp);

    /* ---- ReLU6 / ReLU are dimension-agnostic: 3D and 4D tensors (public API) ---- */
    {
        size_t s3[3] = { 2, 2, 3 };
        float raw3[12] = { 1,-2, 3, 0, -4, 5, -6, 7, 8, -9, 0, 2 };
        void* rn3 = fiv_create_neural_network();
        CHECK(fiv_neural_network_add_node(rn3, FIV_NN_NODE_RELU6, 0, 1, NULL) == FIV_RET_OK, "relu6 3D graph add");
        fiv_tensor3d* in3 = fiv_create_tensor3d(s3, FIV_32F1);
        memcpy(in3->data.fl, raw3, sizeof(raw3));
        fiv_tensor3d* out3 = fiv_create_tensor3d(s3, FIV_32F1);
        CHECK(fiv_neural_network_inference(out3, in3, rn3) == FIV_RET_OK, "relu6 3D graph inference");
        int ok3g = 1;
        for (int i = 0; i < 12; i++) {
            float e = (raw3[i] > 6.0f) ? 6.0f : ((raw3[i] < 0.0f) ? 0.0f : raw3[i]);
            if (out3->data.fl[i] != e) ok3g = 0;
        }
        CHECK(ok3g, "relu6 3D graph: clamp [0,6] correct");
        void* r3p = rn3; fiv_release_neural_network(&r3p);
        fiv_release_tensor((void**)&in3);
        fiv_release_tensor((void**)&out3);
    }
    {
        size_t s3[3] = { 2, 2, 3 };
        float raw3[12] = { 1,-2, 3, 0, -4, 5, -6, 7, 8, -9, 0, 2 };
        void* rn3 = fiv_create_neural_network();
        CHECK(fiv_neural_network_add_node(rn3, FIV_NN_NODE_RELU, 0, 1, NULL) == FIV_RET_OK, "relu 3D graph add");
        fiv_tensor3d* in3 = fiv_create_tensor3d(s3, FIV_32F1);
        memcpy(in3->data.fl, raw3, sizeof(raw3));
        fiv_tensor3d* out3 = fiv_create_tensor3d(s3, FIV_32F1);
        CHECK(fiv_neural_network_inference(out3, in3, rn3) == FIV_RET_OK, "relu 3D graph inference");
        int ok3g = 1;
        for (int i = 0; i < 12; i++) {
            float e = (raw3[i] > 0.0f) ? raw3[i] : 0.0f;
            if (out3->data.fl[i] != e) ok3g = 0;
        }
        CHECK(ok3g, "relu 3D graph: max(0,x) correct (no cap)");
        void* r3p = rn3; fiv_release_neural_network(&r3p);
        fiv_release_tensor((void**)&in3);
        fiv_release_tensor((void**)&out3);
    }
    {
        size_t s4[4] = { 1, 2, 2, 3 };
        float raw4[12] = { 1,-2, 3, 0, -4, 5, -6, 7, 8, -9, 0, 2 };
        void* rn4 = fiv_create_neural_network();
        CHECK(fiv_neural_network_add_node(rn4, FIV_NN_NODE_RELU6, 0, 1, NULL) == FIV_RET_OK, "relu6 4D graph add");
        fiv_tensor4d* in4 = fiv_create_tensor4d(s4, FIV_32F1);
        memcpy(in4->data.fl, raw4, sizeof(raw4));
        fiv_tensor4d* out4 = fiv_create_tensor4d(s4, FIV_32F1);
        CHECK(fiv_neural_network_inference(out4, in4, rn4) == FIV_RET_OK, "relu6 4D graph inference");
        int ok4g = 1;
        for (int i = 0; i < 12; i++) {
            float e = (raw4[i] > 6.0f) ? 6.0f : ((raw4[i] < 0.0f) ? 0.0f : raw4[i]);
            if (out4->data.fl[i] != e) ok4g = 0;
        }
        CHECK(ok4g, "relu6 4D graph: clamp [0,6] correct");
        void* r4p = rn4; fiv_release_neural_network(&r4p);
        fiv_release_tensor((void**)&in4);
        fiv_release_tensor((void**)&out4);
    }
    {
        size_t s4[4] = { 1, 2, 2, 3 };
        float raw4[12] = { 1,-2, 3, 0, -4, 5, -6, 7, 8, -9, 0, 2 };
        void* rn4 = fiv_create_neural_network();
        CHECK(fiv_neural_network_add_node(rn4, FIV_NN_NODE_RELU, 0, 1, NULL) == FIV_RET_OK, "relu 4D graph add");
        fiv_tensor4d* in4 = fiv_create_tensor4d(s4, FIV_32F1);
        memcpy(in4->data.fl, raw4, sizeof(raw4));
        fiv_tensor4d* out4 = fiv_create_tensor4d(s4, FIV_32F1);
        CHECK(fiv_neural_network_inference(out4, in4, rn4) == FIV_RET_OK, "relu 4D graph inference");
        int ok4g = 1;
        for (int i = 0; i < 12; i++) {
            float e = (raw4[i] > 0.0f) ? raw4[i] : 0.0f;
            if (out4->data.fl[i] != e) ok4g = 0;
        }
        CHECK(ok4g, "relu 4D graph: max(0,x) correct (no cap)");
        void* r4p = rn4; fiv_release_neural_network(&r4p);
        fiv_release_tensor((void**)&in4);
        fiv_release_tensor((void**)&out4);
    }

    remove("fiv_nn_m_linear.bin");
    remove("fiv_nn_m_saved.bin");
    remove("fiv_nn_m_relu.bin");
    remove("fiv_nn_m_relu6.bin");
    remove("fiv_nn_m_badmagic.bin");
    remove("fiv_nn_m_badtype.bin");

    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
