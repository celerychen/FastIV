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

/* Correctness tests for the tensor API declared in fiv_ctensor.h.
 * Exercises create/release, header+external-buffer binding, generic dispatch,
 * zero-copy view, reshape, deep copy, header copy and the element-wise
 * binary ops (add/sub/mul/div) for float32 and int32. */

#include "fiv_ctensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static int g_pass = 0;
#define CHECK(c, msg)                                                           \
    do {                                                                        \
        if (!(c)) { printf("  [FAIL] %s @%d\n", (msg), __LINE__); g_fail++; }   \
        else       { g_pass++; }                                                \
    } while (0)
static float fabsf_local(float x) { return x < 0 ? -x : x; }

/* ----------------------------- create / metadata / release ----------------------------- */
static void test_create_2d(void) {
    size_t s[2] = { 3, 4 };
    fiv_tensor2d* t = fiv_create_tensor2d(s, FIV_32F1);
    CHECK(t != NULL, "create 2d returns non-NULL");
    CHECK(t->id == FIV_ID_TENSOR2D, "id is TENSOR2D");
    CHECK(t->dtype == FIV_32F1, "dtype is 32F1");
    CHECK(t->shapes[0] == 3 && t->shapes[1] == 4, "shapes 3x4");
    CHECK(t->element_bytes == 4, "element_bytes 4");
    CHECK(t->strides[1] == 4, "stride[1] == element_bytes");
    CHECK(t->strides[0] == 16, "stride[0] == cols*element_bytes");
    CHECK(t->total_bytes == 48, "total_bytes 48");
    CHECK(t->data_continue == 1 && t->reference == 1, "contiguous + owns data");

    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 4; c++)
            t->data.fl[r * 4 + c] = (float)(r * 4 + c);
    int ok = 1;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 4; c++)
            if (t->data.fl[r * 4 + c] != (float)(r * 4 + c)) ok = 0;
    CHECK(ok, "round-trip float values (row-major)");

    CHECK(fiv_release_tensor2d(&t) == FIV_RET_OK, "release returns OK");
    CHECK(t == NULL, "release nulls the caller pointer");
    CHECK(fiv_release_tensor2d(&t) == FIV_RET_ERR_PARA, "double release on NULL errors");
}

static void test_create_dims(void) {
    /* 1D int32 */
    fiv_tensor1d* a = fiv_create_tensor1d(10, FIV_32S1);
    CHECK(a && a->element_bytes == 4 && a->strides[0] == 4 && a->total_bytes == 40,
          "1D int32 metadata");
    for (int i = 0; i < 10; i++) a->data.ptr32s[i] = i;
    int ok = 1;
    for (int i = 0; i < 10; i++) if (a->data.ptr32s[i] != i) ok = 0;
    CHECK(ok, "1D int32 round-trip");
    fiv_release_tensor1d(&a);

    /* 4D float strides (byte strides, C-order) */
    size_t s4[4] = { 2, 3, 4, 5 };
    fiv_tensor4d* d = fiv_create_tensor4d(s4, FIV_32F1);
    CHECK(d != NULL, "4D create");
    CHECK(d->strides[3] == 4,  "4D stride[3] == eb");
    CHECK(d->strides[2] == 20, "4D stride[2] == 4*5");
    CHECK(d->strides[1] == 80, "4D stride[1] == 4*5*4");
    CHECK(d->strides[0] == 240,"4D stride[0] == 4*5*4*3");
    CHECK(d->total_bytes == 2 * 3 * 4 * 5 * 4, "4D total_bytes");
    fiv_release_tensor4d(&d);
}

/* ----------------------------- header + external buffer binding ----------------------------- */
static void test_header_data(void) {
    size_t s[2] = { 3, 4 };
    fiv_tensor2d* t = fiv_create_tensor2d_header(s, FIV_32F1);
    CHECK(t && t->reference == 0 && t->data.ptr == NULL, "header: no data, not owner");
    CHECK(t->strides[0] == 16 && t->total_bytes == 48, "header fills metadata");

    float buf[12];
    CHECK(fiv_tensor2d_set_data(t, buf, sizeof(buf)) == FIV_RET_OK, "set_data attaches buffer");
    CHECK(t->reference == 0 && t->data.ptr == buf, "set_data keeps external ownership");
    t->data.fl[0]  = 1.0f;
    t->data.fl[11] = 11.0f;
    CHECK(buf[0] == 1.0f && buf[11] == 11.0f, "writes through tensor land in external buffer");

    float small[4];
    CHECK(fiv_tensor2d_set_data(t, small, sizeof(small)) == FIV_RET_ERR_PARA,
          "set_data rejects too-small buffer");

    /* release frees only the struct, never the external buffer */
    CHECK(fiv_release_tensor2d(&t) == FIV_RET_OK, "release header-only tensor");
    CHECK(buf[0] == 1.0f, "external buffer intact after release");
}

/* ----------------------------- generic dispatch ----------------------------- */
static void test_generic(void) {
    size_t s[2] = { 2, 3 };
    void* p = fiv_create_tensor(2, s, FIV_32F1);
    CHECK(p != NULL, "generic create");
    fiv_tensor2d* t = (fiv_tensor2d*)p;
    CHECK(t->id == FIV_ID_TENSOR2D && t->shapes[0] == 2 && t->shapes[1] == 3,
          "generic fields resolved by dim");
    CHECK(fiv_release_tensor(&p) == FIV_RET_OK, "generic release");
    CHECK(p == NULL, "generic release nulls pointer");
}

/* ----------------------------- zero-copy view ----------------------------- */
static void test_view(void) {
    size_t s[2] = { 4, 4 };
    fiv_tensor2d* src = fiv_create_tensor2d(s, FIV_32F1);
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            src->data.fl[r * 4 + c] = (float)(r * 4 + c);

    size_t off[2] = { 1, 1 }, vsz[2] = { 2, 2 };
    fiv_tensor2d view;
    CHECK(fiv_tensor_view(&view, src, off, vsz) == FIV_RET_OK, "view ok");
    CHECK(view.shapes[0] == 2 && view.shapes[1] == 2, "view shapes 2x2");
    CHECK(view.reference == 0, "view does not own data");
    CHECK(view.data_continue == 0, "sub-view is non-contiguous");
    CHECK((iv8u*)view.data.ptr == (iv8u*)src->data.ptr + 1 * src->strides[0] + 1 * src->strides[1],
          "view data.ptr offset to sub-region");

    int ok = 1;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            ivf32 val = *(ivf32*)((iv8u*)view.data.ptr + i * view.strides[0] + j * view.strides[1]);
            if (val != (ivf32)((1 + i) * 4 + (1 + j))) ok = 0;
        }
    CHECK(ok, "view reads correct source elements");

    /* out-of-range offset must be rejected */
    size_t bad_off[2] = { 3, 3 }, bad_sz[2] = { 2, 2 };
    fiv_tensor2d bad;
    CHECK(fiv_tensor_view(&bad, src, bad_off, bad_sz) == FIV_RET_ERR_PARA, "view rejects out-of-range");

    fiv_release_tensor2d(&src);
}

/* ----------------------------- reshape (shared buffer) ----------------------------- */
static void test_reshape(void) {
    size_t s[2] = { 2, 6 };
    fiv_tensor2d* src = fiv_create_tensor2d(s, FIV_32F1);
    for (int i = 0; i < 12; i++) src->data.fl[i] = (float)i;

    fiv_tensor2d r;
    size_t rs[2] = { 3, 4 };
    CHECK(fiv_tensor_reshape(&r, src, 2, rs) == FIV_RET_OK, "reshape ok");
    CHECK(r.shapes[0] == 3 && r.shapes[1] == 4, "reshape shapes 3x4");
    CHECK(r.data_continue == 1 && r.reference == 0, "reshape contiguous + not owner");
    int ok = 1;
    for (int i = 0; i < 12; i++) if (r.data.fl[i] != (float)i) ok = 0;
    CHECK(ok, "reshape preserves row-major element order");

    size_t bad[2] = { 3, 5 };   /* 15 != 12 elements */
    fiv_tensor2d r2;
    CHECK(fiv_tensor_reshape(&r2, src, 2, bad) == FIV_RET_ERR_PARA, "reshape rejects mismatched count");

    fiv_release_tensor2d(&src);
}

/* ----------------------------- deep copy ----------------------------- */
static void test_copy(void) {
    size_t s[2] = { 2, 3 };
    fiv_tensor2d* src = fiv_create_tensor2d(s, FIV_32F1);
    for (int i = 0; i < 6; i++) src->data.fl[i] = (float)(i + 1);

    void* cp = fiv_create_tensor_from_tensor(src);
    CHECK(cp != NULL, "deep copy returns non-NULL");
    fiv_tensor2d* c = (fiv_tensor2d*)cp;
    CHECK(c->reference == 1, "copy owns its own buffer");
    int ok = 1;
    for (int i = 0; i < 6; i++) if (c->data.fl[i] != (float)(i + 1)) ok = 0;
    CHECK(ok, "copy equals source");

    src->data.fl[0] = 999.0f;                 /* mutate source */
    CHECK(c->data.fl[0] == 1.0f, "copy is independent of source");
    CHECK(c->data.ptr != src->data.ptr, "copy has a separate buffer");

    fiv_release_tensor(&cp);
    fiv_release_tensor2d(&src);
}

/* ----------------------------- header-only copy ----------------------------- */
static void test_header_copy(void) {
    size_t s[2] = { 2, 3 };
    fiv_tensor2d* src = fiv_create_tensor2d(s, FIV_32F1);
    void* h = fiv_create_tensor_header_from_tensor(src);
    CHECK(h != NULL, "header copy returns non-NULL");
    fiv_tensor2d* hh = (fiv_tensor2d*)h;
    CHECK(hh->dtype == src->dtype && hh->shapes[0] == 2 && hh->shapes[1] == 3,
          "header copy preserves shape/dtype");
    CHECK(hh->data.ptr == NULL && hh->reference == 0, "header copy has no data, no ownership");
    fiv_release_tensor(&h);
    fiv_release_tensor2d(&src);
}

/* ----------------------------- binary ops: float32 ----------------------------- */
static void test_binop_float(void) {
    fiv_tensor1d* a = fiv_create_tensor1d(4, FIV_32F1);
    fiv_tensor1d* b = fiv_create_tensor1d(4, FIV_32F1);
    fiv_tensor1d* c = fiv_create_tensor1d(4, FIV_32F1);
    float av[4] = { 1, 2, 3, 4 }, bv[4] = { 10, 20, 30, 40 };
    for (int i = 0; i < 4; i++) { a->data.fl[i] = av[i]; b->data.fl[i] = bv[i]; }

    CHECK(fiv_tensor_add(c, a, b) == FIV_RET_OK, "float add");
    { float e[4] = { 11, 22, 33, 44 }; int ok = 1;
      for (int i = 0; i < 4; i++) if (c->data.fl[i] != e[i]) ok = 0;
      CHECK(ok, "float add values"); }

    CHECK(fiv_tensor_sub(c, a, b) == FIV_RET_OK, "float sub");
    { float e[4] = { -9, -18, -27, -36 }; int ok = 1;
      for (int i = 0; i < 4; i++) if (c->data.fl[i] != e[i]) ok = 0;
      CHECK(ok, "float sub values"); }

    CHECK(fiv_tensor_mul(c, a, b) == FIV_RET_OK, "float mul");
    { float e[4] = { 10, 40, 90, 160 }; int ok = 1;
      for (int i = 0; i < 4; i++) if (c->data.fl[i] != e[i]) ok = 0;
      CHECK(ok, "float mul values"); }

    CHECK(fiv_tensor_div(c, a, b) == FIV_RET_OK, "float div");
    { float e[4] = { 0.1f, 0.1f, 0.1f, 0.1f }; int ok = 1;
      for (int i = 0; i < 4; i++) if (fabsf_local(c->data.fl[i] - e[i]) > 1e-5f) ok = 0;
      CHECK(ok, "float div values"); }

    /* in-place: c = a + a (a becomes doubled) */
    fiv_tensor_add(a, a, a);
    { int ok = 1; for (int i = 0; i < 4; i++) if (a->data.fl[i] != av[i] * 2) ok = 0;
      CHECK(ok, "float in-place add"); }

    fiv_release_tensor1d(&a); fiv_release_tensor1d(&b); fiv_release_tensor1d(&c);
}

/* ----------------------------- binary ops: int32 ----------------------------- */
static void test_binop_int(void) {
    fiv_tensor1d* a = fiv_create_tensor1d(4, FIV_32S1);
    fiv_tensor1d* b = fiv_create_tensor1d(4, FIV_32S1);
    fiv_tensor1d* c = fiv_create_tensor1d(4, FIV_32S1);
    iv32s av[4] = { 10, 20, 30, 40 }, bv[4] = { 3, 4, 5, 6 };
    for (int i = 0; i < 4; i++) { a->data.ptr32s[i] = av[i]; b->data.ptr32s[i] = bv[i]; }

    CHECK(fiv_tensor_add(c, a, b) == FIV_RET_OK, "int add");
    { iv32s e[4] = { 13, 24, 35, 46 }; int ok = 1;
      for (int i = 0; i < 4; i++) if (c->data.ptr32s[i] != e[i]) ok = 0;
      CHECK(ok, "int add values"); }

    CHECK(fiv_tensor_sub(c, a, b) == FIV_RET_OK, "int sub");
    { iv32s e[4] = { 7, 16, 25, 34 }; int ok = 1;
      for (int i = 0; i < 4; i++) if (c->data.ptr32s[i] != e[i]) ok = 0;
      CHECK(ok, "int sub values"); }

    CHECK(fiv_tensor_mul(c, a, b) == FIV_RET_OK, "int mul");
    { iv32s e[4] = { 30, 80, 150, 240 }; int ok = 1;
      for (int i = 0; i < 4; i++) if (c->data.ptr32s[i] != e[i]) ok = 0;
      CHECK(ok, "int mul values"); }

    CHECK(fiv_tensor_div(c, a, b) == FIV_RET_OK, "int div");
    { iv32s e[4] = { 3, 5, 6, 6 }; int ok = 1;
      for (int i = 0; i < 4; i++) if (c->data.ptr32s[i] != e[i]) ok = 0;
      CHECK(ok, "int div values (truncating)"); }

    fiv_release_tensor1d(&a); fiv_release_tensor1d(&b); fiv_release_tensor1d(&c);
}

/* ----------------------------- binary ops: error paths ----------------------------- */
static void test_binop_errors(void) {
    /* dtype mismatch */
    fiv_tensor1d* a = fiv_create_tensor1d(4, FIV_32F1);
    fiv_tensor1d* b = fiv_create_tensor1d(4, FIV_32S1);
    fiv_tensor1d* c = fiv_create_tensor1d(4, FIV_32F1);
    CHECK(fiv_tensor_add(c, a, b) == FIV_RET_ERR_PARA, "dtype mismatch -> PARA");
    fiv_release_tensor1d(&a); fiv_release_tensor1d(&b); fiv_release_tensor1d(&c);

    /* shape mismatch */
    fiv_tensor1d* a2 = fiv_create_tensor1d(4, FIV_32F1);
    fiv_tensor1d* b2 = fiv_create_tensor1d(2, FIV_32F1);
    fiv_tensor1d* c2 = fiv_create_tensor1d(4, FIV_32F1);
    CHECK(fiv_tensor_add(c2, a2, b2) == FIV_RET_ERR_PARA, "shape mismatch -> PARA");
    fiv_release_tensor1d(&a2); fiv_release_tensor1d(&b2); fiv_release_tensor1d(&c2);

    /* non-contiguous operand (a 2x2 view of a 4x4) must be rejected */
    size_t s[2] = { 4, 4 };
    fiv_tensor2d* src = fiv_create_tensor2d(s, FIV_32F1);
    size_t off[2] = { 1, 1 }, vsz[2] = { 2, 2 };
    fiv_tensor2d view;
    fiv_tensor_view(&view, src, off, vsz);
    fiv_tensor2d* ca = fiv_create_tensor2d(vsz, FIV_32F1);
    fiv_tensor2d* cb = fiv_create_tensor2d(vsz, FIV_32F1);
    CHECK(fiv_tensor_add(ca, &view, cb) == FIV_RET_ERR_PARA, "non-contiguous -> PARA");
    fiv_release_tensor2d(&src); fiv_release_tensor2d(&ca); fiv_release_tensor2d(&cb);

    /* unsupported dtype (uint8) */
    fiv_tensor1d* ua = fiv_create_tensor1d(4, FIV_8U1);
    fiv_tensor1d* ub = fiv_create_tensor1d(4, FIV_8U1);
    fiv_tensor1d* uc = fiv_create_tensor1d(4, FIV_8U1);
    CHECK(fiv_tensor_add(uc, ua, ub) == FIV_RET_ERR_NOT_SUPPORT, "uint8 -> NOT_SUPPORT");
    fiv_release_tensor1d(&ua); fiv_release_tensor1d(&ub); fiv_release_tensor1d(&uc);
}

int main(void) {
    printf("=== test_ctensor ===\n");
    test_create_2d();
    test_create_dims();
    test_header_data();
    test_generic();
    test_view();
    test_reshape();
    test_copy();
    test_header_copy();
    test_binop_float();
    test_binop_int();
    test_binop_errors();
    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
