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

/* Correctness tests for the header-only dynamic array in fiv_darray.h.
 * Covers both the typed array (FIV_DARRAY_DECL) and the type-erased
 * fiv_ptr_darray (shallow/borrowed vs deep/owning modes). */

#include "fiv_darray.h"   /* pulls in fiv_common.h (fiv_malloc/fiv_free) */
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

/* FIV_DARRAY_DECL emits typedef + static inline function definitions, so it MUST
 * be expanded at file scope, never inside a function body. */
FIV_DARRAY_DECL(int, int_arr);

/* ----------------------------- typed array (int) ----------------------------- */
static void test_typed_int(void) {
    int_arr v;
    fiv_int_arr_darray_init(&v);
    CHECK(fiv_int_arr_darray_size(&v) == 0, "init size 0");
    CHECK(fiv_int_arr_darray_capacity(&v) == 256, "init capacity 256");

    for (int i = 0; i < 300; i++) fiv_int_arr_darray_push_back(&v, i * 2);
    CHECK(fiv_int_arr_darray_size(&v) == 300, "size after 300 pushes");
    CHECK(fiv_int_arr_darray_capacity(&v) >= 300, "capacity grew past 300");
    CHECK(fiv_int_arr_darray_front(&v) == 0, "front == 0");
    CHECK(fiv_int_arr_darray_back(&v) == 598, "back == 598");
    int ok = 1;
    for (int i = 0; i < 300; i++) if (v.data[i] != i * 2) ok = 0;
    CHECK(ok, "values preserved across realloc");

    int p = fiv_int_arr_darray_pop_back(&v);
    CHECK(p == 598, "pop_back returns last value");
    CHECK(fiv_int_arr_darray_size(&v) == 299, "size after pop_back");

    fiv_int_arr_darray_insert(&v, 5, 12345);
    CHECK(fiv_int_arr_darray_size(&v) == 300, "size after insert");
    CHECK(v.data[5] == 12345, "inserted value at index 5");
    CHECK(v.data[6] == 10, "old value shifted to index 6");

    fiv_int_arr_darray_erase(&v, 5);
    CHECK(fiv_int_arr_darray_size(&v) == 299, "size after erase");
    CHECK(v.data[5] == 10, "value shifted back after erase");

    fiv_int_arr_darray_resize(&v, 400);
    CHECK(fiv_int_arr_darray_size(&v) == 400, "resize up size");
    CHECK(v.data[399] == 0, "newly grown slots are zero-filled");
    fiv_int_arr_darray_resize(&v, 10);
    CHECK(fiv_int_arr_darray_size(&v) == 10, "resize down size");

    fiv_int_arr_darray_clear(&v);
    CHECK(fiv_int_arr_darray_size(&v) == 0, "clear sets size 0");
    CHECK(fiv_int_arr_darray_capacity(&v) >= 10, "clear keeps the buffer");

    fiv_int_arr_darray_reserve(&v, 1000);
    CHECK(fiv_int_arr_darray_capacity(&v) >= 1000, "reserve grows capacity");

    fiv_int_arr_darray_uninit(&v);
    CHECK(v.data == NULL, "uninit nulls data");
    CHECK(fiv_int_arr_darray_size(&v) == 0, "uninit sets size 0");
}

/* ----------------------------- typed array (struct) ----------------------------- */
typedef struct { int x; float y; } pt;
FIV_DARRAY_DECL(pt, pt_arr);
static void test_typed_struct(void) {
    pt_arr v;
    fiv_pt_arr_darray_init(&v);
    pt a = { 1, 1.5f }, b = { 2, 2.5f };
    fiv_pt_arr_darray_push_back(&v, a);
    fiv_pt_arr_darray_push_back(&v, b);
    CHECK(fiv_pt_arr_darray_size(&v) == 2, "struct array size 2");
    CHECK(v.data[0].x == 1 && v.data[0].y == 1.5f, "struct elem 0");
    CHECK(v.data[1].x == 2 && v.data[1].y == 2.5f, "struct elem 1");
    fiv_pt_arr_darray_uninit(&v);
}

/* ----------------------------- ptr array (shallow / borrowed) ----------------------------- */
static void test_ptr_shallow(void) {
    fiv_ptr_darray a;
    fiv_ptr_darray_init(&a, NULL, NULL);   /* borrowed mode */
    CHECK(fiv_ptr_darray_capacity(&a) == 256, "ptr init capacity 256");
    int x = 10, y = 20, z = 30;
    fiv_ptr_darray_push_back(&a, &x);
    fiv_ptr_darray_push_back(&a, &y);
    fiv_ptr_darray_push_back(&a, &z);
    CHECK(fiv_ptr_darray_size(&a) == 3, "ptr size 3");
    CHECK(fiv_ptr_darray_at(&a, 0) == &x, "borrowed: stores the raw pointer");
    CHECK(fiv_ptr_darray_at(&a, 2) == &z, "borrowed: stores the raw pointer 2");
    void* popped = fiv_ptr_darray_pop_back(&a);
    CHECK(popped == &z, "pop_back returns the borrowed pointer as-is");
    CHECK(fiv_ptr_darray_size(&a) == 2, "ptr size after pop_back");
    fiv_ptr_darray_uninit(&a);              /* must NOT free the externals */
    CHECK(x == 10 && y == 20 && z == 30, "borrowed externals are NOT freed");
}

/* ----------------------------- ptr array (deep / owning) ----------------------------- */
static void* str_clone(const void* e) {
    const char* s = (const char*)e;
    size_t n = strlen(s) + 1;
    char* c = (char*)fiv_malloc(n);
    memcpy(c, s, n);
    return c;
}
static void str_free(void* e) { fiv_free(e); }

static void test_ptr_deep(void) {
    fiv_ptr_darray a;
    fiv_ptr_darray_init(&a, str_clone, str_free);   /* owning mode */
    fiv_ptr_darray_push_back(&a, "hello");
    fiv_ptr_darray_push_back(&a, "world");
    CHECK(fiv_ptr_darray_size(&a) == 2, "deep size 2");
    CHECK(fiv_ptr_darray_at(&a, 0) != (void*)"hello", "deep keeps a PRIVATE copy");
    CHECK(strcmp((char*)fiv_ptr_darray_at(&a, 0), "hello") == 0, "deep copy content 0");
    CHECK(strcmp((char*)fiv_ptr_darray_at(&a, 1), "world") == 0, "deep copy content 1");

    /* pop_back transfers ownership of the clone to the caller */
    char* popped = (char*)fiv_ptr_darray_pop_back(&a);
    CHECK(strcmp(popped, "world") == 0, "pop returns the owned copy");
    CHECK(fiv_ptr_darray_size(&a) == 1, "deep size after pop_back");
    a.destroy(popped);   /* we now own it -> free it ourselves */

    /* erase frees the still-held clone via the destroy callback */
    fiv_ptr_darray_erase(&a, 0);
    CHECK(fiv_ptr_darray_size(&a) == 0, "deep size after erase");
    fiv_ptr_darray_uninit(&a);   /* nothing left to free */
}

int main(void) {
    printf("=== test_darray ===\n");
    test_typed_int();
    test_typed_struct();
    test_ptr_shallow();
    test_ptr_deep();
    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
