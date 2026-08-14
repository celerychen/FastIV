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

/* =============================================================================
 * fiv_darray.h - header-only dynamic array for C
 *
 * Two kinds of arrays, both compile to zero-overhead native access and need no
 * link step (every function is `static inline` in this header):
 *
 *   1. Typed array via FIV_DARRAY_DECL(T, name):
 *      - `data` is T*; a.data[i] is a compile-time, sizeof(T)-scaled access and
 *        push_back stores the value directly (a.data[size++] = v) - no esize
 *        multiply, no memcpy. For PODs and plain structs only.
 *      - Generated API: fiv_<name>_darray_<op> (e.g. fiv_int_arr_darray_push_back).
 *
 *   2. Type-erased pointer array `fiv_ptr_darray` (void**):
 *      - Holds arbitrary heap pointers; the caller supplies clone/free callbacks
 *        that decide whether the container OWNS each element (deep mode) or just
 *        BORROWS it (shallow mode). See the block above fiv_ptr_darray below.
 *
 * Both ctors pre-allocate FIV_DARRAY_DEFAULT_CAP (256) slots up front, so a
 * handful of pushes never triggers a realloc; reserve() grows from that same
 * default when capacity has dropped to 0.
 *
 * Naming follows C++ std::vector: all lowercase, op name last
 * (fiv_<name>_darray_<op>), so the two APIs are interchangeable in shape.
 *
 * ---------------------------------------------------------------------------
 * Example 1 - typed array of int:
 *
 *   FIV_DARRAY_DECL(int, int_arr);
 *   int_arr v;
 *   fiv_int_arr_darray_init(&v);                 // capacity 256, size 0
 *   fiv_int_arr_darray_push_back(&v, 10);
 *   fiv_int_arr_darray_push_back(&v, 20);
 *   for (size_t i = 0; i < fiv_int_arr_darray_size(&v); i++)
 *       printf("%d\n", fiv_int_arr_darray_at(&v, i));   // prints 10, 20
 *   int last = fiv_int_arr_darray_pop_back(&v); // last == 20, caller owns it
 *   fiv_int_arr_darray_uninit(&v);              // frees the buffer
 *
 * Example 2 - pointer array, shallow / borrowed mode (pass NULL callbacks):
 *
 *   fiv_ptr_darray a;
 *   fiv_ptr_darray_init(&a, NULL, NULL);        // capacity 256, borrows pointers
 *   int x = 1, y = 2;
 *   fiv_ptr_darray_push_back(&a, &x);
 *   fiv_ptr_darray_push_back(&a, &y);
 *   int* p = (int*)fiv_ptr_darray_at(&a, 0);    // p == &x (never freed by a)
 *   int* q = (int*)fiv_ptr_darray_pop_back(&a); // q == &y (returned as-is; do NOT free)
 *   fiv_ptr_darray_uninit(&a);                  // frees only the slot array
 *
 * Example 3 - pointer array, DEEP / owning mode (push heap strings):
 *
 *   // clone + free callbacks: the array owns every string it holds.
 *   static void* str_clone(const void* e) {
 *       const char* s = (const char*)e;
 *       size_t n = strlen(s) + 1;
 *       char* c = (char*)fiv_malloc(n);     // unified allocator
 *       memcpy(c, s, n);
 *       return c;
 *   }
 *   static void str_free(void* e) { fiv_free(e); }
 *
 *   fiv_ptr_darray a;
 *   fiv_ptr_darray_init(&a, str_clone, str_free);  // capacity 256, OWNS elements
 *   fiv_ptr_darray_push_back(&a, "hello");         // library makes + owns a private copy
 *   fiv_ptr_darray_push_back(&a, "world");         // library makes + owns a private copy
 *   const char* s = (const char*)fiv_ptr_darray_at(&a, 0);  // "hello", borrowed: do NOT free
 *   // ... use s ...
 *
 *   // pop_back in owning mode RETURNS the owned copy and transfers ownership to
 *   // you; free it yourself once done (via the array's destroy callback):
 *   char* popped = (char*)fiv_ptr_darray_pop_back(&a);   // popped == "world"
 *   a.destroy(popped);                                    // you own it -> you free it
 *
 *   // Everything STILL HELD when uninit() runs is freed for you via str_free
 *   // (the remaining "hello" copy + slots). Popped elements are already yours.
 *   fiv_ptr_darray_uninit(&a);
 * ---------------------------------------------------------------------------
 * =========================================================================== */
#ifndef FIV_DARRAY_H
#define FIV_DARRAY_H

#include <stdlib.h>
#include <string.h>
#include "fiv_common.h"

/* ------------------------- typed dynamic array (FIV_DARRAY_DECL) ------------------------- */
/* Expands a typed dynamic array of T named `name`. `data` is T*; access is a
 * direct, sizeof(T)-scaled native read/write, and push_back stores the value by
 * copy (no memcpy call, no runtime esize multiply). For PODs and plain structs
 * only - if an element owns a heap pointer, use fiv_ptr_darray instead. The
 * generated API is fiv_<name>_darray_<op>. */
#define FIV_DARRAY_DECL(T, name)                                                             \
typedef struct { T* data; size_t size, capacity; } name;                                     \
                                                                                             \
/* ctor: pre-allocate FIV_DARRAY_DEFAULT_CAP zeroed slots; pairs with uninit. */             \
static inline void fiv_##name##_darray_init(name* a) {                                       \
    a->size = 0; a->capacity = FIV_DARRAY_DEFAULT_CAP;                                       \
    a->data = (T*)fiv_calloc(FIV_DARRAY_DEFAULT_CAP, sizeof(T));                             \
}                                                                                            \
                                                                                             \
/* ensure capacity >= n (double from current cap, or the default when 0) */                  \
static inline void fiv_##name##_darray_reserve(name* a, size_t n) {                          \
    if (n > a->capacity) {                                                                   \
        size_t nc = a->capacity ? a->capacity : FIV_DARRAY_DEFAULT_CAP;                      \
        while (nc < n) nc *= 2;                                                              \
        a->data = (T*)fiv_realloc(a->data, nc * sizeof(T));                                  \
        a->capacity = nc;                                                                    \
    }                                                                                        \
}                                                                                            \
                                                                                             \
/* append v; reserve first once the pre-allocated slots are exhausted */                     \
static inline void fiv_##name##_darray_push_back(name* a, T v) {                             \
    if (a->size >= a->capacity) {                                                            \
        size_t n = a->size ? a->size * 2 : FIV_DARRAY_DEFAULT_CAP;                           \
        fiv_##name##_darray_reserve(a, n);                                                   \
    }                                                                                        \
    a->data[a->size++] = v;                                                                  \
}                                                                                            \
                                                                                             \
/* remove and return the last element (the value is copied out to the caller) */             \
static inline T fiv_##name##_darray_pop_back(name* a) { return a->data[--a->size]; }         \
                                                                                             \
static inline T* fiv_##name##_darray_at(name* a, size_t i) { return &a->data[i]; }           \
static inline T* fiv_##name##_darray_data(name* a) { return a->data; }                       \
static inline T  fiv_##name##_darray_front(name* a) { return a->data[0]; }                   \
static inline T  fiv_##name##_darray_back(name* a) { return a->data[a->size - 1]; }          \
                                                                                             \
static inline size_t fiv_##name##_darray_size(const name* a) { return a->size; }             \
static inline size_t fiv_##name##_darray_capacity(const name* a) { return a->capacity; }     \
static inline int    fiv_##name##_darray_empty(const name* a) { return a->size == 0; }       \
                                                                                             \
/* resize to n: reserve, zero-fill any new slots on grow, then set size */                   \
static inline void fiv_##name##_darray_resize(name* a, size_t n) {                           \
    fiv_##name##_darray_reserve(a, n);                                                       \
    if (n > a->size)                                                                         \
        memset(&a->data[a->size], 0, (n - a->size) * sizeof(T));                             \
    a->size = n;                                                                             \
}                                                                                            \
                                                                                             \
/* drop all elements but keep the allocated buffer (size = 0, nothing freed) */              \
static inline void fiv_##name##_darray_clear(name* a) { a->size = 0; }                       \
                                                                                             \
/* insert v at i, shifting later elements right */                                           \
static inline void fiv_##name##_darray_insert(name* a, size_t i, T v) {                      \
    fiv_##name##_darray_reserve(a, a->size + 1);                                             \
    memmove(&a->data[i + 1], &a->data[i], (a->size - i) * sizeof(T));                        \
    a->data[i] = v; a->size++;                                                               \
}                                                                                            \
                                                                                             \
/* remove the element at i, shifting later elements left (no return value) */                \
static inline void fiv_##name##_darray_erase(name* a, size_t i) {                            \
    memmove(&a->data[i], &a->data[i + 1], (a->size - i - 1) * sizeof(T));                    \
    a->size--;                                                                               \
}                                                                                            \
                                                                                             \
/* free the allocated buffer; pairs with init */                                             \
static inline void fiv_##name##_darray_uninit(name* a) {                                     \
    fiv_free(a->data); a->data = NULL; a->size = a->capacity = 0;                            \
}

/* A growable string is just FIV_DARRAY_DECL(char, name) - a char vector. Keep a
 * trailing NUL yourself, or store independent char* elements in fiv_ptr_darray
 * (see test_ptr_darray_cstr in test_darray.c / test_dir_files.c). */

/* ------------------------- generic pointer array: fiv_ptr_darray ------------------------- */
/* Type-erased dynamic array of void*. Each element's ownership is decided by the
 * `clone` callback given to init:
 *
 *   clone != NULL  (DEEP / owning mode):
 *     - push_back/insert store a CLONE the container owns;
 *     - pop_back returns the last clone and TRANSFERS its ownership to the
 *       caller (container no longer owns it); the caller must free it with
 *       `destroy` once done. The container never frees a popped element.
 *     - erase/clear/resize-shrink/uninit free the still-contained clones via `destroy`.
 *
 *   clone == NULL  (SHALLOW / borrowed mode, i.e. both callbacks NULL):
 *     - push_back/insert store the raw `elem` pointer directly (no clone);
 *     - pop_back returns that raw pointer (caller must NOT free - it is external);
 *     - erase/clear/resize-shrink/uninit NEVER free anything - the container
 *       only borrows the pointers, you manage their lifetime elsewhere.
 *
 * The bulk-removal ops (erase/clear/resize-shrink/uninit) free owned elements
 * only when (clone && destroy); a borrowed pointer is never freed by the
 * container even if a stray `destroy` is supplied. pop_back never frees, because
 * its whole purpose is to HAND the element to the caller.
 *
 *   at/data/front/back ALWAYS return a BORROWED void* you must NOT free in either
 *   mode - the array still holds it (deep) or still points at it (shallow).
 *
 * Interface parity with FIV_DARRAY_DECL: same op names
 * (init/reserve/push_back/pop_back/at/data/front/back/size/capacity/empty/
 * resize/clear/insert/erase/uninit). Differences: init takes the clone/destroy
 * callbacks; pop_back returns void* (the typed one returns T by value); the bulk
 * removals free owned elements when applicable.
 *
 * Implementation: header-only. Ops are static inline; the simple accessors
 * (at/data/front/back/size/capacity/empty) are macros with matching names. */
typedef void* (*fiv_ptr_clone_fn)(const void* elem);
typedef void  (*fiv_ptr_free_fn)(void* elem);

/* Default capacity pre-allocated by both ctors and the floor reserve() grows from. */
#define FIV_DARRAY_DEFAULT_CAP 256

typedef struct {
    void**           data;       /* array of void* slots (owned in deep mode, borrowed in shallow) */
    size_t           size, capacity;
    fiv_ptr_clone_fn clone;      /* user callback: make an owned copy of elem (NULL => shallow) */
    fiv_ptr_free_fn  destroy;    /* user callback: free one owned element */
} fiv_ptr_darray;

/* ------------------------- fiv_ptr_darray implementation (header-only) ------------------------- */
/* static inline so each translation unit gets its own copy; accessors are macros below. */

static inline void fiv_ptr_darray_init(fiv_ptr_darray* a, fiv_ptr_clone_fn clone, fiv_ptr_free_fn destroy) {
    a->clone = clone; a->destroy = destroy;
    a->size = 0;
    a->capacity = FIV_DARRAY_DEFAULT_CAP;
    a->data = (void**)fiv_calloc(FIV_DARRAY_DEFAULT_CAP, sizeof(void*));
}

static inline void fiv_ptr_darray_reserve(fiv_ptr_darray* a, size_t n) {
    if (n > a->capacity) {
        size_t nc = a->capacity ? a->capacity : FIV_DARRAY_DEFAULT_CAP;
        while (nc < n) nc *= 2;
        a->data = (void**)fiv_realloc(a->data, nc * sizeof(void*));
        a->capacity = nc;
    }
}

static inline void fiv_ptr_darray_push_back(fiv_ptr_darray* a, const void* elem) {
    fiv_ptr_darray_reserve(a, a->size + 1);
    a->data[a->size++] = a->clone ? a->clone(elem) : (void*)elem;
}

static inline void* fiv_ptr_darray_pop_back(fiv_ptr_darray* a) {
    if (a->size == 0) return NULL;
    a->size--;
    return a->data[a->size];
}

static inline void fiv_ptr_darray_resize(fiv_ptr_darray* a, size_t n) {
    if (n < a->size) {
        for (size_t i = n; i < a->size; i++)
            if (a->clone && a->destroy && a->data[i]) a->destroy(a->data[i]);
    }
    fiv_ptr_darray_reserve(a, n);
    if (n > a->size) {
        for (size_t i = a->size; i < n; i++) a->data[i] = NULL;
    }
    a->size = n;
}

static inline void fiv_ptr_darray_clear(fiv_ptr_darray* a) {
    for (size_t i = 0; i < a->size; i++)
        if (a->clone && a->destroy && a->data[i]) a->destroy(a->data[i]);
    a->size = 0;
}

static inline void fiv_ptr_darray_insert(fiv_ptr_darray* a, size_t i, const void* elem) {
    fiv_ptr_darray_reserve(a, a->size + 1);
    memmove(&a->data[i + 1], &a->data[i], (a->size - i) * sizeof(void*));
    a->data[i] = a->clone ? a->clone(elem) : (void*)elem;
    a->size++;
}

static inline void fiv_ptr_darray_erase(fiv_ptr_darray* a, size_t i) {
    if (a->clone && a->destroy && a->data[i]) a->destroy(a->data[i]);
    memmove(&a->data[i], &a->data[i + 1], (a->size - i - 1) * sizeof(void*));
    a->size--;
}

static inline void fiv_ptr_darray_uninit(fiv_ptr_darray* a) {
    for (size_t i = 0; i < a->size; i++)
        if (a->clone && a->destroy && a->data[i]) a->destroy(a->data[i]);
    fiv_free(a->data);
    a->data = NULL; a->size = a->capacity = 0;
}

/* member accessors as macros (names strictly match the former functions) */
#define fiv_ptr_darray_at(a, i)        ((a)->data[(i)])
#define fiv_ptr_darray_data(a)         ((a)->data)
#define fiv_ptr_darray_front(a)        ((a)->data[0])
#define fiv_ptr_darray_back(a)         ((a)->data[(a)->size - 1])
#define fiv_ptr_darray_size(a)         ((a)->size)
#define fiv_ptr_darray_capacity(a)     ((a)->capacity)
#define fiv_ptr_darray_empty(a)        ((a)->size == 0)

#endif /* FIV_DARRAY_H */
