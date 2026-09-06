#ifndef FIV_YOLO26_REF_H
#define FIV_YOLO26_REF_H

#include <stddef.h>

#include "fiv_data_typedefs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Reference-tensor loader for the YOLO26 -> FastIV port.
 *
 * gen_ref.py writes, for each named tensor, two files under the ref dir:
 *   <name>.f32   raw float32 little-endian array, no header
 *   <name>.json  {"name","shape":[...],"dtype":"f32","src":"..."}
 *
 * The ref dir is taken from $YOLO26_REF_DIR, or YOLO26_REF_DIR at compile
 * time (default "../app/yolo26/test/ref", i.e. relative to build/). */

#ifndef YOLO26_REF_DIR
#define YOLO26_REF_DIR "../app/yolo26/test/ref"
#endif

/* Load a reference tensor. Returns a buffer allocated with fiv_malloc (the
 * caller releases it with fiv_free) and fills *ndim / shape (shape must hold
 * >= 4 elements). Returns NULL if the json or f32 file is missing. */
const ivf32* ref_load(const char* name, int* ndim, size_t* shape);

/* Load a reference tensor by element count only (skips the json parse).
 * *count receives the element count; returns NULL on a missing f32 file. */
const ivf32* ref_load_n(const char* name, size_t* count);

/* Compare got[] against the reference named `name` element-wise. Returns 0 if
 * every element is within tol (the max absolute error is written to
 * *max_abs_err). Non-zero means mismatch (or reference missing). Prints a
 * one-line report. */
int ref_cmp(const char* name, const ivf32* got, ivf32 tol, ivf32* max_abs_err);

/* Check that the caller's runtime shape matches the reference's recorded
 * shape. Returns 0 if equal, non-zero otherwise. Used by mutation tests (a
 * shape swap in the json must make this fail). */
int ref_shape_eq(const char* name, int ndim, const size_t* shape);

/* Point the loader at an explicit reference/model directory (highest
 * priority; overrides the YOLO26_REF_DIR env var). Pass NULL to clear the
 * override and fall back to env / the compiled-in default. */
void yolo26_ref_set_dir(const char* dir);

/* ------------------------------------------------------------------ *
 * BN-fold table (gen_ref.py export_folds -> fold.json / fold_w.f32 /
 * fold_b.f32). One entry per Conv2d on the inference path, in topo order:
 *   has_bn=1  weights/biases are BN-folded (w' = w*s, b' = beta - s*mean)
 *   has_bn=0  plain Conv2d exported as-is (Detect tail)
 *   act       0 = none/Identity, 1 = SiLU (the Conv wrapper's activation)
 *   kernel/stride/padding/groups  geometry of the conv
 *   w_off/b_off  element offset inside fold_w.f32 / fold_b.f32
 * ------------------------------------------------------------------ */
typedef struct {
    int    seq;
    int    layer;            /* top-level index within model.model (Detect excluded) */
    int    input_channels;
    int    output_channels;
    int    kernel;
    int    stride;
    int    padding;
    int    groups;
    int    has_bn;
    int    act;
    size_t w_off;
    size_t w_cnt;
    size_t b_off;
    size_t b_cnt;
    char   path[192];   /* torch named_modules path, informational */
    char   parent[64];  /* immediate wrapper type (Conv / Bottleneck / C3k2 / ...) */
} yolo26_fold_entry;

/* Parse fold.json into out[0..cap). Returns the entry count, or -1 on an io
 * error (missing fold.json / more than cap entries). */
int yolo26_ref_load_fold(yolo26_fold_entry* out, int cap);

/* Read cnt float32s starting at element offset off of a raw blob file (the
 * fold_w.f32 / fold_b.f32 payloads). Returns an fiv_malloc'd buffer (caller
 * releases with fiv_free), NULL on error. */
const ivf32* yolo26_ref_load_blob(const char* file, size_t off, size_t cnt);

#ifdef __cplusplus
}
#endif

#endif /* FIV_YOLO26_REF_H */
