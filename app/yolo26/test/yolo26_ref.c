/* Reference-tensor loader for the YOLO26 -> FastIV port. See yolo26_ref.h. */

#include "yolo26_ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char* ref_dir(void) {
    const char* d = getenv("YOLO26_REF_DIR");
    return d != NULL ? d : YOLO26_REF_DIR;
}

/* Read a whole file into a malloc'd buffer; returns byte count, -1 on error. */
static long read_file(const char* path, unsigned char** out) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* buf = (unsigned char*)malloc(n > 0 ? (size_t)n : 1u);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return -1; }
    fclose(f);
    *out = buf;
    return n;
}

/* Parse the first integer array after "shape" in the json. */
static int parse_shape(const char* js, size_t jn, int* ndim, size_t* shape) {
    int dim = 0;
    const char* p = (const char*)memchr(js, '"', jn);
    while (p != NULL) {
        if (strncmp(p, "\"shape\"", 7) == 0) break;
        p = (const char*)memchr(p + 1, '"', (const char*)js + jn - (p + 1));
    }
    if (p == NULL) return 0;
    p = (const char*)memchr(p, '[', (const char*)js + jn - p);
    if (p == NULL) return 0;
    p++;
    while (*p != '\0' && *p != ']') {
        while (*p == ' ' || *p == ',' || *p == '\n' || *p == '\t') p++;
        if (*p >= '0' && *p <= '9') {
            if (dim < 8) shape[dim] = (size_t)strtoul(p, (char**)&p, 10);
            else { while (*p >= '0' && *p <= '9') p++; }
            dim++;
        } else if (*p == ']') {
            break;
        } else {
            p++;
        }
    }
    *ndim = dim;
    return dim;
}

const float* ref_load(const char* name, int* ndim, size_t* shape) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.json", ref_dir(), name);
    unsigned char* js = NULL;
    long jn = read_file(path, &js);
    int dim = 0;
    size_t shp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (jn >= 0) {
        parse_shape((const char*)js, (size_t)jn, &dim, shp);
        free(js);
    } else {
        fprintf(stderr, "ref_load: missing json for '%s'\n", name);
        return NULL;
    }

    snprintf(path, sizeof(path), "%s/%s.f32", ref_dir(), name);
    unsigned char* bin = NULL;
    long bn = read_file(path, &bin);
    if (bn < 0) {
        fprintf(stderr, "ref_load: missing f32 for '%s'\n", name);
        return NULL;
    }
    if (ndim != NULL) *ndim = dim;
    if (shape != NULL) {
        for (int i = 0; i < dim && i < 8; i++) shape[i] = shp[i];
    }
    return (const float*)bin; /* caller frees */
}

const float* ref_load_n(const char* name, size_t* count) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.f32", ref_dir(), name);
    unsigned char* bin = NULL;
    long bn = read_file(path, &bin);
    if (bn < 0) {
        fprintf(stderr, "ref_load_n: missing f32 for '%s'\n", name);
        if (count != NULL) *count = 0;
        return NULL;
    }
    if (count != NULL) *count = (size_t)bn / sizeof(float);
    return (const float*)bin;
}

int ref_cmp(const char* name, const float* got, float tol, float* max_abs_err) {
    size_t count = 0;
    const float* ref = ref_load_n(name, &count);
    if (ref == NULL) {
        if (max_abs_err != NULL) *max_abs_err = -1.0f;
        return -1;
    }
    float m = 0.0f;
    long first = -1;
    for (size_t k = 0; k < count; k++) {
        float d = fabsf(ref[k] - got[k]);
        if (d > m) {
            m = d;
            if (first < 0 && d > tol) first = (long)k;
        }
    }
    free((void*)ref);
    if (max_abs_err != NULL) *max_abs_err = m;
    int bad = (m > tol) ? 1 : 0;
    if (bad)
        printf("  [FAIL] %-22s max_abs_err=%.6g tol=%.2g first=%ld\n", name, (double)m, (double)tol, first);
    else
        printf("  [ ok ] %-22s max_abs_err=%.3g\n", name, (double)m);
    return bad;
}

/* ---- BN fold table (fold.json is one compact JSON object per line) ---- */

static int fold_key_int(const char* line, const char* key) {
    char pat[32];
    snprintf(pat, sizeof(pat), "\"%s\":", key);
    const char* hit = strstr(line, pat);
    if (hit == NULL) return -1;
    return (int)strtol(hit + strlen(pat), NULL, 10);
}

static size_t fold_key_size(const char* line, const char* key) {
    char pat[32];
    snprintf(pat, sizeof(pat), "\"%s\":", key);
    const char* hit = strstr(line, pat);
    if (hit == NULL) return (size_t)-1;
    return (size_t)strtoull(hit + strlen(pat), NULL, 10);
}

/* Copy the value of a JSON string field ("path") into buf. */
static void fold_key_str(const char* line, const char* key, char* buf, size_t cap) {
    char pat[32];
    snprintf(pat, sizeof(pat), "\"%s\":\"", key);
    const char* a = strstr(line, pat);
    if (a == NULL) { buf[0] = '\0'; return; }
    a += strlen(pat);
    size_t n = 0;
    while (a[n] != '\0' && a[n] != '"' && n + 1 < cap) n++;
    memcpy(buf, a, n);
    buf[n] = '\0';
}

int yolo26_ref_load_fold(yolo26_fold_entry* out, int cap) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/fold.json", ref_dir());
    unsigned char* bin = NULL;
    long bn = read_file(path, &bin);
    if (bn < 0) { fprintf(stderr, "ref_load_fold: missing %s\n", path); return -1; }

    int count = 0;
    char* line = (char*)bin;
    char* end = (char*)bin + bn;
    while (line < end && count < cap) {
        char* nl = (char*)memchr(line, '\n', (size_t)(end - line));
        if (nl == NULL) nl = end;
        size_t len = (size_t)(nl - line);
        if (len > 2 && line[0] == '{') {
            char* tmp = (char*)malloc(len + 1);
            memcpy(tmp, line, len);
            tmp[len] = '\0';
            yolo26_fold_entry e;
            memset(&e, 0, sizeof(e));
            e.seq    = fold_key_int(tmp, "seq");
            e.layer  = fold_key_int(tmp, "layer");
            e.in_c   = fold_key_int(tmp, "in_c");
            e.out_c  = fold_key_int(tmp, "out_c");
            e.k      = fold_key_int(tmp, "k");
            e.s      = fold_key_int(tmp, "s");
            e.p      = fold_key_int(tmp, "p");
            e.g      = fold_key_int(tmp, "g");
            e.has_bn = fold_key_int(tmp, "has_bn");
            e.act    = fold_key_int(tmp, "act");
            e.w_off  = fold_key_size(tmp, "w_off");
            e.w_cnt  = fold_key_size(tmp, "w_cnt");
            e.b_off  = fold_key_size(tmp, "b_off");
            e.b_cnt  = fold_key_size(tmp, "b_cnt");
            fold_key_str(tmp, "path", e.path, sizeof(e.path));
            fold_key_str(tmp, "parent", e.parent, sizeof(e.parent));
            free(tmp);
            if (e.seq >= 0 && e.in_c > 0 && e.w_cnt > 0) out[count++] = e;
        }
        line = nl + 1;
    }
    free(bin);
    if (line < end && count == cap) {
        fprintf(stderr, "ref_load_fold: more than %d entries\n", cap);
        return -1;
    }
    return count;
}

const float* yolo26_ref_load_blob(const char* file, size_t off, size_t cnt) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", ref_dir(), file);
    unsigned char* bin = NULL;
    long bn = read_file(path, &bin);
    if (bn < 0) { fprintf(stderr, "ref_load_blob: missing %s\n", path); return NULL; }
    size_t avail = (size_t)bn / sizeof(float);
    if (off + cnt > avail) {
        fprintf(stderr, "ref_load_blob: %s range [%zu,%zu) exceeds %zu floats\n",
                file, off, off + cnt, avail);
        free(bin);
        return NULL;
    }
    float* out = (float*)malloc(cnt * sizeof(float));
    memcpy(out, (const float*)bin + off, cnt * sizeof(float));
    free(bin);
    return out;
}

int ref_shape_eq(const char* name, int ndim, const size_t* shape) {
    int rndim = 0;
    size_t rshape[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const float* ref = ref_load(name, &rndim, rshape);
    if (ref == NULL) return -1;
    free((void*)ref);
    if (rndim != ndim) return 1;
    for (int i = 0; i < ndim; i++)
        if (rshape[i] != shape[i]) return 1;
    return 0;
}
