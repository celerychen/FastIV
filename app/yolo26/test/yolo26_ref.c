/* Reference-tensor loader for the YOLO26 -> FastIV port. See yolo26_ref.h. */

#include "yolo26_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fiv_common.h"

/* explicit directory override (yolo26_ref_set_dir); else env, else default. */
static char override_dir[1024] = { 0 };

void yolo26_ref_set_dir(const char* dir)
{
    if (dir != NULL)
        snprintf(override_dir, sizeof(override_dir), "%s", dir);
    else
        override_dir[0] = '\0';
}

static const char* ref_dir(void)
{
    if (override_dir[0] != '\0') return override_dir;
    const char* dir = getenv("YOLO26_REF_DIR");
    return dir != NULL ? dir : YOLO26_REF_DIR;
}

/* Read a whole file into an fiv_malloc'd buffer; returns byte count or -1. */
static long read_file(const char* path, unsigned char** out)
{
    FILE* file = fopen(path, "rb");
    if (file == NULL) return -1;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return -1;
    }
    long byte_count = ftell(file);
    fseek(file, 0, SEEK_SET);
    unsigned char* buf =
        (unsigned char*)fiv_malloc(byte_count > 0 ? (size_t)byte_count : 1u);
    if (fread(buf, 1, (size_t)byte_count, file) != (size_t)byte_count) {
        fclose(file);
        fiv_free(buf);
        return -1;
    }
    fclose(file);
    *out = buf;
    return byte_count;
}

/* Parse the first integer array after the "shape" key in the json text. */
static int parse_shape(const char* json_text, size_t json_len,
                       int* ndim, size_t* shape)
{
    int dim = 0;
    const char* scan = (const char*)memchr(json_text, '"', json_len);
    while (scan != NULL) {
        if (strncmp(scan, "\"shape\"", 7) == 0) break;
        scan = (const char*)memchr(scan + 1, '"',
                                   json_text + json_len - (scan + 1));
    }
    if (scan == NULL) return 0;
    scan = (const char*)memchr(scan, '[', json_text + json_len - scan);
    if (scan == NULL) return 0;
    scan++;
    while (*scan != '\0' && *scan != ']') {
        while (*scan == ' ' || *scan == ',' || *scan == '\n' || *scan == '\t')
            scan++;
        if (*scan >= '0' && *scan <= '9') {
            if (dim < 8)
                shape[dim] = (size_t)strtoul(scan, (char**)&scan, 10);
            else {
                while (*scan >= '0' && *scan <= '9') scan++;
            }
            dim++;
        } else if (*scan == ']') {
            break;
        } else {
            scan++;
        }
    }
    *ndim = dim;
    return dim;
}

const ivf32* ref_load(const char* name, int* ndim, size_t* shape)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.json", ref_dir(), name);
    unsigned char* json_text = NULL;
    long json_len = read_file(path, &json_text);
    int dim = 0;
    size_t ref_shape[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    if (json_len >= 0) {
        parse_shape((const char*)json_text, (size_t)json_len, &dim, ref_shape);
        fiv_free(json_text);
    } else {
        fprintf(stderr, "ref_load: missing json for '%s'\n", name);
        return NULL;
    }

    snprintf(path, sizeof(path), "%s/%s.f32", ref_dir(), name);
    unsigned char* bin = NULL;
    long bin_len = read_file(path, &bin);
    if (bin_len < 0) {
        fprintf(stderr, "ref_load: missing f32 for '%s'\n", name);
        return NULL;
    }
    if (ndim != NULL) *ndim = dim;
    if (shape != NULL) {
        for (int i = 0; i < dim && i < 8; i++) shape[i] = ref_shape[i];
    }
    return (const ivf32*)bin; /* caller frees with fiv_free */
}

const ivf32* ref_load_n(const char* name, size_t* count)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.f32", ref_dir(), name);
    unsigned char* bin = NULL;
    long bin_len = read_file(path, &bin);
    if (bin_len < 0) {
        fprintf(stderr, "ref_load_n: missing f32 for '%s'\n", name);
        if (count != NULL) *count = 0;
        return NULL;
    }
    if (count != NULL) *count = (size_t)bin_len / sizeof(ivf32);
    return (const ivf32*)bin;
}

int ref_cmp(const char* name, const ivf32* got, ivf32 tol, ivf32* max_abs_err)
{
    size_t count = 0;
    const ivf32* ref = ref_load_n(name, &count);
    if (ref == NULL) {
        if (max_abs_err != NULL) *max_abs_err = -1.0f;
        return -1;
    }
    ivf32 worst = 0.0f;
    long first_bad = -1;
    for (size_t k = 0; k < count; k++) {
        ivf32 diff = fabsf(ref[k] - got[k]);
        if (diff > worst) {
            worst = diff;
            if (first_bad < 0 && diff > tol) first_bad = (long)k;
        }
    }
    fiv_free((void*)ref);
    if (max_abs_err != NULL) *max_abs_err = worst;
    int bad = (worst > tol) ? 1 : 0;
    if (bad)
        printf("  [FAIL] %-22s max_abs_err=%.6g tol=%.2g first=%ld\n",
               name, (double)worst, (double)tol, first_bad);
    else
        printf("  [ ok ] %-22s max_abs_err=%.3g\n", name, (double)worst);
    return bad;
}

/* ---- BN fold table (fold.json is one compact JSON object per line) ---- */

static int fold_key_int(const char* row, const char* key)
{
    char pattern[32];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char* hit = strstr(row, pattern);
    if (hit == NULL) return -1;
    return (int)strtol(hit + strlen(pattern), NULL, 10);
}

static size_t fold_key_size(const char* row, const char* key)
{
    char pattern[32];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char* hit = strstr(row, pattern);
    if (hit == NULL) return (size_t)-1;
    return (size_t)strtoull(hit + strlen(pattern), NULL, 10);
}

/* Copy the value of a JSON string field into buf. */
static void fold_key_str(const char* row, const char* key, char* buf, size_t cap)
{
    char pattern[32];
    snprintf(pattern, sizeof(pattern), "\"%s\":\"", key);
    const char* start = strstr(row, pattern);
    if (start == NULL) {
        buf[0] = '\0';
        return;
    }
    start += strlen(pattern);
    size_t len = 0;
    while (start[len] != '\0' && start[len] != '"' && len + 1 < cap) len++;
    memcpy(buf, start, len);
    buf[len] = '\0';
}

int yolo26_ref_load_fold(yolo26_fold_entry* out, int cap)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/fold.json", ref_dir());
    unsigned char* bin = NULL;
    long bin_len = read_file(path, &bin);
    if (bin_len < 0) {
        fprintf(stderr, "ref_load_fold: missing %s\n", path);
        return -1;
    }

    int count = 0;
    char* cursor = (char*)bin;
    char* end    = (char*)bin + bin_len;
    while (cursor < end && count < cap) {
        char* newline = (char*)memchr(cursor, '\n', (size_t)(end - cursor));
        if (newline == NULL) newline = end;
        size_t len = (size_t)(newline - cursor);
        if (len > 2 && cursor[0] == '{') {
            char* row = (char*)fiv_malloc(len + 1);
            memcpy(row, cursor, len);
            row[len] = '\0';

            yolo26_fold_entry entry;
            memset(&entry, 0, sizeof(entry));
            entry.seq              = fold_key_int(row, "seq");
            entry.layer            = fold_key_int(row, "layer");
            entry.input_channels   = fold_key_int(row, "in_c");
            entry.output_channels  = fold_key_int(row, "out_c");
            entry.kernel           = fold_key_int(row, "k");
            entry.stride           = fold_key_int(row, "s");
            entry.padding          = fold_key_int(row, "p");
            entry.groups           = fold_key_int(row, "g");
            entry.has_bn           = fold_key_int(row, "has_bn");
            entry.act              = fold_key_int(row, "act");
            entry.w_off            = fold_key_size(row, "w_off");
            entry.w_cnt            = fold_key_size(row, "w_cnt");
            entry.b_off            = fold_key_size(row, "b_off");
            entry.b_cnt            = fold_key_size(row, "b_cnt");
            fold_key_str(row, "path", entry.path, sizeof(entry.path));
            fold_key_str(row, "parent", entry.parent, sizeof(entry.parent));
            fiv_free(row);

            if (entry.seq >= 0 && entry.input_channels > 0 && entry.w_cnt > 0)
                out[count++] = entry;
        }
        cursor = newline + 1;
    }
    fiv_free(bin);
    if (cursor < end && count == cap) {
        fprintf(stderr, "ref_load_fold: more than %d entries\n", cap);
        return -1;
    }
    return count;
}

const ivf32* yolo26_ref_load_blob(const char* file, size_t off, size_t cnt)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", ref_dir(), file);
    unsigned char* bin = NULL;
    long bin_len = read_file(path, &bin);
    if (bin_len < 0) {
        fprintf(stderr, "ref_load_blob: missing %s\n", path);
        return NULL;
    }
    size_t avail = (size_t)bin_len / sizeof(ivf32);
    if (off + cnt > avail) {
        fprintf(stderr, "ref_load_blob: %s range [%zu,%zu) exceeds %zu floats\n",
                file, off, off + cnt, avail);
        fiv_free(bin);
        return NULL;
    }
    ivf32* out = (ivf32*)fiv_malloc(cnt * sizeof(ivf32));
    memcpy(out, (const ivf32*)bin + off, cnt * sizeof(ivf32));
    fiv_free(bin);
    return out;
}

int ref_shape_eq(const char* name, int ndim, const size_t* shape)
{
    int ref_ndim = 0;
    size_t ref_shape[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    const ivf32* ref = ref_load(name, &ref_ndim, ref_shape);
    if (ref == NULL) return -1;
    fiv_free((void*)ref);
    if (ref_ndim != ndim) return 1;
    for (int i = 0; i < ndim; i++)
        if (ref_shape[i] != shape[i]) return 1;
    return 0;
}
