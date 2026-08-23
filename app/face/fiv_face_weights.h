#ifndef FIV_FACE_WEIGHTS_H
#define FIV_FACE_WEIGHTS_H

#include "fiv_data_typedefs.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FIV_WT_NAME_MAX 48

typedef struct {
    char   name[FIV_WT_NAME_MAX];
    int    ndim;
    int    dims[4];
    ivf32* data;   /* float32, row-major, owned by loader */
} fiv_weight_tensor;

typedef struct {
    fiv_weight_tensor* items;
    int                n;
} fiv_weights;

int          fiv_weights_load(const char* path, fiv_weights* w);
void         fiv_weights_free(fiv_weights* w);
const ivf32* fiv_weights_get(const fiv_weights* w, const char* name,
                             int* ndim, int* dims);

#ifdef __cplusplus
}
#endif

#endif /* FIV_FACE_WEIGHTS_H */
