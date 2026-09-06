/* P0 self-check for the YOLO26 reference loader (app/yolo26/test/yolo26_ref.c).
 * Proves the loader reads gen_ref.py's f32 and json files correctly and that the
 * shape-equality helper rejects a tampered layout.
 *
 * Build: cc yolo26_ref.c test_yolo26_ref.c -I. -o test_yolo26_ref
 *       YOLO26_REF_DIR=.../ref ./test_yolo26_ref
 * (Wired into the project Makefile as test_yolo26_ref in P1.) */

#include "yolo26_ref.h"
#include "fiv_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int g_fail = 0;

#define CHECK(cond)                                              \
    do {                                                         \
        if (!(cond)) {                                           \
            printf("  CHECK FAIL: %s:%d\n", __FILE__, __LINE__); \
            g_fail++;                                            \
        }                                                        \
    } while (0)

static ivf32 silu_c(ivf32 x) { return x / (1.0f + expf(-x)); }

int main(void) {
    /* 1) Loader self-check: load 'input' and compare to itself -> err exactly 0. */
    {
        int ndim;
        size_t shape[8];
        const ivf32* a = ref_load("input", &ndim, shape);
        CHECK(a != NULL);
        ivf32 err;
        CHECK(ref_cmp("input", a, 1e-6f, &err) == 0);
        CHECK(err == 0.0f);
        fiv_free((void*)a);
    }

    /* 2) End-to-end: silu transform of syn_silu_src must match syn_silu_out. */
    {
        size_t count = 0;
        const ivf32* src = ref_load_n("syn_silu_src", &count);
        const ivf32* out = ref_load_n("syn_silu_out", &count);
        CHECK(src != NULL && out != NULL);
        ivf32* mine = (ivf32*)fiv_malloc(count * sizeof(ivf32));
        for (size_t k = 0; k < count; k++) mine[k] = silu_c(src[k]);
        ivf32 err;
        CHECK(ref_cmp("syn_silu_out", mine, 1e-6f, &err) == 0);
        fiv_free(mine);
        fiv_free((void*)src);
        fiv_free((void*)out);
    }

    /* 3) Shape helper: recorded shape matches, a H/W swap must not.
     *    Use a non-square tensor (syn_pool_src = [1,4,9,11]) so the swap
     *    is observable (input is 64x64 square, a swap would be a no-op). */
    {
        int ndim;
        size_t shape[8];
        const ivf32* a = ref_load("syn_pool_src", &ndim, shape);
        CHECK(a != NULL);
        CHECK(ref_shape_eq("syn_pool_src", ndim, shape) == 0);
        size_t swapped[4] = {shape[0], shape[1], shape[3], shape[2]};
        CHECK(ref_shape_eq("syn_pool_src", ndim, swapped) != 0);
        fiv_free((void*)a);
    }

    if (g_fail == 0) {
        printf("P0 ref-loader self-check: PASS\n");
        return 0;
    }
    printf("P0 ref-loader self-check: FAIL (%d)\n", g_fail);
    return 1;
}
