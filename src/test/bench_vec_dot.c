/*
 * FastIV - Fast image and vision
 * Benchmark: naive C scalar dot vs NEON dot across vector lengths.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "fiv_matrix.h"

#if defined(FIV_USE_ARM_NEON)
#include <arm_neon.h>
#endif

static double now_sec(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* ---- naive C scalar baseline (compiler auto-vectorized at -O2) ---- */
static float baseline_dot(const float* a, const float* b, size_t n)
{
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* ---- pure scalar: disable compiler vectorization for a fair floor ---- */
static float scalar_dot(const float* a, const float* b, size_t n)
{
    float s = 0.0f;
#pragma clang loop vectorize(disable) interleave(disable)
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

#if defined(FIV_USE_ARM_NEON)
/* ---- NEON kernel: 8x float32x4_t accumulators (32 floats/iter) + load-ahead ---- */
static float neon_dot(const float* a, const float* b, size_t n)
{
    if (n == 0) return 0.0f;

    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f),
                a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f),
                a4 = vdupq_n_f32(0.0f), a5 = vdupq_n_f32(0.0f),
                a6 = vdupq_n_f32(0.0f), a7 = vdupq_n_f32(0.0f);

    size_t i = 0;
    size_t nb = n / 32;

    if (nb >= 1)
    {
        float32x4_t x0 = vld1q_f32(a +  0), y0 = vld1q_f32(b +  0);
        float32x4_t x1 = vld1q_f32(a +  4), y1 = vld1q_f32(b +  4);
        float32x4_t x2 = vld1q_f32(a +  8), y2 = vld1q_f32(b +  8);
        float32x4_t x3 = vld1q_f32(a + 12), y3 = vld1q_f32(b + 12);
        float32x4_t x4 = vld1q_f32(a + 16), y4 = vld1q_f32(b + 16);
        float32x4_t x5 = vld1q_f32(a + 20), y5 = vld1q_f32(b + 20);
        float32x4_t x6 = vld1q_f32(a + 24), y6 = vld1q_f32(b + 24);
        float32x4_t x7 = vld1q_f32(a + 28), y7 = vld1q_f32(b + 28);

        for (size_t k = 1; k < nb; k++)
        {
            const float* pa = a + k * 32;
            const float* pb = b + k * 32;
            a0 = vfmaq_f32(a0, x0, y0);
            a1 = vfmaq_f32(a1, x1, y1);
            a2 = vfmaq_f32(a2, x2, y2);
            a3 = vfmaq_f32(a3, x3, y3);
            a4 = vfmaq_f32(a4, x4, y4);
            a5 = vfmaq_f32(a5, x5, y5);
            a6 = vfmaq_f32(a6, x6, y6);
            a7 = vfmaq_f32(a7, x7, y7);
            x0 = vld1q_f32(pa +  0); y0 = vld1q_f32(pb +  0);
            x1 = vld1q_f32(pa +  4); y1 = vld1q_f32(pb +  4);
            x2 = vld1q_f32(pa +  8); y2 = vld1q_f32(pb +  8);
            x3 = vld1q_f32(pa + 12); y3 = vld1q_f32(pb + 12);
            x4 = vld1q_f32(pa + 16); y4 = vld1q_f32(pb + 16);
            x5 = vld1q_f32(pa + 20); y5 = vld1q_f32(pb + 20);
            x6 = vld1q_f32(pa + 24); y6 = vld1q_f32(pb + 24);
            x7 = vld1q_f32(pa + 28); y7 = vld1q_f32(pb + 28);
        }

        a0 = vfmaq_f32(a0, x0, y0);
        a1 = vfmaq_f32(a1, x1, y1);
        a2 = vfmaq_f32(a2, x2, y2);
        a3 = vfmaq_f32(a3, x3, y3);
        a4 = vfmaq_f32(a4, x4, y4);
        a5 = vfmaq_f32(a5, x5, y5);
        a6 = vfmaq_f32(a6, x6, y6);
        a7 = vfmaq_f32(a7, x7, y7);
        i = nb * 32;
    }

    float32x4_t sum4  = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    float32x4_t sum4b = vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7));
    float32x4_t acc   = vaddq_f32(sum4, sum4b);

    for (; i + 4 <= n; i += 4)
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));

    float s = vaddvq_f32(acc);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
#endif /* FIV_USE_ARM_NEON */

/* time a kernel: run until >= 0.1s, return seconds per call */
static double bench_kernel(float (*fn)(const float*, const float*, size_t),
                          const float* a, const float* b, size_t n)
{
    /* warmup */
    volatile float sink = 0.0f;
    for (int w = 0; w < 16; w++) sink += fn(a, b, n);

    size_t iters = 1;
    double t = now_sec();
    for (size_t k = 0; k < iters; k++) sink += fn(a, b, n);
    t = now_sec() - t;
    while (t < 0.1)
    {
        iters *= 2;
        t = now_sec();
        for (size_t k = 0; k < iters; k++) sink += fn(a, b, n);
        t = now_sec() - t;
    }
    (void)sink;
    return t / (double)iters;
}

static size_t lengths[] = { 1, 2, 3, 7, 15, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144, 1048576 };

int main(void)
{
    printf("# vec dot benchmark (arm64), dtype=FIV_32F1\n");
    printf("# len      scalar(us)   autoNEON(us)   handNEON(us)   auto/spd   hand/spd   check\n");

    for (size_t li = 0; li < sizeof(lengths)/sizeof(lengths[0]); li++)
    {
        size_t n = lengths[li];
        float* a = (float*)malloc((size_t)n * sizeof(float));
        float* b = (float*)malloc((size_t)n * sizeof(float));
        for (size_t i = 0; i < n; i++)
        {
            a[i] = (float)((i * 3 + 1) % 97) / 32.0f - 1.0f;
            b[i] = (float)((i * 7 + 5) % 53) / 16.0f - 1.5f;
        }

        float auto_v = baseline_dot(a, b, n);
        double t_auto = bench_kernel(baseline_dot, a, b, n);
        float s_v = scalar_dot(a, b, n);
        double t_scalar = bench_kernel(scalar_dot, a, b, n);

        /* high-precision reference (double accumulation) for correctness */
        double ref = 0.0;
        for (size_t i = 0; i < n; i++) ref += (double)a[i] * (double)b[i];

#if defined(FIV_USE_ARM_NEON)
        float neon_v = neon_dot(a, b, n);
        double t_neon = bench_kernel(neon_dot, a, b, n);

        /* verify fiv_vec_dot (library, NEON-backed) against double reference */
        fiv_vec* va = fiv_create_tensor1d((int)n, FIV_32F1);
        fiv_vec* vb = fiv_create_tensor1d((int)n, FIV_32F1);
        for (size_t i = 0; i < n; i++) { va->data.fl[i] = a[i]; vb->data.fl[i] = b[i]; }
        fiv_scalar nv;
        fiv_ret r = fiv_vec_dot(&nv, va, vb);
        double tol = 1e-3 * (fabs(ref) + 1.0);
        int ok = (r == FIV_RET_OK)
                 && (fabs((double)s_v     - ref) <= tol)
                 && (fabs((double)auto_v   - ref) <= tol)
                 && (fabs((double)neon_v   - ref) <= tol)
                 && (fabs((double)nv.data.value_fp32 - ref) <= tol);
        fiv_release_tensor1d(&va);
        fiv_release_tensor1d(&vb);

        printf("%-9zu %12.4f   %12.4f   %12.4f   %8.2fx   %8.2fx   %s\n",
               n, t_scalar * 1e6, t_auto * 1e6, t_neon * 1e6,
               t_scalar / t_auto, t_scalar / t_neon, ok ? "OK" : "MISMATCH");
#else
        printf("%-9zu %12.4f   %12.4f   %12s   %8s   %8s   %s\n",
               n, t_scalar * 1e6, t_auto * 1e6, "n/a", "n/a", "n/a", "NEON off");
#endif

        free(a); free(b);
    }
    return 0;
}
