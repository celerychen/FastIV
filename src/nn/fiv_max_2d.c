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

#include "fiv_max_2d.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "fiv_common.h"

/* SIMD headers are auto-included and detected by api/fiv_data_typedefs.h
   (FIV_USE_ARM_NEON / FIV_USE_AVX2 / FIV_USE_X86_SIMD). */

/* 2x2 stride-2 max pooling over the last two (H, W) dims; leading dims
   (batch/channels) are pooled independently. The training forward caches each
   output's argmax (flat offset inside its channel plane) so backward can route
   grad_output back to the selected element only; the inference forward needs
   no argmax bookkeeping and runs a leaner loop. */

/* Shared dims extraction: (..., C, H, W) -> batch*times, channels, H, W. */
static void fiv_max_2d_dims(const fiv_tensor_hdr* in, size_t* B, size_t* C,
                            size_t* H, size_t* W)
{
    switch (in->id) {
    case FIV_ID_TENSOR3D:
        *B = 1;
        *C = ((const fiv_tensor3d*)in)->channels;
        *H = ((const fiv_tensor3d*)in)->height;
        *W = ((const fiv_tensor3d*)in)->width;
        break;
    case FIV_ID_TENSOR4D:
        *B = ((const fiv_tensor4d*)in)->batch;
        *C = ((const fiv_tensor4d*)in)->channels;
        *H = ((const fiv_tensor4d*)in)->height;
        *W = ((const fiv_tensor4d*)in)->width;
        break;
    default:
        *B = ((const fiv_tensor5d*)in)->batch * ((const fiv_tensor5d*)in)->times;
        *C = ((const fiv_tensor5d*)in)->channels;
        *H = ((const fiv_tensor5d*)in)->height;
        *W = ((const fiv_tensor5d*)in)->width;
        break;
    }
}

void* fiv_max_2d_node_create(void* params)
{
    (void)params;
    fiv_max_2d_node* n = (fiv_max_2d_node*)fiv_malloc(sizeof(fiv_max_2d_node));
    if (!n) return NULL;
    memset(n, 0, sizeof(fiv_max_2d_node));
    n->base.create_fn    = fiv_max_2d_node_create;
    n->base.release_fn   = fiv_max_2d_node_release;
    n->base.forward_fn   = fiv_max_2d_node_forward;
    n->base.backward_fn  = fiv_max_2d_node_backward;
    n->base.inference_fn = fiv_max_2d_node_inference;
    n->base.alloc_out_fn = fiv_max_2d_node_alloc_out;
    return n;
}

void fiv_max_2d_node_release(void* op_state)
{
    fiv_max_2d_node* n = (fiv_max_2d_node*)op_state;
    if (!n) return;
    if (n->argmax) fiv_free(n->argmax);
    fiv_free(n);
}

/* Input: (..., C, H, W) -> output (..., C, H/2, W/2), same ndim. */
void* fiv_max_2d_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{
    *out_ret = FIV_RET_OK;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!in) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }
    if (in->id < FIV_ID_TENSOR3D || in->id > FIV_ID_TENSOR5D) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }
    if (in->dtype != FIV_32F1 || in->data_continue == 0) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }

    size_t B = 0;
    size_t C = 0;
    size_t H = 0;
    size_t W = 0;
    fiv_max_2d_dims(in, &B, &C, &H, &W);
    size_t OH = H / 2;
    size_t OW = W / 2;
    if (OH == 0 || OW == 0) {
        *out_ret = FIV_RET_ERR_PARA;   /* input too small to downsample */
        return NULL;
    }

    fiv_tensor_hdr* out = (fiv_tensor_hdr*)existing_output;
    if (out && out->id == in->id && out->dtype == FIV_32F1 && out->data_continue == 1) {
        size_t oB = 0;
        size_t oC = 0;
        size_t oH = 0;
        size_t oW = 0;
        fiv_max_2d_dims(out, &oB, &oC, &oH, &oW);
        if (oB == B && oC == C && oH == OH && oW == OW) return out;
    }
    if (out) fiv_release_tensor((void**)&out);

    switch (in->id) {
    case FIV_ID_TENSOR3D: {
        size_t sh[3] = { C, OH, OW };
        out = (fiv_tensor_hdr*)fiv_create_tensor3d(sh, FIV_32F1);
        break;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)in;
        size_t sh[4] = { t->batch, C, OH, OW };
        out = (fiv_tensor_hdr*)fiv_create_tensor4d(sh, FIV_32F1);
        break;
    }
    default: {
        const fiv_tensor5d* t = (const fiv_tensor5d*)in;
        size_t sh[5] = { t->batch, t->times, C, OH, OW };
        out = (fiv_tensor_hdr*)fiv_create_tensor5d(sh, FIV_32F1);
        break;
    }
    }
    if (!out) { *out_ret = FIV_RET_ERR_MEM; return NULL; }
    return out;
}

/* Training forward: pool and cache the argmax of every output element. */
static fiv_ret fiv_max_2d_compute(fiv_max_2d_node* n, fiv_tensor_hdr* out, const fiv_tensor_hdr* in)
{
    size_t B = 0;
    size_t C = 0;
    size_t H = 0;
    size_t W = 0;
    fiv_max_2d_dims(in, &B, &C, &H, &W);
    size_t OH = H / 2;
    size_t OW = W / 2;
    size_t n_chan = B * C;
    size_t iHW = H * W;
    size_t oHW = OH * OW;
    size_t need = n_chan * oHW;
    if (need == 0) return FIV_RET_OK;
    if (!n->argmax || n->n_out < need) {
        int* a = (int*)fiv_realloc(n->argmax, need * sizeof(int));
        if (!a) return FIV_RET_ERR_MEM;
        n->argmax = a;
        n->n_out = need;
    }

    const ivf32* ip = in->data.fl;
    ivf32* op = out->data.fl;
    for (size_t ch = 0; ch < n_chan; ch++) {
        const ivf32* src = ip + ch * iHW;
        ivf32* dst = op + ch * oHW;
        int* am = n->argmax + ch * oHW;
        for (size_t oy = 0; oy < OH; oy++) {
            size_t y0 = oy * 2;
            size_t y1 = y0 + 1;   /* window never crosses the edge: 2*(OH-1)+1 <= H-1 */
            for (size_t ox = 0; ox < OW; ox++) {
                size_t x0 = ox * 2;
                size_t x1 = x0 + 1;
                ivf32 m = src[y0 * W + x0];
                int mo = (int)(y0 * W + x0);
                ivf32 v;

                v = src[y0 * W + x1];
                if (v > m) {
                    m = v;
                    mo = (int)(y0 * W + x1);
                }
                v = src[y1 * W + x0];
                if (v > m) {
                    m = v;
                    mo = (int)(y1 * W + x0);
                }
                v = src[y1 * W + x1];
                if (v > m) {
                    m = v;
                    mo = (int)(y1 * W + x1);
                }
                dst[oy * OW + ox] = m;
                am[oy * OW + ox] = mo;
            }
        }
    }
    return FIV_RET_OK;
}

/* Inference forward: same pooling, no argmax bookkeeping (no allocation, no
   position cache, no per-element store of the winning offset). The 2x2 window
   never crosses the input edge (2*(OH-1)+1 <= H-1, 2*(OW-1)+1 <= W-1), so the
   whole pass is branch-free: a SIMD main loop over OW plus a scalar tail. */
static fiv_ret fiv_max_2d_compute_infer(fiv_tensor_hdr* out, const fiv_tensor_hdr* in)
{
    size_t B = 0;
    size_t C = 0;
    size_t H = 0;
    size_t W = 0;
    fiv_max_2d_dims(in, &B, &C, &H, &W);
    size_t OH = H / 2;
    size_t OW = W / 2;
    size_t n_chan = B * C;
    size_t iHW = H * W;
    size_t oHW = OH * OW;
    if (oHW == 0) return FIV_RET_OK;

    const ivf32* ip = in->data.fl;
    ivf32* op = out->data.fl;
#if defined(FIV_USE_AVX2)
    /* cross-lane reorder so the 8 outputs of one iteration land in order */
    const int offset[8] = { 0, 1, 4, 5, 2, 3, 6, 7 };
    const __m256i t_c = _mm256_loadu_si256((const __m256i*)offset);
#endif
    for (size_t ch = 0; ch < n_chan; ch++) {
        const ivf32* src = ip + ch * iHW;
        ivf32* dst = op + ch * oHW;
        for (size_t oy = 0; oy < OH; oy++) {
            const ivf32* r0 = src + 2 * oy * W;   /* window top row */
            const ivf32* r1 = r0 + W;             /* window bottom row */
            ivf32* drow = dst + oy * OW;
            size_t x = 0;
#if defined(FIV_USE_ARM_NEON)
            for (; x + 4 <= OW; x += 4) {
                float32x4_t a0 = vld1q_f32(r0 + 2 * x);
                float32x4_t a1 = vld1q_f32(r0 + 2 * x + 4);
                float32x4_t b0 = vld1q_f32(r1 + 2 * x);
                float32x4_t b1 = vld1q_f32(r1 + 2 * x + 4);
                float32x4_t v0 = vmaxq_f32(a0, b0);       /* vertical: col max */
                float32x4_t v1 = vmaxq_f32(a1, b1);
                float32x4_t h0 = vmaxq_f32(v0, vrev64q_f32(v0));  /* [h01 h01 h23 h23] */
                float32x4_t h1 = vmaxq_f32(v1, vrev64q_f32(v1));  /* [h45 h45 h67 h67] */
                float32x4x2_t uz = vuzpq_f32(h0, h1);      /* [h01 h23 h45 h67] */
                vst1q_f32(drow + x, uz.val[0]);
            }
#elif defined(FIV_USE_AVX2)
            for (; x + 8 <= OW; x += 8) {
                __m256 t1 = _mm256_loadu_ps(r0 + 2 * x);
                __m256 t2 = _mm256_loadu_ps(r0 + 2 * x + 8);
                __m256 t3 = _mm256_loadu_ps(r1 + 2 * x);
                __m256 t4 = _mm256_loadu_ps(r1 + 2 * x + 8);
                t1 = _mm256_max_ps(t1, t3);            /* vertical: col max */
                t2 = _mm256_max_ps(t2, t4);
                __m256 t5 = _mm256_permute_ps(t1, 245); /* 0b11110101 */
                __m256 t6 = _mm256_permute_ps(t2, 245);
                t1 = _mm256_max_ps(t1, t5);            /* horizontal: pair max */
                t2 = _mm256_max_ps(t2, t6);
                t5 = _mm256_shuffle_ps(t1, t2, 136);   /* 0b10001000 */
                t6 = _mm256_permutevar8x32_ps(t5, t_c);
                _mm256_storeu_ps(drow + x, t6);
            }
#elif defined(FIV_USE_X86_SIMD)
            for (; x + 4 <= OW; x += 4) {
                __m128 t1 = _mm_loadu_ps(r0 + 2 * x);
                __m128 t2 = _mm_loadu_ps(r0 + 2 * x + 4);
                __m128 t3 = _mm_loadu_ps(r1 + 2 * x);
                __m128 t4 = _mm_loadu_ps(r1 + 2 * x + 4);
                t1 = _mm_max_ps(t1, t3);                   /* vertical: col max */
                t2 = _mm_max_ps(t2, t4);
                __m128 t5 = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(t1), 4));
                t5 = _mm_max_ps(t5, t1);                   /* horizontal: pair max */
                __m128 t6 = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(t2), 4));
                t6 = _mm_max_ps(t6, t2);
                t3 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(t5), 0x08));
                t4 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(t6), 0x08));
                t3 = _mm_shuffle_ps(t3, t4, 68);           /* 0b01000100 */
                _mm_storeu_ps(drow + x, t3);
            }
#endif
            for (; x < OW; x++) {
                ivf32 t0 = r0[2 * x];
                ivf32 t1 = r0[2 * x + 1];
                ivf32 t2 = r1[2 * x];
                ivf32 t3 = r1[2 * x + 1];
                ivf32 m = t0;
                if (m < t1) m = t1;
                if (m < t2) m = t2;
                if (m < t3) m = t3;
                drow[x] = m;
            }
        }
    }
    return FIV_RET_OK;
}

fiv_ret fiv_max_2d_node_forward(void* op_state, void* output, void* input)
{
    return fiv_max_2d_compute((fiv_max_2d_node*)op_state, (fiv_tensor_hdr*)output,
                              (const fiv_tensor_hdr*)input);
}

fiv_ret fiv_max_2d_node_inference(void* op_state, void* output, void* input)
{
    (void)op_state;
    return fiv_max_2d_compute_infer((fiv_tensor_hdr*)output, (const fiv_tensor_hdr*)input);
}

/* Route each grad_output element to the input element selected by forward. */
fiv_ret fiv_max_2d_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    (void)input;
    fiv_max_2d_node* n = (fiv_max_2d_node*)op_state;
    const fiv_tensor_hdr* go = (const fiv_tensor_hdr*)grad_output;
    fiv_tensor_hdr* gi = (fiv_tensor_hdr*)grad_input;
    if (!n || !go) return FIV_RET_ERR_PARA;
    if (!gi) return FIV_RET_OK;   /* node-0 input: nothing to accumulate */

    size_t B = 0;
    size_t C = 0;
    size_t OH = 0;
    size_t OW = 0;
    fiv_max_2d_dims(go, &B, &C, &OH, &OW);
    size_t H = 0;
    size_t W = 0;
    {
        size_t hB = 0;
        size_t hC = 0;
        fiv_max_2d_dims(gi, &hB, &hC, &H, &W);
        (void)hB;
        (void)hC;
    }
    size_t n_chan = B * C;
    size_t oHW = OH * OW;
    const ivf32* gp = go->data.fl;
    ivf32* gip = gi->data.fl;
    for (size_t ch = 0; ch < n_chan; ch++) {
        const ivf32* goc = gp + ch * oHW;
        ivf32* gic = gip + ch * (H * W);
        const int* am = n->argmax + ch * oHW;
        for (size_t k = 0; k < oHW; k++) gic[(size_t)am[k]] += goc[k];
    }
    return FIV_RET_OK;
}
