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

#include "fiv_prelu_node.h"
#include "fiv_common.h"

#include <string.h>

static fiv_ret fiv_prelu_compute(fiv_tensor_hdr* out, const fiv_tensor_hdr* in,
                                 const ivf32* alpha, int channels)
{
    if (!out || !in || !alpha || channels < 1) return FIV_RET_ERR_PARA;
    if (in->dtype != FIV_32F1 || out->dtype != FIV_32F1) return FIV_RET_ERR_NOT_SUPPORT;
    if (!in->data_continue || !out->data_continue) return FIV_RET_ERR_PARA;
    if (out->total_bytes != in->total_bytes) return FIV_RET_ERR_PARA;

    size_t n = in->total_bytes / sizeof(ivf32);
    size_t hw = n / (size_t)channels;              /* elements per channel plane (NCHW) */
    const ivf32* src = in->data.fl;
    ivf32* dst = out->data.fl;
    /* Hoist the per-channel alpha lookup out of the element loop: the channel
       index only changes once per plane, so iterate plane-by-plane and drop the
       per-element integer division entirely (behaviourally identical). */
#if defined(FIV_USE_AVX2)
    const __m256 zero = _mm256_setzero_ps();
    for (size_t c = 0; c < (size_t)channels; c++) {
        ivf32 a = alpha[c];
        const ivf32* s = src + c * hw;
        ivf32*       d = dst + c * hw;
        __m256 av = _mm256_set1_ps(a);
        size_t i = 0;
        /* out = v>=0 ? v : a*v  ->  neg = v*a; mask = v>=0; blendv(neg, v, mask).
           Unroll the SIMD loop 2x: the mul and cmp of each group only depend on
           the group's own load (independent of each other), and the two groups
           give two such dependency chains to interleave, hiding the load->blendv
           latency instead of stalling the single chain in the 1x body.
           (Channel-loop unrolling was tried and reverted: streaming two channel
           planes at once is memory-bound and hurts cache locality, ~0.2 ms slower.) */
        for (; i + 16 <= hw; i += 16) {
            __m256 v0    = _mm256_loadu_ps(s + i);
            __m256 v1    = _mm256_loadu_ps(s + i + 8);
            __m256 neg0  = _mm256_mul_ps(v0, av);
            __m256 neg1  = _mm256_mul_ps(v1, av);
            __m256 mask0 = _mm256_cmp_ps(v0, zero, _CMP_GE_OQ);
            __m256 mask1 = _mm256_cmp_ps(v1, zero, _CMP_GE_OQ);
            __m256 r0    = _mm256_blendv_ps(neg0, v0, mask0);
            __m256 r1    = _mm256_blendv_ps(neg1, v1, mask1);
            _mm256_storeu_ps(d + i, r0);
            _mm256_storeu_ps(d + i + 8, r1);
        }
        for (; i + 8 <= hw; i += 8) {
            __m256 v    = _mm256_loadu_ps(s + i);
            __m256 neg  = _mm256_mul_ps(v, av);
            __m256 mask = _mm256_cmp_ps(v, zero, _CMP_GE_OQ);
            __m256 r    = _mm256_blendv_ps(neg, v, mask);
            _mm256_storeu_ps(d + i, r);
        }
        for (; i < hw; i++) {                     /* scalar tail */
            ivf32 v = s[i];
            d[i] = (v >= 0.0f) ? v : a * v;
        }
    }
#elif defined(FIV_USE_ARM_NEON)
    for (size_t c = 0; c < (size_t)channels; c++) {
        ivf32 a = alpha[c];
        const ivf32* s = src + c * hw;
        ivf32*       d = dst + c * hw;
        /* out = v + (a-1)*min(v,0): positives pass through, negatives fold to
           a*v. float32x4x2_t carries two quad lanes (8 floats) per load/store
           pair via vld1q_f32_x2/vst1q_f32_x2, halving the body vs bare quads
           and keeping the per-lane formula bit-identical. */
        float32x4_t coef = vdupq_n_f32(a - 1.0f);
        float32x4_t zero = vdupq_n_f32(0.0f);
        size_t i = 0;
        /* load-ahead software pipelining (mirrors fiv_vec_dot's NEON scheme):
           load the first 16-element block ahead of the loop, then in the loop
           compute+store the already-loaded block while issuing the next load so
           the load latency overlaps the fma chain instead of stalling it. */
        size_t nb = hw / 16;
        if (nb >= 1) {
            float32x4x2_t v0 = vld1q_f32_x2(s + i);
            float32x4x2_t v1 = vld1q_f32_x2(s + i + 8);
            for (size_t k = 1; k < nb; k++) {
                float32x4x2_t r0, r1;
                r0.val[0] = vfmaq_f32(v0.val[0], vminq_f32(v0.val[0], zero), coef);
                r0.val[1] = vfmaq_f32(v0.val[1], vminq_f32(v0.val[1], zero), coef);
                r1.val[0] = vfmaq_f32(v1.val[0], vminq_f32(v1.val[0], zero), coef);
                r1.val[1] = vfmaq_f32(v1.val[1], vminq_f32(v1.val[1], zero), coef);
                vst1q_f32_x2(d + i,      r0);
                vst1q_f32_x2(d + i + 8,  r1);
                i += 16;
                v0 = vld1q_f32_x2(s + i);
                v1 = vld1q_f32_x2(s + i + 8);
            }
            float32x4x2_t r0, r1;
            r0.val[0] = vfmaq_f32(v0.val[0], vminq_f32(v0.val[0], zero), coef);
            r0.val[1] = vfmaq_f32(v0.val[1], vminq_f32(v0.val[1], zero), coef);
            r1.val[0] = vfmaq_f32(v1.val[0], vminq_f32(v1.val[0], zero), coef);
            r1.val[1] = vfmaq_f32(v1.val[1], vminq_f32(v1.val[1], zero), coef);
            vst1q_f32_x2(d + i,      r0);
            vst1q_f32_x2(d + i + 8,  r1);
            i += 16;
        }
        for (; i + 8 <= hw; i += 8) {
            float32x4x2_t v = vld1q_f32_x2(s + i);
            float32x4x2_t r;
            r.val[0] = vfmaq_f32(v.val[0], vminq_f32(v.val[0], zero), coef);
            r.val[1] = vfmaq_f32(v.val[1], vminq_f32(v.val[1], zero), coef);
            vst1q_f32_x2(d + i, r);
        }
        for (; i < hw; i++) {                     /* scalar tail */
            ivf32 v = s[i];
            d[i] = (v >= 0.0f) ? v : a * v;
        }
    }
#else
    for (size_t c = 0; c < (size_t)channels; c++) {
        ivf32 a = alpha[c];
        const ivf32* s = src + c * hw;
        ivf32*       d = dst + c * hw;
        for (size_t i = 0; i < hw; i++) {
            ivf32 v = s[i];
            d[i] = (v >= 0.0f) ? v : a * v;
        }
    }
#endif
    return FIV_RET_OK;
}

void* fiv_prelu_node_create(void* params)
{
    const fiv_prelu_node_params* p = (const fiv_prelu_node_params*)params;
    if (!p || p->channels <= 0) return NULL;
    fiv_prelu_node* node = (fiv_prelu_node*)fiv_malloc(sizeof(fiv_prelu_node));
    if (!node) return NULL;
    memset(node, 0, sizeof(fiv_prelu_node));
    node->base.create_fn    = fiv_prelu_node_create;
    node->base.release_fn   = fiv_prelu_node_release;
    node->base.forward_fn   = fiv_prelu_node_forward;
    node->base.backward_fn  = fiv_prelu_node_backward;
    node->base.inference_fn = fiv_prelu_node_inference;
    node->base.alloc_out_fn = fiv_prelu_node_alloc_out;
    node->channels = p->channels;

    node->alpha = (ivf32*)fiv_malloc(sizeof(ivf32) * (size_t)p->channels);
    if (!node->alpha) { fiv_free(node); return NULL; }
    if (p->alpha) {
        memcpy(node->alpha, p->alpha, sizeof(ivf32) * (size_t)p->channels);
    } else {
        for (int c = 0; c < p->channels; c++) node->alpha[c] = 0.25f;
    }
    return node;
}

void fiv_prelu_node_release(void* op_state)
{
    fiv_prelu_node* node = (fiv_prelu_node*)op_state;
    if (!node) return;
    fiv_free(node->alpha);
    fiv_free(node);
}

fiv_ret fiv_prelu_node_forward(void* op_state, void* output, void* input)
{
    fiv_prelu_node* n = (fiv_prelu_node*)op_state;
    return fiv_prelu_compute((fiv_tensor_hdr*)output, (const fiv_tensor_hdr*)input,
                             n->alpha, n->channels);
}

fiv_ret fiv_prelu_node_inference(void* op_state, void* output, void* input)
{
    return fiv_prelu_node_forward(op_state, output, input);
}

/* dL/dx = g*(x>=0) + g*alpha[c]*(x<0); both branches accumulate into grad_input.
   alpha itself is treated as fixed (model weights are injected, not trained). */
fiv_ret fiv_prelu_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    const fiv_prelu_node* n = (const fiv_prelu_node*)op_state;
    fiv_tensor_hdr* gi = (fiv_tensor_hdr*)grad_input;
    const fiv_tensor_hdr* go = (const fiv_tensor_hdr*)grad_output;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!n || !gi || !go || !in) return FIV_RET_ERR_PARA;
    if (in->dtype != FIV_32F1 || go->dtype != FIV_32F1 || gi->dtype != FIV_32F1)
        return FIV_RET_ERR_NOT_SUPPORT;
    if (gi->total_bytes != go->total_bytes || in->total_bytes != go->total_bytes)
        return FIV_RET_ERR_PARA;

    size_t nelt = go->total_bytes / sizeof(ivf32);
    size_t hw = nelt / (size_t)n->channels;          /* elements per channel plane (NCHW) */
    const ivf32* g = go->data.fl;
    const ivf32* x = in->data.fl;
    ivf32* d = gi->data.fl;
    /* Per-plane alpha lookup (drops the per-element integer division). */
    for (size_t c = 0; c < (size_t)n->channels; c++) {
        ivf32 a = n->alpha[c];
        const ivf32* xx = x + c * hw;
        const ivf32* gg = g + c * hw;
        ivf32*       dd = d + c * hw;
        for (size_t i = 0; i < hw; i++) {
            ivf32 v = xx[i];
            dd[i] += (v >= 0.0f) ? gg[i] : a * gg[i];
        }
    }
    return FIV_RET_OK;
}

/* Output mirrors the input shape (same id / dtype / byte count). */
void* fiv_prelu_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{
    *out_ret = FIV_RET_OK;
    (void)op_state;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!in || in->dtype != FIV_32F1 || in->data_continue == 0) {
        *out_ret = FIV_RET_ERR_PARA;
        return NULL;
    }
    fiv_tensor_hdr* out = (fiv_tensor_hdr*)existing_output;
    if (out && out->id == in->id && out->dtype == FIV_32F1 && out->data_continue == 1 &&
        out->total_bytes == in->total_bytes)
        return out;
    if (out) fiv_release_tensor((void**)&out);

    switch (in->id) {
    case FIV_ID_TENSOR3D: {
        const fiv_tensor3d* t = (const fiv_tensor3d*)in;
        size_t shape[3] = { t->channels, t->height, t->width };
        out = (fiv_tensor_hdr*)fiv_create_tensor3d(shape, FIV_32F1);
        break;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)in;
        size_t shape[4] = { t->batch, t->channels, t->height, t->width };
        out = (fiv_tensor_hdr*)fiv_create_tensor4d(shape, FIV_32F1);
        break;
    }
    default:
        out = (fiv_tensor_hdr*)fiv_create_tensor_like_tensor((void*)in);
        break;
    }
    if (!out) { *out_ret = FIV_RET_ERR_MEM; return NULL; }
    return out;
}