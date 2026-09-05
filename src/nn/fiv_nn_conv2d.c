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

#include "fiv_nn_conv2d.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "fiv_common.h"

/* SIMD headers are auto-included and detected by api/fiv_data_typedefs.h
   (FIV_USE_ARM_NEON / FIV_USE_AVX2 / FIV_USE_X86_SIMD). */

#include "fiv_nn_3x3_conv2d.h"
#include "fiv_nn_5x5_conv2d.h"
#include "fiv_nn_2x2_conv2d.h"
#include "fiv_nn_1x1_conv2d.h"
/* ---- generic scalar conv (any kernel / stride / explicit pad) ----
   Mirrors the reference cnn_ops.c accumulation order so results match it
   bit-for-bit: for oc, oy, ox: for ky, kx, ic: acc += v * w. Out-of-range
   taps are skipped (zero pad) or clamped (replicate); only the start pads
   pt/pl enter the index math, exactly like same_pad() in the reference.
   weight layout [c_out, c_in, kh, kw] (permuted from the [co,kh,kw,ci] blob). */
static void fiv_conv2d_generic_std(ivf32* d, const ivf32* s, const ivf32* w,
                                   int c_in, int c_out, int height, int width, int oh, int ow,
                                   int kh, int kw, int st, int pt, int pl, int zero_pad)
{
    const size_t kplane = (size_t)c_in * kh * kw;
    for (int oc = 0; oc < c_out; oc++) {
        const ivf32* wo = w + (size_t)oc * kplane;
        ivf32* dplane = d + (size_t)oc * oh * ow;
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                float acc = 0.0f;
                for (int ky = 0; ky < kh; ky++) {
                    int sy = oy * st - pt + ky;
                    if (sy < 0 || sy >= height) {
                        if (zero_pad) continue;
                        sy = sy < 0 ? 0 : height - 1;
                    }
                    const ivf32* wrow = wo + (size_t)ky * kw;
                    for (int kx = 0; kx < kw; kx++) {
                        int sx = ox * st - pl + kx;
                        if (sx < 0 || sx >= width) {
                            if (zero_pad) continue;
                            sx = sx < 0 ? 0 : width - 1;
                        }
                        const ivf32* wcol = wrow + kx;
                        for (int ic = 0; ic < c_in; ic++) {
                            float v = s[((size_t)ic * height + sy) * width + sx];
                            acc += v * wcol[(size_t)ic * kh * kw];
                        }
                    }
                }
                dplane[(size_t)oy * ow + ox] = acc;
            }
        }
    }
}

static void fiv_conv2d_generic_dw(ivf32* d, const ivf32* s, const ivf32* w,
                                  int c_out, int mult, int height, int width, int oh, int ow,
                                  int kh, int kw, int st, int pt, int pl, int zero_pad)
{
    for (int oc = 0; oc < c_out; oc++) {
        int ic = oc / mult;                        /* output group -> input channel */
        const ivf32* sc = s + (size_t)ic * height * width;
        const ivf32* wc = w + (size_t)oc * kh * kw;
        ivf32* dc = d + (size_t)oc * oh * ow;
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                float acc = 0.0f;
                for (int ky = 0; ky < kh; ky++) {
                    int sy = oy * st - pt + ky;
                    if (sy < 0 || sy >= height) {
                        if (zero_pad) continue;
                        sy = sy < 0 ? 0 : height - 1;
                    }
                    const ivf32* wrow = wc + (size_t)ky * kw;
                    for (int kx = 0; kx < kw; kx++) {
                        int sx = ox * st - pl + kx;
                        if (sx < 0 || sx >= width) {
                            if (zero_pad) continue;
                            sx = sx < 0 ? 0 : width - 1;
                        }
                        acc += sc[(size_t)sy * width + sx] * wrow[kx];
                    }
                }
                dc[(size_t)oy * ow + ox] = acc;
            }
        }
    }
}

/* ---- public API ---- */

fiv_ret fiv_tensor_conv2d(void* dst, void* src, void* kernel, fiv_conv2d_params* params)
{
    fiv_tensor_hdr* sh = (fiv_tensor_hdr*)src;
    fiv_tensor_hdr* dh = (fiv_tensor_hdr*)dst;
    fiv_tensor_hdr* kh = (fiv_tensor_hdr*)kernel;
    if (!dst || !src || !kernel || !params) return FIV_RET_ERR_PARA;

    if (sh->id < FIV_ID_TENSOR3D || sh->id > FIV_ID_TENSOR5D) return FIV_RET_ERR_PARA;  /* need channels, height, width */
    if (kh->id != FIV_ID_TENSOR4D) return FIV_RET_ERR_PARA;
    if (dh->id != sh->id) return FIV_RET_ERR_PARA;
    if (sh->dtype != FIV_32F1 || dh->dtype != FIV_32F1 || kh->dtype != FIV_32F1)
        return FIV_RET_ERR_NOT_SUPPORT;
    if (!sh->data_continue || !dh->data_continue || !kh->data_continue)
        return FIV_RET_ERR_PARA;

    if (params->conv2d_method != FIV_CONV2D_STD &&
        params->conv2d_method != FIV_CONV2D_DEPTHWISE &&
        params->conv2d_method != FIV_CONV2D_POINTWISE)
        return FIV_RET_ERR_NOT_SUPPORT;
    if (params->kernel_size_x < 1 || params->kernel_size_y < 1 || params->stride < 1)
        return FIV_RET_ERR_PARA;
    if (params->pad_top < 0 || params->pad_bottom < 0 || params->pad_left < 0 || params->pad_right < 0)
        return FIV_RET_ERR_PARA;
    if (params->padding_method != 0 && params->padding_method != 1) return FIV_RET_ERR_PARA;

    size_t c_in, height, width, n_batch;
    switch (sh->id) {
    case FIV_ID_TENSOR3D: {
        const fiv_tensor3d* t = (const fiv_tensor3d*)sh;
        c_in = t->channels; height = t->height; width = t->width; n_batch = 1;
        break;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)sh;
        c_in = t->channels; height = t->height; width = t->width; n_batch = t->batch;
        break;
    }
    default: {
        const fiv_tensor5d* t = (const fiv_tensor5d*)sh;
        c_in = t->channels; height = t->height; width = t->width;
        n_batch = t->batch * t->times;
        break;
    }
    }
    if (c_in == 0 || height == 0 || width == 0) return FIV_RET_ERR_PARA;

    int kx = params->kernel_size_x, ky = params->kernel_size_y;
    int st = params->stride;
    const fiv_tensor4d* k = (const fiv_tensor4d*)kernel;
    if (k->shapes[2] != (size_t)ky || k->shapes[3] != (size_t)kx) return FIV_RET_ERR_PARA;
    size_t k_cout = k->shapes[0];
    size_t k_cin  = k->shapes[1];
    if (params->input_channels != (int)c_in) return FIV_RET_ERR_PARA;

    size_t c_out;
    size_t mult = 1;   /* depthwise channel multiplier (C_out = depth_mult * C_in) */
    if (params->conv2d_method == FIV_CONV2D_DEPTHWISE) {
        if (k_cin != 1) return FIV_RET_ERR_PARA;
        if (k_cout < c_in || k_cout % c_in != 0) return FIV_RET_ERR_PARA;
        mult  = k_cout / c_in;   /* depth multiplier; k_cout == c_in keeps mult = 1 */
        c_out = k_cout;
    } else {  /* STD / POINTWISE */
        if (k_cin != c_in) return FIV_RET_ERR_PARA;
        c_out = k_cout;
    }
    if (params->output_channels != (int)c_out) return FIV_RET_ERR_PARA;

    size_t oh = (height + (size_t)st - 1) / (size_t)st;
    size_t ow = (width + (size_t)st - 1) / (size_t)st;
    size_t d_c, d_h, d_w;
    switch (dh->id) {
    case FIV_ID_TENSOR3D: {
        const fiv_tensor3d* t = (const fiv_tensor3d*)dh;
        d_c = t->channels; d_h = t->height; d_w = t->width;
        break;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)dh;
        d_c = t->channels; d_h = t->height; d_w = t->width;
        break;
    }
    default: {
        const fiv_tensor5d* t = (const fiv_tensor5d*)dh;
        d_c = t->channels; d_h = t->height; d_w = t->width;
        break;
    }
    }
    if (d_c != c_out || d_h != oh || d_w != ow) return FIV_RET_ERR_PARA;

    const ivf32* s = sh->data.fl;
    const ivf32* w = kh->data.fl;
    ivf32* d = dh->data.fl;
    size_t hw    = height * width;
    size_t o_hw   = oh * ow;
    size_t s_chan = c_in * hw;
    size_t d_chan = c_out * o_hw;
    int zero_pad = (params->padding_method == 0);

    /* Legacy 3x3 stride-1 keeps its SIMD fast path (implicit p0 = p1 = 1 each
       side) whenever the requested padding is either unset (0) or exactly
       same-padding (1); any other explicit pad / stride / kernel goes through
       the generic scalar kernels above with the literal pad values. */
    int legacy_3x3_s1 = (kx == 3 && ky == 3 && st == 1 &&
                       ((params->pad_top == 0 && params->pad_bottom == 0 &&
                         params->pad_left == 0 && params->pad_right == 0) ||
                        (params->pad_top == 1 && params->pad_bottom == 1 &&
                         params->pad_left == 1 && params->pad_right == 1)));

    /* 3x3 stride-2 keeps its SIMD fast path (implicit p0=p1=1) on the same
       pad conditions as the stride-1 legacy branch. */
    int legacy_3x3_s2 = (kx == 3 && ky == 3 && st == 2 &&
                     ((params->pad_top == 0 && params->pad_bottom == 0 &&
                       params->pad_left == 0 && params->pad_right == 0) ||
                      (params->pad_top == 1 && params->pad_bottom == 1 &&
                       params->pad_left == 1 && params->pad_right == 1)));

    /* 5x5 stride-2 fast path: the BlazeFace stem (3->24) and any kernel of this
       shape. SAME padding always puts pad_top=pad_left=1, so the branchless SIMD
       interior (pt=pl=1) is exact; other pads fall through to the generic scalar. */
    int legacy_5x5_s2 = (kx == 5 && ky == 5 && st == 2);

    /* 1x1 stride-1 (all pointwise layers) has no spatial neighborhood: each
       output pixel only sees the input pixel at the same (y,x), so padding is
       meaningless and is IGNORED -- the dedicated branch never pads. */
    int is_pw = (params->conv2d_method != FIV_CONV2D_DEPTHWISE &&
                 kx == 1 && ky == 1 && st == 1);

    for (size_t b = 0; b < n_batch; b++) {
        const ivf32* sb = s + b * s_chan;
        ivf32* db = d + b * d_chan;
        if (is_pw) {
            fiv_conv2d_pw(db, sb, w, (int)c_in, (int)c_out, hw);
        } else if (legacy_3x3_s1) {
            if (params->conv2d_method == FIV_CONV2D_DEPTHWISE) {
                fiv_conv2d_dw_3x3_s1(db, sb, w, (int)c_out, (int)mult,
                                     (int)width, (int)height, zero_pad);
            } else {
                fiv_conv2d_std_3x3_s1(db, sb, w, (int)c_in, (int)c_out,
                                      (int)width, (int)height, zero_pad);
            }
        } else if (legacy_3x3_s2) {
            if (params->conv2d_method == FIV_CONV2D_DEPTHWISE) {
                fiv_conv2d_dw_3x3_s2(db, sb, w, (int)c_out, (int)mult,
                                     (int)width, (int)height, (int)oh, (int)ow, zero_pad);
            } else {
                fiv_conv2d_std_3x3_s2(db, sb, w, (int)c_in, (int)c_out,
                                      (int)width, (int)height, (int)oh, (int)ow, zero_pad);
            }
        } else if (legacy_5x5_s2) {
            if (params->conv2d_method == FIV_CONV2D_DEPTHWISE) {
                /* 5x5 depthwise is rare; use the generic scalar path (mult-aware) */
                fiv_conv2d_generic_dw(db, sb, w, (int)c_out, (int)mult, (int)height, (int)width,
                                      (int)oh, (int)ow, ky, kx, st,
                                      params->pad_top, params->pad_left, zero_pad);
            } else if (params->pad_top == 1 && params->pad_left == 1) {
                /* 5x5 stride-2 (the BlazeFace stem). SAME padding means pt=pl=1;
                   the SIMD interior of fiv_conv2d_plane_5x5_s2 is then exact and
                   branchless. Other pads fall through to the generic scalar. */
                fiv_conv2d_std_5x5_s2(db, sb, w, (int)c_in, (int)c_out,
                                      (int)width, (int)height, (int)oh, (int)ow, zero_pad,
                                      params->pad_top, params->pad_left);
            } else {
                fiv_conv2d_generic_std(db, sb, w, (int)c_in, (int)c_out, (int)height, (int)width,
                                       (int)oh, (int)ow, ky, kx, st,
                                       params->pad_top, params->pad_left, zero_pad);
            }
        } else if (kx == 2 && ky == 2 && st == 2) {
            /* 2x2 stride-2 (landmark downsample layers): mirror the 3x3 stride-2
               plane, one (oc,ic) plane call per input channel accumulating into
               the output plane (ic>0) so the sum over ic matches the scalar
               reference bit-for-bit. */
            int p2t = params->pad_top, p2l = params->pad_left;
            if (params->conv2d_method == FIV_CONV2D_DEPTHWISE) {
                fiv_conv2d_generic_dw(db, sb, w, (int)c_out, (int)mult, (int)height, (int)width,
                                      (int)oh, (int)ow, 2, 2, 2, p2t, p2l, zero_pad);
            } else {
                fiv_conv2d_std_2x2_s2(db, sb, w, (int)c_in, (int)c_out,
                                      (int)width, (int)height, (int)oh, (int)ow, zero_pad,
                                      p2t, p2l);
            }
        } else {
            int pt = params->pad_top, pl = params->pad_left;
            if (params->conv2d_method == FIV_CONV2D_DEPTHWISE)
                fiv_conv2d_generic_dw(db, sb, w, (int)c_out, (int)mult, (int)height, (int)width,
                                      (int)oh, (int)ow, ky, kx, st, pt, pl, zero_pad);
            else
                fiv_conv2d_generic_std(db, sb, w, (int)c_in, (int)c_out, (int)height, (int)width,
                                       (int)oh, (int)ow, ky, kx, st, pt, pl, zero_pad);
        }
    }
    return FIV_RET_OK;
}

/* ---- CONV2D_STD network node ---- */

void* fiv_conv2d_node_create(void* params)
{
    const fiv_conv2d_params* p = (const fiv_conv2d_params*)params;
    if (!p) return NULL;
    if (p->conv2d_method != FIV_CONV2D_STD &&
        p->conv2d_method != FIV_CONV2D_DEPTHWISE &&
        p->conv2d_method != FIV_CONV2D_POINTWISE)
        return NULL;
    if (p->kernel_size_x < 1 || p->kernel_size_y < 1 || p->stride < 1) return NULL;
    if (p->padding_method != 0 && p->padding_method != 1) return NULL;
    if (p->bias != 0 && p->bias != 1) return NULL;
    if (p->input_channels <= 0 || p->output_channels <= 0) return NULL;
    if (p->pad_top < 0 || p->pad_bottom < 0 || p->pad_left < 0 || p->pad_right < 0) return NULL;
    if (p->conv2d_method == FIV_CONV2D_DEPTHWISE &&
        (p->kernel_size_x != 3 || p->kernel_size_y != 3)) return NULL;
    if (p->conv2d_method == FIV_CONV2D_POINTWISE &&
        (p->kernel_size_x != 1 || p->kernel_size_y != 1 || p->stride != 1)) return NULL;

    int kcin = (p->conv2d_method == FIV_CONV2D_DEPTHWISE) ? 1 : p->input_channels;

    fiv_conv2d_node* n = (fiv_conv2d_node*)fiv_malloc(sizeof(fiv_conv2d_node));
    if (!n) return NULL;
    memset(n, 0, sizeof(fiv_conv2d_node));
    n->base.create_fn    = fiv_conv2d_node_create;
    n->base.release_fn   = fiv_conv2d_node_release;
    n->base.forward_fn   = fiv_conv2d_node_forward;
    n->base.backward_fn  = fiv_conv2d_node_backward;
    n->base.inference_fn = fiv_conv2d_node_inference;
    n->base.alloc_out_fn = fiv_conv2d_node_alloc_out;
    n->params = *p;

    size_t wsh[4] = { (size_t)p->output_channels, (size_t)kcin,
                      (size_t)p->kernel_size_y, (size_t)p->kernel_size_x };
    n->weight = fiv_create_tensor4d(wsh, FIV_32F1);
    if (!n->weight) { fiv_free(n); return NULL; }
    {
        ivf32* w = n->weight->data.fl;
        size_t nw = (size_t)p->output_channels * (size_t)kcin *
                    (size_t)p->kernel_size_y * (size_t)p->kernel_size_x;
        float s = 1.0f / sqrtf((float)(p->input_channels *
                                       p->kernel_size_y * p->kernel_size_x));
        for (size_t k = 0; k < nw; k++) w[k] = fiv_nn_rand() * s;
    }

    n->grad_weight = fiv_create_tensor4d(wsh, FIV_32F1);
    if (!n->grad_weight) { fiv_release_tensor4d(&n->weight); fiv_free(n); return NULL; }
    memset(n->grad_weight->data.ptr, 0, n->grad_weight->total_bytes);

    if (p->bias) {
        n->bias = fiv_create_tensor1d((size_t)p->output_channels, FIV_32F1);
        if (!n->bias) {
            fiv_release_tensor4d(&n->grad_weight);
            fiv_release_tensor4d(&n->weight);
            fiv_free(n);
            return NULL;
        }
        memset(n->bias->data.ptr, 0, n->bias->total_bytes);
        n->grad_bias = fiv_create_tensor1d((size_t)p->output_channels, FIV_32F1);
        if (!n->grad_bias) {
            fiv_release_tensor1d(&n->bias);
            fiv_release_tensor4d(&n->grad_weight);
            fiv_release_tensor4d(&n->weight);
            fiv_free(n);
            return NULL;
        }
        memset(n->grad_bias->data.ptr, 0, n->grad_bias->total_bytes);
    }
    return n;
}

void fiv_conv2d_node_release(void* op_state)
{
    fiv_conv2d_node* n = (fiv_conv2d_node*)op_state;
    if (!n) return;
    if (n->weight)      fiv_release_tensor4d(&n->weight);
    if (n->grad_weight) fiv_release_tensor4d(&n->grad_weight);
    if (n->bias)        fiv_release_tensor1d(&n->bias);
    if (n->grad_bias)   fiv_release_tensor1d(&n->grad_bias);
    fiv_free(n);
}

/* Output keeps the input's dims; channels = output_channels, spatial unchanged. */
void* fiv_conv2d_node_alloc_out(void* op_state, const void* input, void* existing_output, fiv_ret* out_ret)
{
    *out_ret = FIV_RET_OK;
    const fiv_conv2d_node* n = (const fiv_conv2d_node*)op_state;
    const fiv_tensor_hdr* in = (const fiv_tensor_hdr*)input;
    if (!n || !in) {
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

    size_t c_out = (size_t)n->params.output_channels;
    size_t batch   = 1;
    size_t height  = 0;
    size_t width   = 0;
    switch (in->id) {
    case FIV_ID_TENSOR3D:
        height = ((const fiv_tensor3d*)in)->height;
        width = ((const fiv_tensor3d*)in)->width;
        break;
    case FIV_ID_TENSOR4D:
        batch = ((const fiv_tensor4d*)in)->batch;
        height = ((const fiv_tensor4d*)in)->height;
        width = ((const fiv_tensor4d*)in)->width;
        break;
    default:
        batch = ((const fiv_tensor5d*)in)->batch * ((const fiv_tensor5d*)in)->times;
        height = ((const fiv_tensor5d*)in)->height;
        width = ((const fiv_tensor5d*)in)->width;
        break;
    }
    /* output spatial = ceil(in / stride), same as the reference oh = (n+s-1)/s */
    size_t st = (size_t)n->params.stride;
    size_t oh = (height + st - 1) / st;
    size_t ow = (width + st - 1) / st;

    fiv_tensor_hdr* out = (fiv_tensor_hdr*)existing_output;
    if (out && out->id == in->id && out->dtype == FIV_32F1 && out->data_continue == 1) {
        size_t o_b  = 1;
        size_t o_c  = 0;
        size_t o_hh = 0;
        size_t o_ww = 0;
        switch (out->id) {
        case FIV_ID_TENSOR3D:
            o_c = ((fiv_tensor3d*)out)->channels;
            o_hh = ((fiv_tensor3d*)out)->height;
            o_ww = ((fiv_tensor3d*)out)->width;
            break;
        case FIV_ID_TENSOR4D:
            o_b = ((fiv_tensor4d*)out)->batch;
            o_c = ((fiv_tensor4d*)out)->channels;
            o_hh = ((fiv_tensor4d*)out)->height;
            o_ww = ((fiv_tensor4d*)out)->width;
            break;
        default:
            o_b = ((fiv_tensor5d*)out)->batch * ((fiv_tensor5d*)out)->times;
            o_c = ((fiv_tensor5d*)out)->channels;
            o_hh = ((fiv_tensor5d*)out)->height;
            o_ww = ((fiv_tensor5d*)out)->width;
            break;
        }
        if (o_b == batch && o_c == c_out && o_hh == oh && o_ww == ow) return out;
    }
    if (out) fiv_release_tensor((void**)&out);

    switch (in->id) {
    case FIV_ID_TENSOR3D: {
        size_t sh[3] = { c_out, oh, ow };
        out = (fiv_tensor_hdr*)fiv_create_tensor3d(sh, FIV_32F1);
        break;
    }
    case FIV_ID_TENSOR4D: {
        const fiv_tensor4d* t = (const fiv_tensor4d*)in;
        size_t sh[4] = { t->batch, c_out, oh, ow };
        out = (fiv_tensor_hdr*)fiv_create_tensor4d(sh, FIV_32F1);
        break;
    }
    default: {
        const fiv_tensor5d* t = (const fiv_tensor5d*)in;
        size_t sh[5] = { t->batch, t->times, c_out, oh, ow };
        out = (fiv_tensor_hdr*)fiv_create_tensor5d(sh, FIV_32F1);
        break;
    }
    }
    if (!out) { *out_ret = FIV_RET_ERR_MEM; return NULL; }
    return out;
}

/* out = conv(in, weight) then out[b][oc][y][x] += bias[oc] when bias enabled. */
static fiv_ret fiv_conv2d_compute(fiv_conv2d_node* n, fiv_tensor_hdr* out, fiv_tensor_hdr* in)
{
    fiv_ret r = fiv_tensor_conv2d(out, in, (void*)n->weight, &n->params);
    if (r != FIV_RET_OK) return r;
    if (!n->bias) return FIV_RET_OK;

    size_t batch   = 1;
    size_t height  = 0;
    size_t width   = 0;
    switch (in->id) {
    case FIV_ID_TENSOR3D:
        batch = 1;
        height = ((fiv_tensor3d*)in)->height;
        width = ((fiv_tensor3d*)in)->width;
        break;
    case FIV_ID_TENSOR4D:
        batch = ((fiv_tensor4d*)in)->batch;
        height = ((fiv_tensor4d*)in)->height;
        width = ((fiv_tensor4d*)in)->width;
        break;
    default:
        batch = ((fiv_tensor5d*)in)->batch * ((fiv_tensor5d*)in)->times;
        height = ((fiv_tensor5d*)in)->height;
        width = ((fiv_tensor5d*)in)->width;
        break;
    }
    size_t c_out = (size_t)n->params.output_channels;
    size_t st = (size_t)n->params.stride;
    size_t o_hw = ((height + st - 1) / st) * ((width + st - 1) / st);
    ivf32* d = out->data.fl;
    const ivf32* b = n->bias->data.fl;
    for (size_t bb = 0; bb < batch; bb++)
        for (size_t oc = 0; oc < c_out; oc++) {
            ivf32 bv = b[oc];
            ivf32* p = d + (bb * c_out + oc) * o_hw;
            for (size_t k = 0; k < o_hw; k++) p[k] += bv;
        }
    return FIV_RET_OK;
}

fiv_ret fiv_conv2d_node_forward(void* op_state, void* output, void* input)
{
    fiv_conv2d_node* n = (fiv_conv2d_node*)op_state;
    return fiv_conv2d_compute(n, (fiv_tensor_hdr*)output, (fiv_tensor_hdr*)input);
}

fiv_ret fiv_conv2d_node_inference(void* op_state, void* output, void* input)
{
    return fiv_conv2d_compute((fiv_conv2d_node*)op_state, (fiv_tensor_hdr*)output, (fiv_tensor_hdr*)input);
}

/* Conv backward (all accumulations, engine resets grads per step), generalized
   to any kernel / stride / explicit pad:
   d_w[oc][ic][ky][kx] += sum_{b,oy,ox} go[b][oc][oy][ox] * x[b][ic][sy][sx]
   db[oc]             += sum_{b,oy,ox} go[b][oc][oy][ox]
   dIn[b][ic][y][x]   += sum_{oc,ky,kx} go[b][oc][oy][ox] * w[oc][ic][ky][kx]
   where sy = oy*stride - pad_top + ky, sx = ox*stride - pad_left + kx and the
   forward padding rule applies: zero pad skips out-of-range taps, replicate
   clamps them (so a clamped tap's gradient lands on the border pixel). */
fiv_ret fiv_conv2d_node_backward(void* op_state, void* grad_input, const void* grad_output, const void* input)
{
    fiv_conv2d_node* n = (fiv_conv2d_node*)op_state;
    const fiv_tensor_hdr* go = (const fiv_tensor_hdr*)grad_output;
    const fiv_tensor_hdr* x  = (const fiv_tensor_hdr*)input;
    fiv_tensor_hdr* gi = (fiv_tensor_hdr*)grad_input;
    if (!n || !go || !x) return FIV_RET_ERR_PARA;

    size_t batch  = 1;
    size_t c_in   = 0;
    size_t height = 0;
    size_t width  = 0;
    switch (x->id) {
    case FIV_ID_TENSOR3D:
        batch = 1;
        c_in = ((const fiv_tensor3d*)x)->channels;
        height = ((const fiv_tensor3d*)x)->height;
        width = ((const fiv_tensor3d*)x)->width;
        break;
    case FIV_ID_TENSOR4D:
        batch = ((const fiv_tensor4d*)x)->batch;
        c_in = ((const fiv_tensor4d*)x)->channels;
        height = ((const fiv_tensor4d*)x)->height;
        width = ((const fiv_tensor4d*)x)->width;
        break;
    default:
        batch = ((const fiv_tensor5d*)x)->batch * ((const fiv_tensor5d*)x)->times;
        c_in = ((const fiv_tensor5d*)x)->channels;
        height = ((const fiv_tensor5d*)x)->height;
        width = ((const fiv_tensor5d*)x)->width;
        break;
    }
    size_t c_out = (size_t)n->params.output_channels;
    int kh = n->params.kernel_size_y;
    int kw = n->params.kernel_size_x;
    int st = n->params.stride;
    int pt = n->params.pad_top;
    int pl = n->params.pad_left;
    /* Match the forward's legacy rule: a 3x3 stride-1 node with all pads unset
       (0) or all 1 is the historical same-padding with start pad 1. */
    if (kw == 3 && kh == 3 && st == 1 &&
        ((pt == 0 && n->params.pad_bottom == 0 && pl == 0 && n->params.pad_right == 0) ||
         (pt == 1 && n->params.pad_bottom == 1 && pl == 1 && n->params.pad_right == 1))) {
        pt = 1;
        pl = 1;
    }
    int zero_pad = (n->params.padding_method == 0);
    size_t oh = (height + (size_t)st - 1) / (size_t)st;
    size_t ow = (width + (size_t)st - 1) / (size_t)st;
    int kcin = (n->params.conv2d_method == FIV_CONV2D_DEPTHWISE) ? 1 : (int)c_in;
    int is_dw = (n->params.conv2d_method == FIV_CONV2D_DEPTHWISE);

    const ivf32* xp  = x->data.fl;
    const ivf32* gp  = go->data.fl;
    const ivf32* wp  = n->weight->data.fl;
    ivf32*       dw  = n->grad_weight->data.fl;
    ivf32*       gip = gi ? gi->data.fl : NULL;
    const size_t o_hw = oh * ow;
    const size_t ihw  = height * width;

    for (size_t b = 0; b < batch; b++) {
        const ivf32* xb = xp + b * c_in * ihw;
        const ivf32* gb = gp + b * c_out * o_hw;
        for (size_t oc = 0; oc < c_out; oc++) {
            const ivf32* goc = gb + oc * o_hw;
            if (n->grad_bias) {
                float db = 0.0f;
                for (size_t k = 0; k < o_hw; k++) db += goc[k];
                n->grad_bias->data.fl[oc] += db;
            }
            for (size_t ic = 0; ic < (size_t)kcin; ic++) {
                size_t xc = is_dw ? oc : ic;   /* depthwise: out[oc] reads in[oc] */
                const ivf32* xic = xb + xc * ihw;
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        float acc = 0.0f;
                        for (size_t oy = 0; oy < oh; oy++) {
                            int sy = (int)oy * st - pt + ky;
                            int syc;
                            if (sy < 0 || sy >= (int)height) {
                                if (zero_pad) continue;
                                syc = sy < 0 ? 0 : (int)height - 1;
                            } else {
                                syc = sy;
                            }
                            for (size_t xx = 0; xx < ow; xx++) {
                                int sx = (int)xx * st - pl + kx;
                                int sxc;
                                if (sx < 0 || sx >= (int)width) {
                                    if (zero_pad) continue;
                                    sxc = sx < 0 ? 0 : (int)width - 1;
                                } else {
                                    sxc = sx;
                                }
                                acc += goc[oy * ow + xx] * xic[(size_t)syc * width + (size_t)sxc];
                            }
                        }
                        dw[((oc * (size_t)kcin + ic) * (size_t)kh + (size_t)ky) * (size_t)kw + (size_t)kx] += acc;
                    }
                }
            }
        }
    }

    if (gip) {
        for (size_t b = 0; b < batch; b++) {
            const ivf32* gb = gp + b * c_out * o_hw;
            ivf32* gib = gip + b * c_in * ihw;
            for (size_t ic = 0; ic < c_in; ic++) {
                ivf32* giic = gib + ic * ihw;
                for (size_t oc = 0; oc < c_out; oc++) {
                    if (is_dw && oc != ic) continue;   /* depthwise: in[ic] only feeds out[ic] */
                    const ivf32* goc = gb + oc * o_hw;
                    const ivf32* wocic = wp + (oc * (size_t)kcin + ic) * (size_t)kh * kw;
                    for (size_t oy = 0; oy < oh; oy++) {
                        for (int ky = 0; ky < kh; ky++) {
                            int sy = (int)oy * st - pt + ky;
                            int y;
                            if (sy < 0 || sy >= (int)height) {
                                if (zero_pad) continue;
                                y = sy < 0 ? 0 : (int)height - 1;
                            } else {
                                y = sy;
                            }
                            for (size_t xx = 0; xx < ow; xx++) {
                                for (int kx = 0; kx < kw; kx++) {
                                    int sx = (int)xx * st - pl + kx;
                                    int xpos;
                                    if (sx < 0 || sx >= (int)width) {
                                        if (zero_pad) continue;
                                        xpos = sx < 0 ? 0 : (int)width - 1;
                                    } else {
                                        xpos = sx;
                                    }
                                    giic[(size_t)y * width + (size_t)xpos] +=
                                        goc[oy * ow + xx] * wocic[(size_t)ky * kw + (size_t)kx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return FIV_RET_OK;
}

