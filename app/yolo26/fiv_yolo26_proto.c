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

/* Proto26 mask-prototype network (YOLO26 instance segmentation).
 *
 * Mirrors ultralytics/nn/modules/block.py Proto26 + Proto (inference path,
 * real-weight BN-folded blobs, all activations SiLU):
 *
 *   feat_refine0 = SiLU(conv1x1(P4, 128 -> 64))   nearest 2x
 *   feat_refine1 = SiLU(conv1x1(P5, 256 -> 64))   nearest 4x
 *   feat         = P3 + up(feat_refine0) + up(feat_refine1)
 *   feat         = SiLU(conv3x3(feat, 64))                      (feat_fuse)
 *   y            = SiLU(conv3x3(feat, 64))                      (cv1)
 *   y            = ConvTranspose2d 2x2 stride 2 (64)            (upsample)
 *   y            = SiLU(conv3x3(y, 64))                         (cv2)
 *   proto        = SiLU(conv1x1(y, 32))                         (cv3)
 *
 * Every ordinary conv (1x1 or 3x3 s1 SAME) is delegated to the engine's SIMD
 * dispatch (fiv_tensor_conv2d: 1x1 -> fiv_conv2d_pw, 3x3 s1 pad1 ->
 * fiv_conv2d_std_3x3_s1, both vectorized), then bias + SiLU are folded into a
 * single per-plane pass (NEON-8 when available). Only the 2x2 stride-2
 * transposed conv and the nearest-neighbour upsample stay scalar: the former
 * has no SIMD kernel in the engine, the latter is pure data movement.
 *
 * Layouts: input / intermediate / proto buffers are [c, h, w] channel-major
 * flat ivf32 arrays; weights come from the model directory blobs
 * (proto_*.f32), each with its per-channel bias.
 */

#include "fiv_yolo26_proto.h"

#include "fiv_common.h"
#include "fiv_nn_conv2d.h"

#if defined(FIV_USE_ARM_NEON)
#include "fiv_math_kernels.h"
#endif

#include <math.h>
#include <string.h>

/* Scalar SiLU used by the tails and the (unchanged) transposed conv. */
static ivf32 fiv_proto_silu(ivf32 value)
{
    ivf32 sigmoid_value = 1.0f / (1.0f + expf(-value));
    return value * sigmoid_value;
}

/* Per-channel pass over a [channels, plane] buffer: add the channel bias,
   then apply SiLU when requested. Vectorized (NEON 8-wide) on aarch64; the
   shape matches the scalar path (clamp + exp + reciprocal). */
static void fiv_proto_bias_act(ivf32* data, const ivf32* bias,
                               int channels, size_t plane, int apply_silu)
{
    const ivf32 one_v = 1.0f;
    for (int oc = 0; oc < channels; oc++) {
        ivf32* row = data + (size_t)oc * plane;
        ivf32 bias_v = bias[oc];
#if defined(FIV_USE_ARM_NEON)
        size_t index = 0;
        const float32x4_t bias_wide = vdupq_n_f32(bias_v);
        const float32x4_t clamp_hi = vdupq_n_f32(88.0f);
        const float32x4_t clamp_lo = vdupq_n_f32(-88.0f);
        const float32x4_t one_wide = vdupq_n_f32(one_v);
        if (apply_silu) {
            for (; index + 8 <= plane; index += 8) {
                float32x4_t va = vaddq_f32(vld1q_f32(row + index), bias_wide);
                float32x4_t vb = vaddq_f32(vld1q_f32(row + index + 4), bias_wide);
                va = vminq_f32(va, clamp_hi);
                va = vmaxq_f32(va, clamp_lo);
                vb = vminq_f32(vb, clamp_hi);
                vb = vmaxq_f32(vb, clamp_lo);
                fiv_f32x4x2 exp_out =
                    fiv_math_exp128_ps2(vnegq_f32(va), vnegq_f32(vb));
                float32x4_t sig_a =
                    fiv_math_rcp128_ps(vaddq_f32(one_wide, exp_out.a));
                float32x4_t sig_b =
                    fiv_math_rcp128_ps(vaddq_f32(one_wide, exp_out.b));
                vst1q_f32(row + index, vmulq_f32(va, sig_a));
                vst1q_f32(row + index + 4, vmulq_f32(vb, sig_b));
            }
            for (; index + 4 <= plane; index += 4) {
                float32x4_t v = vaddq_f32(vld1q_f32(row + index), bias_wide);
                v = vminq_f32(v, clamp_hi);
                v = vmaxq_f32(v, clamp_lo);
                float32x4_t sig = fiv_math_rcp128_ps(
                    vaddq_f32(one_wide,
                              fiv_math_exp128_ps(vnegq_f32(v))));
                vst1q_f32(row + index, vmulq_f32(v, sig));
            }
            for (; index < plane; index++)
                row[index] = fiv_proto_silu(row[index] + bias_v);
        } else {
            for (; index + 8 <= plane; index += 8) {
                vst1q_f32(row + index,
                          vaddq_f32(vld1q_f32(row + index), bias_wide));
                vst1q_f32(row + index + 4,
                          vaddq_f32(vld1q_f32(row + index + 4), bias_wide));
            }
            for (; index < plane; index++)
                row[index] += bias_v;
        }
#else
        if (apply_silu) {
            for (size_t index = 0; index < plane; index++)
                row[index] = fiv_proto_silu(row[index] + bias_v);
        } else {
            for (size_t index = 0; index < plane; index++)
                row[index] += bias_v;
        }
#endif
    }
}

/* Engine-SIMD conv wrapper. Delegates to fiv_tensor_conv2d (kernel 1x1 ->
 * pointwise path, kernel 3x3 s1 -> 3x3 stride-1 path; both vectorized), then
 * folds bias (+ optional SiLU) in one pass over the output plane. src and dst
 * are [c, h, w] channel-major flat buffers; weight is [oc, ic, ky, kx]. */
static int fiv_proto_conv_engine(const ivf32* src, int input_channels,
                                 ivf32* dst, int output_channels,
                                 int height, int width,
                                 int kernel_size, int pad,
                                 const ivf32* weight, const ivf32* bias,
                                 int apply_silu)
{
    fiv_tensor3d src_view;
    fiv_tensor3d dst_view;
    fiv_tensor4d kernel_view;
    memset(&src_view, 0, sizeof(src_view));
    memset(&dst_view, 0, sizeof(dst_view));
    memset(&kernel_view, 0, sizeof(kernel_view));

    src_view.id = FIV_ID_TENSOR3D;
    src_view.dtype = FIV_32F1;
    src_view.data_continue = 1;
    src_view.channels = (size_t)input_channels;
    src_view.height = (size_t)height;
    src_view.width = (size_t)width;
    src_view.data.ptr = (void*)src;
    src_view.total_bytes = (size_t)input_channels * (size_t)height *
                           (size_t)width * sizeof(ivf32);

    dst_view = src_view;
    dst_view.channels = (size_t)output_channels;
    dst_view.data.ptr = dst;
    dst_view.total_bytes = (size_t)output_channels * (size_t)height *
                           (size_t)width * sizeof(ivf32);

    kernel_view.id = FIV_ID_TENSOR4D;
    kernel_view.dtype = FIV_32F1;
    kernel_view.data_continue = 1;
    kernel_view.shapes[0] = (size_t)output_channels;
    kernel_view.shapes[1] = (size_t)input_channels;
    kernel_view.shapes[2] = (size_t)kernel_size;
    kernel_view.shapes[3] = (size_t)kernel_size;
    kernel_view.data.ptr = (void*)weight;
    kernel_view.total_bytes = (size_t)output_channels * (size_t)input_channels *
                              (size_t)kernel_size * (size_t)kernel_size *
                              sizeof(ivf32);

    fiv_conv2d_params params;
    memset(&params, 0, sizeof(params));
    params.conv2d_method = FIV_CONV2D_STD;
    params.kernel_size_x = kernel_size;
    params.kernel_size_y = kernel_size;
    params.stride = 1;
    params.padding_method = 0;
    params.input_channels = input_channels;
    params.output_channels = output_channels;
    params.pad_top = pad;
    params.pad_bottom = pad;
    params.pad_left = pad;
    params.pad_right = pad;

    if (fiv_tensor_conv2d(&dst_view, &src_view, &kernel_view, &params) !=
        FIV_RET_OK)
        return -1;

    size_t plane = (size_t)height * (size_t)width;
    fiv_proto_bias_act(dst, bias, output_channels, plane, apply_silu);
    return 0;
}

/* 2x2 stride-2 transposed conv (no padding, no activation; the engine has no
 * ConvTranspose2d kernel so this stays scalar). */
/* 2x2 stride-2 transposed conv (no padding, no activation) re-expressed as
 * FOUR standard 1x1 convolutions. A stride-2 transposed conv maps input
 * (y, x) to output (2y+ky, 2x+kx); fixing the phase (ky, kx), the output
 * sub-grid  o[oc, y, x] = sum_ic in[ic, y, x] * W[ic, oc, ky, kx]  is exactly
 * a 1x1 conv with the per-phase weight slice
 *   w_phase[oc][ic] = W[ic][oc][ky][kx]     (engine [oc, ic, 1, 1] layout).
 * Every output element belongs to exactly one phase, so each phase runs the
 * engine 1x1 SIMD kernel (fiv_proto_conv_engine pointwise path) with the
 * per-channel bias applied exactly once per element, then the four sub-grids
 * are interleaved into the [oc, 2H, 2W] output. */
static int fiv_proto_conv_transpose2x2(const ivf32* input, ivf32* output,
                                       const ivf32* weight, const ivf32* bias,
                                       int channels, int height, int width)
{
    size_t plane        = (size_t)height * (size_t)width;
    size_t output_width = (size_t)width * 2;
    size_t output_plane = output_width * (size_t)height * 2;

    ivf32* phase   = (ivf32*)fiv_malloc((size_t)channels * plane * sizeof(ivf32));
    ivf32* phase_w = (ivf32*)fiv_malloc((size_t)channels * (size_t)channels *
                                        sizeof(ivf32));
    if (phase == NULL || phase_w == NULL) {
        fiv_free(phase);
        fiv_free(phase_w);
        return -1;
    }

    for (int ky = 0; ky < 2; ky++) {
        for (int kx = 0; kx < 2; kx++) {
            /* per-phase 1x1 weight: w_phase[oc][ic] = W[ic][oc][ky][kx],
               W laid out [in_c, out_c, 2, 2] as exported from torch. */
            for (int oc = 0; oc < channels; oc++)
                for (int ic = 0; ic < channels; ic++)
                    phase_w[oc * channels + ic] =
                        weight[((size_t)ic * channels + (size_t)oc) * 4 +
                               (size_t)ky * 2 + (size_t)kx];

            if (fiv_proto_conv_engine(input, channels, phase, channels,
                                      height, width, 1, 0, phase_w, bias,
                                      0) != 0) {
                fiv_free(phase);
                fiv_free(phase_w);
                return -1;
            }
            /* interleave phase into output sub-grid (2y+ky, 2x+kx) */
            for (int oc = 0; oc < channels; oc++) {
                const ivf32* phase_channel = phase + (size_t)oc * plane;
                ivf32* output_channel =
                    output + (size_t)oc * output_plane;
                for (int y = 0; y < height; y++) {
                    const ivf32* row_src = phase_channel + (size_t)y * width;
                    ivf32* row_dst = output_channel +
                        ((size_t)y * 2 + (size_t)ky) * output_width +
                        (size_t)kx;
                    for (int x = 0; x < width; x++)
                        row_dst[(size_t)x * 2] = row_src[x];
                }
            }
        }
    }

    fiv_free(phase);
    fiv_free(phase_w);
    return 0;
}

/* nearest-neighbour 2x / 4x upsample into a larger [c, oh, ow] buffer. */
static void fiv_proto_upsample_nearest(const ivf32* input, ivf32* output,
                                       int channels, int height, int width,
                                       int scale_factor)
{
    int out_height = height * scale_factor;
    int out_width = width * scale_factor;
    for (int c = 0; c < channels; c++) {
        const ivf32* src = input + (size_t)c * (size_t)height * width;
        ivf32* dst = output + (size_t)c * (size_t)out_height * out_width;
        for (int y = 0; y < out_height; y++) {
            int src_y = y / scale_factor;
            for (int x = 0; x < out_width; x++) {
                int src_x = x / scale_factor;
                dst[(size_t)y * out_width + x] =
                    src[(size_t)src_y * width + src_x];
            }
        }
    }
}

/* fiv_yolo26_proto_run: full Proto26 forward. P3/P4/P5 are [c,h,w] flat
 * ivf32 arrays (layout: channel-major). proto output is [32, oh, ow] where
 * oh/ow = 2 * P3 h/w. Weights are the BN-folded conv blobs. Returns 0 on
 * success. */
int fiv_yolo26_proto_run(const ivf32* p3, int p3_channels,
                         const ivf32* p4, int p4_channels,
                         const ivf32* p5, int p5_channels,
                         int height, int width,
                         const fiv_yolo26_proto_weights* weights,
                         ivf32* proto, int* out_channels, int* out_height,
                         int* out_width)
{
    size_t p3_plane = (size_t)height * (size_t)width;
    int level_1_w = width / 2;       /* P4: half of P3 (downsample 2x) */
    int level_1_h = height / 2;
    int level_2_w = width / 4;       /* P5: quarter of P3 */
    int level_2_h = height / 4;
    if (level_1_w < 1 || level_1_h < 1 || level_2_w < 1 || level_2_h < 1)
        return -1;

    int channels = p3_channels;       /* 64 */
    int proto_channels = 32;

    ivf32* refine0 = (ivf32*)fiv_malloc(
        (size_t)channels * (size_t)level_1_w * level_1_h * sizeof(ivf32));
    ivf32* refine0_up = (ivf32*)fiv_malloc(
        p3_plane * (size_t)channels * sizeof(ivf32));
    ivf32* refine1 = (ivf32*)fiv_malloc(
        (size_t)channels * (size_t)level_2_w * level_2_h * sizeof(ivf32));
    ivf32* refine1_up = (ivf32*)fiv_malloc(
        p3_plane * (size_t)channels * sizeof(ivf32));
    ivf32* feat = (ivf32*)fiv_malloc(
        p3_plane * (size_t)channels * sizeof(ivf32));
    ivf32* fused = (ivf32*)fiv_malloc(
        p3_plane * (size_t)channels * sizeof(ivf32));
    ivf32* conv1 = (ivf32*)fiv_malloc(
        p3_plane * (size_t)channels * sizeof(ivf32));
    int up_h = height * 2;
    int up_w = width * 2;
    ivf32* upsampled = (ivf32*)fiv_malloc(
        (size_t)channels * (size_t)up_h * up_w * sizeof(ivf32));
    ivf32* conv2 = (ivf32*)fiv_malloc(
        (size_t)channels * (size_t)up_h * up_w * sizeof(ivf32));
    if (refine0 == NULL || refine0_up == NULL || refine1 == NULL ||
        refine1_up == NULL || feat == NULL || fused == NULL || conv1 == NULL ||
        upsampled == NULL || conv2 == NULL) {
        fiv_free(refine0); fiv_free(refine0_up);
        fiv_free(refine1); fiv_free(refine1_up);
        fiv_free(feat); fiv_free(fused); fiv_free(conv1);
        fiv_free(upsampled); fiv_free(conv2);
        return -1;
    }

    /* feat_refine0 / feat_refine1: engine-SIMD 1x1 + SiLU, nearest upsample */
    if (fiv_proto_conv_engine(p4, p4_channels, refine0, channels,
                              level_1_h, level_1_w, 1, 0,
                              weights->refine0_w, weights->refine0_b, 1) != 0 ||
        fiv_proto_conv_engine(p5, p5_channels, refine1, channels,
                              level_2_h, level_2_w, 1, 0,
                              weights->refine1_w, weights->refine1_b, 1) != 0)
        goto fail;
    fiv_proto_upsample_nearest(refine0, refine0_up, channels,
                               level_1_h, level_1_w, 2);
    fiv_proto_upsample_nearest(refine1, refine1_up, channels,
                               level_2_h, level_2_w, 4);

    /* feat = P3 + up(refine0) + up(refine1) */
    size_t p3_count = p3_plane * (size_t)channels;
    for (size_t index = 0; index < p3_count; index++)
        feat[index] = p3[index] + refine0_up[index] + refine1_up[index];

    /* feat_fuse / cv1: engine-SIMD 3x3 s1 + SiLU */
    if (fiv_proto_conv_engine(feat, channels, fused, channels,
                              height, width, 3, 1,
                              weights->fuse_w, weights->fuse_b, 1) != 0 ||
        fiv_proto_conv_engine(fused, channels, conv1, channels,
                              height, width, 3, 1,
                              weights->cv1_w, weights->cv1_b, 1) != 0)
        goto fail;

    /* upsample = convT2x2(cv1): four 1x1 engine-SIMD phases */
    size_t up_plane = (size_t)up_h * up_w;
    memset(upsampled, 0, (size_t)channels * up_plane * sizeof(ivf32));
    if (fiv_proto_conv_transpose2x2(conv1, upsampled, weights->upsample_w,
                                    weights->upsample_b, channels, height,
                                    width) != 0)
        goto fail;

    /* cv2: engine-SIMD 3x3 s1 + SiLU @80x80 (the dominant conv), cv3: 1x1 */
    if (fiv_proto_conv_engine(upsampled, channels, conv2, channels,
                              up_h, up_w, 3, 1,
                              weights->cv2_w, weights->cv2_b, 1) != 0 ||
        fiv_proto_conv_engine(conv2, channels, proto, proto_channels,
                              up_h, up_w, 1, 0,
                              weights->cv3_w, weights->cv3_b, 1) != 0)
        goto fail;

    if (out_channels) *out_channels = proto_channels;
    if (out_height) *out_height = up_h;
    if (out_width) *out_width = up_w;

    fiv_free(refine0); fiv_free(refine0_up);
    fiv_free(refine1); fiv_free(refine1_up);
    fiv_free(feat); fiv_free(fused); fiv_free(conv1);
    fiv_free(upsampled); fiv_free(conv2);
    return 0;

fail:
    fiv_free(refine0); fiv_free(refine0_up);
    fiv_free(refine1); fiv_free(refine1_up);
    fiv_free(feat); fiv_free(fused); fiv_free(conv1);
    fiv_free(upsampled); fiv_free(conv2);
    return -1;
}
