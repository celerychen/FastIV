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
 * Everything runs on plain ivf32 buffers; weights come from the model
 * directory blobs (proto_*.f32), each with its per-channel bias.
 */

#include "fiv_yolo26_proto.h"

#include "fiv_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static ivf32 proto_silu(ivf32 value)
{
    ivf32 sigmoid_value = 1.0f / (1.0f + expf(-value));
    return value * sigmoid_value;
}

/* 1x1 conv: out[n] = act(w.dot(in) + bias) over the channel dim. */
static void proto_conv1x1(const ivf32* input, ivf32* output,
                          const ivf32* weight, const ivf32* bias,
                          int input_channels, int output_channels,
                          size_t spatial_count)
{
    for (size_t s = 0; s < spatial_count; s++) {
        for (int oc = 0; oc < output_channels; oc++) {
            ivf32 sum = bias[oc];
            for (int ic = 0; ic < input_channels; ic++)
                sum += weight[(size_t)oc * input_channels + ic] *
                       input[(size_t)ic * spatial_count + s];
            output[(size_t)oc * spatial_count + s] = proto_silu(sum);
        }
    }
}

/* 3x3 conv with implicit zero padding of 1; input is [channels, h, w]. */
static void proto_conv3x3(const ivf32* input, ivf32* output,
                          const ivf32* weight, const ivf32* bias,
                          int channels, int height, int width,
                          int output_channels)
{
    size_t plane = (size_t)height * (size_t)width;
    for (int oc = 0; oc < output_channels; oc++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                ivf32 sum = bias[oc];
                for (int ic = 0; ic < channels; ic++) {
                    for (int ky = -1; ky <= 1; ky++) {
                        int sy = y + ky;
                        if (sy < 0 || sy >= height) continue;
                        for (int kx = -1; kx <= 1; kx++) {
                            int sx = x + kx;
                            if (sx < 0 || sx >= width) continue;
                            const ivf32* wrow = weight +
                                ((size_t)oc * channels + ic) * 9 +
                                (size_t)(ky + 1) * 3 + (kx + 1);
                            sum += *wrow * input[(size_t)ic * plane +
                                                 (size_t)sy * width + sx];
                        }
                    }
                }
                output[(size_t)oc * plane + (size_t)y * width + x] =
                    proto_silu(sum);
            }
        }
    }
}

/* 2x2 stride-2 transposed conv (no padding, no activation). */
static void proto_conv_transpose2x2(const ivf32* input, ivf32* output,
                                    const ivf32* weight, const ivf32* bias,
                                    int channels, int height, int width)
{
    size_t input_plane = (size_t)height * (size_t)width;
    size_t output_plane = (size_t)(height * 2) * (size_t)(width * 2);
    for (int c = 0; c < channels; c++) {
        ivf32* output_channel = output + (size_t)c * output_plane;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int ky = 0; ky < 2; ky++) {
                    for (int kx = 0; kx < 2; kx++) {
                        int oy = y * 2 + ky;
                        int ox = x * 2 + kx;
                        size_t out_index = (size_t)oy * (width * 2) + ox;
                        /* ConvTranspose2d weight layout [in_c, out_c, ky, kx] */
                        for (int ic = 0; ic < channels; ic++) {
                            const ivf32* wrow = weight +
                                ((size_t)ic * channels + c) * 4;
                            ivf32 weight_value =
                                wrow[(size_t)ky * 2 + kx];
                            output_channel[out_index] +=
                                weight_value * input[(size_t)ic * input_plane +
                                                     (size_t)y * width + x];
                        }
                    }
                }
            }
        }
        for (size_t k = 0; k < output_plane; k++) output_channel[k] += bias[c];
    }
}

/* nearest-neighbour 2x / 4x upsample in place into a larger buffer. */
static void proto_upsample_nearest(const ivf32* input, ivf32* output,
                                   int channels, int height, int width,
                                   int scale_factor)
{
    int out_height = height * scale_factor;
    int out_width = width * scale_factor;
    for (int c = 0; c < channels; c++) {
        const ivf32* src = input + (size_t)c * (size_t)height * width;
        ivf32* dst = output + (size_t)c * (size_t)out_height * out_width;
        for (int y = 0; y < out_height; y++) {
            int sy = y / scale_factor;
            for (int x = 0; x < out_width; x++) {
                int sx = x / scale_factor;
                dst[(size_t)y * out_width + x] =
                    src[(size_t)sy * width + sx];
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
    ivf32* refine0_up = (ivf32*)fiv_malloc(p3_plane * (size_t)channels * sizeof(ivf32));
    ivf32* refine1 = (ivf32*)fiv_malloc(
        (size_t)channels * (size_t)level_2_w * level_2_h * sizeof(ivf32));
    ivf32* refine1_up = (ivf32*)fiv_malloc(p3_plane * (size_t)channels * sizeof(ivf32));
    ivf32* feat = (ivf32*)fiv_malloc(p3_plane * (size_t)channels * sizeof(ivf32));
    ivf32* fused = (ivf32*)fiv_malloc(p3_plane * (size_t)channels * sizeof(ivf32));
    ivf32* conv1 = (ivf32*)fiv_malloc(p3_plane * (size_t)channels * sizeof(ivf32));
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

    /* feat_refine0 / feat_refine1 (1x1 conv + silu) */
    proto_conv1x1(p4, refine0, weights->refine0_w, weights->refine0_b,
                  p4_channels, channels, (size_t)level_1_w * level_1_h);
    proto_upsample_nearest(refine0, refine0_up, channels,
                           level_1_h, level_1_w, 2);
    proto_conv1x1(p5, refine1, weights->refine1_w, weights->refine1_b,
                  p5_channels, channels, (size_t)level_2_w * level_2_h);
    proto_upsample_nearest(refine1, refine1_up, channels,
                           level_2_h, level_2_w, 4);

    /* feat = P3 + up0 + up1 */
    for (size_t i = 0; i < p3_plane * (size_t)channels; i++)
        feat[i] = p3[i] + refine0_up[i] + refine1_up[i];

    /* feat_fuse = silu(conv3x3(feat)) */
    proto_conv3x3(feat, fused, weights->fuse_w, weights->fuse_b,
                  channels, height, width, channels);
    /* cv1 = silu(conv3x3(fused)) */
    proto_conv3x3(fused, conv1, weights->cv1_w, weights->cv1_b,
                  channels, height, width, channels);
    /* upsample = convT2x2(cv1) */
    memset(upsampled, 0,
           (size_t)channels * (size_t)up_h * up_w * sizeof(ivf32));
    proto_conv_transpose2x2(conv1, upsampled, weights->upsample_w,
                            weights->upsample_b, channels, height, width);
    /* cv2 = silu(conv3x3(upsampled)) */
    proto_conv3x3(upsampled, conv2, weights->cv2_w, weights->cv2_b,
                  channels, up_h, up_w, channels);
    /* cv3 = silu(conv1x1(conv2)) -> proto */
    size_t up_plane = (size_t)up_h * up_w;
    for (size_t s = 0; s < up_plane; s++) {
        for (int oc = 0; oc < proto_channels; oc++) {
            ivf32 sum = weights->cv3_b[oc];
            for (int ic = 0; ic < channels; ic++)
                sum += weights->cv3_w[(size_t)oc * channels + ic] *
                       conv2[(size_t)ic * up_plane + s];
            proto[(size_t)oc * up_plane + s] = proto_silu(sum);
        }
    }

    if (out_channels) *out_channels = proto_channels;
    if (out_height) *out_height = up_h;
    if (out_width) *out_width = up_w;

    fiv_free(refine0); fiv_free(refine0_up);
    fiv_free(refine1); fiv_free(refine1_up);
    fiv_free(feat); fiv_free(fused); fiv_free(conv1);
    fiv_free(upsampled); fiv_free(conv2);
    return 0;
}
