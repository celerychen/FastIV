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

/*
 * High-level YuNet (libfacedetection) face detector.
 *
 * This is the simple 3-interface facade over the building blocks in fiv_yunet.h,
 * mirroring the BlazeFace wrapper (fiv_face.h / fiv_face.c):
 *
 *   fiv_create_yunet_detector           build graph + load + inject weights
 *   fiv_yunet_detector_on_image         run detection on a single fiv_mat
 *   fiv_release_yunet_detector          tear everything down
 *
 * The detector is fully self-contained: weights come from the same YWT1 blob
 * (yunet_weights.bin) used by iv_tv_time, and every layer's weight -- including
 * the fused conv_head -- is injected here, so the caller only touches the three
 * functions below.
 */

#ifndef _FIV_YUNET_MODEL_H_
#define _FIV_YUNET_MODEL_H_

#include "fiv_ctensor.h"
#include "fiv_yunet.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FIV_YUNET_MODEL_MAX_FACES FIV_YUNET_MAX_FACES   /* 512, = keep_top_k */

/* Output bundle passed to fiv_yunet_detector_on_image as face_info.
   The detector fills count and detections[0..count-1]. Each detection carries a
   pixel score/box/landmarks in the source image coordinates (see fiv_yunet_result). */
typedef struct {
    int               count;
    fiv_yunet_result  detections[FIV_YUNET_MODEL_MAX_FACES];
} fiv_yunet_model_result;

/* Build the YuNet detector from a YWT1 weights blob produced by the export tool.
   model_name is the path to yunet_weights.bin. Returns an opaque detector
   context, or NULL on failure. */
void* fiv_create_yunet_detector(char* model_name);

/* Run detection on a single image (fiv_mat, 8U3 RGB).
   face_info must point to a fiv_yunet_model_result the detector fills in.
   Returns FIV_RET_OK on success. */
fiv_ret fiv_yunet_detector_on_image(void* face_info, fiv_mat* image, void* detector);

/* Release the detector and all internal state; *detector set to NULL. */
fiv_ret fiv_release_yunet_detector(void** detector);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_YUNET_MODEL_H_ */