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

#ifndef _FIV_FACE_H_
#define _FIV_FACE_H_

#include "fiv_ctensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FIV_FACE_MAX_DETS 64
#define FIV_FACE_KEYPOINTS 6
#define FIV_FACE_TENSOR_SIZE 128
#define FIV_FACE_NUM_BOXES 896   /* BlazeFace short-range anchor count */
#define FIV_FACE_NUM_COORDS 16   /* 4 bbox + 6*2 keypoints */
#define FIV_FACE_MIN_SCORE  0.5f
#define FIV_FACE_NMS_THRESH 0.5f

typedef struct {
    float score;
    float x, y, w, h;                /* pixel coords in the source image */
    float kps[FIV_FACE_KEYPOINTS][2]; /* normalized [0,1] */
} fiv_face_detection;

/* Output bundle passed to fiv_face_detector_on_image as face_info.
   The detector fills count and detections[0..count-1]. */
typedef struct {
    int               count;
    fiv_face_detection detections[FIV_FACE_MAX_DETS];
} fiv_face_result;

/* Build the BlazeFace short-range detector from a weights blob produced by
   export_weights.py. model_name is the path to blazeface_weights.bin.
   Returns an opaque detector context, or NULL on failure. */
void* fiv_create_face_detetor(char* model_name);

/* Run detection on a single image (fiv_mat, 8U3 RGB).
   face_info must point to a fiv_face_result the detector fills in.
   Returns FIV_RET_OK on success. */
fiv_ret fiv_face_detector_on_image(void* face_info, fiv_mat* image, void* detector);

/* Release the detector and all internal state; *detector set to NULL. */
fiv_ret fiv_release_face_detector(void** detector);

#ifdef __cplusplus
}
#endif

#endif /* _FIV_FACE_H_ */
