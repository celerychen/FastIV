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
 * High-level FaceMesh 478-point 3D landmark model.
 *
 * This is a simple 3-interface facade over the landmark building blocks:
 *
 *   fiv_create_face_landmark        build the landmark graph
 *   fiv_face_landmark_from_faces    run the landmark net on a batch of rects
 *   fiv_release_face_landmark       tear the model down
 *
 * Face detection and landmark localization are STRICTLY SEPARATED. This model
 * owns no face detector; the caller runs detection first and hands the landmark
 * stage the normalized rects (fiv_lm_rect) of every detected face:
 *
 *   1. Detect faces with the independent face detector (fiv_face.h).
 *   2. Convert every detection to an expanded normalized rect with
 *      fiv_lm_expanded_face_rect (fiv_landmark_geom.h) - the "Tasks
 *      FaceDetectorGraph" rule (scale_x/y = 1.5, square_long).
 *   3. Feed the rects to fiv_face_landmark_from_faces.
 *
 * For each rect the rotated ROI is perspective-warped to the 256x256 tensor
 * (BORDER_REPLICATE), run through the landmark net, the 1434 raw logits are
 * decoded to 478 (x, y, z) landmarks, and landmarks are projected back to
 * source-image pixel coordinates with the ROI transform (z scaled by the
 * projection's z scale), mirroring MediaPipe's LandmarkProjectionCalculator.
 */

#ifndef _FIV_LANDMARK_MODEL_H_
#define _FIV_LANDMARK_MODEL_H_

#include "fiv_ctensor.h"
#include "fiv_landmark.h"        /* FIV_LM_NUM_LANDMARKS / FIV_LM_TENSOR_SIZE */
#include "fiv_landmark_geom.h"   /* fiv_lm_rect */

#ifdef __cplusplus
extern "C" {
#endif

#define FIV_LM_MODEL_MAX_FACES 64   /* max rects per batch call */

/* One 3D landmark in source-image pixel coordinates (z scaled by the ROI
   projection, as in MediaPipe's LandmarkProjectionCalculator). */
typedef struct {
    ivf32 x, y, z;
} fiv_lm_point;

/* Landmark output for a single face / rect. */
typedef struct {
    fiv_lm_point points[FIV_LM_NUM_LANDMARKS];
    ivf32        presence;   /* sigmoid of the presence logit, [0,1] */
} fiv_lm_face;

/* Output bundle filled by fiv_face_landmark_from_faces: count and
   faces[0..count-1]. `n_landmarks` is the number of valid points per face
   (478 for the 256 model, 468 for the 192 model). */
typedef struct {
    int         count;
    int         n_landmarks;
    fiv_lm_face faces[FIV_LM_MODEL_MAX_FACES];
} fiv_lm_model_result;

/* Build the FaceMesh 478-landmark graph from the landmark network blob.
   lm_model = path to landmark_net.bin (landmark stage only; see the separation
   note above). Returns an opaque model context, or NULL on failure. */
void* fiv_create_face_landmark(const char* lm_model);

/* Run the landmark network on a batch of normalized rects (reused face
   detection results). result must point to a fiv_lm_model_result that is filled
   with count and faces[0..count-1]; n is clamped to FIV_LM_MODEL_MAX_FACES.
   Returns FIV_RET_OK on success. A single face is the n == 1 case. */
fiv_ret fiv_face_landmark_from_faces(void* result, fiv_mat* image,
                                     void* model, const fiv_lm_rect* rects,
                                     iv32s n);

/* Release the model and all internal state; *model set to NULL. */
fiv_ret fiv_release_face_landmark(void** model);

#ifdef __cplusplus
}
#endif

#endif  /* _FIV_LANDMARK_MODEL_H_ */
