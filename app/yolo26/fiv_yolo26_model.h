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
 * YOLO26 high-level model facades for the four detection-family tasks.
 *
 * Each task exposes the same three-interface shape as the face detectors
 * (fiv_face / fiv_yunet):
 *
 *   fiv_create_yolo26_detector / segmenter / pose / classifier
 *   fiv_yolo26_<task>_on_image
 *   fiv_release_yolo26_<task>
 *
 * A model is a directory (fold.json / fold_w.f32 / fold_b.f32 + attention
 * blobs + task extras, exactly what app/yolo26/models/<variant>/ holds).
 * The create call loads the weights, builds the graph and keeps everything in
 * an opaque context; on_image letterboxes the input image to the model size,
 * runs inference, decodes and fills the task result (in model-image pixel
 * coordinates). release tears everything down.
 */

#ifndef FIV_YOLO26_MODEL_H
#define FIV_YOLO26_MODEL_H

#include "fiv_ctensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FIV_YOLO26_MAX_DETECTIONS 300
#define FIV_YOLO26_NUM_CLASSES    80
#define FIV_YOLO26_POSE_KEYPOINTS 17
#define FIV_YOLO26_MASK_DIMS      32   /* proto channels (yolo26n-seg nm) */
#define FIV_YOLO26_TOP_K          5    /* classifier top classes */

/* Common per-detection geometry (pixel coords in the model input image,
 * i.e. after letterboxing the source to the model size). */
typedef struct {
    ivf32 x1, y1, x2, y2;      /* bounding box corners */
    ivf32 score;
    int   class_id;
} fiv_yolo26_box;

/* ---- detector result ---- */
typedef struct {
    int              count;
    fiv_yolo26_box   detections[FIV_YOLO26_MAX_DETECTIONS];
} fiv_yolo26_det_result;

/* ---- segmenter result: detections + per-instance mask on the proto grid
   (proto rows are Hp*Wp floats; Hp/Wp reported). ---- */
typedef struct {
    int            count;
    ivf32          proto_grid_w;    /* grid width (80 @ 320 input) */
    ivf32          proto_grid_h;
    ivf32*         proto;           /* [MASK_DIMS * Hp * Wp], owned by ctx */
    int            mask_rows[FIV_YOLO26_MAX_DETECTIONS]; /* -1 when none */
    fiv_yolo26_box detections[FIV_YOLO26_MAX_DETECTIONS];
    ivf32          coef[FIV_YOLO26_MAX_DETECTIONS][FIV_YOLO26_MASK_DIMS];
    ivf32*         masks;   /* [count * Hp * Wp]: row i is the sigmoid-rebuilt
                               instance mask of detection i; ctx-owned scratch. */
} fiv_yolo26_seg_result;

/* ---- pose result: box + per-keypoint [x, y, visibility] ---- */
typedef struct {
    int              count;
    fiv_yolo26_box   detections[FIV_YOLO26_MAX_DETECTIONS];
    ivf32            keypoints[FIV_YOLO26_MAX_DETECTIONS]
                              [FIV_YOLO26_POSE_KEYPOINTS][3];
} fiv_yolo26_pose_result;

/* ---- classifier result ---- */
typedef struct {
    int    class_id;            /* top-1 */
    ivf32  score;               /* top-1 softmax probability */
    int    top_classes[FIV_YOLO26_TOP_K];
    ivf32  top_scores[FIV_YOLO26_TOP_K];
    ivf32* logits;              /* [nc], owned by ctx */
    int    num_classes;
} fiv_yolo26_cls_result;

/* All creators take the model directory; NULL on failure. */
void* fiv_create_yolo26_detector(const char* model_dir);
void* fiv_create_yolo26_segmenter(const char* model_dir);
void* fiv_create_yolo26_pose(const char* model_dir);
void* fiv_create_yolo26_classifier(const char* model_dir);

/* Run one image (RGB 8U, arbitrary size; letterboxed internally). */
fiv_ret fiv_yolo26_detector_on_image(void* result, fiv_mat* image, void* model);
fiv_ret fiv_yolo26_segmenter_on_image(void* result, fiv_mat* image, void* model);
fiv_ret fiv_yolo26_pose_on_image(void* result, fiv_mat* image, void* model);
fiv_ret fiv_yolo26_classifier_on_image(void* result, fiv_mat* image, void* model);

fiv_ret fiv_release_yolo26_detector(void** model);
fiv_ret fiv_release_yolo26_segmenter(void** model);
fiv_ret fiv_release_yolo26_pose(void** model);
fiv_ret fiv_release_yolo26_classifier(void** model);

#ifdef __cplusplus
}
#endif

#endif /* FIV_YOLO26_MODEL_H */
