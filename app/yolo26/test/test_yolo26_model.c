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

/* YOLO26 model-facade smoke test: exercise the four 3-interface facades
 * (create -> *_on_image -> release) on the real bus image with the official
 * weight bundles shipped under app/yolo26/models/.
 *
 *   ./test_yolo26_model        (run from build/)
 *
 * The deep numeric parity against torch lives in the per-task tests
 * (test_yolo26_net/e2e/seg/pose/cls); this one only checks the public facade
 * contract end to end: a model loads, an image runs, results come back sane. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fiv_ctensor.h"
#include "fiv_image.h"
#include "fiv_yolo26_model.h"
#include "fiv_common.h"
#include "yolo26_ref.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(condition, message)                                      \
    do {                                                               \
        if (condition) { g_pass++; printf("  PASS  %s\n", message); } \
        else          { g_fail++; printf("  FAIL  %s\n", message); } \
    } while (0)

static void check_det_result(const fiv_yolo26_det_result* result)
{
    CHECK(result->count >= 1, "detector found at least one object");
    if (result->count <= 0) return;
    int bad_score = 0;
    int bad_box = 0;
    for (int i = 0; i < result->count && i < 8; i++) {
        const fiv_yolo26_box* detection = &result->detections[i];
        if (!(detection->score >= 0.0f && detection->score <= 1.0f)) bad_score++;
        if (detection->x2 <= detection->x1 || detection->y2 <= detection->y1) bad_box++;
    }
    CHECK(bad_score == 0, "all detection scores within [0,1]");
    CHECK(bad_box == 0, "all detection boxes valid (x2>x1, y2>y1)");
}

static void check_pose_result(const fiv_yolo26_pose_result* result)
{
    CHECK(result->count >= 1, "pose found at least one person");
    if (result->count <= 0) return;
    int bad_vis = 0;
    for (int i = 0; i < result->count && i < 4; i++)
        for (int k = 0; k < FIV_YOLO26_POSE_KEYPOINTS; k++) {
            ivf32 visibility = result->keypoints[i][k][2];
            if (!(visibility >= 0.0f && visibility <= 1.0f)) bad_vis++;
        }
    CHECK(bad_vis == 0, "all keypoint visibilities within [0,1]");
}

static void check_cls_result(const fiv_yolo26_cls_result* result)
{
    CHECK(result->num_classes > 0, "classifier reports a class count");
    CHECK(result->score >= 0.0f && result->score <= 1.0f, "top-1 score within [0,1]");
    printf("  info   top-5: ");
    for (int i = 0; i < FIV_YOLO26_TOP_K && i < result->num_classes; i++)
        printf("%d(%.3f) ", result->top_classes[i], (double)result->top_scores[i]);
    printf("\n");
}

int main(void)
{
    printf("=== YOLO26 model facade smoke test ===\n");

    /* detector */
    {
        void* detector = fiv_create_yolo26_detector(
            (char*)"../app/yolo26/models/yolo26n");
        CHECK(detector != NULL, "detector create");
        fiv_mat* image = fiv_create_image_from_file(
            (char*)"../app/yolo26/test/real/bus.jpg", FIV_RGB24_CS);
        if (detector != NULL && image != NULL) {
            fiv_yolo26_det_result result;
            memset(&result, 0, sizeof(result));
            fiv_ret ret = fiv_yolo26_detector_on_image(&result, image, detector);
            CHECK(ret == FIV_RET_OK, "detector on_image");
            check_det_result(&result);
            printf("  info   detections=%d\n", result.count);
        }
        if (image != NULL) fiv_release_image(image);
        fiv_release_yolo26_detector(&detector);
    }

    /* segmenter (masks are produced by the Proto26 stage) */
    {
        void* segmenter = fiv_create_yolo26_segmenter(
            (char*)"../app/yolo26/models/yolo26n-seg");
        CHECK(segmenter != NULL, "segmenter create");
        fiv_mat* image = fiv_create_image_from_file(
            (char*)"../app/yolo26/test/real/bus.jpg", FIV_RGB24_CS);
        if (segmenter != NULL && image != NULL) {
            fiv_yolo26_seg_result result;
            memset(&result, 0, sizeof(result));
            fiv_ret ret = fiv_yolo26_segmenter_on_image(&result, image, segmenter);
            CHECK(ret == FIV_RET_OK, "segmenter on_image");
            CHECK(result.count >= 1, "segmenter found at least one object");
            CHECK(result.proto != NULL && result.proto_grid_w > 0.0f,
                  "segmenter produced the proto grid");
            CHECK(result.masks != NULL, "segmenter produced per-instance masks");
            printf("  info   segment detections=%d proto_grid=%dx%d\n",
                   result.count, (int)result.proto_grid_w, (int)result.proto_grid_h);
            /* functional mask checks (bit-exact proto parity vs the torch
               reference belongs to the deep test which feeds the reference input
               tensor; this facade letterboxes an arbitrary image instead). */
            if (result.masks != NULL && result.proto != NULL && result.count > 0) {
                int plane = (int)result.proto_grid_w * (int)result.proto_grid_h;
                ivf32 out_of_range = 0.0f;
                ivf32 max_mask = 0.0f;
                for (int p = 0; p < plane && p < 96 * 96; p++) {
                    ivf32 value = result.masks[p];
                    if (value < 0.0f || value > 1.0f) out_of_range = 1.0f;
                    if (value > max_mask) max_mask = value;
                }
                CHECK(out_of_range == 0.0f, "instance mask values within [0,1]");
                CHECK(max_mask > 0.5f, "top-instance mask is non-degenerate (max>0.5)");
                printf("  info   top mask max=%.3f\n", (double)max_mask);
            }
        }
        if (image != NULL) fiv_release_image(image);
        fiv_release_yolo26_segmenter(&segmenter);
    }

    /* pose */
    {
        void* pose = fiv_create_yolo26_pose((char*)"../app/yolo26/models/yolo26n-pose");
        CHECK(pose != NULL, "pose create");
        fiv_mat* image = fiv_create_image_from_file(
            (char*)"../app/yolo26/test/real/bus.jpg", FIV_RGB24_CS);
        if (pose != NULL && image != NULL) {
            fiv_yolo26_pose_result result;
            memset(&result, 0, sizeof(result));
            fiv_ret ret = fiv_yolo26_pose_on_image(&result, image, pose);
            CHECK(ret == FIV_RET_OK, "pose on_image");
            check_pose_result(&result);
            printf("  info   persons=%d\n", result.count);
        }
        if (image != NULL) fiv_release_image(image);
        fiv_release_yolo26_pose(&pose);
    }

    /* classifier */
    {
        void* classifier = fiv_create_yolo26_classifier(
            (char*)"../app/yolo26/models/yolo26n-cls");
        CHECK(classifier != NULL, "classifier create");
        fiv_mat* image = fiv_create_image_from_file(
            (char*)"../app/yolo26/test/real/bus.jpg", FIV_RGB24_CS);
        if (classifier != NULL && image != NULL) {
            fiv_yolo26_cls_result result;
            memset(&result, 0, sizeof(result));
            fiv_ret ret = fiv_yolo26_classifier_on_image(&result, image, classifier);
            CHECK(ret == FIV_RET_OK, "classifier on_image");
            check_cls_result(&result);
        }
        if (image != NULL) fiv_release_image(image);
        fiv_release_yolo26_classifier(&classifier);
    }

    printf("=== model facade: pass=%d fail=%d ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
