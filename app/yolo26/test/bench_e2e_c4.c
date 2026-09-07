/* YOLO26 4-task end-to-end dump + timing on bus.jpg via the model facades.
 * Dumps decoded results (letterbox model-image coords, same schema as torch
 * detect_out: "x1 y1 x2 y2 score class [extras...]") plus wall times.
 * Run from build/: ./e2e_c4 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "fiv_ctensor.h"
#include "fiv_common.h"
#include "fiv_image.h"
#include "fiv_yolo26_model.h"

typedef struct {
    const char* task;
    const char* dir;
    void* (*create)(const char*);
    fiv_ret (*on_image)(void*, fiv_mat*, void*);
    fiv_ret (*release)(void**);
    size_t result_size;
} task_spec;

static const task_spec kTasks[4] = {
    { "detector",  "../app/yolo26/models/yolo26n",
      (void* (*)(const char*))fiv_create_yolo26_detector,
      (fiv_ret (*)(void*, fiv_mat*, void*))fiv_yolo26_detector_on_image,
      (fiv_ret (*)(void**))fiv_release_yolo26_detector,
      sizeof(fiv_yolo26_det_result) },
    { "segmenter", "../app/yolo26/models/yolo26n-seg",
      (void* (*)(const char*))fiv_create_yolo26_segmenter,
      (fiv_ret (*)(void*, fiv_mat*, void*))fiv_yolo26_segmenter_on_image,
      (fiv_ret (*)(void**))fiv_release_yolo26_segmenter,
      sizeof(fiv_yolo26_seg_result) },
    { "pose",      "../app/yolo26/models/yolo26n-pose",
      (void* (*)(const char*))fiv_create_yolo26_pose,
      (fiv_ret (*)(void*, fiv_mat*, void*))fiv_yolo26_pose_on_image,
      (fiv_ret (*)(void**))fiv_release_yolo26_pose,
      sizeof(fiv_yolo26_pose_result) },
    { "classifier", "../app/yolo26/models/yolo26n-cls",
      (void* (*)(const char*))fiv_create_yolo26_classifier,
      (fiv_ret (*)(void*, fiv_mat*, void*))fiv_yolo26_classifier_on_image,
      (fiv_ret (*)(void**))fiv_release_yolo26_classifier,
      sizeof(fiv_yolo26_cls_result) },
};

static void dump_common_rows(FILE* fp, const fiv_yolo26_box* boxes, int count,
                             const char* task, void* raw)
{
    for (int i = 0; i < count; i++) {
        const fiv_yolo26_box* b = &boxes[i];
        fprintf(fp, "%.3f %.3f %.3f %.3f %.6f %d", b->x1, b->y1, b->x2, b->y2,
                b->score, b->class_id);
        if (strcmp(task, "segmenter") == 0) {
            const fiv_yolo26_seg_result* seg = (const fiv_yolo26_seg_result*)raw;
            for (int m = 0; m < FIV_YOLO26_MASK_DIMS; m++)
                fprintf(fp, " %.6f", seg->coef[i][m]);
        } else if (strcmp(task, "pose") == 0) {
            const fiv_yolo26_pose_result* pose = (const fiv_yolo26_pose_result*)raw;
            for (int k = 0; k < FIV_YOLO26_POSE_KEYPOINTS; k++)
                fprintf(fp, " %.3f %.3f %.4f", pose->keypoints[i][k][0],
                        pose->keypoints[i][k][1], pose->keypoints[i][k][2]);
        }
        fprintf(fp, "\n");
    }
}

int main(void)
{
    mkdir("/tmp/c_e2e", 0755);
    fiv_mat* image = fiv_create_image_from_file(
        (char*)"../app/yolo26/test/real/bus.jpg", FIV_RGB24_CS);
    if (!image) { printf("FAIL: load bus.jpg\n"); return 1; }

    const int warmup = 5;
    const int runs = 20;
    for (int t = 0; t < 4; t++) {
        const task_spec* spec = &kTasks[t];
        void* model = spec->create((char*)spec->dir);
        if (!model) { printf("[%s] create FAIL\n", spec->task); continue; }
        unsigned char* result = (unsigned char*)malloc(spec->result_size);

        double best = 1e30, sum = 0.0;
        for (int i = 0; i < warmup + runs; i++) {
            memset(result, 0, spec->result_size);
            ivf64 t0 = fiv_get_current_system_time();
            fiv_ret ret = spec->on_image(result, image, model);
            ivf64 t1 = fiv_get_current_system_time();
            if (ret != FIV_RET_OK) { printf("[%s] on_image FAIL ret=%d\n", spec->task, ret); break; }
            if (i >= warmup) { double dt = t1 - t0; sum += dt; if (dt < best) best = dt; }
        }
        double avg = sum / (double)runs;
        printf("%-10s best %7.2f ms   avg %7.2f ms   (on_image incl letterbox+infer+decode)\n",
               spec->task, best, avg);
        {
            ivf64 stage[3] = { 0.0, 0.0, 0.0 };
            int calls = 0;
            if (fiv_yolo26_get_timing(model, stage, &calls) == FIV_RET_OK && calls > 0) {
                printf("            stage avg: pre(letterbox) %6.2f ms | infer %6.2f ms | post(decode+pp) %6.2f ms  (over %d calls)\n",
                       stage[0], stage[1], stage[2], calls);
            }
        }

        char path[128];
        snprintf(path, sizeof(path), "/tmp/c_e2e/%s.txt", spec->task);
        FILE* fp = fopen(path, "w");
        if (strcmp(spec->task, "detector") == 0) {
            const fiv_yolo26_det_result* r = (const fiv_yolo26_det_result*)result;
            dump_common_rows(fp, r->detections, r->count, spec->task, NULL);
        } else if (strcmp(spec->task, "segmenter") == 0) {
            const fiv_yolo26_seg_result* r = (const fiv_yolo26_seg_result*)result;
            dump_common_rows(fp, r->detections, r->count, spec->task, (void*)r);
            if (r->proto != NULL) {
                int plane = (int)r->proto_grid_w * (int)r->proto_grid_h;
                FILE* pf = fopen("/tmp/c_e2e/seg_proto.bin", "wb");
                fwrite(r->proto, sizeof(ivf32), (size_t)FIV_YOLO26_MASK_DIMS * plane, pf);
                fclose(pf);
            }
        } else if (strcmp(spec->task, "pose") == 0) {
            const fiv_yolo26_pose_result* r = (const fiv_yolo26_pose_result*)result;
            dump_common_rows(fp, r->detections, r->count, spec->task, (void*)r);
        } else {
            const fiv_yolo26_cls_result* r = (const fiv_yolo26_cls_result*)result;
            fprintf(fp, "%d %.6f\n", r->class_id, r->score);
            for (int k = 0; k < FIV_YOLO26_TOP_K; k++)
                fprintf(fp, "%d %.6f\n", r->top_classes[k], r->top_scores[k]);
        }
        fclose(fp);
        free(result);
        spec->release(&model);
    }
    fiv_release_image(image);
    return 0;
}
