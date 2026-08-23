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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fiv_image.h"
#include "fiv_ctensor.h"

/* ---- error paths (no external resources needed) ---- */

static int test_load_error_paths(void) {
    int pass = 1;

    /* NULL file name must yield NULL (no crash). */
    fiv_mat* missing = fiv_create_image_from_file(NULL, FIV_RGB24_CS);
    if (missing != NULL) { printf("FAIL: NULL file_name should return NULL\n"); pass = 0; }

    /* A non-existent file must yield NULL. */
    fiv_mat* nofile = fiv_create_image_from_file("__no_such_file__.png", FIV_RGB24_CS);
    if (nofile != NULL) { printf("FAIL: missing file should return NULL\n"); pass = 0; }

    /* write error paths: NULL inputs. */
    if (fiv_image_write(NULL, NULL)        != FIV_RET_ERR_PARA) { printf("FAIL: write(NULL,NULL)\n"); pass = 0; }
    if (fiv_image_write("x.png", NULL)     != FIV_RET_ERR_PARA) { printf("FAIL: write(name,NULL)\n"); pass = 0; }

    return pass;
}

/* ---- synthetic write -> read round trip (no external resources) ---- */

static int test_write_roundtrip(void) {
    const int  height = 4, width = 3;
    const char tmp_path[] = "fiv_image_io_tmp.png";
    int        row_index, col_index, channel_index;
    int        pass = 1;

    /* Build a deterministic RGB tensor. */
    size_t shape[2] = { (size_t)height, (size_t)width };
    fiv_mat* image  = (fiv_mat*)fiv_create_tensor2d(shape, FIV_8U3);
    if (image == NULL) { printf("FAIL: create_tensor2d for write test\n"); return 0; }
    iv8u* data = image->data.ptr8u;
    for (row_index = 0; row_index < height; row_index++)
        for (col_index = 0; col_index < width; col_index++)
            for (channel_index = 0; channel_index < 3; channel_index++) {
                size_t off = ((size_t)row_index * width + col_index) * 3 + channel_index;
                data[off] = (iv8u)((off * 7 + 3) & 0xff);
            }
    image->color_space_type = FIV_RGB24_CS;

    /* Write as RGB PNG, then read back and compare. PNG is lossless. */
    if (fiv_image_write((char*)tmp_path, image) != FIV_RET_OK) {
        printf("FAIL: fiv_image_write RGB returned error\n");
        fiv_release_image(image);
        return 0;
    }
    fiv_mat* reloaded = fiv_create_image_from_file((char*)tmp_path, FIV_RGB24_CS);
    if (reloaded == NULL) {
        printf("FAIL: reload written PNG returned NULL\n");
        fiv_release_image(image);
        return 0;
    }
    if ((int)reloaded->height != height || (int)reloaded->width != width ||
        reloaded->dtype != FIV_8U3 || reloaded->color_space_type != FIV_RGB24_CS) {
        printf("FAIL: reloaded metadata mismatch\n");
        pass = 0;
    } else {
        iv8u* rdata = reloaded->data.ptr8u;
        for (row_index = 0; row_index < height && pass; row_index++)
            for (col_index = 0; col_index < width && pass; col_index++)
                for (channel_index = 0; channel_index < 3 && pass; channel_index++) {
                    size_t off = ((size_t)row_index * width + col_index) * 3 + channel_index;
                    if (rdata[off] != data[off]) {
                        printf("FAIL: RGB round-trip pixel mismatch at (%d,%d,%d)\n",
                               row_index, col_index, channel_index);
                        pass = 0;
                    }
                }
    }
    fiv_release_image(reloaded);

    /* BGR path: write the same RGB tensor tagged BGR; the writer stores the
       tensor's bytes verbatim (BGR stays BGR). Reload as BGR and confirm the
       bytes are byte-exact (PNG is lossless). */
    image->color_space_type = FIV_BGR24_CS;
    if (fiv_image_write((char*)tmp_path, image) != FIV_RET_OK) {
        printf("FAIL: fiv_image_write BGR returned error\n");
        fiv_release_image(image);
        return 0;
    }
    reloaded = fiv_create_image_from_file((char*)tmp_path, FIV_BGR24_CS);
    if (reloaded == NULL) {
        printf("FAIL: reload BGR-written PNG returned NULL\n");
        fiv_release_image(image);
        return 0;
    }
    if (reloaded->dtype == FIV_8U3 && reloaded->color_space_type == FIV_BGR24_CS) {
        iv8u* rdata = reloaded->data.ptr8u;
        for (row_index = 0; row_index < height && pass; row_index++)
            for (col_index = 0; col_index < width && pass; col_index++)
                for (channel_index = 0; channel_index < 3 && pass; channel_index++) {
                    size_t off = ((size_t)row_index * width + col_index) * 3 + channel_index;
                    /* The writer stores the tensor's bytes verbatim and the
                       loader never reorders, so even a BGR-tagged file round
                       trips the raw bytes byte-exact (no implicit R/B swap). */
                    if (rdata[off] != data[off]) {
                        printf("FAIL: BGR tagged round-trip byte mismatch at (%d,%d,%d)\n",
                               row_index, col_index, channel_index);
                        pass = 0;
                    }
                }
    } else {
        printf("FAIL: BGR reloaded metadata mismatch\n");
        pass = 0;
    }
    fiv_release_image(reloaded);
    fiv_release_image(image);
    remove(tmp_path);
    return pass;
}

/* ---- color space conversion (no external resources) ---- */

static int test_color_space_convert(void) {
    const int  height = 2, width = 3;
    const char tmp_path[] = "fiv_image_cs_tmp.png";
    int        row_index, col_index, channel_index;
    int        pass = 1;

    /* Build a deterministic RGB tensor and persist it via a lossless PNG so we
       can reload an identical copy as the source for the conversions. */
    size_t shape[2] = { (size_t)height, (size_t)width };
    fiv_mat* rgb = (fiv_mat*)fiv_create_tensor2d(shape, FIV_8U3);
    if (rgb == NULL) { printf("FAIL: create_tensor2d for cs test\n"); return 0; }
    iv8u* rgb_data = rgb->data.ptr8u;
    for (row_index = 0; row_index < height; row_index++)
        for (col_index = 0; col_index < width; col_index++)
            for (channel_index = 0; channel_index < 3; channel_index++) {
                size_t off = ((size_t)row_index * width + col_index) * 3 + channel_index;
                rgb_data[off] = (iv8u)((off * 11 + 5) & 0xff);
            }
    rgb->color_space_type = FIV_RGB24_CS;

    if (fiv_image_write((char*)tmp_path, rgb) != FIV_RET_OK) {
        printf("FAIL: write RGB for cs test\n");
        fiv_release_image(rgb);
        return 0;
    }
    fiv_mat* src = fiv_create_image_from_file((char*)tmp_path, FIV_RGB24_CS);
    if (src == NULL) {
        printf("FAIL: reload RGB source for cs test\n");
        fiv_release_image(rgb);
        return 0;
    }

    /* --- RGB2BGR: dst is a separate 3-channel tensor; channels 0/2 swap. --- */
    fiv_mat* bgr = (fiv_mat*)fiv_create_tensor2d(shape, FIV_8U3);
    if (bgr == NULL) { printf("FAIL: create bgr dst\n"); pass = 0; }
    else {
        if (fiv_image_color_space_convertor(bgr, src, FIV_CS_RGB2BGR) != FIV_RET_OK) {
            printf("FAIL: RGB2BGR returned error\n");
            pass = 0;
        } else if (bgr->color_space_type != FIV_BGR24_CS) {
            printf("FAIL: RGB2BGR color_space mismatch\n");
            pass = 0;
        } else {
            iv8u* bd = bgr->data.ptr8u;
            for (row_index = 0; row_index < height && pass; row_index++)
                for (col_index = 0; col_index < width && pass; col_index++) {
                    size_t off = ((size_t)row_index * width + col_index) * 3;
                    if (bd[off + 0] != rgb_data[off + 2] ||
                        bd[off + 1] != rgb_data[off + 1] ||
                        bd[off + 2] != rgb_data[off + 0]) {
                        printf("FAIL: RGB2BGR channel swap mismatch at (%d,%d)\n",
                               row_index, col_index);
                        pass = 0;
                    }
                }
        }
        fiv_release_image(bgr);
    }

    /* --- RGB2GRAY: dst is a 1-channel tensor; Y = 0.299R+0.587G+0.114B. --- */
    fiv_mat* gray = (fiv_mat*)fiv_create_tensor2d(shape, FIV_8U1);
    if (gray == NULL) { printf("FAIL: create gray dst\n"); pass = 0; }
    else {
        if (fiv_image_color_space_convertor(gray, src, FIV_CS_RGB2GRAY) != FIV_RET_OK) {
            printf("FAIL: RGB2GRAY returned error\n");
            pass = 0;
        } else if (gray->color_space_type != FIV_GRAY8_CS || gray->dtype != FIV_8U1) {
            printf("FAIL: RGB2GRAY color_space/dtype mismatch\n");
            pass = 0;
        } else {
            iv8u* gd = gray->data.ptr8u;
            for (row_index = 0; row_index < height && pass; row_index++)
                for (col_index = 0; col_index < width && pass; col_index++) {
                    size_t  off  = ((size_t)row_index * width + col_index) * 3;
                    int     yexp = (77  * rgb_data[off + 0] +
                                   150 * rgb_data[off + 1] +
                                   29  * rgb_data[off + 2]) >> 8;
                    iv8u    ygot = gd[(size_t)row_index * width + col_index];
                    if ((int)ygot != yexp) {
                        printf("FAIL: RGB2GRAY value mismatch at (%d,%d): got %d exp %d\n",
                               row_index, col_index, (int)ygot, yexp);
                        pass = 0;
                    }
                }
        }
        fiv_release_image(gray);
    }

    /* --- BGR2GRAY: same math, source channels are B,G,R. --- */
    fiv_mat* bgr_src = (fiv_mat*)fiv_create_tensor2d(shape, FIV_8U3);
    if (bgr_src == NULL) { printf("FAIL: create bgr_src\n"); pass = 0; }
    else {
        iv8u* bs = bgr_src->data.ptr8u;
        for (row_index = 0; row_index < height; row_index++)
            for (col_index = 0; col_index < width; col_index++) {
                size_t off = ((size_t)row_index * width + col_index) * 3;
                bs[off + 0] = rgb_data[off + 2];  /* B */
                bs[off + 1] = rgb_data[off + 1];  /* G */
                bs[off + 2] = rgb_data[off + 0];  /* R */
            }
        bgr_src->color_space_type = FIV_BGR24_CS;
        fiv_mat* gray2 = (fiv_mat*)fiv_create_tensor2d(shape, FIV_8U1);
        if (gray2 == NULL) { printf("FAIL: create gray2 dst\n"); pass = 0; }
        else {
            if (fiv_image_color_space_convertor(gray2, bgr_src, FIV_CS_BGR2GRAY) != FIV_RET_OK) {
                printf("FAIL: BGR2GRAY returned error\n");
                pass = 0;
            } else if (gray2->color_space_type != FIV_GRAY8_CS) {
                printf("FAIL: BGR2GRAY color_space mismatch\n");
                pass = 0;
            } else {
                iv8u* g2 = gray2->data.ptr8u;
                for (row_index = 0; row_index < height && pass; row_index++)
                    for (col_index = 0; col_index < width && pass; col_index++) {
                        size_t  off  = ((size_t)row_index * width + col_index) * 3;
                        /* BGR source: index 2 = R, index 1 = G, index 0 = B. */
                        int     yexp = (77  * rgb_data[off + 0] +
                                       150 * rgb_data[off + 1] +
                                       29  * rgb_data[off + 2]) >> 8;
                        iv8u    ygot = g2[(size_t)row_index * width + col_index];
                        if ((int)ygot != yexp) {
                            printf("FAIL: BGR2GRAY value mismatch at (%d,%d): got %d exp %d\n",
                                   row_index, col_index, (int)ygot, yexp);
                            pass = 0;
                        }
                    }
            }
            fiv_release_image(gray2);
        }
        fiv_release_image(bgr_src);
    }

    /* --- negative path: wrong dst dtype for a swap must be rejected. --- */
    fiv_mat* bad = (fiv_mat*)fiv_create_tensor2d(shape, FIV_8U1);
    if (bad != NULL) {
        if (fiv_image_color_space_convertor(bad, src, FIV_CS_RGB2BGR) != FIV_RET_ERR_PARA) {
            printf("FAIL: RGB2BGR with 1ch dst should return ERR_PARA\n");
            pass = 0;
        }
        fiv_release_image(bad);
    }

    fiv_release_image(src);
    fiv_release_image(rgb);
    remove(tmp_path);
    return pass;
}


int main(int argc, char** argv) {
    int pass = 1;

    pass &= test_load_error_paths();
    pass &= test_write_roundtrip();
    pass &= test_color_space_convert();

    /* Optional happy path: ./test_image_io <some_image> */
    if (argc > 1) {
        fiv_mat* image = fiv_create_image_from_file(argv[1], FIV_RGB24_CS);
        if (image == NULL) {
            printf("FAIL: could not load %s\n", argv[1]);
            pass = 0;
        } else {
            printf("loaded %s: %zux%zu dtype=%d color_space=%d\n",
                   argv[1], image->height, image->width,
                   image->dtype, (int)image->color_space_type);
            fiv_release_image(image);
        }
    }

    printf("test_image_io: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
