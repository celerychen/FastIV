#ifndef FIV_FACE_WARP_H
#define FIV_FACE_WARP_H

#include "fiv_data_typedefs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* compute the 3x3 homography mapping the centered square ROI of an img_w x
 * img_h image onto a tensor_size x tensor_size output. */
void fiv_detection_warp_matrix(int img_w, int img_h, int tensor_size, ivf32 homography[9]);

/* fixed-point bilinear warp (OpenCV INTER_LINEAR / BORDER_CONSTANT) of an
 * h x w x cn uint8 image into a out_h x out_w buffer, using the forward
 * homography fwd_matrix (as produced by fiv_detection_warp_matrix). */
void fiv_warp_perspective_u8(const iv8u* img, int h, int w, int cn,
                             const ivf32 fwd_matrix[9], int out_w, int out_h,
                             iv8u* out, int border_replicate);

#ifdef __cplusplus
}
#endif

#endif /* FIV_FACE_WARP_H */
