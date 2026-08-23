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

#ifndef _FIV_IMAGE_H_
#define _FIV_IMAGE_H_

#include "fiv_ctensor.h"

#ifdef __cplusplus
extern "C" {
#endif


/* ============================ Color spaces ============================ */
/* iv8u-typed enum of the common interleaved color spaces an image tensor can
   carry. The value is stored in fiv_tensor2d.color_space_type so downstream
   operators know how to interpret the channels. Names encode layout + bit depth
   + channel count, e.g. FIV_BGR24_CS = contiguous B,G,R, 8 bits each (24 bpp).

   Value 0 is reserved as the explicit FIV_UNDEF_CS sentinel ("unset / unknown"),
   matching the color_space_type zero-init done by fiv_create_tensor2d and
   fiv_create_tensor2d_header. All real color spaces therefore start at 1
   (FIV_GRAY8_CS = 1); never assign another color space the value 0. */
typedef enum : iv8u {
    FIV_UNDEF_CS = 0,       /* reserved "unset/unknown" sentinel, matches
                               create_tensor2d zero-init of color_space_type */
    FIV_GRAY8_CS,           /* single 8-bit channel,   dtype FIV_8U1 */
    FIV_RGB24_CS,           /* contiguous R,G,B,        dtype FIV_8U3 */
    FIV_BGR24_CS,           /* contiguous B,G,R,        dtype FIV_8U3 */
    FIV_RGBA32_CS,          /* contiguous R,G,B,A,      dtype FIV_8U4 */
    FIV_BGRA32_CS,          /* contiguous B,G,R,A,      dtype FIV_8U4 */
    FIV_COLOR_SPACE_NUM
} fiv_color_space;


/* Load an image file (PNG/JPG/BMP/TGA/GIF/...) into a 2D tensor of shape
   [height, width] with HWC-interleaved pixels.

   color_flag selects the output color space (fiv_color_space); the tensor's
   dtype is set to match (FIV_8U1 / FIV_8U3 / FIV_8U4) and its color_space_type
   meta field is filled with the same value. An unrecognized color_flag falls
   back to FIV_RGB24_CS.

   The decoded pixel buffer is OWNED by the returned tensor (zero-copy: stb's
   allocator is redirected to fiv_malloc), so release it with fiv_release_image.
   No channel reordering is performed: stb decodes in its native RGB(A) order
   and the bytes are stored verbatim, tagged with the requested color_space_type.
   Converting RGB<->BGR etc. is an explicit, separate transform the caller must
   invoke -- it is never done implicitly during load.

   Returns NULL on error: NULL/empty file_name, unsupported format, or
   allocation failure. */
fiv_mat* fiv_create_image_from_file(char* file_name, int color_flag);


#define fiv_release_image(image)  fiv_release_tensor((void**)&(image))



fiv_ret fiv_image_write(char* file_name, fiv_mat* image);



typedef enum : iv32u{

  FIV_CS_BGR2RGB,
  FIV_CS_RGB2BGR,
  FIV_CS_RGB2GRAY,
  FIV_CS_BGR2GRAY,

}fiv_cs_convertor_type;




fiv_ret fiv_image_color_space_convertor(fiv_mat* image_dst, fiv_mat* image_src, fiv_cs_convertor_type type);


/* Element-type conversion for a tensor's data. Orthogonal to color-space
   conversion: this changes the numeric dtype (e.g. 8U -> 32F, 32S -> 32F)
   without touching channel layout. When the source and destination element
   sizes match (e.g. 32S<->32F, both 4 bytes) the conversion may run in-place
   (dst == src); when they differ (e.g. 8U -> 32F, 1 -> 4 bytes) a separate
   dst buffer is required and dst must not alias src. */
typedef enum : iv32u{

  FIV_8U_TO_32F,
  FIV_8U_TO_32F_NORM01,   // div 255, [0, 1]
  FIV_8U_TO_32F_NORM_N1_P1,  // [-1, 1]
  FIV_8U_TO_32F_NORM_MU_SIGMA, // sub mu div sigma
  FIV_32S_TO_32F,

}fiv_data_convertor_type;



fiv_ret fiv_tensor_data_convert(void* image_dst, void* image_src, fiv_data_convertor_type type);




#ifdef __cplusplus
}
#endif

#endif  /* _FIV_IMAGE_H_ */
