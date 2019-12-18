#ifndef IM2COL_H
#define IM2COL_H

void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

#ifdef GPU

void im2col_ongpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);

#endif

#ifdef OCL
#include"oclutils.h"
extern cl_int im2col_on_ocl(cl_command_queue command, cl_mem im, int offset, int channels, int height, int width, int ksize, int stride, int pad, cl_mem data_col);
#endif
#endif
