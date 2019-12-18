#ifndef COL2IM_H
#define COL2IM_H

void col2im_cpu(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im);


#ifdef OCL
#include"oclutils.h"
extern cl_int col2im_on_ocl(cl_command_queue command, cl_mem data_col, int channels, int height, int width, int ksize, int stride, int pad, cl_mem data_im, size_t data_im_offset);
#endif


#endif
