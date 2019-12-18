#include "im2col.h"
#include <stdio.h>
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

#ifdef OCL
cl_int im2col_on_ocl(cl_command_queue command, cl_mem im, int offset, int channels, int height, int width, int ksize, int stride, int pad, cl_mem data_col)
{
    cl_int errNum;
    int out_h = (height + 2 * pad - ksize) / stride + 1;
    int out_w  = (width + 2 * pad - ksize) / stride + 1;
    int n = out_h * out_w * channels;
    size_t global_work_size1d[1];
    cl_program program = Program;
    cl_kernel  kernel = clCreateKernel(program, "im2col_kernel", NULL);
    errNum = clSetKernelArg(kernel, 0, sizeof(int), &n);

    errNum |= clSetKernelArg(kernel, 1, sizeof(int), &offset);

    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &im);

    errNum |= clSetKernelArg(kernel, 3, sizeof(int), &height);

    errNum |= clSetKernelArg(kernel, 4, sizeof(int), &width);

    errNum |= clSetKernelArg(kernel, 5, sizeof(int), &ksize);

    errNum |= clSetKernelArg(kernel, 6, sizeof(int), &pad);

    errNum |= clSetKernelArg(kernel, 7, sizeof(int), &stride);

    errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &data_col);
    if(!check_errNum(errNum)){
	   clReleaseKernel(kernel);
	   return errNum;
    }
    get_global_work_size1d(n, global_work_size1d);

    errNum = clEnqueueNDRangeKernel(command, kernel, 1, NULL, global_work_size1d, local_work_size1d, 0, NULL, NULL);

    clReleaseKernel(kernel);

    return errNum;
}
#endif
