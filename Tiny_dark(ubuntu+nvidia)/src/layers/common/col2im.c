#include <stdio.h>
#include <math.h>
#include "col2im.h"
void col2im_add_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
    im[col + width*(row + height*channel)] += val;
}
void col2im_cpu(float* data_col,
         int channels,  int height,  int width,
         int ksize,  int stride, int pad, float* data_im) 
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
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad, val);
            }
        }
    }
}
#ifdef OCL
cl_int col2im_on_ocl(cl_command_queue command, cl_mem data_col, int channels, int height, int width, int ksize, int stride, int pad, cl_mem data_im, size_t data_im_offset)
{
	cl_int errNum;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height * width;
	//cl_program program = Program[COL2IM_PROGRAM];
	cl_program program = Program;
    cl_kernel  kernel = clCreateKernel(program, "col2im_kernel", NULL);
    errNum = clSetKernelArg(kernel, 0, sizeof(int), &num_kernels);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &data_col);
    errNum |= clSetKernelArg(kernel, 2, sizeof(int), &height);
    errNum |= clSetKernelArg(kernel, 3, sizeof(int), &width);
    errNum |= clSetKernelArg(kernel, 4, sizeof(int), &ksize);
    errNum |= clSetKernelArg(kernel, 5, sizeof(int), &pad);
    errNum |= clSetKernelArg(kernel, 6, sizeof(int), &stride);
    errNum |= clSetKernelArg(kernel, 7, sizeof(int), &height_col);
    errNum |= clSetKernelArg(kernel, 8, sizeof(int), &width_col);
    errNum |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &data_im);
    errNum |= clSetKernelArg(kernel, 10, sizeof(size_t), &data_im_offset);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
    size_t global_work_size1d[1];
    get_global_work_size1d(num_kernels, global_work_size1d);
    errNum = clEnqueueNDRangeKernel(command, kernel, 1, NULL, global_work_size1d, local_work_size1d, 0, NULL, NULL);
    clReleaseKernel(kernel);
    return errNum;
}
#endif
