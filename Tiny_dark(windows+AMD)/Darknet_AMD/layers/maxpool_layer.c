#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}

image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = {0};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + 2*padding)/stride;
    l.out_h = (h + 2*padding)/stride;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = alignedCalloc(output_size, sizeof(int));
    l.output =  alignedCalloc(output_size, sizeof(float));
    l.delta =   alignedCalloc(output_size, sizeof(float));
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    #ifdef GPU
    l.forward_gpu = forward_maxpool_layer_gpu;
    l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif

#ifdef OCL
    l.forward_ocl = forward_maxpool_layer_ocl;
    l.backward_ocl = backward_maxpool_layer_ocl;
    l.indexes_ocl = cl_make_array(context, NULL, output_size * sizeof(int));
    l.output_ocl = cl_make_array(context, l.output, output_size * sizeof(int));
    l.delta_ocl = cl_make_array(context, l.delta, output_size * sizeof(int));
#endif
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + 2*l->pad)/l->stride;
    l->out_h = (h + 2*l->pad)/l->stride;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
}

void forward_maxpool_layer(const maxpool_layer l, network_state state)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad;
    int h_offset = -l.pad;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? state.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
}

void backward_maxpool_layer(const maxpool_layer l, network_state state)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        state.delta[index] += l.delta[i];
    }
}

#ifdef OCL
void forward_maxpool_layer_ocl(maxpool_layer layer, network_state state)
{
	printf("forward_maxpool_layer_ocl*****************\n");
	cl_int errNum;
	static double total_time = 0.0;
    int h = layer.out_h;
    int w = layer.out_w;
    int c = layer.c;
    size_t n = h * w * c * layer.batch;
	//cl_program program = Program[MAXPOOL_PROGRAM];
	cl_program program = Program;
    cl_kernel kernel = clCreateKernel(program, "forward_maxpool_layer_kernel", NULL);
    errNum = clSetKernelArg(kernel, 0, sizeof(int), &n);
    errNum |= clSetKernelArg(kernel, 1, sizeof(int), &layer.h);
    errNum |= clSetKernelArg(kernel, 2, sizeof(int), &layer.w);
    errNum |= clSetKernelArg(kernel, 3, sizeof(int), &layer.c);
    errNum |= clSetKernelArg(kernel, 4, sizeof(int), &layer.stride);
    errNum |= clSetKernelArg(kernel, 5, sizeof(int), &layer.size);
    errNum |= clSetKernelArg(kernel, 6, sizeof(int), &layer.pad);
    errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &state.input_ocl);
    errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &layer.output_ocl);
    errNum |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &layer.indexes_ocl);
    if(!check_errNum(errNum))
    {
    	fprintf(stderr, "forward_layer_ocl 设置内核参数失败, 退出程序\n");
    	clReleaseKernel(kernel);
    	exit(1);
    }
    size_t global_work_size1d[1];
    get_global_work_size1d(n, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
    if(!check_errNum(errNum))
    {
    	fprintf(stderr, "forward_layer_ocl 执行内核失败, 退出程序\n");
    	exit(1);
    }

	//float *result = (float*)alignedMalloc(10 * sizeof(float));   //l.outputs*l.batch
	//errNum = clEnqueueReadBuffer(commandQueue, layer.output_ocl, CL_TRUE, 0, 10 * sizeof(float), result, 0, NULL, NULL);
	//printf("###########################################################\n");
	//for (int q = 0; q < 10; q++)
	//{
	//	printf(" maxool layer.output_ocl[%d]:%f\n", q, result[q]);
	//}
	//alignedFree(result);

    //printf("forward_convolutional_layer_ocl 用时：%f\n", total_time);
}
void backward_maxpool_layer_ocl(maxpool_layer layer, network_state state)
{
	cl_int errNum;
	size_t n = layer.h * layer.w * layer.c * layer.batch;
	//cl_program program = Program[MAXPOOL_PROGRAM];
	cl_program program = Program;
    cl_kernel kernel = clCreateKernel(program, "backward_maxpool_layer_kernel", NULL);
    errNum = clSetKernelArg(kernel, 0, sizeof(size_t), &n);
    errNum |= clSetKernelArg(kernel, 1, sizeof(int), &layer.h);
    errNum |= clSetKernelArg(kernel, 2, sizeof(int), &layer.w);
    errNum |= clSetKernelArg(kernel, 3, sizeof(int), &layer.c);
    errNum |= clSetKernelArg(kernel, 4, sizeof(int), &layer.stride);
    errNum |= clSetKernelArg(kernel, 5, sizeof(int), &layer.size);
    errNum |= clSetKernelArg(kernel, 6, sizeof(int), &layer.pad);
    errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &layer.delta_ocl);
    errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &state.delta_ocl);
    errNum |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &layer.indexes_ocl);
    if(!check_errNum(errNum))
    {
    	fprintf(stderr, "backward_maxpool_layer_ocl 设置内核参数失败, 退出程序\n");
    	clReleaseKernel(kernel);
    	exit(1);
    }
    size_t global_work_size1d[1];
    get_global_work_size1d(n, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
    if(!check_errNum(errNum))
    {
    	fprintf(stderr, "backward_maxpool_layer_ocl 执行内核失败, 退出程序\n");
    	exit(1);
    }
}
#endif
