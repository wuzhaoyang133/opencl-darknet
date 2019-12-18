#include "avgpool_layer.h"
#include <stdio.h>

avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    avgpool_layer l = {0};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  alignedCalloc(output_size, sizeof(float));
    l.delta =   alignedCalloc(output_size, sizeof(float));
    l.forward = forward_avgpool_layer;
    l.backward = backward_avgpool_layer;
#ifdef OCL
    l.forward_ocl = forward_avgpool_layer_ocl;
    l.output_ocl  = cl_make_array(context, l.output, output_size * sizeof(float));
    l.delta_ocl   = cl_make_array(context, l.delta, output_size * sizeof(float));
#endif
    return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}

void forward_avgpool_layer(const avgpool_layer l, network_state state)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += state.input[in_index];
            }
            l.output[out_index] /= l.h*l.w;
        }
    }
}

void backward_avgpool_layer(const avgpool_layer l, network_state state)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                state.delta[in_index] += l.delta[out_index] / (l.h*l.w);
            }
        }
    }
}

#ifdef OCL
void forward_avgpool_layer_ocl(avgpool_layer l, network_state state)
{
	cl_int errNum;
	int n = l.c * l.batch;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "forward_avgpool_layer_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &n);
	errNum |= clSetKernelArg(kernel, 1, sizeof(int), &l.w);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &l.h);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &l.c);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &state.input_ocl);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &l.output_ocl);
	if(!check_errNum(errNum))
	{
		fprintf(stderr, "forward_avgpool_layer_ocl 设置内核参数失败\n");
		clReleaseKernel(kernel);
		exit(1);
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(n, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	if(!check_errNum(errNum))
	{
		fprintf(stderr, "forward_avgpool_layer_ocl 执行内核失败\n");
		exit(1);
	}
}
#endif
