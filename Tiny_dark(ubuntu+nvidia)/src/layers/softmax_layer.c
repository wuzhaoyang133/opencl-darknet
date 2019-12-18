#include "softmax_layer.h"
#include "blas.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n",  inputs);
    softmax_layer l = {0};
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = alignedCalloc(inputs*batch, sizeof(float));
    l.delta = alignedCalloc(inputs*batch, sizeof(float));

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;

#ifdef OCL
    l.forward_ocl = forward_softmax_layer_ocl;
    l.output_ocl = cl_make_array(context, l.output, inputs * batch * sizeof(float));
    l.delta_ocl = cl_make_array(context, l.delta, inputs * batch * sizeof(float));
#endif
    return l;
}

void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output)
{
    int b;
    for(b = 0; b < batch; ++b){
        int i;
        int count = 0;
        for(i = 0; i < hierarchy->groups; ++i){
            int group_size = hierarchy->group_size[i];
            softmax(input+b*inputs + count, group_size, temp, output+b*inputs + count);
            count += group_size;
        }
    }
}

void forward_softmax_layer(const softmax_layer l, network_state state)
{
    int b;
    int inputs = l.inputs / l.groups;
    int batch = l.batch * l.groups;
    if(l.softmax_tree){
        softmax_tree(state.input, batch, inputs, l.temperature, l.softmax_tree, l.output);
    } else {
        for(b = 0; b < batch; ++b){
            softmax(state.input+b*inputs, inputs, l.temperature, l.output+b*inputs);
        }
    }
}

void backward_softmax_layer(const softmax_layer l, network_state state)
{
    int i;
    for(i = 0; i < l.inputs*l.batch; ++i){
        state.delta[i] += l.delta[i];
    }
}

#ifdef OCL
void forward_softmax_layer_ocl(softmax_layer l, network_state state)
{
	cl_int errNum;
        int inputs = l.inputs / l.groups;
        int batch = l.batch * l.groups;
	state.input = (float*)alignedMalloc(l.batch * l.inputs * sizeof(float));
	errNum = clEnqueueReadBuffer(commandQueue, state.input_ocl, CL_TRUE, 0, l.batch * l.inputs * sizeof(float), state.input, 0, NULL, NULL);
	if (!check_errNum(errNum))
	{
		fprintf(stderr, "forward_softmax_layer_ocl中读取buffer执行失败, 退出程序\n");
		exit(1);
	}
	if (l.softmax_tree) {
		softmax_tree(state.input, batch, inputs, l.temperature, l.softmax_tree, l.output);
	}
	else {
		for (int b = 0; b < batch; ++b) {
			softmax(state.input + b*inputs, inputs, l.temperature, l.output + b*inputs);
		}
	}
}

cl_int softmax_offset_ocl(cl_command_queue command, cl_mem input, int input_offset, int n, int offset, int groups, float temp, cl_mem output, int output_offset)
{
	cl_int errNum;
        int inputs = n;
        int batch = groups;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "softmax_offset_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &inputs);
	errNum |= clSetKernelArg(kernel, 1, sizeof(int), &offset);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &batch);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &input);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &input_offset);
	errNum |= clSetKernelArg(kernel, 5, sizeof(float), &temp);
	errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &output);
	errNum |= clSetKernelArg(kernel, 7, sizeof(int), &output_offset);

	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(batch, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	return errNum;
}

cl_int softmax_ocl(cl_command_queue command, cl_mem input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem output, int offset)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "softmax_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	errNum |= clSetKernelArg(kernel, 1, sizeof(int), &n);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &batch);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &batch_offset);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &groups);
	errNum |= clSetKernelArg(kernel, 5, sizeof(int), &group_offset);
	errNum |= clSetKernelArg(kernel, 6, sizeof(int), &stride);
	errNum |= clSetKernelArg(kernel, 7, sizeof(float), &temp);
	errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &output);
	errNum |= clSetKernelArg(kernel, 9, sizeof(int), &offset);

	if (!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(batch, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	return errNum;
}
#endif
