#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"
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
	l.loss = alignedCalloc(inputs*batch, sizeof(float));
    l.output = alignedCalloc(inputs*batch, sizeof(float));
    l.delta = alignedCalloc(inputs*batch, sizeof(float));
	l.cost = alignedCalloc(1, sizeof(float));

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;
    #ifdef GPU
    l.forward_gpu = forward_softmax_layer_gpu;
    l.backward_gpu = backward_softmax_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
    #endif

#ifdef OCL
    l.forward_ocl = forward_softmax_layer_ocl;
    //l.backward_gpu = backward_softmax_layer_gpu;

    l.output_ocl = cl_make_array(context, l.output, inputs * batch * sizeof(float));
	l.loss_ocl = cl_make_array(context, l.loss, inputs*batch* sizeof(float));
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

#ifdef GPU

void pull_softmax_layer_output(const softmax_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
}

void forward_softmax_layer_gpu(const softmax_layer l, network_state state)
{
    int inputs = l.inputs / l.groups;
    int batch = l.batch * l.groups;
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_gpu(state.input+count, group_size, inputs, batch, l.temperature, l.output_gpu + count);
            count += group_size;
        }
    } else {
        softmax_gpu(state.input, inputs, inputs, batch, l.temperature, l.output_gpu);
    }
}

void backward_softmax_layer_gpu(const softmax_layer layer, network_state state)
{
    axpy_ongpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, state.delta, 1);
}

#endif

#ifdef OCL
void forward_softmax_layer_ocl(const softmax_layer l, network_state state)
{
	//printf("l.softmax_tree?=NULL:%d, state.truth?NULL: %d, l.noloss %d\n", l.softmax_tree == NULL, state.truth == NULL, l.noloss);
	cl_int errNum;
	int inputs = l.inputs / l.groups;
	int batch = l.batch * l.groups;
	//cl_program program = Program[SOFTMAX_PROGRAM];
	// --------------- 使用CPU计算softmax-----------------
	state.input = (float*)malloc(l.batch * l.inputs * sizeof(float));
	errNum = clEnqueueReadBuffer(commandQueue, state.input_ocl, CL_TRUE, 0, l.batch * l.inputs * sizeof(float), state.input, 0, NULL, NULL);
	if (!check_errNum(errNum))
	{
		fprintf(stderr, "forward_softmax_layer_ocl中读取buffer执行失败, 退出程序\n");
		exit(1);
	}
	if (l.softmax_tree) {   // l.softmax_tree:NULL
		softmax_tree(state.input, batch, inputs, l.temperature, l.softmax_tree, l.output);
	}
	else {
		for (int b = 0; b < batch; ++b) {
			softmax(state.input + b*inputs, inputs, l.temperature, l.output + b*inputs);
			//softmax_cpu(state.input, l.inputs / l.groups, l.batch, l.inputs, l.groups, l.inputs / l.groups, 1, l.temperature, l.output);
		}
		
	}
	//printf("##############%d  %d   ***********************\n", state.truth==NULL, l.noloss);
	//for (int i = 0; i < 20; i++)
		//printf("i:%d, l.output[%d]:%f\n", i, i, l.output[i]);
	if (state.truth && !l.noloss) {   //state.truth:NULL, l.noloss:0
		//printf("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n");
		softmax_x_ent_cpu(l.batch*l.inputs, l.output, state.truth, l.delta, l.loss);
		l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
	}
	//printf("forward_softmax_layer_ocl completed\n");
	
	//---------------------------------------------------

	/*
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "softmax_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &inputs);
	errNum |= clSetKernelArg(kernel, 1, sizeof(int), &inputs);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &batch);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &state.input_ocl);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &l.temperature);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &l.output_ocl);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		fprintf(stderr, "forward_soft_max_layer_ocl 设置内核参数失败, 退出程序\n");
		exit(1);
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(batch, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	if(!check_errNum(errNum))
	{
		fprintf(stderr, "forward_soft_max_layer_ocl 执行内核失败, 退出程序\n");
		exit(1);
	}
	*/
}

//cl_int softmax_ocl(cl_command_queue command, cl_mem input, int n, int offset, int groups, float temp, cl_mem output)
//{
//	cl_int errNum;
//    int inputs = n;
//    int batch = groups;
//	//cl_program program = Program[SOFTMAX_PROGRAM];
//	cl_program program = Program;
//	cl_kernel kernel = clCreateKernel(program, "softmax_kernel", NULL);
//	errNum = clSetKernelArg(kernel, 0, sizeof(int), &inputs);
//	errNum |= clSetKernelArg(kernel, 1, sizeof(int), &offset);
//	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &batch);
//	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &input);
//	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &temp);
//	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &output);
//	if(!check_errNum(errNum))
//	{
//		clReleaseKernel(kernel);
//		return errNum;
//	}
//	size_t global_work_size1d[1];
//	get_global_work_size1d(batch, global_work_size1d);
//	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
//	clReleaseKernel(kernel);
//	return errNum;
//}

//__kernel void softmax_offset_kernel(int n, int offset, int batch, __global float *input, int input_offset, float temp, __global float *output, int output_offset)
//{
//	int b = get_global_id(0) + get_global_id(1) * get_global_size(0);
//	if (b < batch)
//		softmax_device(n, input + b*offset + input_offset, temp, output + b*offset + output_offset);
//}


cl_int softmax_offset_ocl(cl_command_queue command, cl_mem input, int input_offset, int n, int offset, int groups, float temp, cl_mem output, int output_offset)
{
	cl_int errNum;
    int inputs = n;
    int batch = groups;
	//cl_program program = Program[SOFTMAX_PROGRAM];
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


//cl_int softmax_offset_ocl(cl_command_queue command, cl_mem input, int input_offset, int n, int offset, int groups, float temp, cl_mem output, int output_offset)
//{
//	cl_int errNum;
//	int inputs = n;
//	int batch = groups;
//	//cl_program program = Program[SOFTMAX_PROGRAM];
//	cl_program program = Program;
//	cl_kernel kernel = clCreateKernel(program, "softmax_offset_kernel", NULL);
//	errNum = clSetKernelArg(kernel, 0, sizeof(int), &inputs);
//	errNum |= clSetKernelArg(kernel, 1, sizeof(int), &offset);
//	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &batch);
//	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &input);
//	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &input_offset);
//	errNum |= clSetKernelArg(kernel, 5, sizeof(float), &temp);
//	errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &output);
//	errNum |= clSetKernelArg(kernel, 7, sizeof(int), &output_offset);
//	if (!check_errNum(errNum))
//	{
//		clReleaseKernel(kernel);
//		return errNum;
//	}
//	size_t global_work_size1d[1];
//	get_global_work_size1d(batch, global_work_size1d);
//	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
//	clReleaseKernel(kernel);
//	return errNum;
//}


#endif
