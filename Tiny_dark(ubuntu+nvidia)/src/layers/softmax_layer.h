#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "layer.h"
#include "network.h"

typedef layer softmax_layer;

void softmax_array(float *input, int n, float temp, float *output);
softmax_layer make_softmax_layer(int batch, int inputs, int groups);
void forward_softmax_layer(const softmax_layer l, network_state state);
void backward_softmax_layer(const softmax_layer l, network_state state);


#ifdef OCL
extern void forward_softmax_layer_ocl(softmax_layer l, network_state state);
//extern cl_int softmax_ocl(cl_command_queue command, cl_mem input, int n, int offset, int groups, float temp, cl_mem output);
//extern cl_int softmax_offset_ocl(cl_command_queue command, cl_mem input, int input_offset, int n, int offset, int groups, float temp, cl_mem output);
extern cl_int softmax_offset_ocl(cl_command_queue command, cl_mem input, int input_offset, int n, int offset, int groups, float temp, cl_mem output, int output_offset);
extern cl_int softmax_ocl(cl_command_queue command, cl_mem input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem output, int offset);
#endif

#endif
