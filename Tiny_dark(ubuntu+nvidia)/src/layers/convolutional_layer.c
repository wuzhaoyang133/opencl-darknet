#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>
#ifdef AI2
#include "xnor_layer.h"
#endif
#ifndef AI2
#define AI2 0
void forward_xnor_layer(layer l, network_state state);
#endif
void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l)
{
    int h,w,c;
    h = convolutional_out_height(l);
    w = convolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    int h,w,c;
    h = convolutional_out_height(l);
    w = convolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.delta);
}

size_t get_workspace_size(layer l){
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);
}

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;
    
    l.weights = alignedCalloc(c*n*size*size, sizeof(float));
    l.weight_updates = alignedCalloc(c*n*size*size, sizeof(float));
    l.biases = alignedCalloc(n, sizeof(float));
    l.bias_updates = alignedCalloc(n, sizeof(float));
    float scale = sqrt(2./(size*size*c));
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = alignedCalloc(l.batch*l.outputs, sizeof(float));
    l.delta  = alignedCalloc(l.batch*l.outputs, sizeof(float));


    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = alignedCalloc(c*n*size*size, sizeof(float));
        l.cweights = alignedCalloc(c*n*size*size, sizeof(char));
        l.scales = alignedCalloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = alignedCalloc(c*n*size*size, sizeof(float));
        l.binary_input = alignedCalloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = alignedCalloc(n, sizeof(float));
        l.scale_updates = alignedCalloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = alignedCalloc(n, sizeof(float));
        l.variance = alignedCalloc(n, sizeof(float));

        l.mean_delta = alignedCalloc(n, sizeof(float));
        l.variance_delta = alignedCalloc(n, sizeof(float));

        l.rolling_mean = alignedCalloc(n, sizeof(float));
        l.rolling_variance = alignedCalloc(n, sizeof(float));
        l.x = alignedCalloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = alignedCalloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.adam = 1;
        l.m = alignedCalloc(c*n*size*size, sizeof(float));
        l.v = alignedCalloc(c*n*size*size, sizeof(float));
    }
#ifdef OCL
    l.forward_ocl = forward_convolutional_layer_ocl;
    l.backward_ocl = backward_convolutional_layer_ocl;
    l.update_ocl = update_convolutional_layer_ocl;

    if(ocl_index >= 0)
    {
        l.weights_ocl = cl_make_array(context,l.weights, c*n*size*size * sizeof(float));
        l.weight_updates_ocl = cl_make_array(context, l.weight_updates, c*n*size*size * sizeof(float));

        l.biases_ocl = cl_make_array(context, l.biases, n * sizeof(float));
        l.bias_updates_ocl = cl_make_array(context, l.bias_updates, n * sizeof(float));

        l.delta_ocl = cl_make_array(context, l.delta, l.batch*out_h*out_w*n * sizeof(float));
        l.output_ocl = cl_make_array(context, l.output, l.batch*out_h*out_w*n * sizeof(float));
        if(batch_normalize)
        {
            l.mean_ocl = cl_make_array(context, l.mean, n * sizeof(float));
            l.variance_ocl = cl_make_array(context, l.variance, n * sizeof(float));

            l.rolling_mean_ocl = cl_make_array(context, l.mean, n * sizeof(float));
            l.rolling_variance_ocl = cl_make_array(context, l.variance, n * sizeof(float));

            l.mean_delta_ocl = cl_make_array(context, l.mean, n * sizeof(float));
            l.variance_delta_ocl = cl_make_array(context, l.variance, n * sizeof(float));

            l.scales_ocl = cl_make_array(context, l.scales, n * sizeof(float));
            l.scale_updates_ocl = cl_make_array(context, l.scale_updates, n * sizeof(float));

            l.x_ocl = cl_make_array(context, l.output, l.batch*out_h*out_w*n * sizeof(float));
            l.x_norm_ocl = cl_make_array(context, l.output, l.batch*out_h*out_w*n * sizeof(float));
        }
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c*l.size*l.size; ++j){
            l.weights[i*l.c*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    network_state state = {0};
    state.input = data;
    forward_convolutional_layer(l, state);
}

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }
    l->workspace_size = get_workspace_size(*l);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

void forward_convolutional_layer(convolutional_layer l, network_state state)
{
	static double total_time = 0.0;
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    int i;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input);
        state.input = l.binary_input;
    }

    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;


    float *a = l.weights;
    float *b = state.workspace;
    float *c = l.output;

    for(i = 0; i < l.batch; ++i){
        im2col_cpu(state.input, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        c += n*m;
        state.input += l.c*l.h*l.w;
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, state);
    }
    add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
    activate_array(l.output, m*n*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}

void backward_convolutional_layer(convolutional_layer l, network_state state)
{
    int i;
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = convolutional_out_height(l)*
    convolutional_out_width(l);
    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);
    backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    if(l.batch_normalize){
        backward_batchnorm_layer(l, state);
    }
    for(i = 0; i < l.batch; ++i){
        float *a = l.delta + i*m*k;
        float *b = state.workspace;
        float *c = l.weight_updates;
        float *im = state.input+i*l.c*l.h*l.w;
        im2col_cpu(im, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
        if(state.delta){
            a = l.weights;
            b = l.delta + i*m*k;
            c = state.workspace;
            gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);
            col2im_cpu(state.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta+i*l.c*l.h*l.w);
        }
    }
}

void update_convolutional_layer(convolutional_layer l, int batch, float learning_rate, float momentum, float decay)
{
    int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);
    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }
    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = alignedCalloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
    }
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    free_image(dc);
    return single_weights;
}

#ifdef OCL
void forward_convolutional_layer_ocl(convolutional_layer l, network_state state)
{
    static double total_time = 0.0;
    cl_int errNum;
    int i;
    int m = l.n;  
    int n = l.size * l.size * l.c;
    int k = l.out_w * l.out_h;
    for(i = 0; i < l.batch; ++i)  
    {
        errNum = im2col_on_ocl(commandQueue, state.input_ocl , i*l.c*l.h*l.w, l.c,  l.h, l.w, l.size, l.stride, l.pad, state.workspace_ocl);
        if(errNum != CL_SUCCESS)
        {
        	fprintf(stderr, "im2col_on_ocl 执行失败, 退出程序\n");
        	exit(1);
        }
        cl_mem a = l.weights_ocl;
        cl_mem b = state.workspace_ocl;
        cl_mem c = l.output_ocl;
		clFinish(commandQueue);
		time_start();
        errNum = gemm_ocl(commandQueue, 0, 0, m, n, k, 1.0, a, 0, b, 0, 0.0, c,  i * m * k);
        if(errNum != CL_SUCCESS)
        {
        	fprintf(stderr, "gemm_ocl 执行失败, 退出程序\n");
        	exit(1);
        }
		clFinish(commandQueue);
		total_time = time_elapse();
    }

    if (l.batch_normalize) {
        forward_batchnorm_layer_ocl(l, state);
    }
	else 
	{
		errNum = add_bias_ocl(commandQueue, l.output_ocl, l.biases_ocl, l.batch, l.n, l.out_w * l.out_h);
		if (errNum != CL_SUCCESS)
		{
			fprintf(stderr, "add_bias_ocl 执行失败, 退出程序\n");
			exit(1);
		}
	}
    errNum = activate_array_on_ocl(commandQueue, l.output_ocl, l.outputs * l.batch, l.activation);
    if(errNum != CL_SUCCESS)
    {
    	fprintf(stderr, "activate_array_on_ocl执行失败, 退出程序\n");
    	exit(1);
    }

}
void push_convolutional_layer_ocl(convolutional_layer layer)
{
	layer.weights_ocl = cl_push_array(commandQueue, layer.weights, layer.weights_ocl, layer.c*layer.n*layer.size*layer.size * sizeof(float));
	layer.biases_ocl = cl_push_array(commandQueue, layer.biases , layer.biases_ocl,  layer.n * sizeof(float));
	layer.weight_updates_ocl = cl_push_array(commandQueue, layer.weight_updates, layer.weight_updates_ocl, layer.c*layer.n*layer.size*layer.size * sizeof(float));
	layer.bias_updates_ocl = cl_push_array(commandQueue, layer.bias_updates, layer.bias_updates_ocl, layer.n * sizeof(float));
    if (layer.batch_normalize)
    {
    	layer.scales_ocl = cl_push_array(commandQueue, layer.scales, layer.scales_ocl, layer.n * sizeof(float));
    	layer.rolling_mean_ocl = cl_push_array(commandQueue, layer.rolling_mean, layer.rolling_mean_ocl, layer.n * sizeof(float));
    	layer.rolling_variance_ocl = cl_push_array(commandQueue, layer.rolling_variance, layer.rolling_variance_ocl, layer.n * sizeof(float));
    }
    if (layer.adam){
    }
}
void backward_convolutional_layer_ocl(convolutional_layer l, network_state state)
{
    cl_int errNum;
    errNum = gradient_array_on_ocl(commandQueue, l.output_ocl, l.outputs * l.batch, l.activation, l.delta_ocl);
    if(errNum != CL_SUCCESS) {fprintf(stderr, "backward_convolutional_layer_ocl中的gradient_array_on_ocl执行失败,退出程序\n"); exit(1);}
    errNum = backward_bias_ocl(commandQueue, l.bias_updates_ocl, l.delta_ocl,  l.batch, l.n, l.out_w*l.out_h);
    if(errNum != CL_SUCCESS) {fprintf(stderr, "backward_convolutional_layer_ocl中的backward_bias_ocl执行失败,退出程序\n"); exit(1);}
    if(l.batch_normalize)
        backward_batchnorm_layer_ocl(l, state);
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = l.out_w*l.out_h;
    int i;
    for(i = 0; i < l.batch; ++i)
    {
        cl_mem a = l.delta_ocl;
        cl_mem b = state.workspace_ocl;
        cl_mem c = l.weight_updates_ocl;
        errNum = im2col_on_ocl(commandQueue, state.input_ocl , i * l.c * l.h * l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace_ocl);
        if(errNum != CL_SUCCESS) {fprintf(stderr, "backward_convolutional_layer_ocl中的im2col_on_ocl执行失败,退出程序\n"); exit(1);}
        errNum = gemm_ocl(commandQueue, 0, 1, m, k, n, 1.0, a, i*m*k, b, 0, 1.0, c, 0);
        if(errNum != CL_SUCCESS) {fprintf(stderr, "backward_convolutional_layer_ocl中的gemm_ocl执行失败,退出程序\n"); exit(1);}
        if(state.delta_ocl)
        {
            cl_mem a = l.weights_ocl;
            cl_mem b = l.delta_ocl;
            cl_mem c = state.workspace_ocl;
            errNum = gemm_ocl(commandQueue, 1, 0, n, m, k, 1.0, a, 0, b, i * k * m, 0.0 , c, 0);
            if(errNum != CL_SUCCESS) {fprintf(stderr, "backward_convolutional_layer_ocl中的gemm_ocl执行失败,退出程序\n"); exit(1);}
            errNum = col2im_on_ocl(commandQueue, state.workspace_ocl, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta_ocl, i * l.c * l.h * l.w);
            if(errNum != CL_SUCCESS) {fprintf(stderr, "backward_convolutional_layer_ocl中的col2im执行失败,退出程序\n"); exit(1);}
        }
    }
}
void update_convolutional_layer_ocl(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay)
{
    cl_int errNum;
    int size = layer.size * layer.size * layer.c * layer.n;
    errNum = axpy_on_ocl(commandQueue, layer.n, learning_rate/batch, layer.bias_updates_ocl, 1, layer.biases_ocl, 1);
    if(errNum != CL_SUCCESS) {fprintf(stderr, "update_convolutional_layer_ocl中的axpy_on_ocl执行失败,退出程序\n"); exit(1);}
    errNum = scal_on_ocl(commandQueue, layer.n, momentum, layer.bias_updates_ocl, 1);
    if(errNum != CL_SUCCESS) {fprintf(stderr, "update_convolutional_layer_ocl中的scal_on_ocl执行失败,退出程序\n"); exit(1);}
    if(layer.scales_ocl)
    {
        errNum = axpy_on_ocl(commandQueue, layer.n, learning_rate/batch, layer.scale_updates_ocl, 1, layer.scales_ocl, 1);
        if(errNum != CL_SUCCESS) {fprintf(stderr, "update_convolutional_layer_ocl中的axpy_on_ocl执行失败,退出程序\n"); exit(1);}
        errNum = scal_on_ocl(commandQueue, layer.n, momentum, layer.scale_updates_ocl, 1);
        if(errNum != CL_SUCCESS) {fprintf(stderr, "update_convolutional_layer_ocl中的scal_on_ocl执行失败,退出程序\n"); exit(1);}
    }
    errNum = axpy_on_ocl(commandQueue, size, -decay * batch, layer.weights_ocl, 1, layer.weight_updates_ocl, 1);
    if(errNum != CL_SUCCESS) {fprintf(stderr, "update_convolutional_layer_ocl中的axpy_on_ocl执行失败,退出程序\n"); exit(1);}
    errNum = axpy_on_ocl(commandQueue, size, learning_rate/batch, layer.weight_updates_ocl, 1, layer.weights_ocl, 1);
    if(errNum != CL_SUCCESS) {fprintf(stderr, "update_convolutional_layer_ocl中的axpy_on_ocl执行失败,退出程序\n"); exit(1);}
    errNum = scal_on_ocl(commandQueue, size, momentum, layer.weight_updates_ocl, 1);
    if(errNum != CL_SUCCESS) {fprintf(stderr, "update_convolutional_layer_ocl中的scal_on_ocl执行失败,退出程序\n"); exit(1);}
}
#endif

