#include "batchnorm_layer.h"
#include "blas.h"
#include <stdio.h>

layer make_batchnorm_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w,h,c);
    layer layer = {0};
    layer.type = BATCHNORM;
    layer.batch = batch;
    layer.h = layer.out_h = h;
    layer.w = layer.out_w = w;
    layer.c = layer.out_c = c;
    layer.output = alignedCalloc(h * w * c * batch, sizeof(float));
    layer.delta  = alignedCalloc(h * w * c * batch, sizeof(float));
    layer.inputs = w*h*c;
    layer.outputs = layer.inputs;

    layer.scales = alignedCalloc(c, sizeof(float));
    layer.scale_updates = alignedCalloc(c, sizeof(float));
    int i;
    for(i = 0; i < c; ++i){
        layer.scales[i] = 1;
    }

    layer.mean = alignedCalloc(c, sizeof(float));
    layer.variance = alignedCalloc(c, sizeof(float));

    layer.rolling_mean = alignedCalloc(c, sizeof(float));
    layer.rolling_variance = alignedCalloc(c, sizeof(float));

    layer.forward = forward_batchnorm_layer;
    layer.backward = backward_batchnorm_layer;
    return layer;
}

void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    int i,b,f;
    for(f = 0; f < n; ++f){
        float sum = 0;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
                int index = i + size*(f + n*b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}

void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j*filters*spatial + i*spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int f, j, k;
    for(j = 0; j < batch; ++j){
        for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(sqrt(variance[f]) + .00001f) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}

void resize_batchnorm_layer(layer *layer, int w, int h)
{
    fprintf(stderr, "Not implemented\n");
}

void forward_batchnorm_layer(layer l, network_state state)
{
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    if(l.type == CONNECTED){
        l.out_c = l.outputs;
        l.out_h = l.out_w = 1;
    }
    if(state.train){
        mean_cpu(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
        variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);

        scal_cpu(l.out_c, .9, l.rolling_mean, 1);
        axpy_cpu(l.out_c, .1, l.mean, 1, l.rolling_mean, 1);
        scal_cpu(l.out_c, .9, l.rolling_variance, 1);
        axpy_cpu(l.out_c, .1, l.variance, 1, l.rolling_variance, 1);

        copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
        normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);   
        copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
    } else {
        normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    }
    scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
}

void backward_batchnorm_layer(const layer l, network_state state)
{
    backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates);

    scale_bias(l.delta, l.scales, l.batch, l.out_c, l.out_h*l.out_w);

    mean_delta_cpu(l.delta, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta);
    variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta);
    normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, l.delta, 1, state.delta, 1);
}

#ifdef OCL

void forward_batchnorm_layer_ocl(layer l, network_state state)
{
    cl_int errNum;
    if(l.type == BATCHNORM)
    {
	errNum = copy_on_ocl(commandQueue, l.outputs*l.batch, state.input_ocl, 1, l.output_ocl, 1);
    	if(errNum != CL_SUCCESS)
    	{
    		fprintf(stderr, "copy_on_ocl 执行失败, 退出程序\n");
    		exit(1);
    	}
    }
    if(l.type == CONNECTED){
        l.out_c = l.outputs;
        l.out_h = l.out_w = 1;
    }
    if (state.train)
    {
	errNum = fast_mean_ocl(commandQueue, l.output_ocl, l.batch, l.out_c, l.out_h*l.out_w, l.mean_ocl);
        if(errNum != CL_SUCCESS) {fprintf(stderr, "forward_batchnorm_layer_ocl train中的fast_scale_ocl执行失败,退出程序\n"); exit(1);}
        errNum = fast_variance_ocl(commandQueue, l.output_ocl, l.mean_ocl, l.batch, l.out_c, l.out_h*l.out_w, l.variance_ocl);
        if(errNum != CL_SUCCESS) {fprintf(stderr, "forward_batchnorm_layer_ocl train中的fast_variance_ocl执行失败,退出程序\n"); exit(1);}
        errNum = scal_on_ocl(commandQueue, l.out_c, .99, l.rolling_mean_ocl, 1);
        if(errNum != CL_SUCCESS) {fprintf(stderr, "forward_batchnorm_layer_ocl train中的scal_on_ocl执行失败,退出程序\n"); exit(1);}
        errNum = axpy_on_ocl(commandQueue, l.out_c, .01, l.mean_ocl, 1, l.rolling_mean_ocl, 1);
        if(errNum != CL_SUCCESS) {fprintf(stderr, "forward_batchnorm_layer_ocl train中的axpy_on_ocl执行失败,退出程序\n"); exit(1);}
        errNum = scal_on_ocl(commandQueue, l.out_c, .99, l.rolling_variance_ocl, 1);
        if(errNum != CL_SUCCESS) {fprintf(stderr, "forward_batchnorm_layer_ocl train中的scale_on_ocl执行失败,退出程序\n"); exit(1);}
        errNum = axpy_on_ocl(commandQueue, l.out_c, .01, l.variance_ocl, 1, l.rolling_variance_ocl, 1);
        if(errNum != CL_SUCCESS) {fprintf(stderr, "forward_batchnorm_layer_ocl train中的axpy_on_ocl执行失败,退出程序\n"); exit(1);}
        errNum = copy_on_ocl(commandQueue, l.outputs * l.batch, l.output_ocl, 1, l.x_ocl, 1);
        if(errNum != CL_SUCCESS) {fprintf(stderr, "forward_batchnorm_layer_ocl train中的copy_on_ocl执行失败,退出程序\n"); exit(1);}
        errNum = normalize_ocl(commandQueue, l.output_ocl, l.mean_ocl, l.variance_ocl, l.batch, l.out_c, l.out_h * l.out_w);
        if(errNum != CL_SUCCESS) {fprintf(stderr, "forward_batchnorm_layer_ocl train中的normalize_ocll执行失败,退出程序\n"); exit(1);}
        errNum = copy_on_ocl(commandQueue, l.outputs * l.batch, l.output_ocl, 1, l.x_norm_ocl, 1);
        if(errNum != CL_SUCCESS) {fprintf(stderr, "forward_batchnorm_layer_ocl train中的copy_on_ocl执行失败,退出程序\n"); exit(1);}
    }
    else
    {
	errNum = normalize_ocl(commandQueue, l.output_ocl, l.rolling_mean_ocl, l.rolling_variance_ocl, l.batch, l.out_c, l.out_h * l.out_w);
    	if(errNum != CL_SUCCESS)
    	{
    		fprintf(stderr, "normalize_ocl 执行失败, 退出程序\n");
    		exit(1);
    	}
    }
    errNum = scale_bias_ocl(commandQueue, l.output_ocl, l.scales_ocl, l.batch, l.out_c, l.out_h*l.out_w);
    if(errNum != CL_SUCCESS)
    {
	fprintf(stderr, "scale_bias_ocl 执行失败, 退出程序\n");
	exit(1);
    }
    errNum = add_bias_ocl(commandQueue, l.output_ocl, l.biases_ocl, l.batch, l.out_c, l.out_h * l.out_w);
    if (errNum != CL_SUCCESS)
    {
	fprintf(stderr, "add_bias_ocl 执行失败, 退出程序\n");
	exit(1);
    }
}
void push_batchnorm_layer_ocl(layer l)
{
	l.scales_ocl = cl_push_array(commandQueue, l.scales, l.scales_ocl, l.c * sizeof(float));
	l.rolling_mean_ocl = cl_push_array(commandQueue, l.rolling_mean, l.rolling_mean_ocl, l.c * sizeof(float));
	l.rolling_variance_ocl = cl_push_array(commandQueue, l.rolling_variance, l.rolling_variance_ocl, l.c * sizeof(float));
}
void backward_batchnorm_layer_ocl(const layer l, network_state state)
{
    cl_int errNum;
    errNum = backward_scale_ocl(commandQueue, l.x_norm_ocl, l.delta_ocl, l.batch, l.out_c, l.out_w * l.out_h, l.scale_updates_ocl);
    if(errNum != CL_SUCCESS) {fprintf(stderr, "backward_batchnorm_layer_ocl中的backward_scale_ocl执行失败,退出程序\n"); exit(1);}
    errNum = scale_bias_ocl(commandQueue, l.delta_ocl, l.scales_ocl, l.batch, l.out_c, l.out_h * l.out_w);
    if(errNum != CL_SUCCESS) {fprintf(stderr, "backward_batchnorm_layer_ocl中的scale_bias_ocl执行失败,退出程序\n"); exit(1);}
    errNum = fast_mean_delta_ocl(commandQueue, l.delta_ocl, l.variance_ocl, l.batch, l.out_c, l.out_w * l.out_h, l.mean_delta_ocl);
    if(errNum != CL_SUCCESS) {fprintf(stderr, "backward_batchnorm_layer_ocl中的fast_mean_delta_ocl执行失败,退出程序\n"); exit(1);}
    errNum = fast_variance_delta_ocl(commandQueue, l.x_ocl, l.delta_ocl, l.mean_ocl, l.variance_ocl, l.batch, l.out_c, l.out_w * l.out_h, l.variance_delta_ocl);
    if(errNum != CL_SUCCESS) {fprintf(stderr, "backward_batchnorm_layer_ocl中的fast_variance_delta_ocl执行失败,退出程序\n"); exit(1);}
    errNum = normalize_delta_ocl(commandQueue, l.x_ocl, l.mean_ocl, l.variance_ocl, l.mean_delta_ocl, l.variance_delta_ocl, l.batch, l.out_c, l.out_w * l.out_h, l.delta_ocl);
    if(errNum != CL_SUCCESS) {fprintf(stderr, "backward_batchnorm_layer_ocl中的normalize_delta_ocl执行失败,退出程序\n"); exit(1);}
    if(l.type == BATCHNORM)
    {
    	errNum = copy_on_ocl(commandQueue, l.outputs * l.batch, l.delta_ocl, 1, state.delta_ocl, 1);
    	if(errNum != CL_SUCCESS) {fprintf(stderr, "backward_batchnorm_layer_ocl中的copy_on_ocl执行失败,退出程序\n"); exit(1);}
    }
}
#endif
