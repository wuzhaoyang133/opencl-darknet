#include "connected_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "blas.h"
#include "gemm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

connected_layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize)
{
    int i;
    connected_layer l = {0};
    l.type = CONNECTED;
    l.inputs = inputs;
    l.outputs = outputs;
    l.batch=batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;
    l.output = alignedCalloc(batch*outputs, sizeof(float));
    l.delta = alignedCalloc(batch*outputs, sizeof(float));
    l.weight_updates = alignedCalloc(inputs*outputs, sizeof(float));
    l.bias_updates = alignedCalloc(outputs, sizeof(float));
    l.weights = alignedCalloc(outputs*inputs, sizeof(float));
    l.biases = alignedCalloc(outputs, sizeof(float));
    l.forward = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update = update_connected_layer;
    float scale = sqrt(2./inputs);
    for(i = 0; i < outputs*inputs; ++i){
        l.weights[i] = scale*rand_uniform(-1, 1);
    }
    for(i = 0; i < outputs; ++i){
        l.biases[i] = 0;
    }
    if(batch_normalize){
        l.scales = alignedCalloc(outputs, sizeof(float));
        l.scale_updates = alignedCalloc(outputs, sizeof(float));
        for(i = 0; i < outputs; ++i){
            l.scales[i] = 1;
        }
        l.mean = alignedCalloc(outputs, sizeof(float));
        l.mean_delta = alignedCalloc(outputs, sizeof(float));
        l.variance = alignedCalloc(outputs, sizeof(float));
        l.variance_delta = alignedCalloc(outputs, sizeof(float));
        l.rolling_mean = alignedCalloc(outputs, sizeof(float));
        l.rolling_variance = alignedCalloc(outputs, sizeof(float));
        l.x = alignedCalloc(batch*outputs, sizeof(float));
        l.x_norm = alignedCalloc(batch*outputs, sizeof(float));
    }
#ifdef OCL
    l.forward_ocl = forward_connected_layer_ocl;

    if(ocl_index >= 0)
    {
        l.weights_ocl = cl_make_array(context,l.weights, outputs*inputs*sizeof(float));
        l.biases_ocl = cl_make_array(context,l.biases, outputs*sizeof(float));

        l.weight_updates_ocl = cl_make_array(context,l.weight_updates, outputs*inputs*sizeof(float));
        l.bias_updates_ocl = cl_make_array(context,l.bias_updates, outputs*sizeof(float));

        l.output_ocl = cl_make_array(context,l.output, outputs*batch*sizeof(float));
        l.delta_ocl = cl_make_array(context,l.delta, outputs*batch*sizeof(float));

        if(batch_normalize){
            l.mean_ocl = cl_make_array(context,l.mean, outputs*sizeof(float));
            l.variance_ocl = cl_make_array(context,l.variance, outputs*sizeof(float));

            l.rolling_mean_ocl = cl_make_array(context,l.mean, outputs*sizeof(float));
            l.rolling_variance_ocl = cl_make_array(context,l.variance, outputs*sizeof(float));

            l.mean_delta_ocl = cl_make_array(context,l.mean, outputs*sizeof(float));
            l.variance_delta_ocl = cl_make_array(context,l.variance, outputs*sizeof(float));

            l.scales_ocl = cl_make_array(context,l.scales, outputs*sizeof(float));
            l.scale_updates_ocl = cl_make_array(context,l.scale_updates, outputs*sizeof(float));

            l.x_ocl = cl_make_array(context,l.output, l.batch*outputs*sizeof(float));
            l.x_norm_ocl = cl_make_array(context,l.output, l.batch*outputs*sizeof(float));
        }
    }
#endif
    l.activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}
#ifdef OCL
void push_connected_layer_ocl(connected_layer layer)
{
    layer.weights_ocl = cl_push_array(commandQueue, layer.weights, layer.weights_ocl, layer.outputs*layer.inputs* sizeof(float));
    layer.biases_ocl = cl_push_array(commandQueue, layer.biases , layer.biases_ocl,  layer.outputs * sizeof(float));
    layer.weight_updates_ocl = cl_push_array(commandQueue, layer.weight_updates, layer.weight_updates_ocl, layer.outputs*layer.inputs*sizeof(float));
    layer.bias_updates_ocl = cl_push_array(commandQueue, layer.bias_updates, layer.bias_updates_ocl, layer.outputs*sizeof(float));

}
void forward_connected_layer_ocl(connected_layer l, network_state state){
    cl_int errNum;
    fill_on_ocl(commandQueue, l.outputs*l.batch, 0, l.output_ocl, 1);
    int m = l.batch;
    int n = l.inputs;
    int k = l.outputs;
    cl_mem a = state.input_ocl;
    cl_mem b = l.weights_ocl;
    cl_mem c = l.output_ocl;
    errNum = gemm_ocl(commandQueue, 0, 1, m, n, k, 1.0, a, 0, b, 0, 0.0, c,  0);

    if (l.batch_normalize) {
        forward_batchnorm_layer_ocl(l, state);
    } else {
        errNum = add_bias_ocl(commandQueue, l.output_ocl, l.biases_ocl, l.batch, l.outputs, 1);
    }
    activate_array_on_ocl(commandQueue,l.output_ocl, l.outputs*l.batch, l.activation);
}
#endif
void update_connected_layer(connected_layer l, int batch, float learning_rate, float momentum, float decay)
{
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    if(l.batch_normalize){
        axpy_cpu(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}

void forward_connected_layer(connected_layer l, network_state state)
{
    int i;
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = state.input;
    float *b = l.weights;
    float *c = l.output;
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    if(l.batch_normalize){
        if(state.train){
            mean_cpu(l.output, l.batch, l.outputs, 1, l.mean);
            variance_cpu(l.output, l.mean, l.batch, l.outputs, 1, l.variance);

            scal_cpu(l.outputs, .95, l.rolling_mean, 1);
            axpy_cpu(l.outputs, .05, l.mean, 1, l.rolling_mean, 1);
            scal_cpu(l.outputs, .95, l.rolling_variance, 1);
            axpy_cpu(l.outputs, .05, l.variance, 1, l.rolling_variance, 1);

            copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
            normalize_cpu(l.output, l.mean, l.variance, l.batch, l.outputs, 1);   
            copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
        } else {
            normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.outputs, 1);
        }
        scale_bias(l.output, l.scales, l.batch, l.outputs, 1);
    }
    for(i = 0; i < l.batch; ++i){
        axpy_cpu(l.outputs, 1, l.biases, 1, l.output + i*l.outputs, 1);
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_connected_layer(connected_layer l, network_state state)
{
    int i;
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    for(i = 0; i < l.batch; ++i){
        axpy_cpu(l.outputs, 1, l.delta + i*l.outputs, 1, l.bias_updates, 1);
    }
    if(l.batch_normalize){
        backward_scale_cpu(l.x_norm, l.delta, l.batch, l.outputs, 1, l.scale_updates);

        scale_bias(l.delta, l.scales, l.batch, l.outputs, 1);

        mean_delta_cpu(l.delta, l.variance, l.batch, l.outputs, 1, l.mean_delta);
        variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.outputs, 1, l.variance_delta);
        normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.outputs, 1, l.delta);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta;
    float *b = state.input;
    float *c = l.weight_updates;
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = state.delta;

    if(c) gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
}


void denormalize_connected_layer(layer l)
{
    int i, j;
    for(i = 0; i < l.outputs; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .000001);
        for(j = 0; j < l.inputs; ++j){
            l.weights[i*l.inputs + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}


void statistics_connected_layer(layer l)
{
    if(l.batch_normalize){
        printf("Scales ");
        print_statistics(l.scales, l.outputs);
    }
    printf("Biases ");
    print_statistics(l.biases, l.outputs);
    printf("Weights ");
    print_statistics(l.weights, l.outputs);
}
