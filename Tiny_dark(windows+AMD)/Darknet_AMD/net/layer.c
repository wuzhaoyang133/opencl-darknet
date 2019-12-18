#include "layer.h"
#include "cuda.h"
#include <stdlib.h>

void free_layer(layer l)
{
    if(l.type == DROPOUT){
        if(l.rand)           alignedFree(l.rand);
#ifdef GPU
        if(l.rand_gpu)             cuda_free(l.rand_gpu);
#endif
        return;
    }
    if(l.cweights)           free(l.cweights);
    if(l.indexes)            alignedFree(l.indexes);
    if(l.input_layers)       alignedFree(l.input_layers);
    if(l.input_sizes)        alignedFree(l.input_sizes);
    if(l.map)                alignedFree(l.map);
    if(l.rand)               alignedFree(l.rand);
    if(l.cost)               alignedFree(l.cost);
    if(l.state)              alignedFree(l.state);
    if(l.prev_state)         free(l.prev_state);
    if(l.forgot_state)       free(l.forgot_state);
    if(l.forgot_delta)       free(l.forgot_delta);
    if(l.state_delta)        free(l.state_delta);
    if(l.concat)             free(l.concat);
    if(l.concat_delta)       free(l.concat_delta);
    if(l.binary_weights)     free(l.binary_weights);
    if(l.biases)             alignedFree(l.biases);
    if(l.bias_updates)       alignedFree(l.bias_updates);
    if(l.scales)             alignedFree(l.scales);
    if(l.scale_updates)      alignedFree(l.scale_updates);
    if(l.weights)            alignedFree(l.weights);
    if(l.weight_updates)     alignedFree(l.weight_updates);
    if(l.col_image)          alignedFree(l.col_image);
    if(l.delta)              alignedFree(l.delta);
    if(l.output)             alignedFree(l.output);
    if(l.squared)            free(l.squared);
    if(l.norms)              alignedFree(l.norms);
    if(l.spatial_mean)       free(l.spatial_mean);
    if(l.mean)               alignedFree(l.mean);
    if(l.variance)           alignedFree(l.variance);
    if(l.mean_delta)         alignedFree(l.mean_delta);
    if(l.variance_delta)     alignedFree(l.variance_delta);
    if(l.rolling_mean)       alignedFree(l.rolling_mean);
    if(l.rolling_variance)   alignedFree(l.rolling_variance);
    if(l.x)                  alignedFree(l.x);
    if(l.x_norm)             alignedFree(l.x_norm);
    if(l.m)                  alignedFree(l.m);
    if(l.v)                  free(l.v);
    if(l.z_cpu)              free(l.z_cpu);
    if(l.r_cpu)              free(l.r_cpu);
    if(l.h_cpu)              free(l.h_cpu);
    if(l.binary_input)       free(l.binary_input);

#ifdef GPU
    if(l.indexes_gpu)           cuda_free((float *)l.indexes_gpu);

    if(l.z_gpu)                   cuda_free(l.z_gpu);
    if(l.r_gpu)                   cuda_free(l.r_gpu);
    if(l.h_gpu)                   cuda_free(l.h_gpu);
    if(l.m_gpu)                   cuda_free(l.m_gpu);
    if(l.v_gpu)                   cuda_free(l.v_gpu);
    if(l.prev_state_gpu)          cuda_free(l.prev_state_gpu);
    if(l.forgot_state_gpu)        cuda_free(l.forgot_state_gpu);
    if(l.forgot_delta_gpu)        cuda_free(l.forgot_delta_gpu);
    if(l.state_gpu)               cuda_free(l.state_gpu);
    if(l.state_delta_gpu)         cuda_free(l.state_delta_gpu);
    if(l.gate_gpu)                cuda_free(l.gate_gpu);
    if(l.gate_delta_gpu)          cuda_free(l.gate_delta_gpu);
    if(l.save_gpu)                cuda_free(l.save_gpu);
    if(l.save_delta_gpu)          cuda_free(l.save_delta_gpu);
    if(l.concat_gpu)              cuda_free(l.concat_gpu);
    if(l.concat_delta_gpu)        cuda_free(l.concat_delta_gpu);
    if(l.binary_input_gpu)        cuda_free(l.binary_input_gpu);
    if(l.binary_weights_gpu)      cuda_free(l.binary_weights_gpu);
    if(l.mean_gpu)                cuda_free(l.mean_gpu);
    if(l.variance_gpu)            cuda_free(l.variance_gpu);
    if(l.rolling_mean_gpu)        cuda_free(l.rolling_mean_gpu);
    if(l.rolling_variance_gpu)    cuda_free(l.rolling_variance_gpu);
    if(l.variance_delta_gpu)      cuda_free(l.variance_delta_gpu);
    if(l.mean_delta_gpu)          cuda_free(l.mean_delta_gpu);
    if(l.col_image_gpu)           cuda_free(l.col_image_gpu);
    if(l.x_gpu)                   cuda_free(l.x_gpu);
    if(l.x_norm_gpu)              cuda_free(l.x_norm_gpu);
    if(l.weights_gpu)             cuda_free(l.weights_gpu);
    if(l.weight_updates_gpu)      cuda_free(l.weight_updates_gpu);
    if(l.biases_gpu)              cuda_free(l.biases_gpu);
    if(l.bias_updates_gpu)        cuda_free(l.bias_updates_gpu);
    if(l.scales_gpu)              cuda_free(l.scales_gpu);
    if(l.scale_updates_gpu)       cuda_free(l.scale_updates_gpu);
    if(l.output_gpu)              cuda_free(l.output_gpu);
    if(l.delta_gpu)               cuda_free(l.delta_gpu);
    if(l.rand_gpu)                cuda_free(l.rand_gpu);
    if(l.squared_gpu)             cuda_free(l.squared_gpu);
    if(l.norms_gpu)               cuda_free(l.norms_gpu);
#endif
#ifdef OCL
	if (l.indexes_ocl) clReleaseMemObject(l.indexes_ocl);
	if(l.output_ocl) clReleaseMemObject(l.output_ocl);
	if(l.delta_ocl) clReleaseMemObject(l.delta_ocl);
	if(l.weights_ocl) clReleaseMemObject(l.weights_ocl);
	if(l.weight_updates_ocl) clReleaseMemObject(l.weight_updates_ocl);
	if(l.biases_ocl) clReleaseMemObject(l.weight_updates_ocl);
	if(l.bias_updates_ocl) clReleaseMemObject(l.bias_updates_ocl);
	if(l.mean_ocl) clReleaseMemObject(l.mean_ocl);
	if(l.variance_ocl) clReleaseMemObject(l.variance_ocl);
	if(l.rolling_mean_ocl) clReleaseMemObject(l.rolling_mean_ocl);
	if(l.rolling_variance_ocl) clReleaseMemObject(l.rolling_variance_ocl);
	if(l.mean_delta_ocl) clReleaseMemObject(l.mean_delta_ocl);
	if(l.variance_delta_ocl) clReleaseMemObject(l.variance_delta_ocl);
	if(l.scales_ocl) clReleaseMemObject(l.scales_ocl);
	if(l.scale_updates_ocl) clReleaseMemObject(l.scale_updates_ocl);
	if(l.rolling_mean_ocl) clReleaseMemObject(l.rolling_mean_ocl);
	if(l.x_ocl) clReleaseMemObject(l.x_ocl);
	if(l.x_norm_ocl) clReleaseMemObject(l.x_norm_ocl);
	if(l.squared_ocl) clReleaseMemObject(l.squared_ocl);
#endif
}
