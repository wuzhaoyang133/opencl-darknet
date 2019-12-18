#include "blas.h"
#include "math.h"
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int out_c = c/(stride*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}

void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] += add[add_index];
                }
            }
        }
    }
}

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}

void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff < 0) ? -1 : 1;
        }
    }
}

void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}

void softmax(float *input, int n, float temp, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i] > largest) largest = input[i];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i]/temp - largest/temp);
        sum += e;
        output[i] = e;
    }
    for(i = 0; i < n; ++i){
        output[i] /= sum;
    }
}

void softmax_stride(float *input, int n, float temp, int stride, float *output)
{
	int i;
	float sum = 0;
	float largest = -FLT_MAX;
	for (i = 0; i < n; ++i) {
		if (input[i*stride] > largest) largest = input[i*stride];
	}
	for (i = 0; i < n; ++i) {
		float e = exp(input[i*stride] / temp - largest / temp);
		sum += e;
		output[i*stride] = e;
	}
	for (i = 0; i < n; ++i) {
		output[i*stride] /= sum;
	}
}


void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
	int g, b;
	for (b = 0; b < batch; ++b) {
		for (g = 0; g < groups; ++g) {
			softmax_stride(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
		}
	}
}
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
	int i, j, k, b;
	for (b = 0; b < batch; ++b) {
		for (k = 0; k < c; ++k) {
			for (j = 0; j < h*stride; ++j) {
				for (i = 0; i < w*stride; ++i) {
					int in_index = b*w*h*c + k*w*h + (j / stride)*w + i / stride;
					int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
					if (forward) out[out_index] = scale*in[in_index];
					else in[in_index] += scale*out[out_index];
				}
			}
		}
	}
}
#ifdef OCL
cl_int fill_on_ocl(cl_command_queue command, int N, float ALPHA, cl_mem  X, int INCX)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "fill_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &N);
	errNum |= clSetKernelArg(kernel, 1, sizeof(float), &ALPHA);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &X);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &INCX);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(N, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	check_errNum(errNum);
	clReleaseKernel(kernel);
	return errNum;
}

cl_int const_on_ocl(cl_command_queue command, int N, float ALPHA, cl_mem X, int INCX)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "const_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &N);
	errNum |= clSetKernelArg(kernel, 1, sizeof(float), &ALPHA);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &X);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &INCX);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(N, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	return errNum;
}

cl_int scal_on_ocl(cl_command_queue command, int N, float ALPHA, cl_mem  X, int INCX)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "scal_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &N);
	errNum |= clSetKernelArg(kernel, 1, sizeof(float), &ALPHA);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &X);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &INCX);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(N, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	return errNum;
}

cl_int pow_on_ocl(cl_command_queue command, int N, float ALPHA, cl_mem X, int INCX, cl_mem Y, int INCY)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "pow_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &N);
	errNum |= clSetKernelArg(kernel, 1, sizeof(float), &ALPHA);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &X);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &INCX);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &Y);
	errNum |= clSetKernelArg(kernel, 5, sizeof(int), &INCY);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(N, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	return errNum;
}

cl_int axpy_on_ocl(cl_command_queue command, int N, float ALPHA, cl_mem  X, int INCX, cl_mem Y, int INCY)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "axpy_kernel", NULL);
	cl_int OFFX = 0; cl_int OFFY = 0;
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &N);
	errNum |= clSetKernelArg(kernel, 1, sizeof(float), &ALPHA);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &X);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &OFFX);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &INCX);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &Y);
	errNum |= clSetKernelArg(kernel, 6, sizeof(int), &OFFY);
	errNum |= clSetKernelArg(kernel, 7, sizeof(int), &INCY);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(N, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	return errNum;
}

cl_int axpy_offset_on_ocl(cl_command_queue command, int N, float ALPHA, cl_mem  X, int OFFX, int INCX, cl_mem Y, int OFFY, int INCY)
{
	cl_int errNum;
	//cl_program program = Program[UTILS1_PROGRAM];
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "axpy_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &N);
	errNum |= clSetKernelArg(kernel, 1, sizeof(float), &ALPHA);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &X);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &OFFX);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &INCX);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &Y);
	errNum |= clSetKernelArg(kernel, 6, sizeof(int), &OFFY);
	errNum |= clSetKernelArg(kernel, 7, sizeof(int), &INCY);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(N, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	return errNum;
}
cl_int copy_on_ocl(cl_command_queue command, int N, cl_mem X, int INCX, cl_mem Y, int INCY)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "copy_kernel", NULL);
	cl_int OFFX = 0; cl_int OFFY = 0;
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &N);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &X);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &OFFX);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &INCX);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &Y);
	errNum |= clSetKernelArg(kernel, 5, sizeof(int), &OFFY);
	errNum |= clSetKernelArg(kernel, 6, sizeof(int), &INCY);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(N, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	check_errNum(errNum);
	clReleaseKernel(kernel);
	return errNum;
}
cl_int copy_on_ocl_offset(cl_command_queue command, int N, cl_mem  X, int OFFX, int INCX, cl_mem Y, int OFFY, int INCY)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "copy_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &N);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &X);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &OFFX);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &INCX);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &Y);
	errNum |= clSetKernelArg(kernel, 5, sizeof(int), &OFFY);
	errNum |= clSetKernelArg(kernel, 6, sizeof(int), &INCY);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(N, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	return errNum;
}

cl_int mul_on_ocl(cl_command_queue command, int N, cl_mem X, int INCX, cl_mem Y, int INCY)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "mul_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &N);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &X);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &INCX);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &Y);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &INCY);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(N, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	return errNum;
}

cl_int add_bias_ocl(cl_command_queue command, cl_mem output, cl_mem biases, int batch, int n, int size)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "add_bias_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &biases);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &batch);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &n);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &size);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size3d[3];
	get_global_work_size3d(size, n, batch, global_work_size3d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 3, 0, global_work_size3d, local_work_size3d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	return errNum;
}

cl_int scale_bias_ocl(cl_command_queue command, cl_mem output, cl_mem biases, int batch, int n, int size)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "scale_bias_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &biases);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &batch);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &n);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &size);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size3d[3];
	get_global_work_size3d(size, n, batch, global_work_size3d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 3, 0, global_work_size3d, local_work_size3d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	return errNum;
}

cl_int normalize_ocl(cl_command_queue command, cl_mem x, cl_mem mean, cl_mem variance, int batch, int filters, int spatial)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "normalize_kernel", NULL);
	size_t N = batch * filters * spatial;
	errNum = clSetKernelArg(kernel, 0, sizeof(size_t), &N);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &x);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &mean);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &variance);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &batch);
	errNum |= clSetKernelArg(kernel, 5, sizeof(int), &filters);
	errNum |= clSetKernelArg(kernel, 6, sizeof(int), &spatial);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(N, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	return errNum;
}

cl_int l2_ocl(cl_command_queue command, int N, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "l2_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &N);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &pred);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &truth);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &delta);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &error);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(N, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	return errNum;
}

cl_int flatten_on_ocl(cl_command_queue command, cl_mem x, int spatial, int layers, int batch, int forward, cl_mem out)
{
	cl_int errNum;
	int size = spatial * batch * layers;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "flatten_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &size);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &x);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_int), &spatial);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_int), &layers);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_int), &batch);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_int), &forward);
	errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &out);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(size, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	return errNum;
}

cl_int backward_bias_ocl(cl_command_queue command, cl_mem bias_updates, cl_mem delta, int batch, int n, int size)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "backward_bias_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bias_updates);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &delta);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &batch);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &n);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &size);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(n * local_work_size1d[0], global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	check_errNum(errNum);
	clReleaseKernel(kernel);
	return errNum;
}

cl_int backward_scale_ocl(cl_command_queue command, cl_mem x_norm, cl_mem delta, int batch, int n, int size, cl_mem scale_updates)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "backward_scale_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &x_norm);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &delta);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &batch);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &n);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &size);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &scale_updates);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(n * local_work_size1d[0], global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	check_errNum(errNum);
	clReleaseKernel(kernel);
	return errNum;
}
cl_int fast_mean_delta_ocl(cl_command_queue command, cl_mem delta, cl_mem variance, int batch, int filters, int spatial, cl_mem mean_delta)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "fast_mean_delta_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &delta);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &variance);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &batch);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &filters);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &spatial);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &mean_delta);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(filters * local_work_size1d[0], global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	check_errNum(errNum);
	clReleaseKernel(kernel);
	return errNum;
}
cl_int fast_variance_delta_ocl(cl_command_queue command, cl_mem x, cl_mem delta,cl_mem mean, cl_mem variance, int batch, int filters, int spatial,cl_mem variance_delta)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "fast_variance_delta_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &x);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &delta);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &mean);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &variance);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &batch);
	errNum |= clSetKernelArg(kernel, 5, sizeof(int), &filters);
	errNum |= clSetKernelArg(kernel, 6, sizeof(int), &spatial);
	errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &variance_delta);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(filters * local_work_size1d[0], global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	check_errNum(errNum);
	clReleaseKernel(kernel);
	return errNum;
}

cl_int normalize_delta_ocl(cl_command_queue command, cl_mem x, cl_mem mean, cl_mem variance, cl_mem mean_delta, cl_mem variance_delta, int batch, int filters, int spatial, cl_mem delta)
{
	cl_int errNum;
	size_t N = batch * filters * spatial;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "normalize_delta_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(size_t), &N);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &x);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &mean);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &variance);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &mean_delta);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &variance_delta);
	errNum |= clSetKernelArg(kernel, 6, sizeof(int), &batch);
	errNum |= clSetKernelArg(kernel, 7, sizeof(int), &filters);
	errNum |= clSetKernelArg(kernel, 8, sizeof(int), &spatial);
	errNum |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &delta);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(N, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	check_errNum(errNum);
	clReleaseKernel(kernel);
	return errNum;
}
cl_int fast_mean_ocl(cl_command_queue command, cl_mem x, int batch, int filters, int spatial, cl_mem mean)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "fast_mean_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &x);
	errNum |= clSetKernelArg(kernel, 1, sizeof(int), &batch);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &filters);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &spatial);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &mean);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(filters * local_work_size1d[0], global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	check_errNum(errNum);
	clReleaseKernel(kernel);
	return errNum;
}
cl_int fast_variance_ocl(cl_command_queue command, cl_mem x, cl_mem mean, int batch, int filters, int spatial, cl_mem variance)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "fast_variance_kernel", NULL);
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &x);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mean);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &batch);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &filters);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &spatial);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &variance);
	if(!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(filters * local_work_size1d[0], global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	check_errNum(errNum);
	clReleaseKernel(kernel);
	return errNum;
}
cl_int shortcut_ocl(cl_command_queue command, int batch, int w1, int h1, int c1, cl_mem add, int w2, int h2, int c2, cl_mem out)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "shortcut_kernel", NULL);
	int minw = (w1 < w2) ? w1 : w2;
	int minh = (h1 < h2) ? h1 : h2;
	int minc = (c1 < c2) ? c1 : c2;
	int stride = w1 / w2;
	int sample = w2 / w1;
	assert(stride == h1 / h2);
	assert(sample == h2 / h1);
	if (stride < 1) stride = 1;
	if (sample < 1) sample = 1;
	int size = batch * minw * minh * minc;
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &size);
	errNum |= clSetKernelArg(kernel, 1, sizeof(int), &minw);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &minh);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &minc);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &stride);
	errNum |= clSetKernelArg(kernel, 5, sizeof(int), &sample);
	errNum |= clSetKernelArg(kernel, 6, sizeof(int), &batch);
	errNum |= clSetKernelArg(kernel, 7, sizeof(int), &w1);
	errNum |= clSetKernelArg(kernel, 8, sizeof(int), &h1);
	errNum |= clSetKernelArg(kernel, 9, sizeof(int), &c1);
	errNum |= clSetKernelArg(kernel, 10, sizeof(cl_mem), &add);
	errNum |= clSetKernelArg(kernel, 11, sizeof(int), &w2);
	errNum |= clSetKernelArg(kernel, 12, sizeof(int), &h2);
	errNum |= clSetKernelArg(kernel, 13, sizeof(int), &c2);
	errNum |= clSetKernelArg(kernel, 14, sizeof(cl_mem), &out);
	if (!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(size, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	check_errNum(errNum);
	clReleaseKernel(kernel);
	return errNum;
}
cl_int upsample_ocl(cl_command_queue command, cl_mem in, int w, int h, int c, int batch, int stride, int forward, float scale, cl_mem out)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "upsample_kernel", NULL);
	int size = w*h*c*batch*stride*stride;
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &size);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &in);
	errNum |= clSetKernelArg(kernel, 2, sizeof(int), &w);
	errNum |= clSetKernelArg(kernel, 3, sizeof(int), &h);
	errNum |= clSetKernelArg(kernel, 4, sizeof(int), &c);
	errNum |= clSetKernelArg(kernel, 5, sizeof(int), &batch);
	errNum |= clSetKernelArg(kernel, 6, sizeof(int), &stride);
	errNum |= clSetKernelArg(kernel, 7, sizeof(int), &forward);
	errNum |= clSetKernelArg(kernel, 8, sizeof(float), &scale);
	errNum |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &out);
	if (!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	size_t global_work_size1d[1];
	get_global_work_size1d(size, global_work_size1d);
	errNum = clEnqueueNDRangeKernel(command, kernel, 1, 0, global_work_size1d, local_work_size1d, 0, NULL, NULL);
	check_errNum(errNum);
	clReleaseKernel(kernel);
	return errNum;
}

#endif
