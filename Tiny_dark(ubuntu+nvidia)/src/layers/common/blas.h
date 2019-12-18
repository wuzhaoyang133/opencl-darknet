#ifndef BLAS_H
#define BLAS_H
void flatten(float *x, int size, int layers, int batch, int forward);
void pm(int M, int N, float *A);
float *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

void test_blas();

void const_cpu(int N, float ALPHA, float *X, int INCX);
void constrain_ongpu(int N, float ALPHA, float * X, int INCX);
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void test_gpu_blas();
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out);

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);

void scale_bias(float *output, float *scales, int batch, int n, int size);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_sum_cpu(float *a, float *b, float *s, int num, float *c);

void softmax(float *input, int n, float temp, float *output);
void softmax_stride(float *input, int n, float temp, int stride, float *output);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

#ifdef OCL
#include "oclutils.h"
extern cl_int upsample_ocl(cl_command_queue command, cl_mem in, int w, int h, int c, int batch, int stride, int forward, float scale, cl_mem out);
extern cl_int shortcut_ocl(cl_command_queue command, int batch, int w1, int h1, int c1, cl_mem add, int w2, int h2, int c2, cl_mem out);
extern cl_int fill_on_ocl(cl_command_queue command, int N, float ALPHA, cl_mem  X, int INCX);
extern cl_int const_on_ocl(cl_command_queue command, int N, float ALPHA, cl_mem X, int INCX);
extern cl_int scal_on_ocl(cl_command_queue command, int N, float ALPHA, cl_mem  X, int INCX);
extern cl_int pow_on_ocl(cl_command_queue command, int N, float ALPHA, cl_mem X, int INCX, cl_mem Y, int INCY);
extern cl_int axpy_on_ocl(cl_command_queue command, int N, float ALPHA, cl_mem  X, int INCX, cl_mem Y, int INCY);
extern cl_int axpy_offset_on_ocl(cl_command_queue command, int N, float ALPHA, cl_mem  X, int OFFX, int INCX, cl_mem Y, int OFFY, int INCY);
extern cl_int copy_on_ocl(cl_command_queue command, int N, cl_mem X, int INCX, cl_mem Y, int INCY);
extern cl_int copy_on_ocl_offset(cl_command_queue command, int N, cl_mem  X, int OFFX, int INCX, cl_mem Y, int OFFY, int INCY);
extern cl_int mul_on_ocl(cl_command_queue command, int N, cl_mem X, int INCX, cl_mem Y, int INCY);
extern cl_int add_bias_ocl(cl_command_queue command, cl_mem output, cl_mem biases, int batch, int n, int size);
extern cl_int scale_bias_ocl(cl_command_queue command, cl_mem output, cl_mem biases, int batch, int n, int size);
extern cl_int normalize_ocl(cl_command_queue command, cl_mem x, cl_mem mean, cl_mem variance, int batch, int filters, int spatial);
extern cl_int l2_ocl(cl_command_queue command, int N, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error);
extern cl_int flatten_on_ocl(cl_command_queue command, cl_mem x, int spatial, int layers, int batch, int forward, cl_mem out);
extern cl_int backward_bias_ocl(cl_command_queue command, cl_mem bias_updates, cl_mem delta, int batch, int n, int size);
extern cl_int backward_scale_ocl(cl_command_queue command, cl_mem x_norm, cl_mem delta, int batch, int n, int size, cl_mem scale_updates);
extern cl_int fast_mean_delta_ocl(cl_command_queue command, cl_mem delta, cl_mem variance, int batch, int filters, int spatial, cl_mem mean_delta);
extern cl_int fast_variance_delta_ocl(cl_command_queue command, cl_mem x, cl_mem delta,cl_mem mean, cl_mem variance, int batch, int filters, int spatial,cl_mem variance_delta);
extern cl_int normalize_delta_ocl(cl_command_queue command, cl_mem x, cl_mem mean, cl_mem variance, cl_mem mean_delta, cl_mem variance_delta, int batch, int filters, int spatial, cl_mem delta);
extern cl_int fast_mean_ocl(cl_command_queue command, cl_mem x, int batch, int filters, int spatial, cl_mem mean);
extern cl_int fast_variance_ocl(cl_command_queue command, cl_mem x, cl_mem mean, int batch, int filters, int spatial, cl_mem variance);
#endif
#endif
