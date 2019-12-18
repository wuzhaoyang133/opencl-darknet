/*
 * oclutils.h
 *
 *  Created on: 2017年5月3日
 *      Author: yiicy
 */

#ifndef _OCLUTILS_H_
#define _OCLUTILS_H_

#include<stdlib.h>
#include<stdio.h>
#include <stdbool.h>
#include<time.h>
#include<sys/time.h>
//#include<sys/time.h>
//#include <windows.h>
#ifdef _APPLE_
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#define SHARE_MEMORY_SIZE (16 * 1024 - 256)
#define INF 9999999
#define PROGRAM_NUM 10
#define DMA_ALIGNMENT 64
#define BLOCK_SIZE 16  //10
#define MAT_MUL_BLOCK_SIZE 16 //10
typedef unsigned uint;
typedef unsigned char uchar;
typedef enum {MAXPOOL_PROGRAM, UTILS_PROGRAM, UTILS1_PROGRAM, IM2COL_PROGRAM, ACTIVATION_PROGRAM, 
			  AVGPOOL_PROGRAM, SOFTMAX_PROGRAM, REORG_PROGRAM, COL2IM_PROGRAM, TRANSPOSE_PROGRAM};
extern const size_t local_work_size1d[1];
extern const size_t local_work_size2d[2];
extern const size_t local_work_size3d[3];
extern cl_context context;
extern cl_device_id device;
extern cl_command_queue commandQueue;
extern cl_program Program;
extern int ocl_index;
extern void time_start(void);
extern double time_elapse(void);
extern double get_kernel_runtime(cl_event kernel_event);
extern bool check_errNum(cl_int errNum);
extern cl_mem cl_make_array(cl_context context, void* host_data, size_t size);
extern cl_mem cl_push_array(cl_command_queue command, void* host_data, cl_mem device_data, size_t size);
extern cl_mem cl_pull_array(cl_command_queue command, void* host_data, cl_mem device_data, size_t size);
extern void get_global_work_size1d(size_t n, size_t* global_work_size1d);
extern void get_global_work_size2d(size_t x, size_t y, size_t* global_work_size2d);
extern void get_global_work_size3d(size_t x, size_t y, size_t z,  size_t* global_work_size3d);
extern char* read_kernel(const char* filename, size_t* kernel_length);
extern cl_program createProgramWithSource(cl_context context, cl_device_id * devices, size_t device_num, char** kernel_name, size_t kernel_num, const char* options);
extern bool saveProgramBinary(cl_program program);
extern cl_program createProgramWithBinary(cl_context context, cl_device_id device, const char* binary_path);
extern void *alignedMalloc(size_t size);
extern void *alignedCalloc(size_t n, size_t size);
extern void alignedFree(void * ptr);



cl_context CreateContext(cl_device_id *device);
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device);
void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem memObjects[3]);
//void forward_maxpool_layer(int batch, int in_h, int in_w, int in_c, int stride, int size, int pad,  float *input,  float *output, int *indexes);
//void im2col_cpu(float* data_im,
 //       int channels, int height, int width,
 //      int ksize, int stride, int pad, float* data_col);
void mat_mul(int m, int n, int k, float* A, float* B, float* C);
//void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
//void softmax(float *input, int n, float temp, float *output);
#endif /* OCLINCLUDE_OCLUTILS_H_ */
