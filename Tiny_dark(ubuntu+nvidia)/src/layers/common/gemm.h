#ifndef GEMM_H
#define GEMM_H

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);


#ifdef OCL
#include"oclutils.h"
extern cl_int mat_mul_ocl(cl_command_queue command, int M, int N, int K, float ALPHA, cl_mem A, size_t A_offset, cl_mem B, size_t B_offset, float BETA, cl_mem C, size_t C_offset);
extern cl_int gemm_ocl(cl_command_queue command, int TA, int TB, int M, int N, int K, float ALPHA, cl_mem A, size_t A_offset, cl_mem B,  size_t B_offset, float BETA, cl_mem C,  size_t C_offset);
#endif

#endif
