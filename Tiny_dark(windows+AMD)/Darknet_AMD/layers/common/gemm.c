#include "gemm.h"
#include "utils.h"
#include "blas.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "sgemm.h"
void gemm_bin(int M, int N, int K, float ALPHA,
	char  *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			char A_PART = A[i*lda + k];
			if (A_PART) {
				for (j = 0; j < N; ++j) {
					C[i*ldc + j] += B[k*ldb + j];
				}
			}
			else {
				for (j = 0; j < N; ++j) {
					C[i*ldc + j] -= B[k*ldb + j];
				}
			}
		}
	}
}

float *random_matrix(int rows, int cols)
{
	int i;
	float *m = alignedCalloc(rows*cols, sizeof(float));
	for (i = 0; i < rows*cols; ++i) {
		m[i] = (float)rand() / RAND_MAX;
	}
	return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
	float *a;
	if (!TA) a = random_matrix(m, k);
	else a = random_matrix(k, m);
	int lda = (!TA) ? k : m;
	float *b;
	if (!TB) b = random_matrix(k, n);
	else b = random_matrix(n, k);
	int ldb = (!TB) ? n : k;

	float *c = random_matrix(m, n);
	int i;
	clock_t start = clock(), end;
	for (i = 0; i<10; ++i) {
		gemm_cpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
	}
	end = clock();
	printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n", m, k, k, n, TA, TB, (float)(end - start) / CLOCKS_PER_SEC);
	free(a);
	free(b);
	free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc)
{
	printf("M N K %d %d %d \n", M, N, K);
	gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

void gemm_nn(int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register float A_PART = ALPHA*A[i*lda + k];
			for (j = 0; j < N; ++j) {
				C[i*ldc + j] += A_PART*B[k*ldb + j];
			}
		}
	}
}

void gemm_nt(int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			register float sum = 0;
			for (k = 0; k < K; ++k) {
				sum += ALPHA*A[i*lda + k] * B[j*ldb + k];
			}
			C[i*ldc + j] += sum;
		}
	}
}

void gemm_tn(int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register float A_PART = ALPHA*A[k*lda + i];
			for (j = 0; j < N; ++j) {
				C[i*ldc + j] += A_PART*B[k*ldb + j];
			}
		}
	}
}

void gemm_tt(int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			register float sum = 0;
			for (k = 0; k < K; ++k) {
				sum += ALPHA*A[i + k*lda] * B[k + j*ldb];
			}
			C[i*ldc + j] += sum;
		}
	}
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc)
{
	int i, j;
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			C[i*ldc + j] *= BETA;
		}
	}
	if (!TA && !TB)
		gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else if (TA && !TB)
		gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else if (!TA && TB)
		gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else
		gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}
#ifdef OCL
cl_int mat_mul_ocl(cl_command_queue command, int M, int N, int K, float ALPHA, cl_mem A, size_t A_offset, cl_mem B, size_t B_offset, float BETA, cl_mem C, size_t C_offset)
{
	cl_int errNum;
	cl_program program = Program;
	cl_kernel kernel = clCreateKernel(program, "matrixMult", NULL);
	size_t global_work_size2d[2];
	size_t local_size2d[2] = { MAT_MUL_BLOCK_SIZE,MAT_MUL_BLOCK_SIZE };
	global_work_size2d[0] = (K + MAT_MUL_BLOCK_SIZE - 1) / MAT_MUL_BLOCK_SIZE * MAT_MUL_BLOCK_SIZE;
	global_work_size2d[1] = (M + MAT_MUL_BLOCK_SIZE - 1) / MAT_MUL_BLOCK_SIZE * MAT_MUL_BLOCK_SIZE;
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &C);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &A);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &B);
	errNum = clSetKernelArg(kernel, 3, sizeof(int), &N);
	errNum = clSetKernelArg(kernel, 4, sizeof(int), &K);
	errNum = clSetKernelArg(kernel, 5, sizeof(int), &M);
	errNum = clSetKernelArg(kernel, 6, sizeof(size_t), &C_offset);
	if (!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	errNum = clEnqueueNDRangeKernel(command, kernel, 2, NULL, global_work_size2d, local_size2d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	return errNum;
}
cl_int mat_transpose_ocl(cl_command_queue command, int M, int N, cl_mem mat, size_t mat_offset, cl_mem mat_t)
{
	cl_int errNum;
	cl_program program = Program;
	printf("err1 %d\n",errNum);
	cl_kernel kernel = clCreateKernel(program, "mat_transpose", NULL);
	printf("err2 %d\n", errNum);
	size_t global_work_size2d[2];
	get_global_work_size2d(N, M, global_work_size2d);
	printf("err3 %d\n", errNum);
	errNum = clSetKernelArg(kernel, 0, sizeof(int), &M);
	errNum |= clSetKernelArg(kernel, 1, sizeof(int), &N);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &mat);
	errNum |= clSetKernelArg(kernel, 3, sizeof(size_t), &mat_offset);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &mat_t);
	printf("err4 %d\n", errNum);
	if (!check_errNum(errNum))
	{
		clReleaseKernel(kernel);
		return errNum;
	}
	errNum = clEnqueueNDRangeKernel(command, kernel, 2, NULL, global_work_size2d, local_work_size2d, 0, NULL, NULL);
	clReleaseKernel(kernel);
	check_errNum(errNum);
	return errNum;
}

cl_int gemm_nn_ocl(cl_command_queue command, int M, int N, int K, float ALPHA, cl_mem A, size_t A_offset, cl_mem B, size_t B_offset, float BETA, cl_mem C, size_t C_offset)
{
	cl_int errNum;
	errNum = mat_mul_ocl(command, M, N, K, ALPHA, A, A_offset, B, B_offset, BETA, C, C_offset);
	if (!check_errNum(errNum))
		fprintf(stderr, "gemm_nn_ocl 中的 mat_mul_ocl执行失败\n");
	return errNum;
}
cl_int gemm_nt_ocl(cl_command_queue command, int M, int N, int K, float ALPHA, cl_mem A, size_t A_offset, cl_mem B, size_t B_offset, float BETA, cl_mem C, size_t C_offset)
{
	cl_int errNum;
	cl_context context;
	errNum = clGetCommandQueueInfo(command,
		CL_QUEUE_CONTEXT,
		sizeof(cl_context),
		&context,
		NULL);
	if (!check_errNum(errNum))
	{
		fprintf(stderr, "gemm_nt_ocl中获取命令队列信息失败\n");
		return errNum;
	}
	cl_mem B_t = cl_make_array(context, NULL, K * N * sizeof(float));
	errNum = mat_transpose_ocl(command, K, N, B, B_offset, B_t);  // B原来是K * N 的矩阵
	if (!check_errNum(errNum))
	{
		//clReleaseMemObject(B_t);
		//fprintf(stderr, "gemm_nt_ocl fail_transpose中执行矩阵转置失败\n");
		//return errNum;
	}
	errNum = mat_mul_ocl(command, M, N, K, ALPHA, A, A_offset, B_t, 0, BETA, C, C_offset);
	if (!check_errNum(errNum))
		fprintf(stderr, "gemm_nt_ocl 中的 mat_mul_ocl执行失败\n");
	clReleaseMemObject(B_t);
	return errNum;
}
cl_int gemm_tn_ocl(cl_command_queue command, int M, int N, int K, float ALPHA, cl_mem A, size_t A_offset, cl_mem B, size_t B_offset, float BETA, cl_mem C, size_t C_offset)
{
	cl_int errNum;
	cl_context context;
	errNum = clGetCommandQueueInfo(command,
		CL_QUEUE_CONTEXT,
		sizeof(cl_context),
		&context,
		NULL);
	if (!check_errNum(errNum))
	{
		fprintf(stderr, "gemm_tn_ocl中获取命令队列信息失败\n");
		return errNum;
	}
	cl_mem A_t = cl_make_array(context, NULL, N * M * sizeof(float));
	errNum = mat_transpose_ocl(command, N, M, A, A_offset, A_t);  // A原来是N * M 的矩阵
	if (!check_errNum(errNum))
	{
		clReleaseMemObject(A_t);
		fprintf(stderr, "gemm_tn_ocl中执行矩阵转置失败\n");
		return errNum;
	}
	errNum = mat_mul_ocl(command, M, N, K, ALPHA, A_t, 0, B, B_offset, BETA, C, C_offset);
	if (!check_errNum(errNum))
		fprintf(stderr, "gemm_tn_ocl 中的 mat_mul_ocl执行失败\n");
	clReleaseMemObject(A_t);
	return errNum;
}
cl_int gemm_tt_ocl(cl_command_queue command, int M, int N, int K, float ALPHA, cl_mem A, size_t A_offset, cl_mem B, size_t B_offset, float BETA, cl_mem C, size_t C_offset)
{
	cl_int errNum;
	cl_context context;
	errNum = clGetCommandQueueInfo(command,
		CL_QUEUE_CONTEXT,
		sizeof(cl_context),
		&context,
		NULL);
	if (!check_errNum(errNum))
	{
		fprintf(stderr, "gemm_tt_ocl中获取命令队列信息失败\n");
		return errNum;
	}
	cl_mem A_t = cl_make_array(context, NULL, N * M * sizeof(float));
	cl_mem B_t = cl_make_array(context, NULL, K * N * sizeof(float));
	errNum = mat_transpose_ocl(command, N, M, A, A_offset, A_t);  // A原来是N * M 的矩阵
	if (!check_errNum(errNum))
	{
		clReleaseMemObject(A_t);
		clReleaseMemObject(B_t);
		fprintf(stderr, "gemm_tt_ocl中执行A矩阵转置失败\n");
		return errNum;
	}
	errNum = mat_transpose_ocl(command, K, N, B, B_offset, B_t);  // B原来是K * N 的矩阵
	if (!check_errNum(errNum))
	{
		clReleaseMemObject(A_t);
		clReleaseMemObject(B_t);
		fprintf(stderr, "gemm_tt_ocl中执行B矩阵转置失败\n");
		return errNum;
	}
	errNum = mat_mul_ocl(command, M, N, K, ALPHA, A_t, 0, B_t, 0, BETA, C, C_offset);
	if (!check_errNum(errNum))
		fprintf(stderr, "gemm_tt_ocl 中的 mat_mul_ocl执行失败\n");
	clReleaseMemObject(A_t);
	clReleaseMemObject(B_t);
	return errNum;
}
cl_int gemm_ocl(cl_command_queue command, int TA, int TB, int M, int N, int K, float ALPHA,
	cl_mem A, size_t A_offset,
	cl_mem B, size_t B_offset,
	float BETA,
	cl_mem C, size_t C_offset)
{
	printf("M N K %d %d %d\n", M, N, K);
	cl_int errNum = 0;
	if (!TA  && !TB)
		errNum = gemm_nn_ocl(command, M, N, K, ALPHA, A, A_offset, B, B_offset, BETA, C, C_offset);
	else if (TA && !TB)
		errNum = gemm_tn_ocl(command, M, N, K, ALPHA, A, A_offset, B, B_offset, BETA, C, C_offset);
	else if (!TA && TB)
		errNum = gemm_nt_ocl(command, M, N, K, ALPHA, A, A_offset, B, B_offset, BETA, C, C_offset);
	else if (TA && TB)
		errNum = gemm_tt_ocl(command, M, N, K, ALPHA, A, A_offset, B, B_offset, BETA, C, C_offset);

	return errNum;
}
#endif
