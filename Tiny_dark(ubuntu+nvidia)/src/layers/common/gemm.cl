#define BLOCK_SIZE 16
#ifndef SIMD_WORK_ITEMS
#define SIMD_WORK_ITEMS 4 // default value
#endif

__kernel 
__attribute((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
__attribute((num_simd_work_items(SIMD_WORK_ITEMS)))
void matrixMult( // Input and output matrices
                 __global float *restrict C,
                 __global float *restrict A,
                 __global float *restrict B, 
                 // Widths of matrices.
                 int A_width, int B_width, int A_height, unsigned long C_offset)
{
    __local float A_local[BLOCK_SIZE][BLOCK_SIZE];
    __local float B_local[BLOCK_SIZE][BLOCK_SIZE];

    int block_x = get_group_id(0);
    int block_y = get_group_id(1);

    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

	int global_x = get_global_id(0);
	int global_y = get_global_id(1);

    int a_start = A_width * BLOCK_SIZE * block_y;// A_weidth shi lieshu ,b*B shi qian mian hang shu
    int a_end   = a_start + A_width - 1;
    int b_start = BLOCK_SIZE * block_x;

    float running_sum = 0.0f;
	bool A_F = BLOCK_SIZE * block_y + local_y < A_height;
	bool B_F = b_start + local_x < B_width;
	int offset;

    for (int a = a_start, b = b_start; a <= a_end; a += BLOCK_SIZE, b += (BLOCK_SIZE * B_width))
    {
		offset = a - a_start;
		if(offset + local_x < A_width && A_F)
			A_local[local_y][local_x] = A[a + A_width * local_y + local_x];
		else
			A_local[local_y][local_x] = 0.0f;
		if(B_F && offset + local_y < A_width)
			B_local[local_x][local_y] = B[b + B_width * local_y + local_x];
		else
			B_local[local_x][local_y] = 0.0f;
	
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            running_sum += A_local[local_y][k] * B_local[local_x][k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
	if(global_x < B_width && global_y < A_height)
		C[global_y * B_width + global_x + C_offset] = running_sum;
}

__kernel void mat_transpose(int M, int N, __global float *restrict X, unsigned long offset, __global float *restrict Y)
{
    int i = get_global_id(1) * get_global_size(0) + get_global_id(0);
    int row = i/N;
    int col = i%N;
    if (i < N*M) {Y[col*M + row] = X[i];}
}
