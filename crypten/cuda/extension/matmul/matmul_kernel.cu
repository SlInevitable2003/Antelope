#include <cstdio>

#define BLOCK_SIZE 16
__global__ void matmulLong(long long* c, const long long* a, const long long* b, int m, int n, int k)
{
	//计算这个 thread 应该计算的 row 和 col
	const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	//显式声明共享内存a，b子矩阵块
	
	__shared__ long long shareA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ long long shareB[BLOCK_SIZE][BLOCK_SIZE];
	
	shareA[threadIdx.y][threadIdx.x] = 0;
	shareB[threadIdx.y][threadIdx.x] = 0;
	__syncthreads();
	
	long long t = 0;
	//计算矩阵乘法 Kahan’s Summation Formula
	for (int i = 0; i < (n - 1) / BLOCK_SIZE + 1; i++)
	{
		// load data from global memory to shared memory
		if(row * n + (i * BLOCK_SIZE + threadIdx.x) < m*n)
			shareA[threadIdx.y][threadIdx.x] = a[row * n + (i * BLOCK_SIZE + threadIdx.x)];

		if((i * BLOCK_SIZE + threadIdx.y) * k + col < n*k)
			shareB[threadIdx.y][threadIdx.x] = b[(i * BLOCK_SIZE + threadIdx.y) * k + col];
		// sync to wait for all threads in one block to finish loading datas
		__syncthreads();

	#pragma unroll
		for (int j = 0; j < BLOCK_SIZE; j++)
		{
			if(i * BLOCK_SIZE + j < n )
				t += shareA[threadIdx.y][j] * shareB[j][threadIdx.x];
		}
		// sync to wait for all threads in one block to finish compute
		__syncthreads();
	}
	// store results into global memory
	if (row < m && col < k)
		c[row * k + col] = t;
	
}

void launch_matmul(long long *C, const long long *A, const long long *B, int m, int n, int k)
{
	dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE,1);
	matmulLong<<<grid, blk>>>(C, A, B, m, n, k);
}