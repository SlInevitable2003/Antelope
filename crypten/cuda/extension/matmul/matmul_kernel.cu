#define BLOCK_SIZE 16
#define BM 64
#define BN 64
#define BK 4
__global__ void matmulLong22(long long* c, const long long* a, const long long* b, int m, int n, int k)
{
	//计算这个 thread 应该计算的 row 和 col
	const int col = blockIdx.x * BM;
	const int row = blockIdx.y * BM;

	const int tx=(threadIdx.y*BM+threadIdx.x)%16;
	const int ty=(threadIdx.y*BM+threadIdx.x)/16;
	//显式声明共享内存a，b子矩阵块
	__shared__ __align__(16*1024) long long shareA[BM][BK];
	__shared__ __align__(16*1024) long long shareB[BK][BN];
	
	long long reg_a[BK] = {0};
	long long reg_b[BK] = {0};
	long long reg_c[BK][BK] = {0};
	//计算矩阵乘法
	for (int i = 0; i < (n - 1) / BK + 1; i++)
	{
		// load data from global memory to shared memory
		if((row + threadIdx.x) * n + (i * BK + threadIdx.y) < m*n)
			shareA[threadIdx.y][threadIdx.x] = a[(row + threadIdx.x) * n + (i * BK + threadIdx.y)];

		if((i * BK + threadIdx.y) * k + threadIdx.x + col < n*k)
			shareB[threadIdx.y][threadIdx.x] = b[(i * BK + threadIdx.y) * k + threadIdx.x + col];

		//shareA[threadIdx.x][threadIdx.y] = a[(row + threadIdx.x) * n + (i * BK + threadIdx.y)];
		//shareB[threadIdx.y][threadIdx.x] = b[(i * BK + threadIdx.y) * k + threadIdx.x + col];
		// sync to wait for all threads in one block to finish loading datas
		__syncthreads();
		for(int j = 0; j < BK; j++){
			reg_a[0] = shareA[ty*BK][j];
			reg_a[1] = shareA[ty*BK+1][j];
			reg_a[2] = shareA[ty*BK+2][j];
			reg_a[3] = shareA[ty*BK+3][j];
			reg_b[0] = shareB[j][tx*BK];
			reg_b[1] = shareB[j][tx*BK+1];
			reg_b[2] = shareB[j][tx*BK+2];
			reg_b[3] = shareB[j][tx*BK+3];
			//cal
		#pragma unroll
			for(int z=0;z<BK;z++){
				#pragma unroll
				for(int x=0;x<BK;x++){
					reg_c[z][x]+=reg_a[z]*reg_b[x];
				}
			}
		}
		// sync to wait for all threads in one block to finish compute
		__syncthreads();
	}
	// store results into global memory
	for(int i=0;i<BK;i++){
		for(int j=0;j<BK;j++){
			if(row + ty*BK +i < m && col + tx*BK +j < k)
				c[(row + ty*BK +i)*k + col + tx*BK +j]=reg_c[i][j];
		}
	}
}
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
void launch_matmul(long long* c,
                 const long long* a,
                 const long long* b,
                 int m,
                 int n,
                 int k) {
	//dim3 block(BM, BK);
	//dim3 grid((k + BM - 1) / BM, (m + BM - 1) / BM,1);
	//matmulLong22<<<grid, block>>>(c, a, b, m, n, k);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE,1);
	matmulLong<<<grid, block>>>(c, a, b, m, n, k);
}