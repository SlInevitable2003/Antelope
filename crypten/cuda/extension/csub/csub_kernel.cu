#define BLOCK_SIZE 256
__global__ void csub(long long* a, const long long* b, int n, int rank)
{
	const int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	
	if (index < n){
		if (a[index]==0)
		{
			a[index]+=rank-b[index];
		}else{
			a[index]=b[index];
		}
	}

}

void launch_csub(long long* a,
                 const long long* b,
                 int n,
                 int rank) {
    dim3 block(BLOCK_SIZE);
	dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	csub<<<grid, block>>>(a, b, n, rank);
}