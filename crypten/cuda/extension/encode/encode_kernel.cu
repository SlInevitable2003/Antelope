#define BLOCK_SIZE 256
__global__ void encode(int *a,int *r, int n){
	                            
	const int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	const int r_bit=index/n;       
	
	if(index<32*n){
		int tmp=(a[index]>>r_bit);
	
		if(tmp%2==0){
			r[index]=tmp+1;
		}
	}
	
}          
void launch_encode(int *a, int *r, int n){
    dim3 block(BLOCK_SIZE,1);
	dim3 grid((32*n + BLOCK_SIZE - 1) / BLOCK_SIZE,1);
	encode<<<grid, block>>>(a,r,n);

}