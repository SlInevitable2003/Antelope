#define BLOCK_SIZE 256
__global__ void encode1(int *a,int *r, int n){
	                            
	const int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	const int r_bit=index/n;       
	
	if(index<32*n){
		int tmp=(a[index]>>r_bit);
	
		if(tmp%2==1){
			r[index]=tmp;
		}
	}
	
}          
void launch_encode1(int *a, int *r, int n){
    dim3 block(BLOCK_SIZE,1);
	dim3 grid((32*n + BLOCK_SIZE - 1) / BLOCK_SIZE,1);
	encode1<<<grid, block>>>(a,r,n);

}