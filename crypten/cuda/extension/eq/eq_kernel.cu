#define BLOCK_SIZE 256
__global__ void eq(int *a, int n){
	                            
	const int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;     
	
	if(index<32*n){

		if(a[index]==0){
			a[index]=1;
		}else{
			a[index]=0;
		}
	}
	
}          
void launch_eq(int *a, int n){
    dim3 block(BLOCK_SIZE,1);
	dim3 grid((32*n + BLOCK_SIZE - 1) / BLOCK_SIZE,1);
	eq<<<grid, block>>>(a,n);

}