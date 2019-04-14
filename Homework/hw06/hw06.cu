// To compile: nvcc hw06.cu -o hw06
#include <sys/time.h>
#include <stdio.h>

#define     N       100000
#define     FORMAT  "%f\n"
#define     TYPE    float

__global__ void dotProduct(TYPE *a, TYPE *b, TYPE *c){
    unsigned long id = (blockIdx.x*blockDim.x)+threadIdx.x;
    __shared__ TYPE cache[1024];
    cache[threadIdx.x] = 0;

    if(id > N){ return; }

    cache[threadIdx.x] = a[id]*b[id];
    __syncthreads();

    int i = blockDim.x/2;
    while(i != 0){
        if(threadIdx.x < i){ cache[threadIdx.x] += cache[threadIdx.x+i]; }
        __syncthreads();
        i /= 2;
    }

    if(threadIdx.x == 0){ atomicAdd(c, cache[0]); }
}


void CUDAErrorCheck(const char *message){
  cudaError_t  error;
  error = cudaGetLastError();

  if(error != cudaSuccess){
    printf("\n CUDA ERROR in: %s -> %s\n", message, cudaGetErrorString(error));
    exit(0);
  }
}

int main(){
	TYPE *A_CPU, *B_CPU, *C_CPU; //Pointers for memory on the Host
	long n = N;

	// Your variables start here.
	TYPE *A_GPU, *B_GPU, *C_GPU;
    dim3 gridDim, blockDim;
	// Your variables stop here.

	//Allocating and loading Host (CPU) Memory
	A_CPU = (TYPE*)malloc(n*sizeof(TYPE));
	B_CPU = (TYPE*)malloc(n*sizeof(TYPE));
    C_CPU = (TYPE*)malloc(sizeof(TYPE));
    *C_CPU = 0;
	for(int i = 0; i < n; i++) {
        A_CPU[i] = 2;
        B_CPU[i] = 1;
    }

	// Your code starts here.
    //gridDim.x = (n < 1024) ? n:1024;
    gridDim.x = 1+(n-1)/1024;
    gridDim.y = 1;
    gridDim.z = 1;

    //blockDim.x = 1+(n-1)/1024;
    blockDim.x = 1024;
    blockDim.y = 1;
    blockDim.z = 1;

	cudaMalloc(&A_GPU, n*sizeof(TYPE));
    CUDAErrorCheck("a cuda malloc...");
    cudaMalloc(&B_GPU, n*sizeof(TYPE));
    CUDAErrorCheck("b cuda malloc...");
    cudaMalloc(&C_GPU, sizeof(TYPE));
    CUDAErrorCheck("c cuda malloc...");

    cudaMemcpyAsync(A_GPU, A_CPU, n*sizeof(TYPE), cudaMemcpyHostToDevice);
    CUDAErrorCheck("a cuda memcpy from host...");
    cudaMemcpyAsync(B_GPU, B_CPU, n*sizeof(TYPE), cudaMemcpyHostToDevice);
    CUDAErrorCheck("b cuda memcpy from host...");
    cudaMemcpyAsync(C_GPU, C_CPU, sizeof(TYPE), cudaMemcpyHostToDevice);
    CUDAErrorCheck("c cuda memcpy from host...");

    free(A_CPU);
    free(B_CPU);

    dotProduct<<<gridDim, blockDim>>>(A_GPU, B_GPU, C_GPU);
    CUDAErrorCheck("kernel...");

    cudaMemcpyAsync(C_CPU, C_GPU, sizeof(TYPE), cudaMemcpyDeviceToHost);
    CUDAErrorCheck("c cuda memcpy from device..");

    cudaFree(A_GPU);
    CUDAErrorCheck("a cuda free...");
    cudaFree(B_GPU);
    CUDAErrorCheck("b cuda free...");
    cudaFree(C_GPU);
    CUDAErrorCheck("c cuda free...");

    printf(FORMAT, *C_CPU);
    free(C_CPU);

	return(0);
}
