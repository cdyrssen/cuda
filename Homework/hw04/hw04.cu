// To compile: nvcc hw04.cu -o hw04
#include <sys/time.h>
#include <stdio.h>

#define N  1000000

__global__ void dotProduct(float *a, float *b, float *c){
    unsigned long id = (blockIdx.x*blockDim.x)+threadIdx.x;
    __shared__ float cache[1024];
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

    c[blockIdx.x] = cache[0];
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
	float *A_CPU, *B_CPU, *C_CPU; //Pointers for memory on the Host
	long n = N;

	// Your variables start here.
	float *A_GPU, *B_GPU, *C_GPU;
    dim3 gridDim, blockDim;
	// Your variables stop here.

	//Allocating and loading Host (CPU) Memory
	A_CPU = (float*)malloc(n*sizeof(float));
	B_CPU = (float*)malloc(n*sizeof(float));
	C_CPU = (float*)malloc((1+(n-1)/1024)*sizeof(float));
	for(int i = 0; i < n; i++) {
        A_CPU[i] = 2.5;
        B_CPU[i] = 1.0;
    }

	// Your code starts here.
    //gridDim.x = (n < 1024) ? n:1024;
    gridDim.x = 1+(n-1)/1024;
    gridDim.y = 1;
    gridDim.z = 1;

    //blockDim.x = 1+(n-1)/1024;
    blockDim.x = (n < 1024) ? n:1024;
    blockDim.y = 1;
    blockDim.z = 1;

	cudaMalloc(&A_GPU, n*sizeof(float));
    CUDAErrorCheck("a cuda malloc...");
    cudaMalloc(&B_GPU, n*sizeof(float));
    CUDAErrorCheck("b cuda malloc...");
    cudaMalloc(&C_GPU, gridDim.x*sizeof(float));
    CUDAErrorCheck("c cuda malloc...");

    cudaMemcpyAsync(A_GPU, A_CPU, n*sizeof(float), cudaMemcpyHostToDevice);
    CUDAErrorCheck("a cuda memcpy from host...");
    cudaMemcpyAsync(B_GPU, B_CPU, n*sizeof(float), cudaMemcpyHostToDevice);
    CUDAErrorCheck("b cuda memcpy from host...");
    cudaMemcpyAsync(C_GPU, C_CPU, gridDim.x*sizeof(float), cudaMemcpyHostToDevice);
    CUDAErrorCheck("c cuda memcpy from host...");

    free(A_CPU);
    free(B_CPU);

    dotProduct<<<gridDim, blockDim>>>(A_GPU, B_GPU, C_GPU);
    CUDAErrorCheck("kernel...");

    cudaMemcpyAsync(C_CPU, C_GPU, gridDim.x*sizeof(float), cudaMemcpyDeviceToHost);
    CUDAErrorCheck("c cuda memcpy from device..");

    cudaFree(A_GPU);
    CUDAErrorCheck("a cuda free...");
    cudaFree(B_GPU);
    CUDAErrorCheck("b cuda free...");
    cudaFree(C_GPU);
    CUDAErrorCheck("c cuda free...");

    float ans = 0;
    for(int i=0; i<gridDim.x; i++){ ans += C_CPU[i]; }

    free(C_CPU);

    printf("%f\n", ans);
	// Your code stops here.
	return(0);
}
