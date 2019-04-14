// To compile: nvcc CPUAndGPUVectorAdditionClass.cu -o temp2
// To run: ./temp2
#include <sys/time.h>
#include <stdio.h>

//This is the CUDA kernel that will add the two vectors.
__global__ void Addition(unsigned char *A, unsigned char *B, unsigned char *C){
	unsigned long id = (blockIdx.x * blockDim.x) + threadIdx.x;
	C[id] = A[id] + B[id];
}

int main()
{
	int dev_cnt;
	long unsigned int max_thread_cnt, threads_per_block;
    long unsigned int input_cnt, id, sum;
	unsigned char *A_CPU, *B_CPU, *C_CPU; //Pointers for memory on the Host
	unsigned char *A_GPU, *B_GPU, *C_GPU; //Pointers for memory on the Device
	float time;
	dim3 dimBlock; //This variable will hold the Dimensions of your block
	dim3 dimGrid; //This variable will hold the Dimensions of your grid
	timeval start, end;

	cudaDeviceProp prop;
	cudaGetDeviceCount(&dev_cnt);

	max_thread_cnt = 0;
	for (int i=0; i<dev_cnt; i++){
		cudaGetDeviceProperties(&prop, i);
		threads_per_block = prop.maxThreadsPerBlock;
		printf("Threads per block for device %d:  %d\n", i, threads_per_block);

		long unsigned int blocks = prop.maxGridSize[0];
		for(int j=1; j<3; j++)
			blocks = (prop.maxGridSize[j] > blocks) ?
				prop.maxGridSize[j] : blocks;
        printf("Blocks on device %d: %d\n", i, blocks);

		max_thread_cnt += threads_per_block*blocks;
	}

	printf("\n");
	do { printf("Size of array (< %lu)? \n", max_thread_cnt); }
	while (scanf("%lu", &input_cnt) > max_thread_cnt);

	//Threads in a block
	dimBlock.x = (input_cnt < 1024) ? input_cnt:1024;
	dimBlock.y = 1;
	dimBlock.z = 1;

	//Blocks in a grid
	dimGrid.x = 1+(input_cnt-1)/threads_per_block;
	dimGrid.y = 1;
	dimGrid.z = 1;

	//Allocate Host (CPU) Memory
	A_CPU = (unsigned char*)malloc(input_cnt*sizeof(unsigned char));
	B_CPU = (unsigned char*)malloc(input_cnt*sizeof(unsigned char));
	C_CPU = (unsigned char*)malloc(input_cnt*sizeof(unsigned char));

	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,input_cnt*sizeof(unsigned char));
	cudaMalloc(&B_GPU,input_cnt*sizeof(unsigned char));
	cudaMalloc(&C_GPU,input_cnt*sizeof(unsigned char));


	//Loads values into vectors that we will add.
	for(id = 0; id < input_cnt; id++){
		A_CPU[id] = 1;
		B_CPU[id] = 2;
	}

	//********************** GPU addition start ****************************************
	//Starting a timer
	gettimeofday(&start, NULL);

	//Copying vectors A_CPU and B_CPU that were loaded on the CPU up to the GPU
	cudaMemcpyAsync(A_GPU, A_CPU, input_cnt*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(B_GPU, B_CPU, input_cnt*sizeof(unsigned char), cudaMemcpyHostToDevice);

	//Addition<<<1, input_cnt>>>(A_GPU, B_GPU, C_GPU);
    Addition<<<dimGrid, dimBlock>>>(A_GPU, B_GPU, C_GPU);

	//Copy C_GPU that was calculated on the GPU down to the CPU
	cudaMemcpyAsync(C_CPU, C_GPU, input_cnt*sizeof(unsigned char), cudaMemcpyDeviceToHost);


	//Stopping the timer
	gettimeofday(&end, NULL);
	//********************** GPU addition finish ****************************************

	//Calculating the total time used in the addition on the GPU and converting it to milliseconds and printing it to the screen.
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf("\n------ GPU Results ------\n");
	printf("GPU Time in milliseconds= %.15f\n", (time/1000.0));

	//Summing up the vector C and printing it so we can have a spot check for the correctness of the GPU addition.
	sum = 0;
	for(id = 0; id < input_cnt; id++)
        sum += C_CPU[id];

	printf("Sum of C_CPU from GPU addition= %lu\n", sum);

	//Your done so cleanup your mess.
	free(A_CPU); free(B_CPU); free(C_CPU);
	cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(C_GPU);

	return(0);
}
