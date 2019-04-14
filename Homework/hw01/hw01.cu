#include <iostream>

int main(void){
    cudaDeviceProp prop;

    int count;
    cudaGetDeviceCount(&count);
    printf("Number of devices: %d\n", count);
    printf("\n");
    for (int i=0; i<count; i++){
        cudaGetDeviceProperties(&prop, i);
        printf("Information for device %d\n", i);
        printf("=========================\n");
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %d\n", prop.clockRate);
        printf("Total global memory: %u\n", prop.totalGlobalMem);
        printf("Total constant memory: %u\n", prop.totalConstMem);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Number of 32-bit registers per block: %d\n", prop.regsPerBlock);
        printf("Max shared memory per block: %d\n", prop.sharedMemPerBlock);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        if(prop.integrated){ printf("Integrated GPU...\n"); }
        else{ printf("Discrete GPU...\n"); }
        printf("\n\n");
    }

    return 0;
}
