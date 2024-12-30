// #include "../../github/CUDA-by-Example-source-code-for-the-book-s-examples-/common/book.h"
#include <stdio.h>

int main(void) {
    cudaDeviceProp prop;

    int count;
    cudaGetDeviceCount( &count);
    printf("Device count: %d\n", count);
    for (int i=0; i<count; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);

        printf("##### MEMORY INFO FOR DEVICE %d ---\n", i);
        printf("Total global memory: %ld\n", prop.totalGlobalMem);
        printf("Total constant memory: %ld\n", prop.totalConstMem);
        printf("Max shared memory in byte a single block may use: %ld\n", prop.sharedMemPerBlock);
        printf("Total 32-bits registers per block: %d\n", prop.regsPerBlock);
        printf("Number of threads in a warp: %d\n", prop.warpSize);
        printf("Max pitch allowed for memory copies in bytes: %ld\n", prop.memPitch);     

        printf("##### MP INFORMATION FOR DEVICE %d ---\n", i);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max threads dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        

    }
}