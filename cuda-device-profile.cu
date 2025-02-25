// #include "../../github/CUDA-by-Example-source-code-for-the-book-s-examples-/common/book.h"
#include <stdio.h>

int main(void) {
    cudaDeviceProp prop;

    int count;
    int total_CUDA_cores = 3072; // https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/
    cudaGetDeviceCount( &count);
    printf("Device count: %d\n", count);
    for (int i=0; i<count; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);

        printf("##### MEMORY INFO FOR DEVICE %d ---\n", i);
        printf("Total global memory in bytes: %ld\n", prop.totalGlobalMem);
        printf("Total constant memory in bytes: %ld\n", prop.totalConstMem);
        printf("Max shared memory in byte a single block may use: %ld\n", prop.sharedMemPerBlock);
        printf("Total 32-bits registers per block: %d\n", prop.regsPerBlock);
        printf("Number of threads in a warp: %d\n", prop.warpSize);
        printf("Max pitch allowed for memory copies in bytes: %ld\n", prop.memPitch);     
        if (prop.deviceOverlap) {
            printf("Device handles CUDA streaming overlap and speedup. \n");
        } else {
            printf("Device does not handle overlap, no CUDA streams capabilities. \n");
        };
        if (prop.concurrentKernels) {
            printf("Device can execute multiple kernels within the same context simultaneously. \n");
        } else {
            printf("Device cannot execute multiple kernels within the same context simultaneously. \n");
        }
        if (prop.integrated) {
            printf("Device i an integrated GPU and not standalone. \n");
        } else {
            printf("Device is a discrete GPU. \n");
        }
        printf("##### MP INFORMATION FOR DEVICE %d ---\n", i);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max threads dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Total CUDA Cores: %d\n", total_CUDA_cores);
        printf("CUDA Cores per SM: %d\n", total_CUDA_cores / prop.multiProcessorCount);
        

    

    }
}