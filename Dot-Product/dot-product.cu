#include "../common/book.h"
#include <iostream> 
using namespace std;



#define imin(a,b) (a<b?a:b) // macro and placeholders a and b. 


const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock);

__global__ void dot( float *a, float *b, float *c) {

    // declare shared memory for each thread in block to store temporary results
    __shared__ float cache[threadsPerBlock];

    // compute data indices
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;

    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    //set cache values
    cache[cacheIndex] = temp; 

    // Synchronize threads in this block
    __syncthreads();

    // For reduction, now each thread adds two values in cache and store the sum in cache.
    // So threadsPerBlock must be power of 2
    int i = blockDim.x/2;
    while (i != 0 ){
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i]; // add thread from next block
        __syncthreads();
        i /= 2; // cache occupied is now half of what it started, so block increment i is also halfed.
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0]; // at last round, assign sum to array c.

}


int main(void) {
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    //allocate memory in CPU
    a = (float*)malloc(N*sizeof(float));
    b = (float*)malloc(N*sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid*sizeof(float));

    //allocate memory for GPU
    HANDLE_ERROR( cudaMalloc((void**)&dev_a, N*sizeof(float)));
    HANDLE_ERROR( cudaMalloc((void**)&dev_b, N*sizeof(float)));
    HANDLE_ERROR( cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float)));

    // fill in host memory with data
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i*2;
    }
    
    // copy array a and b to GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);
    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
    // Record the stop event
    cudaEventRecord(stop);
    // Synchronize to make sure the events have completed
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time taken by the kernel: " << milliseconds << " ms" << std::endl;

    // copy array c from GPU to CPU
    HANDLE_ERROR( cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost));

    // finish CPU
    c = 0;
    for(int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
    }

    #define sum_squares(x) (x*(x+1)*(2*x+1)/6)
    printf("Does GPU value %.6g = %.6g?\n", c, 2*sum_squares((float)(N-1)));
    
    // free memory on GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    // Free memory on CPU
    free(a);
    free(b);
    free(partial_c);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

