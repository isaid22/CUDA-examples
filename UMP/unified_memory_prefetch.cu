#include <iostream>
#include <math.h>
 
// CUDA kernel to add elements of two arrays
__global__ void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

 
int main(void)
{
 int N = 1<<20; // 1x2^20 = 1048576 bitwise left shift
 float *x, *y;
 
 // Allocate Unified Memory -- pointers accessible from CPU or GPU
 cudaMallocManaged(&x, N*sizeof(float));
 cudaMallocManaged(&y, N*sizeof(float));
 
 // initialize x and y arrays on the host (CPU)
 for (int i = 0; i < N; i++) {
   x[i] = 1.0f;
   y[i] = 2.0f;
 }
 
 // Launch kernel on 1M elements on the GPU
 int blockSize = 256;
 int numBlocks = (N + blockSize - 1) / blockSize;


 int device = -1; // let runtime pick the GPU
 cudaGetDevice(&device); // Retrieves the current device index
 cudaMemPrefetchAsync(x, N*sizeof(float), device, NULL); // Prefetches data to the selected device
 cudaMemPrefetchAsync(y, N*sizeof(float), device, NULL); // Prefetches data to the same selected device

 add<<<numBlocks, blockSize>>>(N, x, y);
 
 // Wait for GPU to finish before accessing on host**
 cudaDeviceSynchronize();
 
 // Check for errors (all values should be 3.0f)
 float maxError = 0.0f;
 for (int i = 0; i < N; i++)
   maxError = fmax(maxError, fabs(y[i]-3.0f));
 std::cout << "Max error: " << maxError << std::endl;
 
 // Free memory
 cudaFree(x);
 cudaFree(y);
 
 return 0;
}