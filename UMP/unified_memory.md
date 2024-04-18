## Unified Memory Programming
It the simplest term, "Unified Memory Programming" (UMP), or simply "Unified Memory" (UM), refers to a memory space where all processors, GPU or CPU, may access as a coherent memory with a common address space. This means that as long as the data address is allocated in the memory, the data is accessible by either GPU or CPU, without the need for `cudaMemcpy` or `cudaMalloc`. To allocate unified memory, we will use `cudaMallocManaged` instead of `cudaMalloc`. 

The example source code is in [here.](unified_memory.cu)

In some cases, when your applications requires large data transfer between CPU and GPU, or that GPU requires frequent data access. An option to help with data transfer latency may be via `cudaMemPrefetchAsync` API. Under the hood, this API prefetch data from host memory and transfer it to the GPU memory. This API call has to take place before the kernel call.

The example source code for using `cudaMemPrefetchAsync` [is in here](unified_memory_prefetch.cu)

## Instruction

To run the source code, go to `UMP` directory, and first compile the source code into an executable:

```
nvcc -o unified_memory unified_memory.cu
```

Then in the same directory, run the executable:

```
./unified_memory
```

and expect output such as this:

```
Max error: 0
```

## Explanation
Here we take a look at element-wise addition as an example to demonstrate how to allocate unified memory for GPU and CPU. While GPU is responsible for executing the addition kernel, CPU is responsible for executing output display:

```
std::cout << "Max error: " << maxError << std::endl;
```

### Kernel
This is a very simple kernel for element-wise addition:

```
__global__ void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}
```

The two arrays, `y` array will be reused to hold the sum.

### Unified Memory
In this example, we are going to perform element-wise addition of two arrays, each array has approximately 1M floating point elements (2e20, or 1048576 to be exact).

So once we allocated the two pointers to access the array,

```
float *x, *y;
```

we may allocate the necesary unified memory:

```
cudaMallocManaged(&x, N*sizeof(float));
cudaMallocManaged(&y, N*sizeof(float));
```

With `cudaMallocManaged`, we have allocated coherent memory for both arrays. This memory is coherent and accessible by either GPU or CPU.

###  Kernel launch
To launch the kernel, we need to define block size and number of blocks for the kernel:

```
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;
add<<<numBlocks, blockSize>>>(N, x, y);
 ```
This kernel will execute the addition, and the results will stored in `y` array in the allocated memory. This kernel is executed as multiple parallel threads in a GPU. All threads has to finish before the runtime execute next line of the code. Therefore we need [`cudaDeviceSynchronize`](../Synchronize/sync.md) to hold the runtime until all threads are complete before executing next line of code. 

Once all GPU threads are complete, the next step is to check for errors and print maximum error:

```
std::cout << "Max error: " << maxError << std::endl;
```

This is executed by CPU, which can acess the same memory space as GPU. 

Finally, before the program exits, you need to free up the memory:

```
cudaFree(x);
cudaFree(y);
```

In the older CUDA, there is no UMP feature. To output results by CPU, you need to copy the results from GPU memory to host memory with such call:

```
cudaMemcpy(&y, dev_y, sizeof(float), cudaMemcpyDeviceToHost)
```

Since we have UMP in CUDA now, we no longer need to explicitly copy data between GPU and host.

### Prefetch
To try out if prefetching memory helps with a particular CUDA program, before the kernel call, insert the following code:

```
 int device = -1; // let runtime pick the GPU
 cudaGetDevice(&device); // Retrieves the current device index
 cudaMemPrefetchAsync(x, N*sizeof(float), device, NULL); // Prefetches data to the selected device
 cudaMemPrefetchAsync(y, N*sizeof(float), device, NULL); // Prefetches data to the same selected device
 ```

 By setting `device` to -1, you explicitly let CUDA runtime to determine the GPU to use for this application. 


## Reference
[CUDA Memory Management](https://developer.ridgerun.com/wiki/index.php/NVIDIA_CUDA_Memory_Management)