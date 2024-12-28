## Memory Allocation
This section demonstrates the result of memory allocation by `cudaMalloc` function. At a very general sense, this function allocates GPU's shared memory for object in the kernel execution. Here, we are going to take a look at a deeper level as to the memory allocation. Specifically, we want to see the memory address being allocated.

From a data structure perspective, an array is the most efficient data structure for `set` or `get` operation. This is because an elements in an array are allocated in a contiguous memory space. To do a `get` or `set` operation, one only need to know the address offset from the start of the array. To get the base address of the start of the array, one only need to assign a pointer to the array, and the offset may be calculated based on the index of interest, multiply by number of bytes allocated for each element pointed by the index. 

## Instruction

The source code is in `Memory-Allocation` directory.

To run the source code, go to `Memory-Allocation` directory:
```
nvcc -o allcation allocation.cu
```

Then in the same directory, run the executable:
```
allocation
```

and expect output such as this:
```
printAddresses<<<1, N>>>(d_data);
Thread 0: Address of d_data[0] = 0x70f569e00000, Value = 0
Thread 1: Address of d_data[1] = 0x70f569e00004, Value = 10
Thread 2: Address of d_data[2] = 0x70f569e00008, Value = 20
Thread 3: Address of d_data[3] = 0x70f569e0000c, Value = 30
Thread 4: Address of d_data[4] = 0x70f569e00010, Value = 40
Thread 5: Address of d_data[5] = 0x70f569e00014, Value = 50
Thread 6: Address of d_data[6] = 0x70f569e00018, Value = 60
Thread 7: Address of d_data[7] = 0x70f569e0001c, Value = 70
Thread 8: Address of d_data[8] = 0x70f569e00020, Value = 80
Thread 9: Address of d_data[9] = 0x70f569e00024, Value = 90
```



## Explanation

In this example program, we want to allocate GPU memory for an array of ten elements of integers. To verify it is done properly, we want to examine the memory address of each element in the array as it is allocated in the GPU memory. Since integer is four bytes, and that elements are placed in a contiguous memoryspace, we expect the address of each consecutive element should be incremented by 4 from the previous element.

Specifically, let's take a look at the program we use to demonstrate the memory allocation. 

1. We set the array size to 10, declare a pointer for CPU (host) to the integer array of the size we specified, and declare a pointer for GPU (device) to the integer array, which we will copy from the host to the device:

```
    int N = 10;
    int *h_data = new int[N];
    int *d_data;
```

2. We allocate the amount of GPU memory specified by the number of the elements and the respecitve size of each element. We specify `cudaMalloc` function to write the allocated device memory address into device pointer:

```
    cudaMalloc(&d_data, N * sizeof(int));
```

Here, we want the pointer `d_data` to specifically points to the address of the array allocated by `cudaMalloc`. Therefore the way this is done is by spefying a double pointer `&d_data` as the address of the device pointer.

3. Now create and array of ten integers in host:

```
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 10;
    }
```

4. Copy the array from host (CPU) to device (GPU). This is done by 

```
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
```

In this function, the array is copied from host to device, with specified size of the memory required, and the direction of copy operation. 

5. Print the address of each element in the array:
```
    printAddresses<<<1, N>>>(d_data);
```
This funtion takes device pointer as an input, which represents the base address of the array in GPU, and reference a particular index of the array, and print out the thread ID, address of the array elelment, and the corresponding value at each index of the array:

```
    printAddresses<<<1, N>>>(d_data);
```
Each element of the array is accessed by a thread in this case, so that we are sure that each element is accessed independently and there is no dependency of access. Below are example results:

```
printAddresses<<<1, N>>>(d_data);
Thread 0: Address of d_data[0] = 0x70f569e00000, Value = 0
Thread 1: Address of d_data[1] = 0x70f569e00004, Value = 10
Thread 2: Address of d_data[2] = 0x70f569e00008, Value = 20
Thread 3: Address of d_data[3] = 0x70f569e0000c, Value = 30
Thread 4: Address of d_data[4] = 0x70f569e00010, Value = 40
Thread 5: Address of d_data[5] = 0x70f569e00014, Value = 50
Thread 6: Address of d_data[6] = 0x70f569e00018, Value = 60
Thread 7: Address of d_data[7] = 0x70f569e0001c, Value = 70
Thread 8: Address of d_data[8] = 0x70f569e00020, Value = 80
Thread 9: Address of d_data[9] = 0x70f569e00024, Value = 90
```

Arrays are zero-indexed. Therefore the first element of the array is indexed by 0. The address of this element is the base address of the array. Subsequent array element address are incremented by 4, corresponding to four bytes of each element.

6. Finally, let's synchronize all the threads, and release the memory and clear the pointers before we exit this program
```
    cudaDeviceSynchronize(); // Wait for the kernel to finish
    cudaFree(d_data);
    delete[] h_data;
```
These functions effectively releases pointers and memory in the host as well as device. 

## Summary
From this example, we verify that when `cudaMalloc` allocates memory in the shared memory in the GPU, the memory spaces are contiguous, and that address offset corresponds to the memory space allocated per element. 