
## [Device configuration](#intro)
A GPU's configuration can be determined by reading information from [`cudaDeviceProp`](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html). This object is accessible from a CUDA runtime running on a system with Nvidia GPU. 

The source code is [here](./cuda-device-profile.cu).

### [Instruction](#run)
To run the source code:
```
nvcc -o cuda-device-profile cuda-device-profile.cu
```
This should generate an excuble:
```
cuda-device-profile
```
in the same directory.

Now run the executable:

```
./cuda-device-profile
```
and the output should be similar to this:

```
Device count: 1
Name: NVIDIA GeForce RTX 4060 Laptop GPU
Compute capability: 8.9
##### MEMORY INFO FOR DEVICE 0 ---
Total global memory in bytes: 8328511488
Total constant memory in bytes: 65536
Max shared memory in byte a single block may use: 49152
Total 32-bits registers per block: 65536
Number of threads in a warp: 32
Max pitch allowed for memory copies in bytes: 2147483647
##### MP INFORMATION FOR DEVICE 0 ---
Multiprocessor count: 24
Max threads per block: 1024
Max threads dimensions: (1024, 1024, 64)
Max grid dimensions: (2147483647, 65535, 65535)
Total CUDA Cores: 3072
CUDA Cores per SM: 128
```

The os system is:

```
casablanca@casablanca-xps:~/Documents/projects/cuda/github/CUDA-examples$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 22.04.3 LTS
Release:        22.04
Codename:       jammy
```

### CUDA Compiler
The command to determine the compiler used for the example is `nvcc --version`.

Following is my output:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0
```

### Notes

A `cudaDeviceProp` object is defined by:

`cudaDeviceProp prop;`

A comprehensive list of attributes accessible by this object is [already mentioned](#intro). 

Some of important attributes are shown in this example:

1. Device info: `count`, `name`.
2. Max number of threads per block: `maxThreadsPerBlock`.
3. Max thread dimensions: `maxThreadsDim`.
4. Max grid dimensions: `maxGridSize`.


### Blocks and threads info
Information about blocks and threads are important for setting up how a function will be parallelized in GPU. 

In function invocation or kernel definition, we will see this notation:

```
<<<P, Q>>>
```
where  
P denotes number of blocks to run in parallel;  
Q denotes number of threads in each block.

The system in this example is from a laptop equipped with Nvidia RTX4060 GPU. Compute capability index for each GPU series is published [here](https://developer.nvidia.com/cuda-gpus). You may compare it to your GPU output.
