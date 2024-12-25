## [Run a function in GPU](#intro)
In CUDA programming, it is necessary to specify which part of your code runs in CPU (host), and which parts runs in 
GPU (device). Therefore it is a good idea to encapsulate your code for GPU in a function. This function is also known as a "kernel". In a typical CUDA program, host code makes a kernel call to the function, and this function will be executed in the device. Once it's done, the results will be copied from the device to the host.

## Explanation
In order for a function to execute on the device, it needs to be labeled with a qualifier `__global__`:

```
__global__ void gpufunc () {
    .....
}
```

and in the host code, where such function is invoked:

```
gpufunc<<<p, q>>>()
```
Such function or kernell call needs to have `<<<p, q>>>` in the call, where

p is number of blocks\
q is number of threads

Blocks are the basic units that execute in parallel in a GPU program.

The source code for this exercise is [here](./kernel-demo.cu).

**Note** 
`__global__` is often confused with another qualifier `__device__`. In either case, such qualifier designated the function fo be executed in GPU, and therefore the fonctions are kernel functions. However, `__global__` function can only be called from the CPU. This means that typically, there is a `main` program that will call and execute the `__global__` function. 

In contrast, `__device__` qualifier designates the function to be called from another GPU function. This means that a `__device__` funciton cannot be called by a `main` program or any program running on the CPU host, rather, only another GPU function, either `__global__` or `__device__` function, can call the `__device__` function.
 

## Instruction

To run the source code, go to `kernel` directory:

```
nvcc -o kernel-demo kernel-demo.cu
```

Then in the same directory, run the executable:

```
./kernel-demo
```

and expect output such as this:

```
...
-9997 + 99940009 = 99930012
-9998 + 99960004 = 99950006
-9999 + 99980001 = 99970002
```


